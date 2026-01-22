"""
Tile-based deconvolution for spatially-varying PSFs.

Implements deconvolution using different PSFs for different regions of the
image, with smooth blending at tile boundaries.
"""

import numpy as np
from pathlib import Path
import re

from .pd_joint_deconv import pd_joint_deconv
from .utils import img_to_norm_grayscale


def load_tiled_psfs(pattern_or_dir, n_tiles_h=None, n_tiles_w=None):
    """
    Load tiled PSFs from files.

    Parameters
    ----------
    pattern_or_dir : str or Path
        Either a directory containing PSF files named like 'psf_tile_0_0.png',
        or a file pattern like 'psf_tile_{row}_{col}.png'
    n_tiles_h : int, optional
        Number of tiles vertically. Auto-detected if not specified.
    n_tiles_w : int, optional
        Number of tiles horizontally. Auto-detected if not specified.

    Returns
    -------
    psfs : list of ndarray
        List of PSFs in row-major order
    tile_grid : tuple
        (n_tiles_h, n_tiles_w)
    """
    import imageio.v3 as iio

    pattern_or_dir = Path(pattern_or_dir)

    if pattern_or_dir.is_dir():
        # Find all tile PSF files in directory
        tile_files = list(pattern_or_dir.glob('*_tile_*_*.png'))
        if not tile_files:
            tile_files = list(pattern_or_dir.glob('*_tile_*_*.tif'))
        if not tile_files:
            raise ValueError(f"No tile PSF files found in {pattern_or_dir}")
    else:
        # Treat as a base path pattern
        base_dir = pattern_or_dir.parent
        base_name = pattern_or_dir.stem
        tile_files = list(base_dir.glob(f'{base_name}_tile_*_*.png'))
        if not tile_files:
            tile_files = list(base_dir.glob(f'{base_name}_tile_*_*.tif'))

    # Parse tile coordinates from filenames
    tile_pattern = re.compile(r'_tile_(\d+)_(\d+)\.')
    tiles = {}
    max_row, max_col = 0, 0

    for f in tile_files:
        match = tile_pattern.search(str(f))
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            tiles[(row, col)] = f
            max_row = max(max_row, row)
            max_col = max(max_col, col)

    if not tiles:
        raise ValueError(f"Could not parse tile coordinates from filenames")

    # Determine grid size
    detected_h = max_row + 1
    detected_w = max_col + 1

    if n_tiles_h is None:
        n_tiles_h = detected_h
    if n_tiles_w is None:
        n_tiles_w = detected_w

    # Load PSFs in row-major order
    psfs = []
    for row in range(n_tiles_h):
        for col in range(n_tiles_w):
            if (row, col) not in tiles:
                raise ValueError(f"Missing tile PSF for position ({row}, {col})")

            psf = iio.imread(tiles[(row, col)])
            psf = img_to_norm_grayscale(psf)
            psf = np.maximum(psf, 0)
            psf = psf / (psf.sum() + 1e-10)
            psfs.append(psf)

    return psfs, (n_tiles_h, n_tiles_w)


def deconvolve_tiled(image, psfs, tile_grid, overlap=0.25,
                     lambda_residual=200, lambda_tv=2.0, lambda_cross=3.0,
                     max_iterations=200, tolerance=1e-4, verbose=False):
    """
    Deconvolve an image using spatially-varying PSFs.

    Divides the image into tiles, deconvolves each with its corresponding
    PSF, and blends the results using weighted averaging in overlap regions.

    Parameters
    ----------
    image : ndarray
        Input image (H, W) for grayscale or (H, W, C) for color
    psfs : list of ndarray
        List of PSFs in row-major order, one per tile
    tile_grid : tuple
        (n_tiles_h, n_tiles_w) specifying the tile grid dimensions
    overlap : float
        Fraction of tile size to overlap with neighbors (0-0.5, default: 0.25)
    lambda_residual : float
        Data fidelity weight
    lambda_tv : float
        Total variation weight
    lambda_cross : float
        Cross-channel coupling weight
    max_iterations : int
        Maximum iterations per tile/channel
    tolerance : float
        Convergence tolerance
    verbose : bool
        Print progress information

    Returns
    -------
    result : ndarray
        Deconvolved image
    """
    image = np.asarray(image, dtype=np.float64)
    n_tiles_h, n_tiles_w = tile_grid

    if len(psfs) != n_tiles_h * n_tiles_w:
        raise ValueError(f"Number of PSFs ({len(psfs)}) must match "
                        f"tile grid ({n_tiles_h}x{n_tiles_w}={n_tiles_h*n_tiles_w})")

    # Handle grayscale vs color
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    h, w, n_channels = image.shape

    # Calculate tile dimensions
    base_tile_h = h // n_tiles_h
    base_tile_w = w // n_tiles_w
    overlap_h = int(base_tile_h * overlap)
    overlap_w = int(base_tile_w * overlap)

    if verbose:
        print(f"Tile-based deconvolution")
        print(f"  Image: {w}x{h}, {n_channels} channel(s)")
        print(f"  Tiles: {n_tiles_w}x{n_tiles_h}")
        print(f"  Tile size: ~{base_tile_w}x{base_tile_h} (with {int(overlap*100)}% overlap)")

    # Output accumulator and weight accumulator for blending
    output = np.zeros_like(image)
    weights = np.zeros((h, w), dtype=np.float64)

    # Process each tile
    for tile_idx in range(n_tiles_h * n_tiles_w):
        tile_row = tile_idx // n_tiles_w
        tile_col = tile_idx % n_tiles_w
        psf = psfs[tile_idx]

        # Calculate tile boundaries with overlap
        y_start = max(0, tile_row * base_tile_h - overlap_h)
        y_end = min(h, (tile_row + 1) * base_tile_h + overlap_h)
        x_start = max(0, tile_col * base_tile_w - overlap_w)
        x_end = min(w, (tile_col + 1) * base_tile_w + overlap_w)

        tile_h = y_end - y_start
        tile_w = x_end - x_start

        if verbose:
            print(f"\n--- Tile ({tile_row}, {tile_col}) ---")
            print(f"  Region: [{y_start}:{y_end}, {x_start}:{x_end}]")

        # Extract tile from image
        tile_image = image[y_start:y_end, x_start:x_end, :]

        # Deconvolve tile
        tile_result = _deconvolve_single(
            tile_image, psf, n_channels,
            lambda_residual, lambda_tv, lambda_cross,
            max_iterations, tolerance, verbose
        )

        # Create blending weights (cosine taper at edges that overlap with neighbors)
        tile_weight = _create_blend_weights(
            tile_h, tile_w,
            y_start == 0, y_end == h,  # is_top, is_bottom
            x_start == 0, x_end == w,  # is_left, is_right
            overlap_h, overlap_w
        )

        # Accumulate weighted results
        for ch in range(n_channels):
            output[y_start:y_end, x_start:x_end, ch] += tile_result[:, :, ch] * tile_weight
        weights[y_start:y_end, x_start:x_end] += tile_weight

    # Normalize by accumulated weights
    for ch in range(n_channels):
        output[:, :, ch] /= np.maximum(weights, 1e-10)

    # Remove extra dimension for grayscale
    if output.shape[2] == 1:
        output = output[:, :, 0]

    return output


def _deconvolve_single(image, kernel, n_channels, lambda_residual, lambda_tv,
                       lambda_cross, max_iterations, tolerance, verbose):
    """Deconvolve a single tile."""
    # Prepare channel data
    channels = []
    for ch in range(n_channels):
        channels.append({
            'image': image[:, :, ch],
            'kernel': kernel
        })

    # Build lambda parameters
    lambda_params = []
    for ch in range(n_channels):
        row = [ch + 1, lambda_residual, lambda_tv, 0.0]
        cross_weights = [0.0] * n_channels
        if ch > 0 and n_channels > 1:
            cross_weights[0] = lambda_cross
        row.extend(cross_weights)
        row.append(1 if ch == 0 else 0)
        lambda_params.append(row)

    lambda_params = np.array(lambda_params)

    # Run deconvolution
    verbose_mode = 'brief' if verbose else 'none'
    result = pd_joint_deconv(channels, lambda_params,
                             max_it=max_iterations, tol=tolerance,
                             verbose=verbose_mode)

    # Gather results
    output = np.zeros_like(image)
    for ch in range(n_channels):
        output[:, :, ch] = result[ch]['image']

    return output


def _create_blend_weights(tile_h, tile_w, is_top, is_bottom, is_left, is_right,
                          overlap_h, overlap_w):
    """
    Create blending weights for a tile using cosine tapering.

    Edges that border other tiles get a smooth taper; edges at image
    boundaries get full weight.
    """
    weight = np.ones((tile_h, tile_w), dtype=np.float64)

    # Create 1D taper functions
    def cosine_taper(length):
        """Cosine taper from 0 to 1 over given length."""
        if length <= 0:
            return np.array([])
        t = np.linspace(0, np.pi / 2, length)
        return np.sin(t) ** 2

    # Apply vertical tapers
    if not is_top and overlap_h > 0:
        taper = cosine_taper(min(overlap_h, tile_h))
        weight[:len(taper), :] *= taper[:, np.newaxis]

    if not is_bottom and overlap_h > 0:
        taper = cosine_taper(min(overlap_h, tile_h))
        weight[-len(taper):, :] *= taper[::-1, np.newaxis]

    # Apply horizontal tapers
    if not is_left and overlap_w > 0:
        taper = cosine_taper(min(overlap_w, tile_w))
        weight[:, :len(taper)] *= taper[np.newaxis, :]

    if not is_right and overlap_w > 0:
        taper = cosine_taper(min(overlap_w, tile_w))
        weight[:, -len(taper):] *= taper[np.newaxis, ::-1]

    return weight
