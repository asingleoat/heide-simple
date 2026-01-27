"""
Tile-based deconvolution for spatially-varying PSFs.

Implements deconvolution using different PSFs for different regions of the
image, with smooth blending at tile boundaries.
"""

import numpy as np
from pathlib import Path
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .pd_joint_deconv import pd_joint_deconv
from .utils import img_to_norm_grayscale
from .tracing import trace


def _process_tile_worker(args):
    """
    Worker function to process a single tile (must be module-level for pickling).

    Returns the tile result along with metadata needed for blending.
    """
    (tile_idx, tile_image, psf, n_channels, tile_row, tile_col,
     y_start, y_end, x_start, x_end, h, w,
     overlap_h, overlap_w,
     lambda_residual, lambda_tv, lambda_cross,
     max_iterations, tolerance) = args

    # Deconvolve tile
    tile_result = _deconvolve_single(
        tile_image, psf, n_channels,
        lambda_residual, lambda_tv, lambda_cross,
        max_iterations, tolerance, verbose=False
    )

    # Create blending weights
    tile_h = y_end - y_start
    tile_w = x_end - x_start
    tile_weight = _create_blend_weights(
        tile_h, tile_w,
        y_start == 0, y_end == h,
        x_start == 0, x_end == w,
        overlap_h, overlap_w
    )

    return {
        'tile_idx': tile_idx,
        'tile_row': tile_row,
        'tile_col': tile_col,
        'result': tile_result,
        'weight': tile_weight,
        'y_start': y_start,
        'y_end': y_end,
        'x_start': x_start,
        'x_end': x_end,
    }


def load_tiled_psfs(pattern_or_dir, n_tiles_h=None, n_tiles_w=None, per_channel=True):
    """
    Load tiled PSFs from files.

    Parameters
    ----------
    pattern_or_dir : str or Path
        Either a directory containing PSF files named like 'psf_tile_0_0.png',
        or a file pattern like 'psf_tile_{row}_{col}.png'. For per-channel PSFs,
        expects files like 'psf_red_tile_0_0.png', 'psf_green_tile_0_0.png', etc.
    n_tiles_h : int, optional
        Number of tiles vertically. Auto-detected if not specified.
    n_tiles_w : int, optional
        Number of tiles horizontally. Auto-detected if not specified.
    per_channel : bool
        If True (default), try to load per-channel PSFs first, falling back to
        grayscale. If False, only load grayscale PSFs.

    Returns
    -------
    psfs : list of ndarray or dict
        If grayscale: List of PSFs in row-major order.
        If per-channel: Dict with 'red', 'green', 'blue' keys, each containing
        a list of PSFs in row-major order.
    tile_grid : tuple
        (n_tiles_h, n_tiles_w)
    is_per_channel : bool
        Whether per-channel PSFs were loaded.
    """
    import imageio.v3 as iio

    pattern_or_dir = Path(pattern_or_dir)
    channel_names = ['red', 'green', 'blue']

    # Determine base directory and name
    if pattern_or_dir.is_dir():
        base_dir = pattern_or_dir
        base_name = None
    else:
        base_dir = pattern_or_dir.parent
        base_name = pattern_or_dir.stem

    # Try to find per-channel PSFs first if requested
    per_channel_tiles = {}
    if per_channel:
        for ch_name in channel_names:
            if base_name:
                pattern = f'{base_name}_{ch_name}_tile_*_*.png'
            else:
                pattern = f'*_{ch_name}_tile_*_*.png'

            tile_files = list(base_dir.glob(pattern))
            if not tile_files:
                pattern_tif = pattern.replace('.png', '.tif')
                tile_files = list(base_dir.glob(pattern_tif))

            if tile_files:
                per_channel_tiles[ch_name] = tile_files

    # Check if we have per-channel PSFs for all channels
    has_per_channel = len(per_channel_tiles) == 3

    if has_per_channel:
        # Load per-channel PSFs
        result = {}
        detected_h, detected_w = 0, 0

        for ch_name in channel_names:
            tile_pattern = re.compile(rf'_{ch_name}_tile_(\d+)_(\d+)\.')
            tiles = {}
            max_row, max_col = 0, 0

            for f in per_channel_tiles[ch_name]:
                match = tile_pattern.search(str(f))
                if match:
                    row, col = int(match.group(1)), int(match.group(2))
                    tiles[(row, col)] = f
                    max_row = max(max_row, row)
                    max_col = max(max_col, col)

            if not tiles:
                raise ValueError(f"Could not parse tile coordinates from {ch_name} PSF filenames")

            detected_h = max(detected_h, max_row + 1)
            detected_w = max(detected_w, max_col + 1)

            # Store for later loading
            result[ch_name] = tiles

        # Determine grid size
        if n_tiles_h is None:
            n_tiles_h = detected_h
        if n_tiles_w is None:
            n_tiles_w = detected_w

        # Load PSFs for each channel
        for ch_name in channel_names:
            tiles = result[ch_name]
            psfs = []
            for row in range(n_tiles_h):
                for col in range(n_tiles_w):
                    if (row, col) not in tiles:
                        raise ValueError(f"Missing {ch_name} tile PSF for position ({row}, {col})")

                    psf = iio.imread(tiles[(row, col)])
                    psf = img_to_norm_grayscale(psf)
                    psf = np.maximum(psf, 0)
                    psf = psf / (psf.sum() + 1e-10)
                    psfs.append(psf)
            result[ch_name] = psfs

        return result, (n_tiles_h, n_tiles_w), True

    else:
        # Fall back to grayscale PSF loading
        if base_name:
            tile_files = list(base_dir.glob(f'{base_name}_tile_*_*.png'))
            if not tile_files:
                tile_files = list(base_dir.glob(f'{base_name}_tile_*_*.tif'))
        else:
            # Find files that are NOT per-channel (don't have _red_, _green_, _blue_ before _tile_)
            all_files = list(base_dir.glob('*_tile_*_*.png'))
            if not all_files:
                all_files = list(base_dir.glob('*_tile_*_*.tif'))
            # Filter out per-channel files
            tile_files = [f for f in all_files
                         if not any(f'_{ch}_tile_' in str(f) for ch in channel_names)]

        if not tile_files:
            raise ValueError(f"No tile PSF files found in {base_dir}")

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

        return psfs, (n_tiles_h, n_tiles_w), False


def deconvolve_tiled(image, psfs, tile_grid, overlap=0.25,
                     lambda_residual=200, lambda_tv=2.0, lambda_cross=3.0,
                     max_iterations=200, tolerance=1e-4, n_workers=1,
                     verbose=False):
    """
    Deconvolve an image using spatially-varying PSFs.

    Divides the image into tiles, deconvolves each with its corresponding
    PSF, and blends the results using weighted averaging in overlap regions.

    Parameters
    ----------
    image : ndarray
        Input image (H, W) for grayscale or (H, W, C) for color
    psfs : list of ndarray or dict
        If list: PSFs in row-major order, one per tile (same PSF for all channels).
        If dict: Keys are 'red', 'green', 'blue' and values are lists of PSFs
        in row-major order (per-channel PSFs).
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
    n_workers : int
        Number of parallel workers. 1 = sequential, 0 = auto (use all CPUs),
        >1 = use specified number of workers. Default: 1
    verbose : bool
        Print progress information

    Returns
    -------
    result : ndarray
        Deconvolved image
    """
    image = np.asarray(image, dtype=np.float64)
    n_tiles_h, n_tiles_w = tile_grid
    n_tiles = n_tiles_h * n_tiles_w

    # Determine if we have per-channel PSFs
    is_per_channel = isinstance(psfs, dict)

    if is_per_channel:
        # Validate per-channel PSFs
        channel_names = ['red', 'green', 'blue']
        for ch_name in channel_names:
            if ch_name not in psfs:
                raise ValueError(f"Missing '{ch_name}' key in per-channel PSFs dict")
            if len(psfs[ch_name]) != n_tiles:
                raise ValueError(f"Number of {ch_name} PSFs ({len(psfs[ch_name])}) must match "
                                f"tile grid ({n_tiles_h}x{n_tiles_w}={n_tiles})")
    else:
        # Validate grayscale PSFs
        if len(psfs) != n_tiles:
            raise ValueError(f"Number of PSFs ({len(psfs)}) must match "
                            f"tile grid ({n_tiles_h}x{n_tiles_w}={n_tiles})")

    # Handle grayscale vs color
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    h, w, n_channels = image.shape

    # Calculate tile dimensions
    base_tile_h = h // n_tiles_h
    base_tile_w = w // n_tiles_w
    overlap_h = int(base_tile_h * overlap)
    overlap_w = int(base_tile_w * overlap)

    # Determine number of workers
    if n_workers == 0:
        n_workers = os.cpu_count() or 1
    use_parallel = n_workers > 1 and n_tiles > 1

    if verbose:
        print(f"Tile-based deconvolution")
        print(f"  Image: {w}x{h}, {n_channels} channel(s)")
        print(f"  Tiles: {n_tiles_w}x{n_tiles_h}")
        print(f"  Tile size: ~{base_tile_w}x{base_tile_h} (with {int(overlap*100)}% overlap)")
        print(f"  PSFs: {'per-channel' if is_per_channel else 'grayscale'}")
        if use_parallel:
            print(f"  Workers: {n_workers} (parallel)")
        else:
            print(f"  Workers: 1 (sequential)")

    # Output accumulator and weight accumulator for blending
    output = np.zeros_like(image)
    weights = np.zeros((h, w), dtype=np.float64)

    # Prepare tile arguments
    tile_args = []
    channel_names = ['red', 'green', 'blue']
    for tile_idx in range(n_tiles):
        tile_row = tile_idx // n_tiles_w
        tile_col = tile_idx % n_tiles_w

        # Get PSF(s) for this tile
        if is_per_channel:
            # Build list of PSFs for each channel
            tile_psfs = [psfs[channel_names[ch]][tile_idx]
                        for ch in range(min(n_channels, 3))]
            # If more than 3 channels, use the last PSF for remaining
            while len(tile_psfs) < n_channels:
                tile_psfs.append(tile_psfs[-1])
            psf = tile_psfs
        else:
            psf = psfs[tile_idx]

        # Calculate tile boundaries with overlap
        y_start = max(0, tile_row * base_tile_h - overlap_h)
        y_end = min(h, (tile_row + 1) * base_tile_h + overlap_h)
        x_start = max(0, tile_col * base_tile_w - overlap_w)
        x_end = min(w, (tile_col + 1) * base_tile_w + overlap_w)

        # Extract tile image (copy for multiprocessing)
        tile_image = image[y_start:y_end, x_start:x_end, :].copy()

        tile_args.append((
            tile_idx, tile_image, psf, n_channels, tile_row, tile_col,
            y_start, y_end, x_start, x_end, h, w,
            overlap_h, overlap_w,
            lambda_residual, lambda_tv, lambda_cross,
            max_iterations, tolerance
        ))

    if use_parallel:
        # Parallel processing using threads
        # NumPy/SciPy release the GIL during heavy computation, so threading works well
        with trace("parallel_tiles"):
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_tile_worker, args): args[0]
                          for args in tile_args}

                for future in as_completed(futures):
                    tile_result = future.result()
                    tile_row = tile_result['tile_row']
                    tile_col = tile_result['tile_col']

                    if verbose:
                        print(f"  Completed tile ({tile_row}, {tile_col})")

                    # Accumulate
                    y_start = tile_result['y_start']
                    y_end = tile_result['y_end']
                    x_start = tile_result['x_start']
                    x_end = tile_result['x_end']
                    tile_res = tile_result['result']
                    tile_weight = tile_result['weight']

                    for ch in range(n_channels):
                        output[y_start:y_end, x_start:x_end, ch] += tile_res[:, :, ch] * tile_weight
                    weights[y_start:y_end, x_start:x_end] += tile_weight
    else:
        # Sequential processing (original behavior)
        for args in tile_args:
            tile_idx = args[0]
            tile_row = args[4]
            tile_col = args[5]

            if verbose:
                y_start, y_end = args[6], args[7]
                x_start, x_end = args[8], args[9]
                print(f"\n--- Tile ({tile_row}, {tile_col}) ---")
                print(f"  Region: [{y_start}:{y_end}, {x_start}:{x_end}]")

            with trace(f"deconv_tile_{tile_row}_{tile_col}"):
                tile_result = _process_tile_worker(args)

            # Accumulate
            y_start = tile_result['y_start']
            y_end = tile_result['y_end']
            x_start = tile_result['x_start']
            x_end = tile_result['x_end']
            tile_res = tile_result['result']
            tile_weight = tile_result['weight']

            with trace("accumulate"):
                for ch in range(n_channels):
                    output[y_start:y_end, x_start:x_end, ch] += tile_res[:, :, ch] * tile_weight
                weights[y_start:y_end, x_start:x_end] += tile_weight

    # Normalize by accumulated weights
    with trace("normalize"):
        for ch in range(n_channels):
            output[:, :, ch] /= np.maximum(weights, 1e-10)

    # Remove extra dimension for grayscale
    if output.shape[2] == 1:
        output = output[:, :, 0]

    return output


def _deconvolve_single(image, kernels, n_channels, lambda_residual, lambda_tv,
                       lambda_cross, max_iterations, tolerance, verbose):
    """Deconvolve a single tile.

    Parameters
    ----------
    image : ndarray
        Tile image (H, W, C)
    kernels : ndarray or list of ndarray
        Either a single kernel (used for all channels) or a list of kernels
        (one per channel).
    """
    # Handle single kernel vs per-channel kernels
    if isinstance(kernels, np.ndarray) and kernels.ndim == 2:
        kernels = [kernels] * n_channels
    elif len(kernels) != n_channels:
        raise ValueError(f"Number of kernels ({len(kernels)}) must match channels ({n_channels})")

    # Prepare channel data
    channels = []
    for ch in range(n_channels):
        channels.append({
            'image': image[:, :, ch],
            'kernel': kernels[ch]
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
