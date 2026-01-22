#!/usr/bin/env python3
"""
Command-line tool for primal-dual cross-channel deconvolution.

Usage:
    ./deconvolve.py input.jpg --kernel psf.png -o output.png
    ./deconvolve.py input.jpg --gaussian 2.5 -o output.png
    ./deconvolve.py input.jpg --kernel psf.png --channels rgb

Based on:
    F. Heide, M. Rouf, M. Hullin, B. Labitzke, W. Heidrich, A. Kolb.
    "High-Quality Computational Imaging Through Simple Lenses."
    ACM ToG 2013
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from scipy.ndimage import gaussian_filter

from deconv import pd_joint_deconv, img_to_norm_grayscale, deconvolve_tiled, load_tiled_psfs


def create_gaussian_kernel(sigma, size=None):
    """Create a Gaussian blur kernel."""
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1

    kernel = np.zeros((size, size))
    kernel[size // 2, size // 2] = 1.0
    kernel = gaussian_filter(kernel, sigma=sigma)
    kernel = kernel / kernel.sum()
    return kernel


def load_kernel(kernel_path, size=None):
    """Load and normalize a kernel from an image file."""
    kernel = iio.imread(kernel_path)
    kernel = img_to_norm_grayscale(kernel)

    if size is not None:
        from skimage.transform import resize
        kernel = resize(kernel, (size, size), order=3, anti_aliasing=True)

    kernel = np.maximum(kernel, 0)
    kernel = kernel / kernel.sum()
    return kernel


def deconvolve_image(image, kernels, lambda_residual=200, lambda_tv=2.0,
                     lambda_cross=3.0, max_iterations=200, tolerance=1e-4,
                     verbose=False):
    """
    Deconvolve an image using primal-dual optimization.

    Parameters
    ----------
    image : ndarray
        Input image (H, W) for grayscale or (H, W, C) for color
    kernels : list of ndarray or ndarray
        Blur kernel(s). If single kernel, used for all channels.
        If list, one kernel per channel.
    lambda_residual : float
        Data fidelity weight (lower = more regularization)
    lambda_tv : float
        Total variation weight (higher = smoother)
    lambda_cross : float
        Cross-channel coupling weight (higher = more channel correlation)
    max_iterations : int
        Maximum iterations per channel
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

    # Handle grayscale vs color
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    n_channels = image.shape[2]

    # Handle kernel input
    if isinstance(kernels, np.ndarray) and kernels.ndim == 2:
        kernels = [kernels] * n_channels

    if len(kernels) != n_channels:
        raise ValueError(f"Number of kernels ({len(kernels)}) must match "
                        f"number of channels ({n_channels})")

    # Prepare channel data
    channels = []
    for ch in range(n_channels):
        channels.append({
            'image': image[:, :, ch],
            'kernel': kernels[ch]
        })

    # Build lambda parameters
    # Format: [ch_idx (1-based), lambda_residual, lambda_tv, lambda_black,
    #          lambda_cross_ch..., n_detail_layers]
    lambda_params = []
    for ch in range(n_channels):
        row = [ch + 1, lambda_residual, lambda_tv, 0.0]
        # Cross-channel weights (couple to first channel if multiple channels)
        cross_weights = [0.0] * n_channels
        if ch > 0 and n_channels > 1:
            cross_weights[0] = lambda_cross
        row.extend(cross_weights)
        # Detail layers (1 for first channel, 0 for others)
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

    # Remove extra dimension for grayscale
    if output.shape[2] == 1:
        output = output[:, :, 0]

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Deconvolve an image using primal-dual optimization.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jpg --kernel psf.png
  %(prog)s input.jpg --gaussian 2.5 -o sharp.png
  %(prog)s input.jpg --kernel psf.png --lambda-res 100 --lambda-tv 3.0
  %(prog)s blurry.png --kernel psf.png --channels r --verbose

  # Deconvolve with spatially-varying PSFs (tiled)
  %(prog)s blurry.png --kernel-tiles ./psf_tiles/ --tiles 3x3 -o sharp.png

Kernel specification (one required):
  --kernel        Load PSF from image file
  --gaussian      Generate Gaussian PSF with given sigma
  --kernel-tiles  Directory or base path for tiled PSFs
        """
    )

    parser.add_argument('input', type=Path,
                        help='Input image file')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Output file (default: input_deconv.ext)')

    # Kernel options (mutually exclusive)
    kernel_group = parser.add_mutually_exclusive_group(required=True)
    kernel_group.add_argument('--kernel', type=Path,
                              help='PSF/kernel image file')
    kernel_group.add_argument('--gaussian', type=float, metavar='SIGMA',
                              help='Gaussian blur sigma (generates kernel)')
    kernel_group.add_argument('--kernel-tiles', type=Path,
                              help='Directory or base path for tiled PSFs (e.g., psf_tile_*.png)')

    # Kernel size
    parser.add_argument('--kernel-size', type=int, default=None,
                        help='Resize kernel to this size (must be odd)')

    # Tile options (for --kernel-tiles)
    parser.add_argument('--tiles', type=str, default=None,
                        help='Tile grid for spatially-varying deconvolution, e.g., 3x3')
    parser.add_argument('--tile-overlap', type=float, default=0.25,
                        help='Tile overlap fraction 0-0.5 (default: 0.25)')

    # Channel options
    parser.add_argument('--channels', type=str, default='rgb',
                        choices=['rgb', 'r', 'g', 'b', 'gray'],
                        help='Channels to process (default: rgb)')

    # Algorithm parameters
    parser.add_argument('--lambda-res', type=float, default=200,
                        help='Data fidelity weight (default: 200)')
    parser.add_argument('--lambda-tv', type=float, default=2.0,
                        help='Total variation weight (default: 2.0)')
    parser.add_argument('--lambda-cross', type=float, default=3.0,
                        help='Cross-channel coupling weight (default: 3.0)')
    parser.add_argument('--max-iter', type=int, default=200,
                        help='Maximum iterations (default: 200)')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Convergence tolerance (default: 1e-4)')

    # Other options
    parser.add_argument('--linear', action='store_true',
                        help='Input is linear (skip gamma decoding)')
    parser.add_argument('--gamma', type=float, default=2.2,
                        help='Gamma value for encoding/decoding (default: 2.2)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print progress information')
    parser.add_argument('--16bit', dest='bit16', action='store_true',
                        help='Save as 16-bit image')

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.kernel and not args.kernel.exists():
        print(f"Error: Kernel file not found: {args.kernel}", file=sys.stderr)
        sys.exit(1)

    if args.kernel_size is not None and args.kernel_size % 2 == 0:
        print("Error: Kernel size must be odd", file=sys.stderr)
        sys.exit(1)

    # Set output path
    if args.output is None:
        args.output = args.input.with_stem(args.input.stem + '_deconv')

    # Load image
    if args.verbose:
        print(f"Loading image: {args.input}")

    image = iio.imread(args.input)
    original_dtype = image.dtype

    # Convert to float [0, 1]
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        image = image.astype(np.float64) / info.max
    else:
        image = image.astype(np.float64)
        if image.max() > 1.0:
            image = image / image.max()

    if args.verbose:
        print(f"Image size: {image.shape[1]} x {image.shape[0]}")
        if image.ndim == 3:
            print(f"Channels: {image.shape[2]}")

    # Apply inverse gamma (linearize) if needed
    if not args.linear:
        if args.verbose:
            print(f"Applying inverse gamma (gamma={args.gamma})")
        image = np.power(image, args.gamma)

    # Handle channel selection
    if image.ndim == 3 and image.shape[2] >= 3:
        if args.channels == 'r':
            image = image[:, :, 0]
        elif args.channels == 'g':
            image = image[:, :, 1]
        elif args.channels == 'b':
            image = image[:, :, 2]
        elif args.channels == 'gray':
            image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        # 'rgb' keeps all channels

    # Handle tiled vs single-kernel deconvolution
    if args.kernel_tiles:
        # Tiled PSF deconvolution
        if args.verbose:
            print(f"Loading tiled PSFs: {args.kernel_tiles}")

        # Parse tile grid if specified
        tiles_h, tiles_w = None, None
        if args.tiles:
            try:
                tiles_h, tiles_w = map(int, args.tiles.lower().split('x'))
            except ValueError:
                print(f"Error: Invalid tiles format '{args.tiles}'. Use format like '3x3'",
                      file=sys.stderr)
                sys.exit(1)

        psfs, tile_grid = load_tiled_psfs(args.kernel_tiles, tiles_h, tiles_w)

        if args.verbose:
            print(f"Loaded {len(psfs)} tile PSFs ({tile_grid[1]}x{tile_grid[0]} grid)")
            print(f"PSF size: {psfs[0].shape[0]} x {psfs[0].shape[1]}")

        # Run tiled deconvolution
        if args.verbose:
            print(f"\nRunning tiled deconvolution...")
            print(f"  lambda_residual: {args.lambda_res}")
            print(f"  lambda_tv: {args.lambda_tv}")
            print(f"  lambda_cross: {args.lambda_cross}")
            print(f"  tile_overlap: {args.tile_overlap}")
            print(f"  max_iterations: {args.max_iter}")
            print()

        result = deconvolve_tiled(
            image, psfs, tile_grid,
            overlap=args.tile_overlap,
            lambda_residual=args.lambda_res,
            lambda_tv=args.lambda_tv,
            lambda_cross=args.lambda_cross,
            max_iterations=args.max_iter,
            tolerance=args.tolerance,
            verbose=args.verbose
        )
    else:
        # Single kernel deconvolution
        if args.kernel:
            if args.verbose:
                print(f"Loading kernel: {args.kernel}")
            kernel = load_kernel(args.kernel, args.kernel_size)
        else:
            if args.verbose:
                print(f"Creating Gaussian kernel (sigma={args.gaussian})")
            kernel = create_gaussian_kernel(args.gaussian, args.kernel_size)

        if args.verbose:
            print(f"Kernel size: {kernel.shape[0]} x {kernel.shape[1]}")

        # Run deconvolution
        if args.verbose:
            print(f"\nRunning deconvolution...")
            print(f"  lambda_residual: {args.lambda_res}")
            print(f"  lambda_tv: {args.lambda_tv}")
            print(f"  lambda_cross: {args.lambda_cross}")
            print(f"  max_iterations: {args.max_iter}")
            print()

        result = deconvolve_image(
            image, kernel,
            lambda_residual=args.lambda_res,
            lambda_tv=args.lambda_tv,
            lambda_cross=args.lambda_cross,
            max_iterations=args.max_iter,
            tolerance=args.tolerance,
            verbose=args.verbose
        )

    # Clip to valid range
    result = np.clip(result, 0, None)

    # Apply gamma correction for display
    if not args.linear:
        result = np.power(result, 1.0 / args.gamma)

    # Clip to [0, 1]
    result = np.clip(result, 0, 1)

    # Convert to output format
    if args.bit16:
        result = (result * 65535).astype(np.uint16)
    else:
        result = (result * 255).astype(np.uint8)

    # Save result
    if args.verbose:
        print(f"\nSaving result: {args.output}")

    iio.imwrite(args.output, result)

    print(f"Deconvolved image saved to: {args.output}")


if __name__ == '__main__':
    main()
