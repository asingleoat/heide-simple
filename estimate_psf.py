#!/usr/bin/env python3
"""
Command-line tool for PSF estimation.

Estimates the Point Spread Function (PSF) of an optical system from
calibration images. Requires a sharp reference image (captured with
pinhole aperture) and a blurred image (captured with wide aperture).

Based on:
    F. Heide, M. Rouf, M. Hullin, B. Labitzke, W. Heidrich, A. Kolb.
    "High-Quality Computational Imaging Through Simple Lenses."
    ACM ToG 2013

Usage:
    # Estimate PSF from sharp/blurred image pair
    ./estimate_psf.py --sharp sharp.png --blurred blurred.png -o psf.png

    # Generate a calibration pattern
    ./estimate_psf.py --generate-pattern -o calibration.png

    # Estimate from calibration pattern images
    ./estimate_psf.py --sharp-pattern sharp_calib.png --blurred-pattern blurred_calib.png -o psf.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from deconv import (
    estimate_psf,
    estimate_psf_from_patches,
    create_calibration_pattern,
    extract_patches_from_images,
    img_to_norm_grayscale,
)


def main():
    parser = argparse.ArgumentParser(
        description='Estimate PSF from calibration images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate PSF from image pair (e.g., same scene with different apertures)
  %(prog)s --sharp pinhole.png --blurred wide_aperture.png --size 31 -o psf.png

  # Generate calibration pattern to print
  %(prog)s --generate-pattern --patch-size 128 --grid 4x4 -o calibration.png

  # Estimate from photos of calibration pattern
  %(prog)s --sharp-pattern sharp_photo.png --blurred-pattern blurred_photo.png \\
           --size 43 --grid 4x4 -o psf.png

  # Estimate per-channel PSFs for chromatic aberration
  %(prog)s --sharp sharp.png --blurred blurred.png --size 31 --per-channel -o psf
        """
    )

    # Input modes (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--generate-pattern', action='store_true',
                            help='Generate calibration pattern image')
    input_group.add_argument('--sharp', type=Path,
                            help='Sharp reference image (pinhole aperture)')

    # Additional inputs for estimation
    parser.add_argument('--blurred', type=Path,
                        help='Blurred image (wide aperture) - for single pair mode')
    parser.add_argument('--sharp-pattern', type=Path,
                        help='Photo of calibration pattern with pinhole aperture')
    parser.add_argument('--blurred-pattern', type=Path,
                        help='Photo of calibration pattern with wide aperture')

    # Output
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output file (PSF image or calibration pattern)')

    # PSF estimation parameters
    parser.add_argument('--size', type=int, default=31,
                        help='PSF size in pixels (must be odd, default: 31)')
    parser.add_argument('--lambda-tv', type=float, default=0.001,
                        help='TV regularization weight (default: 0.001)')
    parser.add_argument('--mu-sum', type=float, default=50.0,
                        help='Sum-to-one constraint weight (default: 50.0)')
    parser.add_argument('--max-iter', type=int, default=500,
                        help='Maximum iterations (default: 500)')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Convergence tolerance (default: 1e-5)')
    parser.add_argument('--per-channel', action='store_true',
                        help='Estimate separate PSF for each color channel')

    # Calibration pattern parameters
    parser.add_argument('--patch-size', type=int, default=128,
                        help='Size of noise patches in calibration pattern (default: 128)')
    parser.add_argument('--grid', type=str, default='4x4',
                        help='Grid size for calibration pattern, e.g., 4x4 (default: 4x4)')
    parser.add_argument('--border', type=int, default=20,
                        help='Border width around patches (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for pattern generation (default: 42)')

    # Patch extraction parameters
    parser.add_argument('--crop-border', type=int, default=5,
                        help='Extra border to crop from patches (default: 5)')

    # Other options
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print progress information')
    parser.add_argument('--16bit', dest='bit16', action='store_true',
                        help='Save PSF as 16-bit image')

    args = parser.parse_args()

    # Validate PSF size
    if args.size % 2 == 0:
        print("Error: PSF size must be odd", file=sys.stderr)
        sys.exit(1)

    # Parse grid size
    try:
        grid_h, grid_w = map(int, args.grid.lower().split('x'))
    except ValueError:
        print(f"Error: Invalid grid format '{args.grid}'. Use format like '4x4'",
              file=sys.stderr)
        sys.exit(1)

    verbose = 'brief' if args.verbose else 'none'

    # Mode 1: Generate calibration pattern
    if args.generate_pattern:
        if args.verbose:
            print(f"Generating calibration pattern...")
            print(f"  Patch size: {args.patch_size}x{args.patch_size}")
            print(f"  Grid: {grid_h}x{grid_w}")
            print(f"  Border: {args.border}")

        pattern, coords = create_calibration_pattern(
            patch_size=args.patch_size,
            n_patches_h=grid_h,
            n_patches_w=grid_w,
            border_width=args.border,
            seed=args.seed
        )

        # Save pattern
        if args.bit16:
            pattern_out = (pattern * 65535).astype(np.uint16)
        else:
            pattern_out = (pattern * 255).astype(np.uint8)

        iio.imwrite(args.output, pattern_out)
        print(f"Calibration pattern saved to: {args.output}")
        print(f"Pattern size: {pattern.shape[1]}x{pattern.shape[0]} pixels")
        print(f"\nInstructions:")
        print(f"  1. Print this pattern")
        print(f"  2. Photograph it twice:")
        print(f"     - Once with pinhole/small aperture (sharp)")
        print(f"     - Once with wide aperture (blurred)")
        print(f"  3. Run: {sys.argv[0]} --sharp-pattern <sharp.png> \\")
        print(f"          --blurred-pattern <blurred.png> --size {args.size} -o psf.png")
        return

    # Mode 2: Estimate from single image pair
    if args.sharp and args.blurred:
        if not args.sharp.exists():
            print(f"Error: Sharp image not found: {args.sharp}", file=sys.stderr)
            sys.exit(1)
        if not args.blurred.exists():
            print(f"Error: Blurred image not found: {args.blurred}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"Loading images...")
            print(f"  Sharp: {args.sharp}")
            print(f"  Blurred: {args.blurred}")

        sharp = iio.imread(args.sharp)
        blurred = iio.imread(args.blurred)

        # Handle per-channel estimation
        if args.per_channel and sharp.ndim == 3:
            psfs = []
            channel_names = ['red', 'green', 'blue']
            for ch in range(min(3, sharp.shape[2])):
                if args.verbose:
                    print(f"\nEstimating PSF for {channel_names[ch]} channel...")

                sharp_ch = img_to_norm_grayscale(sharp[:, :, ch])
                blurred_ch = img_to_norm_grayscale(blurred[:, :, ch])

                psf = estimate_psf(
                    sharp_ch, blurred_ch, args.size,
                    lambda_tv=args.lambda_tv,
                    mu_sum=args.mu_sum,
                    max_it=args.max_iter,
                    tol=args.tolerance,
                    verbose=verbose
                )
                psfs.append(psf)

            # Save individual PSFs
            output_base = args.output.stem
            output_ext = args.output.suffix or '.png'
            for ch, (psf, name) in enumerate(zip(psfs, channel_names)):
                out_path = args.output.parent / f"{output_base}_{name}{output_ext}"
                _save_psf(psf, out_path, args.bit16)
                print(f"PSF ({name}) saved to: {out_path}")
        else:
            # Single grayscale PSF
            sharp_gray = img_to_norm_grayscale(sharp)
            blurred_gray = img_to_norm_grayscale(blurred)

            if args.verbose:
                print(f"\nEstimating PSF...")
                print(f"  PSF size: {args.size}x{args.size}")
                print(f"  lambda_tv: {args.lambda_tv}")
                print(f"  mu_sum: {args.mu_sum}")

            psf = estimate_psf(
                sharp_gray, blurred_gray, args.size,
                lambda_tv=args.lambda_tv,
                mu_sum=args.mu_sum,
                max_it=args.max_iter,
                tol=args.tolerance,
                verbose=verbose
            )

            _save_psf(psf, args.output, args.bit16)
            print(f"PSF saved to: {args.output}")

        return

    # Mode 3: Estimate from calibration pattern photos
    if args.sharp_pattern and args.blurred_pattern:
        if not args.sharp_pattern.exists():
            print(f"Error: Sharp pattern image not found: {args.sharp_pattern}",
                  file=sys.stderr)
            sys.exit(1)
        if not args.blurred_pattern.exists():
            print(f"Error: Blurred pattern image not found: {args.blurred_pattern}",
                  file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"Loading calibration pattern images...")
            print(f"  Sharp: {args.sharp_pattern}")
            print(f"  Blurred: {args.blurred_pattern}")

        sharp = iio.imread(args.sharp_pattern)
        blurred = iio.imread(args.blurred_pattern)

        # Generate patch coordinates (assuming same pattern parameters)
        _, coords = create_calibration_pattern(
            patch_size=args.patch_size,
            n_patches_h=grid_h,
            n_patches_w=grid_w,
            border_width=args.border,
            seed=args.seed
        )

        # Handle per-channel estimation
        if args.per_channel and sharp.ndim == 3:
            channel_names = ['red', 'green', 'blue']
            for ch in range(min(3, sharp.shape[2])):
                if args.verbose:
                    print(f"\nProcessing {channel_names[ch]} channel...")

                sharp_ch = img_to_norm_grayscale(sharp[:, :, ch])
                blurred_ch = img_to_norm_grayscale(blurred[:, :, ch])

                sharp_patches, blurred_patches = extract_patches_from_images(
                    sharp_ch, blurred_ch, coords, crop_border=args.crop_border
                )

                if args.verbose:
                    print(f"  Extracted {len(sharp_patches)} patches")

                psf = estimate_psf_from_patches(
                    sharp_patches, blurred_patches, args.size,
                    lambda_tv=args.lambda_tv,
                    mu_sum=args.mu_sum,
                    max_it=args.max_iter,
                    tol=args.tolerance,
                    verbose=verbose
                )

                output_base = args.output.stem
                output_ext = args.output.suffix or '.png'
                out_path = args.output.parent / f"{output_base}_{channel_names[ch]}{output_ext}"
                _save_psf(psf, out_path, args.bit16)
                print(f"PSF ({channel_names[ch]}) saved to: {out_path}")
        else:
            # Single grayscale PSF
            sharp_gray = img_to_norm_grayscale(sharp)
            blurred_gray = img_to_norm_grayscale(blurred)

            sharp_patches, blurred_patches = extract_patches_from_images(
                sharp_gray, blurred_gray, coords, crop_border=args.crop_border
            )

            if args.verbose:
                print(f"\nExtracted {len(sharp_patches)} patches")
                print(f"Estimating PSF...")

            psf = estimate_psf_from_patches(
                sharp_patches, blurred_patches, args.size,
                lambda_tv=args.lambda_tv,
                mu_sum=args.mu_sum,
                max_it=args.max_iter,
                tol=args.tolerance,
                verbose=verbose
            )

            _save_psf(psf, args.output, args.bit16)
            print(f"PSF saved to: {args.output}")

        return

    # If we get here, missing required arguments
    if args.sharp and not args.blurred:
        print("Error: --blurred is required when using --sharp", file=sys.stderr)
        sys.exit(1)

    parser.print_help()
    sys.exit(1)


def _save_psf(psf, output_path, bit16=False):
    """Save PSF to image file with proper normalization."""
    # Normalize to [0, 1] for visualization
    psf_vis = psf / psf.max() if psf.max() > 0 else psf

    if bit16:
        psf_out = (psf_vis * 65535).astype(np.uint16)
    else:
        psf_out = (psf_vis * 255).astype(np.uint8)

    iio.imwrite(output_path, psf_out)


if __name__ == '__main__':
    main()
