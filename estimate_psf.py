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
    ./estimate_psf.py --sharp sharp.png --blurred blurred.png --output-dir ./psf_out

    # Generate a calibration pattern
    ./estimate_psf.py --generate-pattern --output-dir ./

    # Estimate from calibration pattern images
    ./estimate_psf.py --sharp-pattern sharp_calib.png --blurred-pattern blurred_calib.png --output-dir ./psf_out
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from deconv import (
    estimate_psf,
    estimate_psf_multiscale,
    estimate_psf_from_patches,
    estimate_psf_tiled,
    create_calibration_pattern,
    extract_patches_from_images,
    img_to_norm_grayscale,
)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate PSF from calibration images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate PSF from image pair (e.g., same scene with different apertures)
  %(prog)s --sharp pinhole.png --blurred wide_aperture.png --size 31 --output-dir ./

  # Generate calibration pattern to print
  %(prog)s --generate-pattern --patch-size 128 --grid 4x4 --output-dir ./

  # Estimate from photos of calibration pattern
  %(prog)s --sharp-pattern sharp_photo.png --blurred-pattern blurred_photo.png \\
           --size 43 --grid 4x4 --output-dir ./psf_out

  # Estimate single grayscale PSF instead of per-channel
  %(prog)s --sharp sharp.png --blurred blurred.png --size 31 --grayscale --output-dir ./

  # Estimate spatially-varying PSF (3x3 tile grid)
  %(prog)s --sharp sharp.png --blurred blurred.png --size 31 --tiles 3x3 --output-dir ./psf_tiles
        """,
    )

    # Input modes (mutually exclusive groups)
    input_mode_group = parser.add_mutually_exclusive_group()
    input_mode_group.add_argument(
        "--generate-pattern",
        action="store_true",
        help="Generate calibration pattern image",
    )

    image_pair_group = parser.add_argument_group(
        "Image Pair Estimation",
        "Estimate PSF from a sharp/blurred image pair (e.g., same scene, different apertures).",
    )
    image_pair_group.add_argument(
        "--sharp", type=Path, help="Sharp reference image (pinhole aperture)"
    )
    image_pair_group.add_argument(
        "--blurred", type=Path, help="Blurred image (wide aperture)"
    )

    pattern_photo_group = parser.add_argument_group(
        "Calibration Pattern Estimation",
        "Estimate PSF from photos of a printed calibration pattern.",
    )
    pattern_photo_group.add_argument(
        "--sharp-pattern",
        type=Path,
        help="Photo of calibration pattern with pinhole aperture",
    )
    pattern_photo_group.add_argument(
        "--blurred-pattern",
        type=Path,
        help="Photo of calibration pattern with wide aperture",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated files (PSFs or calibration pattern)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="psf",
        help='Prefix for generated PSF filenames (e.g., "psf" for "psf_red_tile_0_0.png")',
    )

    # PSF estimation parameters
    parser.add_argument(
        "--size",
        type=int,
        default=31,
        help="PSF size in pixels (must be odd, default: 31)",
    )
    parser.add_argument(
        "--tv",
        "--lambda-tv",
        type=float,
        dest="lambda_tv",
        default=0.001,
        help="TV regularization weight (default: 0.001)",
    )
    parser.add_argument(
        "--mu",
        "--mu-sum",
        type=float,
        dest="mu_sum",
        default=50.0,
        help="Sum-to-one constraint weight (default: 50.0)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=500, help="Maximum iterations (default: 500)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Convergence tolerance (default: 1e-5)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Estimate single grayscale PSF (default: per-channel for color images)",
    )
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Use scale-space estimation for faster convergence",
    )
    parser.add_argument(
        "--n-scales",
        type=int,
        default=None,
        help="Number of scales for multiscale (auto if not set)",
    )

    # Tile-based estimation for spatially-varying PSF
    tile_group = parser.add_argument_group(
        "Spatially-Varying PSF Estimation (Tiled)",
        "Options for estimating PSFs across a grid of tiles.",
    )
    tile_group.add_argument(
        "--tiles",
        type=str,
        default=None,
        help="Tile grid for spatially-varying PSF, e.g., 3x3 (default: disabled)",
    )
    tile_group.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        help="Tile overlap fraction 0-0.5 (default: 0.25)",
    )
    tile_group.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Spatial smoothing sigma for tile PSFs (default: 1.0, 0=disabled)",
    )

    # Calibration pattern parameters
    calibration_group = parser.add_argument_group(
        "Calibration Pattern Generation & Patch Extraction",
        "Options for generating a calibration pattern and extracting patches from its photos.",
    )
    calibration_group.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Size of noise patches in calibration pattern (default: 128)",
    )
    calibration_group.add_argument(
        "--grid",
        type=str,
        default="4x4",
        help="Grid size for calibration pattern, e.g., 4x4 (default: 4x4)",
    )
    calibration_group.add_argument(
        "--border",
        type=int,
        default=20,
        help="Border width around patches (default: 20)",
    )
    calibration_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pattern generation (default: 42)",
    )
    calibration_group.add_argument(
        "--crop-border",
        type=int,
        default=5,
        help="Extra border to crop from patches (default: 5)",
    )

    # Other options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print progress information"
    )
    parser.add_argument(
        "--16bit", dest="bit16", action="store_true", help="Save PSF as 16-bit image"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Manual check for mutually exclusive input modes
    mode_selected_count = 0
    if args.generate_pattern:
        mode_selected_count += 1
    if args.sharp and args.blurred:
        mode_selected_count += 1
    if args.sharp_pattern and args.blurred_pattern:
        mode_selected_count += 1

    if mode_selected_count == 0:
        parser.error(
            "One of --generate-pattern, or (--sharp and --blurred), or (--sharp-pattern and --blurred-pattern) must be provided."
        )
    elif mode_selected_count > 1:
        parser.error(
            "Only one mode of operation can be selected (--generate-pattern, --sharp/--blurred, or --sharp-pattern/--blurred-pattern)."
        )

    # Add checks for incomplete pairs
    if (args.sharp and not args.blurred) or (not args.sharp and args.blurred):
        parser.error(
            "Both --sharp and --blurred must be provided together for image pair estimation."
        )
    if (args.sharp_pattern and not args.blurred_pattern) or (
        not args.sharp_pattern and args.blurred_pattern
    ):
        parser.error(
            "Both --sharp-pattern and --blurred-pattern must be provided together for calibration pattern estimation."
        )

    # Validate PSF size
    if args.size % 2 == 0:
        print("Error: PSF size must be odd", file=sys.stderr)
        sys.exit(1)

    # Parse grid size
    try:
        grid_h, grid_w = map(int, args.grid.lower().split("x"))
    except ValueError:
        print(
            f"Error: Invalid grid format '{args.grid}'. Use format like '4x4'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse tiles size if provided
    tiles_h, tiles_w = None, None
    if args.tiles:
        try:
            tiles_h, tiles_w = map(int, args.tiles.lower().split("x"))
        except ValueError:
            print(
                f"Error: Invalid tiles format '{args.tiles}'. Use format like '3x3'",
                file=sys.stderr,
            )
            sys.exit(1)

    verbose = "brief" if args.verbose else "none"

    # Mode 1: Generate calibration pattern
    if args.generate_pattern:
        if args.verbose:
            print("Generating calibration pattern...")
            print(f"  Patch size: {args.patch_size}x{args.patch_size}")
            print(f"  Grid: {grid_h}x{grid_w}")
            print(f"  Border: {args.border}")

        pattern, coords = create_calibration_pattern(
            patch_size=args.patch_size,
            n_patches_h=grid_h,
            n_patches_w=grid_w,
            border_width=args.border,
            seed=args.seed,
        )

        # Save pattern
        if args.bit16:
            pattern_out = (pattern * 65535).astype(np.uint16)
        else:
            pattern_out = (pattern * 255).astype(np.uint8)

        iio.imwrite(args.output_dir / "calibration_pattern.png", pattern_out)
        print(
            f"Calibration pattern saved to: {args.output_dir / 'calibration_pattern.png'}"
        )
        print(f"Pattern size: {pattern.shape[1]}x{pattern.shape[0]} pixels")
        print("\nInstructions:")
        print("  1. Print this pattern")
        print("  2. Photograph it twice:")
        print("     - Once with pinhole/small aperture (sharp)")
        print("     - Once with wide aperture (blurred)")
        print(f"  3. Run: {sys.argv[0]} --sharp-pattern <sharp.png> \\")
        print(
            f"          --blurred-pattern <blurred.png> --size {args.size} --output-dir <output_folder> --output-prefix psf"
        )
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
            print("Loading images...")
            print(f"  Sharp: {args.sharp}")
            print(f"  Blurred: {args.blurred}")

        sharp = iio.imread(args.sharp)
        blurred = iio.imread(args.blurred)

        # Handle tile-based estimation for spatially-varying PSF
        if tiles_h is not None and tiles_w is not None:
            output_base = args.output_prefix
            output_ext = ".png"

            # Per-channel tiled estimation (default for color images)
            if not args.grayscale and sharp.ndim == 3 and sharp.shape[2] >= 3:
                channel_names = ["red", "green", "blue"]
                all_channel_psfs = {}

                for ch in range(min(3, sharp.shape[2])):
                    if args.verbose:
                        print(
                            f"\nEstimating spatially-varying PSF for {channel_names[ch]} channel..."
                        )
                        print(f"  Tiles: {tiles_w}x{tiles_h}")
                        print(f"  PSF size: {args.size}x{args.size}")

                    sharp_ch = img_to_norm_grayscale(sharp[:, :, ch])
                    blurred_ch = img_to_norm_grayscale(blurred[:, :, ch])

                    psfs, tile_centers, tile_grid = estimate_psf_tiled(
                        sharp_ch,
                        blurred_ch,
                        args.size,
                        n_tiles_h=tiles_h,
                        n_tiles_w=tiles_w,
                        overlap=args.tile_overlap,
                        lambda_tv=args.lambda_tv,
                        mu_sum=args.mu_sum,
                        max_it=args.max_iter,
                        tol=args.tolerance,
                        multiscale=args.multiscale,
                        n_scales=args.n_scales,
                        smooth_sigma=args.smooth_sigma,
                        verbose=verbose,
                    )

                    all_channel_psfs[channel_names[ch]] = psfs

                    # Save individual tile PSFs for this channel
                    for idx, (row, col) in enumerate(
                        [(i, j) for i in range(tiles_h) for j in range(tiles_w)]
                    ):
                        out_path = (
                            args.output_dir
                            / f"{output_base}_{channel_names[ch]}_tile_{row}_{col}{output_ext}"
                        )
                        _save_psf(psfs[idx], out_path, args.bit16)
                        if args.verbose:
                            print(
                                f"PSF ({channel_names[ch]} tile {row},{col}) saved to: {out_path}"
                            )

                    # Save combined grid for this channel
                    grid_path = (
                        args.output_dir
                        / f"{output_base}_{channel_names[ch]}{output_ext}"
                    )
                    _save_psf_grid(psfs, tiles_h, tiles_w, grid_path, args.bit16)

                print(f"\nSaved {tiles_h * tiles_w * 3} tile PSFs:")
                for ch_name in channel_names:
                    print(f"  {ch_name}: {output_base}_{ch_name}_tile_*{output_ext}")
                print(
                    f"Combined grids: {output_base}_red{output_ext}, {output_base}_green{output_ext}, {output_base}_blue{output_ext}"
                )

            else:
                # Grayscale tiled estimation
                sharp_gray = img_to_norm_grayscale(sharp)
                blurred_gray = img_to_norm_grayscale(blurred)

                if args.verbose:
                    print("\nEstimating spatially-varying PSF (grayscale)...")
                    print(f"  Tiles: {tiles_w}x{tiles_h}")
                    print(f"  PSF size: {args.size}x{args.size}")

                psfs, tile_centers, tile_grid = estimate_psf_tiled(
                    sharp_gray,
                    blurred_gray,
                    args.size,
                    n_tiles_h=tiles_h,
                    n_tiles_w=tiles_w,
                    overlap=args.tile_overlap,
                    lambda_tv=args.lambda_tv,
                    mu_sum=args.mu_sum,
                    max_it=args.max_iter,
                    tol=args.tolerance,
                    multiscale=args.multiscale,
                    n_scales=args.n_scales,
                    smooth_sigma=args.smooth_sigma,
                    verbose=verbose,
                )

                # Save individual tile PSFs
                for idx, (row, col) in enumerate(
                    [(i, j) for i in range(tiles_h) for j in range(tiles_w)]
                ):
                    out_path = (
                        args.output_dir
                        / f"{args.output_prefix}_tile_{row}_{col}{output_ext}"
                    )
                    _save_psf(psfs[idx], out_path, args.bit16)
                    if args.verbose:
                        print(f"PSF (tile {row},{col}) saved to: {out_path}")

                print(
                    f"\nSaved {len(psfs)} tile PSFs to: {args.output_dir}/{args.output_prefix}_tile_*{output_ext}"
                )

                # Also save a combined visualization
                _save_psf_grid(
                    psfs,
                    tiles_h,
                    tiles_w,
                    args.output_dir / f"{args.output_prefix}{output_ext}",
                    args.bit16,
                )
                print(
                    f"Combined PSF grid saved to: {args.output_dir / f'{args.output_prefix}{output_ext}'}"
                )

            return

        # Handle per-channel estimation (default for color images)
        if not args.grayscale and sharp.ndim == 3 and sharp.shape[2] >= 3:
            psfs = []
            channel_names = ["red", "green", "blue"]
            output_base = args.output_prefix
            output_ext = ".png"
            for ch in range(min(3, sharp.shape[2])):
                if args.verbose:
                    print(f"\nEstimating PSF for {channel_names[ch]} channel...")

                sharp_ch = img_to_norm_grayscale(sharp[:, :, ch])
                blurred_ch = img_to_norm_grayscale(blurred[:, :, ch])

                if args.multiscale:
                    psf = estimate_psf_multiscale(
                        sharp_ch,
                        blurred_ch,
                        args.size,
                        lambda_tv=args.lambda_tv,
                        mu_sum=args.mu_sum,
                        max_it=args.max_iter,
                        tol=args.tolerance,
                        n_scales=args.n_scales,
                        verbose=verbose,
                    )
                else:
                    psf = estimate_psf(
                        sharp_ch,
                        blurred_ch,
                        args.size,
                        lambda_tv=args.lambda_tv,
                        mu_sum=args.mu_sum,
                        max_it=args.max_iter,
                        tol=args.tolerance,
                        verbose=verbose,
                    )
                psfs.append(psf)

            # Save individual PSFs
            for ch, (psf, name) in enumerate(zip(psfs, channel_names)):
                out_path = args.output_dir / f"{output_base}_{name}{output_ext}"
                _save_psf(psf, out_path, args.bit16)
                print(f"PSF ({name}) saved to: {out_path}")
        else:
            # Single grayscale PSF
            sharp_gray = img_to_norm_grayscale(sharp)
            blurred_gray = img_to_norm_grayscale(blurred)

            if args.verbose:
                print("\nEstimating PSF...")
                print(f"  PSF size: {args.size}x{args.size}")
                print(f"  lambda_tv: {args.lambda_tv}")
                print(f"  mu_sum: {args.mu_sum}")

            if args.multiscale:
                psf = estimate_psf_multiscale(
                    sharp_gray,
                    blurred_gray,
                    args.size,
                    lambda_tv=args.lambda_tv,
                    mu_sum=args.mu_sum,
                    max_it=args.max_iter,
                    tol=args.tolerance,
                    n_scales=args.n_scales,
                    verbose=verbose,
                )
            else:
                psf = estimate_psf(
                    sharp_gray,
                    blurred_gray,
                    args.size,
                    lambda_tv=args.lambda_tv,
                    mu_sum=args.mu_sum,
                    max_it=args.max_iter,
                    tol=args.tolerance,
                    verbose=verbose,
                )
            _save_psf(psf, args.output_dir / f"{args.output_prefix}.png", args.bit16)
            print(f"PSF saved to: {args.output_dir / f'{args.output_prefix}.png'}")
        return

    # Mode 3: Estimate from calibration pattern photos
    if args.sharp_pattern and args.blurred_pattern:
        if not args.sharp_pattern.exists():
            print(
                f"Error: Sharp pattern image not found: {args.sharp_pattern}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.blurred_pattern.exists():
            print(
                f"Error: Blurred pattern image not found: {args.blurred_pattern}",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.verbose:
            print("Loading calibration pattern images...")
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
            seed=args.seed,
        )

        # Handle per-channel estimation (default for color images)
        if not args.grayscale and sharp.ndim == 3 and sharp.shape[2] >= 3:
            channel_names = ["red", "green", "blue"]
            output_base = args.output_prefix
            output_ext = ".png"
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
                    sharp_patches,
                    blurred_patches,
                    args.size,
                    lambda_tv=args.lambda_tv,
                    mu_sum=args.mu_sum,
                    max_it=args.max_iter,
                    tol=args.tolerance,
                    multiscale=args.multiscale,
                    n_scales=args.n_scales,
                    verbose=verbose,
                )

                out_path = (
                    args.output_dir / f"{output_base}_{channel_names[ch]}{output_ext}"
                )
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
                print("Estimating PSF...")

            psf = estimate_psf_from_patches(
                sharp_patches,
                blurred_patches,
                args.size,
                lambda_tv=args.lambda_tv,
                mu_sum=args.mu_sum,
                max_it=args.max_iter,
                tol=args.tolerance,
                multiscale=args.multiscale,
                n_scales=args.n_scales,
                verbose=verbose,
            )

            _save_psf(psf, args.output_dir / f"{args.output_prefix}.png", args.bit16)
            print(f"PSF saved to: {args.output_dir / f'{args.output_prefix}.png'}")
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


def _save_psf_grid(psfs, n_rows, n_cols, output_path, bit16=False):
    """Save a grid of PSFs as a combined visualization."""
    psf_h, psf_w = psfs[0].shape
    border = 2

    # Create combined image with borders
    combined_h = n_rows * psf_h + (n_rows + 1) * border
    combined_w = n_cols * psf_w + (n_cols + 1) * border
    combined = np.ones((combined_h, combined_w))  # White background

    # Place each PSF
    for idx, psf in enumerate(psfs):
        row = idx // n_cols
        col = idx % n_cols
        y_start = border + row * (psf_h + border)
        x_start = border + col * (psf_w + border)

        # Normalize individual PSF for visualization
        psf_vis = psf / psf.max() if psf.max() > 0 else psf
        combined[y_start : y_start + psf_h, x_start : x_start + psf_w] = psf_vis

    if bit16:
        combined_out = (combined * 65535).astype(np.uint16)
    else:
        combined_out = (combined * 255).astype(np.uint8)

    iio.imwrite(output_path, combined_out)


if __name__ == "__main__":
    main()
