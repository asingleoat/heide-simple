#!/usr/bin/env python3
"""
A wrapper for estimate_psf.py to simplify its interface.

This script takes a sharp image, a blurred image, and an output folder,
and runs the PSF estimation with sensible defaults for spatially-varying,
multiscale PSF estimation.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Simplified wrapper for PSF estimation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --sharp sharp_image.png --blurred blurred_image.png --output ./psf_folder
        """
    )

    parser.add_argument('--sharp', type=Path, required=True,
                        help='Sharp reference image (e.g., pinhole aperture)')
    parser.add_argument('--blurred', type=Path, required=True,
                        help='Blurred image (e.g., wide aperture)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output folder to save the estimated PSFs')
    
    args = parser.parse_args()

    # --- Sensible Defaults ---
    psf_size = 31
    tiles = '9x12'
    # The `estimate_psf.py` script does not have a multi-threading option itself.
    # The `--workers` option is available in `deconvolve.py`.
    # We will use --multiscale for faster convergence on large PSFs as requested.
    use_multiscale = True

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    print("--- Starting PSF Estimation with Sensible Defaults ---")
    print(f"  Sharp Image: {args.sharp}")
    print(f"  Blurred Image: {args.blurred}")
    print(f"  Output Folder: {args.output}")
    print(f"  PSF Size: {psf_size}")
    print(f"  Tile Grid: {tiles}")
    print(f"  Multiscale: {use_multiscale}")
    print("----------------------------------------------------")

    # Build the command to call the original script
    command = [
        './dev',
        'python',
        'estimate_psf.py',
        '--sharp', str(args.sharp),
        '--blurred', str(args.blurred),
        '--size', str(psf_size),
        '--tiles', tiles,
        '--output-dir', str(args.output), # Map wrapper's --output to new --output-dir
        '--output-prefix', 'psf',         # Explicitly set prefix to 'psf'
        '--verbose'
    ]
    if use_multiscale:
        command.append('--multiscale')

    try:
        # Execute the command
        process = subprocess.run(command, check=True, text=True, capture_output=False)
        print("\n--- PSF Estimation Completed Successfully ---")
        print(f"PSFs saved in: {args.output}")

    except subprocess.CalledProcessError as e:
        print("\n--- Error during PSF Estimation ---", file=sys.stderr)
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("\n--- Error: './dev' script not found. ---", file=sys.stderr)
        print("Please ensure you are running this wrapper from the root of the project directory.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
