#!/usr/bin/env python3
"""
A wrapper for deconvolve.py to simplify its interface.

This script takes a blurred image, a folder of PSFs, and an output path.
It automatically infers the tile grid from the PSF filenames and runs
tiled deconvolution using the maximum number of available workers.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import glob
import re

def main():
    parser = argparse.ArgumentParser(
        description='Simplified wrapper for tiled deconvolution.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --blurred blurred_image.png --psf_folder ./psf_folder --output sharp_image.png
        """
    )

    parser.add_argument('--blurred', type=Path, required=True,
                        help='Blurred image to deconvolve')
    parser.add_argument('--psf_folder', type=Path, required=True,
                        help='Folder containing the estimated PSF files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output path for the deblurred image')
    
    args = parser.parse_args()

    # --- Infer Tile Grid from PSF folder ---
    psf_files = glob.glob(str(args.psf_folder / '*_tile_*.png'))
    if not psf_files:
        print(f"Error: No tiled PSF files (e.g., 'psf_red_tile_0_0.png') found in '{args.psf_folder}'", file=sys.stderr)
        sys.exit(1)

    max_row, max_col = 0, 0
    base_name = ''
    for f in psf_files:
        match = re.search(r'(.+)_tile_(\d+)_(\d+)\.png', Path(f).name)
        if match:
            if not base_name:
                # This will find the base name, e.g., 'psf_red'
                base_name_match = re.search(r'(.+)_[a-z]+_tile_', Path(f).name)
                if base_name_match:
                    base_name = base_name_match.group(1)

            max_row = max(max_row, int(match.group(2)))
            max_col = max(max_col, int(match.group(3)))
    
    if not base_name:
        print(f"Error: Could not determine the base name of the PSF files (e.g., 'psf') from folder '{args.psf_folder}'", file=sys.stderr)
        sys.exit(1)

    tiles_h = max_row + 1
    tiles_w = max_col + 1
    tiles = f"{tiles_h}x{tiles_w}"

    # --- Sensible Defaults ---
    # Use 0 to auto-detect and use all available CPUs
    workers = 1 
    kernel_tiles_base_arg = args.psf_folder

    print("--- Starting Deconvolution with Sensible Defaults ---")
    print(f"  Blurred Image: {args.blurred}")
    print(f"  PSF Folder: {args.psf_folder}")
    print(f"  Output Image: {args.output}")
    print(f"  Inferred Tile Grid: {tiles}")
    print(f"  Parallel Workers: {workers} (0=auto)")
    print(f"  Kernel Base Path: {kernel_tiles_base_arg}")
    print("-------------------------------------------------------")

    # Build the command to call the original script
    command = [
        './dev',
        'python',
        'deconvolve.py',
        str(args.blurred),
        '--kernel-tiles', str(kernel_tiles_base_arg),
        '--tiles', tiles,
        '--workers', str(workers),
        '-o', str(args.output)
    ]

    try:
        # Execute the command
        process = subprocess.run(command, check=True, text=True, capture_output=False)
        print("\n--- Deconvolution Completed Successfully ---")
        print(f"Sharp image saved to: {args.output}")

    except subprocess.CalledProcessError as e:
        print("\n--- Error during Deconvolution ---", file=sys.stderr)
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("\n--- Error: './dev' script not found. ---", file=sys.stderr)
        print("Please ensure you are running this wrapper from the root of the project directory.", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
