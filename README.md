# Primal-Dual Cross-Channel Deconvolution

Claude Code and Google Gemini driven Python port of the primal-dual cross-channel deconvolution algorithm from:

> F. Heide, M. Rouf, M. Hullin, B. Labitzke, W. Heidrich, A. Kolb.
> "High-Quality Computational Imaging Through Simple Lenses."
> ACM ToG 2013

## Installation

This project uses a Nix shell for dependencies. If you have Nix installed:

```bash
# Enter the development environment
nix-shell shell.nix

# Or use the helper script
./dev python deconvolve.py --help
```

Required Python packages (if not using Nix):
- numpy
- scipy
- scikit-image
- imageio

To install these, you can use pip, you probably also want to use a python venv, ask an LLM for advice:
```bash
pip install -r requirements.txt
```

## Quick Start

Deconvolve an image with a known PSF:

```bash
./dev python deconvolve.py blurry.jpg --kernel psf.png -o sharp.png
```

Deconvolve with a Gaussian blur estimate:

```bash
./dev python deconvolve.py blurry.jpg --gaussian 2.5 -o sharp.png
```

## Example Workflow

This repository includes example images to demonstrate the full workflow. The images `sharp-f8.jpg` (shot at f/8) and `blurry-f1.2.jpg` (shot at f/1.2 with the same lens) can be used to estimate the lens PSF and then deconvolve the blurry image. WARNING: this will take a very long time, this is horrendously computationally expensive, and these are very high resolution pics.

### Step 1: Estimate the PSF

Use the simplified wrapper to estimate spatially-varying PSFs from the image pair:

```bash
./dev python estimate_psf_wrapper.py \
    --sharp sharp-f8.jpg \
    --blurred blurry-f1.2.jpg \
    --output ./psf_out
```

This estimates a 3x4 grid of per-channel PSFs (accounting for chromatic aberration) and saves them to `./psf_out/`.

### Step 2: Deconvolve the Blurry Image

Use the deconvolution wrapper with the estimated PSFs:

```bash
./dev python deconvolve_wrapper.py \
    --blurred blurry-f1.2.jpg \
    --psf_folder ./psf_out \
    --output sharp-restored.jpg
```

The wrapper automatically detects the tile grid from the PSF filenames and runs tiled deconvolution.

## PSF Estimation

If you don't have a known PSF for your lens, you can estimate it using calibration images.

### Quick PSF Estimation

```bash
# Generate a calibration pattern to print
./dev python estimate_psf.py --generate-pattern --output-dir ./

# Estimate PSF from sharp/blurred image pair
./dev python estimate_psf.py --sharp pinhole.png --blurred wide_aperture.png --size 31 --output-dir ./
```

### Calibration Workflow

1. **Generate calibration pattern**:
   ```bash
   ./dev python estimate_psf.py --generate-pattern --patch-size 128 --grid 4x4 --output-dir ./
   ```

2. **Print the pattern** and mount it flat

3. **Capture two photos**:
   - Sharp reference: Use pinhole/small aperture (e.g., f/22)
   - Blurred image: Use wide aperture (e.g., f/2.8)

4. **Estimate PSF**:
   ```bash
   ./dev python estimate_psf.py --sharp-pattern sharp.png --blurred-pattern blurred.png \
       --size 43 --grid 4x4 --output-dir ./psf_out
   ```

### Per-Channel PSF Estimation (Default)

For color images, per-channel PSF estimation is enabled by default. This accounts for chromatic aberration where each color channel has a different blur:

```bash
./dev python estimate_psf.py --sharp sharp.png --blurred blurred.png \
    --size 31 --output-dir ./
# Creates: psf_red.png, psf_green.png, psf_blue.png
```

To estimate a single grayscale PSF instead:

```bash
./dev python estimate_psf.py --sharp sharp.png --blurred blurred.png \
    --size 31 --grayscale --output-dir ./
# Creates: psf.png
```

### Multiscale PSF Estimation

For faster convergence on large PSFs, use scale-space estimation:

```bash
./dev python estimate_psf.py --sharp sharp.png --blurred blurred.png \
    --size 63 --multiscale --output-dir ./
# Creates: psf_red.png, psf_green.png, psf_blue.png (or psf.png with --grayscale)
```

### Spatially-Varying PSF (Tile-based)

For lenses with spatially-varying blur (e.g., wide-angle lenses with field curvature), estimate PSFs across a tile grid:

```bash
./dev python estimate_psf.py --sharp sharp.png --blurred blurred.png \
    --size 31 --tiles 3x3 --output-dir ./psf_out
# For color images (default):
# Creates: psf_out/psf_red_tile_0_0.png, psf_out/psf_green_tile_0_0.png, ...
# Plus combined grids: psf_out/psf_red.png, psf_out/psf_green.png, psf_out/psf_blue.png

# For grayscale (--grayscale flag):
# Creates: psf_out/psf_tile_0_0.png, psf_out/psf_tile_0_1.png, ..., and psf_out/psf.png (combined grid)
```

### PSF Estimation Options

| Option | Default | Description |
|---|---|---|
| `--size` | 31 | PSF size in pixels (must be odd) |
| **Basic Parameters** | | |
| `--tv`, `--lambda-tv` | 0.001 | TV regularization on PSF |
| `--mu`, `--mu-sum` | 50.0 | Sum-to-one constraint weight |
| `--max-iter` | 500 | Maximum iterations |
| `--tolerance` | 1e-5 | Convergence tolerance |
| `--grayscale` | off | Estimate single grayscale PSF (default: per-channel for color) |
| `--multiscale` | off | Use scale-space estimation (faster for large PSFs) |
| `--n-scales` | auto | Number of scales for multiscale |
| **Spatially-Varying PSF Estimation (Tiled)** | | |
| `--tiles` | off | Tile grid for spatially-varying PSF (e.g., 3x3) |
| `--tile-overlap` | 0.25 | Tile overlap fraction (0-0.5) |
| `--smooth-sigma` | 1.0 | Spatial smoothing between tile PSFs |
| **Calibration Pattern Generation & Patch Extraction** | | |
| `--patch-size` | 128 | Size of noise patches in calibration pattern |
| `--grid` | 4x4 | Grid of patches in calibration pattern |
| `--border` | 20 | Border width (should be ≥ expected blur radius) |
| `--seed` | 42 | Random seed for pattern generation |
| `--crop-border` | 5 | Extra border to crop from patches |
| **Other Options** | | |
| `-v`, `--verbose` | off | Print progress information |
| `--16bit` | off | Save PSF as 16-bit image |

## Command-Line Usage (Deconvolution)

```
usage: deconvolve.py [-h] [-o OUTPUT] (--kernel KERNEL | --gaussian SIGMA)
                     [--kernel-size KERNEL_SIZE] [--channels {rgb,r,g,b,gray}]
                     [--lambda-res LAMBDA_RES] [--lambda-tv LAMBDA_TV]
                     [--lambda-cross LAMBDA_CROSS] [--max-iter MAX_ITER]
                     [--tolerance TOLERANCE] [--linear] [--gamma GAMMA] [-q]
                     [--16bit]
                     input
```

### Required Arguments

- `input` - Input image file (JPEG, PNG, etc.)
- `--kernel KERNEL`, `--gaussian SIGMA`, or `--kernel-tiles PATH` - Specify the blur kernel(s)

### Options

| Option | Default | Description |
|---|---|---|
| `-o, --output` | `input_deconv.ext` | Output file path |
| `--kernel-size` | auto | Resize kernel to this size (must be odd) |
| **Tiled Deconvolution** | | |
| `--kernel-tiles` | - | Directory or base path for tiled PSFs |
| `--tiles` | auto | Tile grid for tiled deconvolution (e.g., 3x3) |
| `--tile-overlap` | 0.25 | Tile overlap fraction (0-0.5) |
| `--workers` | 1 | Parallel workers for tiled deconvolution (0=auto) |
| **Algorithm Parameters** | | |
| `--channels` | `rgb` | Channels to process: `rgb`, `r`, `g`, `b`, `gray` |
| `--res`, `--lambda-res` | 200 | Data fidelity weight (lower = more regularization) |
| `--tv`, `--lambda-tv` | 2.0 | Total variation weight (higher = smoother) |
| `--cross`, `--lambda-cross` | 3.0 | Cross-channel coupling weight |
| `--max-iter` | 200 | Maximum iterations per channel |
| `--tolerance` | 1e-4 | Convergence tolerance |
| **Other Options** | | |
| `--linear` | off | Input is already linear (skip gamma decoding) |
| `--gamma` | 2.2 | Gamma value for encoding/decoding |
| `-q`, `--quiet` | off | Suppress progress output |
| `--16bit` | off | Save as 16-bit image |

### Examples

```bash
# Basic deconvolution with a PSF image
./dev python deconvolve.py photo.jpg --kernel measured_psf.png

# Gaussian blur removal (use -q for quiet mode)
./dev python deconvolve.py blurry.png --gaussian 3.0 -o sharp.png

# Deconvolve with spatially-varying PSFs (from tiled estimation)
# Automatically detects and uses per-channel PSFs if present
./dev python deconvolve.py photo.jpg --kernel-tiles ./psf --tiles 3x3 -o sharp.png

# Parallel tiled deconvolution (use all CPUs)
./dev python deconvolve.py photo.jpg --kernel-tiles ./psf --tiles 3x3 --workers 0 -o sharp.png

# Process only the red channel with stronger regularization
./dev python deconvolve.py image.jpg --kernel psf.png --channels r --res 100

# High-quality 16-bit output
./dev python deconvolve.py raw_photo.png --kernel psf.png --16bit --linear

# Adjust regularization for challenging images
./dev python deconvolve.py noisy.jpg --kernel psf.png --tv 5.0 --res 100
```

## Python API

### Deconvolution

```python
from deconv import pd_joint_deconv, img_to_norm_grayscale
import numpy as np

# Load your image and kernel
image = ...  # (H, W, C) float array in [0, 1]
kernel = ...  # (K, K) float array, normalized to sum=1

# Prepare channel data
channels = [
    {'image': image[:, :, 0], 'kernel': kernel},
    {'image': image[:, :, 1], 'kernel': kernel},
    {'image': image[:, :, 2], 'kernel': kernel},
]

# Lambda parameters: [ch_idx, lambda_residual, lambda_tv, lambda_cross_ch..., n_detail]
lambda_params = np.array([
    [1, 200, 2.0, 0.0, 0.0, 0.0, 1],
    [2, 200, 2.0, 3.0, 0.0, 0.0, 0],
    [3, 200, 2.0, 3.0, 0.0, 0.0, 0],
])

# Run deconvolution
result = pd_joint_deconv(channels, lambda_params, max_it=200, tol=1e-4)

# Extract results
output = np.stack([result[i]['image'] for i in range(3)], axis=2)
```

### PSF Estimation

```python
from deconv import (
    estimate_psf, estimate_psf_multiscale, estimate_psf_tiled,
    create_calibration_pattern, extract_patches_from_images,
    get_psf_at_position
)
import numpy as np

# Option 1: Estimate from a single sharp/blurred pair
sharp = ...   # Sharp reference image (pinhole aperture)
blurred = ... # Blurred image (wide aperture)

psf = estimate_psf(sharp, blurred, psf_size=31,
                   tv=0.001, mu=50.0)

# Option 2: Multiscale estimation (faster for large PSFs)
psf = estimate_psf_multiscale(sharp, blurred, psf_size=63,
                              tv=0.001, mu=50.0)

# Option 3: Tile-based estimation for spatially-varying blur
psfs, tile_centers, tile_grid = estimate_psf_tiled(
    sharp, blurred, psf_size=31,
    n_tiles_h=3, n_tiles_w=3, overlap=0.25
)
# Interpolate PSF at any position
psf_at_center = get_psf_at_position(
    psfs, tile_centers, tile_grid,
    position=(h//2, w//2), image_shape=(h, w)
)

# Option 4: Generate calibration pattern
pattern, patch_coords = create_calibration_pattern(
    patch_size=128, n_patches_h=4, n_patches_w=4, border_width=20
)

# Option 5: Estimate from calibration pattern photos
sharp_patches, blurred_patches = extract_patches_from_images(
    sharp_image, blurred_image, patch_coords
)
psf = estimate_psf_from_patches(sharp_patches, blurred_patches, psf_size=31)
```

## Algorithm Overview

The algorithm uses primal-dual optimization to solve:

```
minimize  λ_res * ||Kx - y||² + λ_tv * ||∇x||₁ + λ_cross * ||∇x - ∇x_ref||₁
```

Where:
- `K` is the blur operator (convolution with PSF)
- `y` is the observed blurry image
- `∇` is the gradient operator (total variation)
- `x_ref` is a reference channel for cross-channel coupling

The cross-channel coupling allows sharper channels to guide the deconvolution of blurrier channels, which is useful for chromatic aberration correction.

## Parameter Tuning

If you see artifacts in the output:

| Artifact | Solution |
|----------|----------|
| Horizontal/vertical banding | Decrease `--res` (try 100-150) |
| Too blurry result | Increase `--res` (try 300-500) |
| Noisy/grainy result | Increase `--tv` (try 3.0-5.0) |
| Over-smoothed details | Decrease `--tv` (try 0.5-1.0) |
| Color fringing | Increase `--cross` (try 5.0-10.0) |

## Files

- `deconvolve.py` - Command-line deconvolution tool
- `estimate_psf.py` - Command-line PSF estimation tool
- `test_algorithms.py` - Test script to verify algorithms
- `deconv/` - Core algorithm package
  - `pd_joint_deconv.py` - Main primal-dual deconvolution algorithm
  - `psf_estimation.py` - PSF estimation algorithm
  - `operator_norm.py` - Operator norm computation
  - `utils.py` - Utility functions (psf2otf, edgetaper, etc.)
- `DIFFERENCES_FROM_MATLAB.md` - Documentation of differences from original MATLAB

## Testing

Run the test script to verify the algorithms:

```bash
./dev python test_algorithms.py
```

## License

This code is released under the GNU General Public License v3.0 (GPL-3.0). See [LICENSE](LICENSE) for details.

This is an independent Python implementation based on the algorithms described in the original paper. If you use this code in research, please cite the original paper:

> F. Heide, M. Rouf, M. Hullin, B. Labitzke, W. Heidrich, A. Kolb.
> "High-Quality Computational Imaging Through Simple Lenses."
> ACM Transactions on Graphics (ToG), 2013
