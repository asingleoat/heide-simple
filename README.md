# Primal-Dual Cross-Channel Deconvolution

Python port of the primal-dual cross-channel deconvolution algorithm from:

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

## Quick Start

Deconvolve an image with a known PSF:

```bash
./dev python deconvolve.py blurry.jpg --kernel psf.png -o sharp.png
```

Deconvolve with a Gaussian blur estimate:

```bash
./dev python deconvolve.py blurry.jpg --gaussian 2.5 -o sharp.png
```

## Command-Line Usage

```
usage: deconvolve.py [-h] [-o OUTPUT] (--kernel KERNEL | --gaussian SIGMA)
                     [--kernel-size KERNEL_SIZE] [--channels {rgb,r,g,b,gray}]
                     [--lambda-res LAMBDA_RES] [--lambda-tv LAMBDA_TV]
                     [--lambda-cross LAMBDA_CROSS] [--max-iter MAX_ITER]
                     [--tolerance TOLERANCE] [--linear] [--gamma GAMMA] [-v]
                     [--16bit]
                     input
```

### Required Arguments

- `input` - Input image file (JPEG, PNG, etc.)
- `--kernel KERNEL` or `--gaussian SIGMA` - Specify the blur kernel

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `input_deconv.ext` | Output file path |
| `--kernel-size` | auto | Resize kernel to this size (must be odd) |
| `--channels` | `rgb` | Channels to process: `rgb`, `r`, `g`, `b`, `gray` |
| `--lambda-res` | 200 | Data fidelity weight (lower = more regularization) |
| `--lambda-tv` | 2.0 | Total variation weight (higher = smoother) |
| `--lambda-cross` | 3.0 | Cross-channel coupling weight |
| `--max-iter` | 200 | Maximum iterations per channel |
| `--tolerance` | 1e-4 | Convergence tolerance |
| `--linear` | off | Input is already linear (skip gamma decoding) |
| `--gamma` | 2.2 | Gamma value for encoding/decoding |
| `-v, --verbose` | off | Print progress information |
| `--16bit` | off | Save as 16-bit image |

### Examples

```bash
# Basic deconvolution with a PSF image
./dev python deconvolve.py photo.jpg --kernel measured_psf.png

# Gaussian blur removal with verbose output
./dev python deconvolve.py blurry.png --gaussian 3.0 -v -o sharp.png

# Process only the red channel with stronger regularization
./dev python deconvolve.py image.jpg --kernel psf.png --channels r --lambda-res 100

# High-quality 16-bit output
./dev python deconvolve.py raw_photo.png --kernel psf.png --16bit --linear

# Adjust regularization for challenging images
./dev python deconvolve.py noisy.jpg --kernel psf.png --lambda-tv 5.0 --lambda-res 100
```

## Python API

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

# Lambda parameters: [ch_idx, lambda_res, lambda_tv, lambda_black, lambda_cross..., n_detail]
lambda_params = np.array([
    [1, 200, 2.0, 0.0, 0.0, 0.0, 0.0, 1],
    [2, 200, 2.0, 0.0, 3.0, 0.0, 0.0, 0],
    [3, 200, 2.0, 0.0, 3.0, 0.0, 0.0, 0],
])

# Run deconvolution
result = pd_joint_deconv(channels, lambda_params, max_it=200, tol=1e-4)

# Extract results
output = np.stack([result[i]['image'] for i in range(3)], axis=2)
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
| Horizontal/vertical banding | Decrease `--lambda-res` (try 100-150) |
| Too blurry result | Increase `--lambda-res` (try 300-500) |
| Noisy/grainy result | Increase `--lambda-tv` (try 3.0-5.0) |
| Over-smoothed details | Decrease `--lambda-tv` (try 0.5-1.0) |
| Color fringing | Increase `--lambda-cross` (try 5.0-10.0) |

## Running the Demo

```bash
./dev python demo.py
```

This creates synthetic test data with chromatic aberration and demonstrates the cross-channel deconvolution. Results are saved to `output/`.

## Files

- `deconvolve.py` - Command-line tool
- `demo.py` - Demonstration script with synthetic data
- `deconv/` - Core algorithm package
  - `pd_joint_deconv.py` - Main primal-dual algorithm
  - `operator_norm.py` - Operator norm computation
  - `utils.py` - Utility functions (psf2otf, edgetaper, etc.)
- `DIFFERENCES_FROM_MATLAB.md` - Documentation of differences from original MATLAB

## License

The original MATLAB code is by Felix Heide (fheide@cs.ubc.ca). This Python port follows the same academic use terms. Please cite the original paper if you use this code in research.
