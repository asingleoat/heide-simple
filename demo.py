#!/usr/bin/env python3
"""
Demo script for primal-dual cross-channel deconvolution.

Python port of deblurring_launch.m from:
F. Heide, M. Rouf, M. Hullin, B. Labitzke, W. Heidrich, A. Kolb.
"High-Quality Computational Imaging Through Simple Lenses."
ACM ToG 2013
"""

import numpy as np
from pathlib import Path
import imageio.v3 as iio
from skimage.transform import resize
from skimage.util import random_noise

from deconv import pd_joint_deconv, img_to_norm_grayscale


def create_blur_kernel(image_path, size):
    """Load and resize blur kernel from image."""
    kernel = iio.imread(image_path)
    kernel = img_to_norm_grayscale(kernel)
    kernel = resize(kernel, (size, size), order=3, anti_aliasing=True)
    kernel = kernel / kernel.sum()  # Normalize to sum to 1
    return kernel


def compute_psnr(original, reconstructed, border_pad=0):
    """Compute Peak Signal-to-Noise Ratio."""
    if border_pad > 0:
        original = original[border_pad:-border_pad, border_pad:-border_pad]
        reconstructed = reconstructed[border_pad:-border_pad, border_pad:-border_pad]

    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def main():
    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'

    # Load image
    image_path = base_dir / 'images' / 'houses_big.jpg'
    print(f"Loading image: {image_path}")
    I = iio.imread(image_path)
    I = resize(I, (int(I.shape[0] * 0.15), int(I.shape[1] * 0.15)), anti_aliasing=True)
    I = I.astype(np.float64)
    I = I / I.max()

    print(f"Processing image with size {I.shape[1]} x {I.shape[0]}")

    # Apply inverse gamma (linearize)
    I = I ** 2.0

    # Store sharp reference
    I_sharp = I.copy()
    n_channels = I.shape[2]

    # Create blur kernels (different sizes per channel to simulate chromatic aberration)
    blur_size = 15
    inc_blur_exp = 1.7
    kernel_path = base_dir / 'kernels' / 'fading.png'

    K_blur = []
    for ch in range(n_channels):
        curr_blur_size = round(blur_size * (inc_blur_exp ** ch))
        curr_blur_size = curr_blur_size + (1 - curr_blur_size % 2)  # Make odd

        kernel = create_blur_kernel(kernel_path, curr_blur_size)
        K_blur.append(kernel)

    # Set first channel to sharp (delta function) - simulates one sharp channel
    center = blur_size // 2
    K_blur[0] = np.zeros((blur_size, blur_size))
    K_blur[0][center, center] = 1.0

    # Apply blur and noise to create synthetic test data
    noise_sd = 0.005
    kernel_var = 0.0000001
    I_blurred = np.zeros_like(I)
    K_blur_noisy = []

    for ch in range(n_channels):
        # Convolve with kernel (using scipy for proper boundary handling)
        from scipy.ndimage import convolve
        I_blurred[:, :, ch] = convolve(I_sharp[:, :, ch], K_blur[ch], mode='reflect')

        # Add Gaussian noise
        I_blurred[:, :, ch] = random_noise(I_blurred[:, :, ch], mode='gaussian',
                                            var=noise_sd ** 2, clip=False)

        # Add noise to kernel (simulating kernel uncertainty)
        K_noisy = K_blur[ch] + np.random.randn(*K_blur[ch].shape) * np.sqrt(kernel_var / (ch + 1) ** 2)
        K_noisy = np.maximum(K_noisy, 0)
        K_noisy = K_noisy / K_noisy.sum()
        K_blur_noisy.append(K_noisy)

    # Prepare channel data for deconvolution
    channels = []
    for ch in range(n_channels):
        channels.append({
            'image': I_blurred[:, :, ch],
            'kernel': K_blur_noisy[ch]
        })

    # Lambda parameters for cross-channel deconvolution
    # Format: [ch_idx (1-based), lambda_residual, lambda_tv, lambda_black, lambda_cross_ch..., n_detail_layers]
    # Note: Original MATLAB used lambda_residual=750, lambda_tv=0.5, lambda_cross=1.0
    # We use stronger regularization to avoid banding artifacts in Python
    lambda_params = np.array([
        [1, 200, 2.0, 0.0, 0.0, 0.0, 0.0, 1],  # Channel 1 (sharp reference)
        [2, 200, 2.0, 0.0, 3.0, 0.0, 0.0, 0],  # Channel 2, coupled to channel 1
        [3, 200, 2.0, 0.0, 3.0, 0.0, 0.0, 0],  # Channel 3, coupled to channel 1
    ])

    # Run primal-dual cross-channel deconvolution
    print("\nComputing cross-channel deconvolution...")
    result = pd_joint_deconv(channels, lambda_params, max_it=200, tol=1e-4, verbose='brief')

    # Gather results
    I_deconv = np.zeros_like(I)
    for ch in range(n_channels):
        I_deconv[:, :, ch] = result[ch]['image']

    # Compute PSNR
    psnr_pad = round(blur_size * 1.5)
    psnr = compute_psnr(I_sharp, I_deconv, psnr_pad)
    print(f"\nPSNR: {psnr:.2f} dB")

    # Apply gamma correction for display
    I_deconv = np.clip(I_deconv, 0, None)
    I_deconv = I_deconv ** 0.5

    I_blurred_disp = np.clip(I_blurred, 0, None) ** 0.5
    I_sharp_disp = I_sharp ** 0.5

    # Save results
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    iio.imwrite(output_dir / 'I_sharp.png', (np.clip(I_sharp_disp, 0, 1) * 255).astype(np.uint8))
    iio.imwrite(output_dir / 'I_blurred.png', (np.clip(I_blurred_disp, 0, 1) * 255).astype(np.uint8))
    iio.imwrite(output_dir / 'I_deconv.png', (np.clip(I_deconv, 0, 1) * 255).astype(np.uint8))

    print(f"\nResults saved to {output_dir}/")
    print("  I_sharp.png   - Original sharp image")
    print("  I_blurred.png - Synthetically blurred image")
    print("  I_deconv.png  - Deconvolved result")


if __name__ == '__main__':
    main()
