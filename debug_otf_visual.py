#!/usr/bin/env python3
"""Visualize OTF structure to understand banding."""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from pathlib import Path
import imageio.v3 as iio
from skimage.transform import resize

from deconv import psf2otf, img_to_norm_grayscale


def create_blur_kernel(image_path, size):
    """Load and resize blur kernel from image."""
    kernel = iio.imread(image_path)
    kernel = img_to_norm_grayscale(kernel)
    kernel = resize(kernel, (size, size), order=3, anti_aliasing=True)
    kernel = kernel / kernel.sum()
    return kernel


def visualize_otf(kernel, shape, name, output_dir):
    """Visualize OTF magnitude pattern."""
    otf = psf2otf(kernel, shape)
    abs_otf = np.abs(otf)

    # Save OTF magnitude (shifted for visualization)
    otf_shifted = fftshift(abs_otf)

    # Normalize for visualization
    otf_vis = otf_shifted / otf_shifted.max()
    iio.imwrite(output_dir / f'{name}_otf_magnitude.png',
                (otf_vis * 255).astype(np.uint8))

    # Log scale for better visibility
    otf_log = np.log10(otf_shifted + 1e-10)
    otf_log = (otf_log - otf_log.min()) / (otf_log.max() - otf_log.min())
    iio.imwrite(output_dir / f'{name}_otf_log_magnitude.png',
                (otf_log * 255).astype(np.uint8))

    # Threshold view - show where OTF is small
    threshold = 0.01 * abs_otf.max()
    otf_thresh = (abs_otf < threshold).astype(np.uint8) * 255
    iio.imwrite(output_dir / f'{name}_otf_nearzero.png',
                fftshift(otf_thresh))

    # Save kernel itself
    kernel_vis = kernel / kernel.max()
    iio.imwrite(output_dir / f'{name}_kernel.png',
                (kernel_vis * 255).astype(np.uint8))

    print(f"Saved {name} OTF visualizations")

    return otf


def test_psf2otf_correctness():
    """Test that psf2otf is computing correctly."""
    print("\n=== Testing psf2otf correctness ===")

    # Simple test: Gaussian kernel
    from scipy.ndimage import gaussian_filter
    size = 15
    gaussian = np.zeros((size, size))
    gaussian[size//2, size//2] = 1.0
    gaussian = gaussian_filter(gaussian, sigma=3.0)
    gaussian = gaussian / gaussian.sum()

    shape = (64, 64)
    otf = psf2otf(gaussian, shape)

    # For a Gaussian, OTF should also be roughly Gaussian (no zeros/rings)
    abs_otf = np.abs(otf)
    print(f"Gaussian OTF min: {abs_otf.min():.6f}")
    print(f"Gaussian OTF max: {abs_otf.max():.6f}")

    # Check for near-zeros
    threshold = 0.01 * abs_otf.max()
    near_zero_pct = 100 * np.sum(abs_otf < threshold) / abs_otf.size
    print(f"Gaussian OTF near-zero %: {near_zero_pct:.1f}%")

    # Delta function test
    delta = np.zeros((3, 3))
    delta[1, 1] = 1.0
    otf_delta = psf2otf(delta, shape)
    abs_otf_delta = np.abs(otf_delta)
    print(f"\nDelta OTF (should be all 1s):")
    print(f"  min: {abs_otf_delta.min():.6f}")
    print(f"  max: {abs_otf_delta.max():.6f}")
    print(f"  all close to 1: {np.allclose(abs_otf_delta, 1.0)}")


def compare_psf2otf_methods():
    """Compare different psf2otf implementations."""
    print("\n=== Comparing psf2otf methods ===")

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'

    kernel = create_blur_kernel(kernel_path, 25)
    shape = (100, 150)

    # Method 1: Our implementation
    otf1 = psf2otf(kernel, shape)

    # Method 2: Different centering approach
    def psf2otf_alt(psf, shape):
        """Alternative psf2otf with different centering."""
        psf = np.asarray(psf)
        padded = np.zeros(shape, dtype=psf.dtype)

        # Place PSF in center of padded array
        start_h = (shape[0] - psf.shape[0]) // 2
        start_w = (shape[1] - psf.shape[1]) // 2
        padded[start_h:start_h+psf.shape[0], start_w:start_w+psf.shape[1]] = psf

        # Shift to put center at (0,0)
        shift_h = shape[0] // 2
        shift_w = shape[1] // 2
        padded = np.roll(padded, (-shift_h, -shift_w), axis=(0, 1))

        return fft2(padded)

    otf2 = psf2otf_alt(kernel, shape)

    # Method 3: Even simpler - place at top-left, shift by PSF center
    def psf2otf_simple(psf, shape):
        """Simplest psf2otf."""
        padded = np.zeros(shape)
        padded[:psf.shape[0], :psf.shape[1]] = psf
        # Shift so that center of PSF is at (0,0)
        center = (psf.shape[0] // 2, psf.shape[1] // 2)
        padded = np.roll(padded, (-center[0], -center[1]), axis=(0, 1))
        return fft2(padded)

    otf3 = psf2otf_simple(kernel, shape)

    print(f"OTF1 (our impl) DC: {otf1[0,0]:.6f}")
    print(f"OTF2 (alt center) DC: {otf2[0,0]:.6f}")
    print(f"OTF3 (simple) DC: {otf3[0,0]:.6f}")

    print(f"\nOTF1 vs OTF3 diff: {np.abs(otf1 - otf3).max():.10f}")

    # Check OTF row structure
    abs_otf1 = np.abs(otf1)
    print(f"\nOTF1 structure check:")
    print(f"  Row 0 mean: {abs_otf1[0, :].mean():.6f}")
    print(f"  Row 1 mean: {abs_otf1[1, :].mean():.6f}")
    print(f"  Row 50 mean: {abs_otf1[50, :].mean():.6f}")


def test_wiener_regularization():
    """Test if regularization strength affects banding."""
    print("\n=== Testing regularization effect on banding ===")

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'
    image_path = base_dir / 'images' / 'houses_big.jpg'

    # Load image
    I = iio.imread(image_path)
    I = resize(I, (int(I.shape[0] * 0.15), int(I.shape[1] * 0.15)), anti_aliasing=True)
    I = I.astype(np.float64)
    I = I / I.max()
    I = I ** 2.0

    img = I[:, :, 2]
    kernel = create_blur_kernel(kernel_path, 43)
    shape = img.shape

    otf = psf2otf(kernel, shape)
    Nomin1 = np.conj(otf) * fft2(img)
    Denom1 = np.abs(otf) ** 2

    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    tau = 0.1

    for lambda_val in [10, 100, 750, 5000]:
        # Wiener filter: x = (c*Nomin1 + fft2(f)) / (c*Denom1 + 1)
        # where c = tau * 2 * lambda
        c = tau * 2 * lambda_val
        x = (c * Nomin1 + fft2(img)) / (c * Denom1 + 1)
        result = np.real(ifft2(x))

        row_means = result.mean(axis=1)
        row_diffs = np.diff(row_means)

        print(f"\nlambda={lambda_val}:")
        print(f"  Row-to-row max jump: {np.abs(row_diffs).max():.4f}")
        print(f"  Row mean std: {row_means.std():.4f}")

        # Save result
        result_clipped = np.clip(result, 0, 1)
        iio.imwrite(output_dir / f'debug_lambda_{lambda_val}.png',
                    (result_clipped * 255).astype(np.uint8))


def test_pure_wiener():
    """Test pure Wiener filter without primal-dual iteration."""
    print("\n=== Testing pure Wiener filter ===")

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'
    image_path = base_dir / 'images' / 'houses_big.jpg'

    # Load image
    I = iio.imread(image_path)
    I = resize(I, (int(I.shape[0] * 0.15), int(I.shape[1] * 0.15)), anti_aliasing=True)
    I = I.astype(np.float64)
    I = I / I.max()

    img = I[:, :, 0]  # Test on red channel
    kernel = create_blur_kernel(kernel_path, 25)
    shape = img.shape

    # Blur the image first
    from scipy.ndimage import convolve
    blurred = convolve(img, kernel, mode='reflect')

    # Now deconvolve with Wiener
    otf = psf2otf(kernel, shape)
    Denom = np.abs(otf) ** 2

    output_dir = Path(__file__).parent / 'output'

    for snr in [0.001, 0.01, 0.1, 1.0]:
        # Classic Wiener: H* / (|H|^2 + 1/SNR)
        wiener = np.conj(otf) / (Denom + snr)
        result = np.real(ifft2(wiener * fft2(blurred)))

        row_means = result.mean(axis=1)
        row_diffs = np.diff(row_means)

        print(f"\nSNR={snr}:")
        print(f"  Row-to-row max jump: {np.abs(row_diffs).max():.4f}")

        result_clipped = np.clip(result, 0, 1)
        iio.imwrite(output_dir / f'debug_wiener_snr_{snr}.png',
                    (result_clipped * 255).astype(np.uint8))


if __name__ == '__main__':
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'

    # Visualize OTFs for different kernel sizes
    for size in [15, 25, 43]:
        kernel = create_blur_kernel(kernel_path, size)
        visualize_otf(kernel, (288, 384), f'kernel_{size}', output_dir)

    test_psf2otf_correctness()
    compare_psf2otf_methods()
    test_wiener_regularization()
    test_pure_wiener()
