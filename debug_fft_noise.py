#!/usr/bin/env python3
"""Debug FFT noise amplification."""

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


def analyze_noise_amplification():
    """Analyze how noise gets amplified in the FFT solve."""
    print("=== Analyzing noise amplification in FFT solve ===")

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'
    image_path = base_dir / 'images' / 'houses_big.jpg'

    # Load image
    I = iio.imread(image_path)
    I = resize(I, (int(I.shape[0] * 0.15), int(I.shape[1] * 0.15)), anti_aliasing=True)
    I = I.astype(np.float64)
    I = I / I.max()
    I = I ** 2.0

    img = I[:, :, 2]  # Use blue channel (largest blur)
    kernel = create_blur_kernel(kernel_path, 43)
    shape = img.shape

    print(f"Image shape: {shape}")
    print(f"Kernel shape: {kernel.shape}")

    # Compute OTF
    otf = psf2otf(kernel, shape)
    abs_otf = np.abs(otf)

    print(f"\nOTF statistics:")
    print(f"  min |OTF|: {abs_otf.min():.6f}")
    print(f"  max |OTF|: {abs_otf.max():.6f}")
    print(f"  mean |OTF|: {abs_otf.mean():.6f}")

    # Compute Nomin1 and Denom1
    Nomin1 = np.conj(otf) * fft2(img)
    Denom1 = abs_otf ** 2

    print(f"\nNomin1 (H* Y) statistics:")
    print(f"  min |Nomin1|: {np.abs(Nomin1).min():.6f}")
    print(f"  max |Nomin1|: {np.abs(Nomin1).max():.6f}")
    print(f"  mean |Nomin1|: {np.abs(Nomin1).mean():.6f}")

    print(f"\nDenom1 (|H|²) statistics:")
    print(f"  min: {Denom1.min():.6f}")
    print(f"  max: {Denom1.max():.6f}")
    print(f"  mean: {Denom1.mean():.6f}")

    # Find frequencies where OTF is near zero but Nomin1 is not
    threshold = 0.01 * abs_otf.max()
    small_otf_mask = abs_otf < threshold

    print(f"\nAt frequencies where |OTF| < {threshold:.4f}:")
    small_nomin = np.abs(Nomin1[small_otf_mask])
    print(f"  Number of frequencies: {small_otf_mask.sum()} / {Denom1.size}")
    print(f"  |Nomin1| range: [{small_nomin.min():.6f}, {small_nomin.max():.6f}]")
    print(f"  |Nomin1| mean: {small_nomin.mean():.6f}")

    # The solve_fft computes: (c*Nomin1 + fft(f)) / (c*Denom1 + 1)
    # At small OTF frequencies: ≈ c*Nomin1 + fft(f)
    # If Nomin1 is noisy here, it passes through!

    # Check if the issue is that Nomin1 has a non-zero pattern at these frequencies
    # that shouldn't be there (i.e., it's noise from the observed image)

    # Let's check the ratio Nomin1 / OTF at various frequencies
    # At valid frequencies, this should equal fft(img)
    # At small OTF, this ratio will be huge

    # Instead, let's check what the theoretical "true" signal should be
    # If we had perfect deconvolution: X = Y / H (in freq domain)
    # But this blows up at OTF zeros

    # Let's see what c (the regularization constant) does
    tau = 0.9  # typical value
    lambda_residual = 750

    c = tau * 2 * lambda_residual
    print(f"\nWith tau={tau}, lambda={lambda_residual}:")
    print(f"  c = tau * 2 * lambda = {c:.1f}")

    # Effective denominator at small OTF:
    # c * Denom1 + 1 ≈ c * 0 + 1 = 1
    # So numerator passes through essentially unchanged

    # At these frequencies, x ≈ c * Nomin1 + fft(f)
    # If Nomin1 has amplitude ~0.001 and c = 2100, then c*Nomin1 ~ 2.1

    print(f"\nAt small OTF frequencies, c * |Nomin1| statistics:")
    c_nomin = c * small_nomin
    print(f"  range: [{c_nomin.min():.2f}, {c_nomin.max():.2f}]")
    print(f"  mean: {c_nomin.mean():.2f}")

    # Compare with fft(f) at same frequencies
    fft_img = fft2(img)
    fft_at_small = np.abs(fft_img[small_otf_mask])
    print(f"\n|fft(img)| at small OTF frequencies:")
    print(f"  range: [{fft_at_small.min():.2f}, {fft_at_small.max():.2f}]")
    print(f"  mean: {fft_at_small.mean():.2f}")

    # So the ratio of c*Nomin1 to fft(img) at small OTF frequencies tells us
    # how much the "noise" from deconvolution contributes relative to the input
    ratio = c_nomin / (fft_at_small + 1e-10)
    print(f"\nRatio c*|Nomin1| / |fft(img)| at small OTF:")
    print(f"  range: [{ratio.min():.2f}, {ratio.max():.2f}]")
    print(f"  mean: {ratio.mean():.2f}")
    print(f"  If this >> 1, deconvolution artifacts dominate")


def test_lower_lambda():
    """Test if lower lambda reduces banding."""
    print("\n=== Testing with lower lambda ===")

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'
    image_path = base_dir / 'images' / 'houses_big.jpg'

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

    tau = 0.9

    for lambda_val in [10, 50, 100, 300, 750]:
        c = tau * 2 * lambda_val
        x = (c * Nomin1 + fft2(img)) / (c * Denom1 + 1)
        result = np.real(ifft2(x))

        row_means = result.mean(axis=1)
        row_jump = np.abs(np.diff(row_means)).max()

        print(f"lambda={lambda_val}: c={c:.0f}, row-to-row max jump = {row_jump:.4f}")

        result_clipped = np.clip(result, 0, 1)
        iio.imwrite(output_dir / f'debug_lambda_{lambda_val}_direct.png',
                    (result_clipped * 255).astype(np.uint8))


def test_with_synthetic_clean():
    """Test with synthetically blurred image (no acquisition noise)."""
    print("\n=== Testing with synthetic clean image ===")

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'
    image_path = base_dir / 'images' / 'houses_big.jpg'

    # Load sharp image
    I = iio.imread(image_path)
    I = resize(I, (int(I.shape[0] * 0.15), int(I.shape[1] * 0.15)), anti_aliasing=True)
    I = I.astype(np.float64)
    I = I / I.max()
    I = I ** 2.0

    sharp = I[:, :, 2]
    kernel = create_blur_kernel(kernel_path, 43)
    shape = sharp.shape

    # Blur the image ourselves (perfectly, no noise)
    from scipy.ndimage import convolve
    blurred = convolve(sharp, kernel, mode='reflect')

    print(f"Sharp image range: [{sharp.min():.4f}, {sharp.max():.4f}]")
    print(f"Blurred image range: [{blurred.min():.4f}, {blurred.max():.4f}]")

    # Now try to deconvolve
    otf = psf2otf(kernel, shape)
    Nomin1 = np.conj(otf) * fft2(blurred)
    Denom1 = np.abs(otf) ** 2

    tau = 0.9
    lambda_val = 750
    c = tau * 2 * lambda_val

    x = (c * Nomin1 + fft2(blurred)) / (c * Denom1 + 1)
    result = np.real(ifft2(x))

    row_jump = np.abs(np.diff(result.mean(axis=1))).max()
    print(f"\nWith synthetic clean blurred image:")
    print(f"  row-to-row max jump = {row_jump:.4f}")

    output_dir = Path(__file__).parent / 'output'
    result_clipped = np.clip(result, 0, 1)
    iio.imwrite(output_dir / 'debug_synthetic_clean.png',
                (result_clipped * 255).astype(np.uint8))

    # Compare with original noisy blurred
    # Actually the demo adds noise... let me use the original image directly
    Nomin1_orig = np.conj(otf) * fft2(I[:, :, 2])
    x_orig = (c * Nomin1_orig + fft2(I[:, :, 2])) / (c * Denom1 + 1)
    result_orig = np.real(ifft2(x_orig))

    row_jump_orig = np.abs(np.diff(result_orig.mean(axis=1))).max()
    print(f"With original (unblurred) image:")
    print(f"  row-to-row max jump = {row_jump_orig:.4f}")


def check_frequency_structure():
    """Check if the banding is at specific frequencies."""
    print("\n=== Checking frequency structure of banding ===")

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'
    image_path = base_dir / 'images' / 'houses_big.jpg'

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

    tau = 0.9
    lambda_val = 750
    c = tau * 2 * lambda_val

    x_fft = (c * Nomin1 + fft2(img)) / (c * Denom1 + 1)
    result = np.real(ifft2(x_fft))

    # Take FFT of result to see frequency content
    result_fft = fft2(result)
    input_fft = fft2(img)

    # Compute ratio
    ratio = np.abs(result_fft) / (np.abs(input_fft) + 1e-10)

    print("Frequency amplification ratio (result/input):")
    print(f"  At DC: {ratio[0, 0]:.2f}")
    print(f"  At (1, 0): {ratio[1, 0]:.2f}")
    print(f"  At (0, 1): {ratio[0, 1]:.2f}")
    print(f"  At (H//4, 0): {ratio[shape[0]//4, 0]:.2f}")
    print(f"  At (0, W//4): {ratio[0, shape[1]//4]:.2f}")
    print(f"  At (H//2, 0): {ratio[shape[0]//2, 0]:.2f}")
    print(f"  At (0, W//2): {ratio[0, shape[1]//2]:.2f}")

    # Horizontal banding = variation in vertical direction = ky ≠ 0 frequencies
    # Let's look at the column 0 (all ky, kx=0) of the FFT
    col0_input = np.abs(input_fft[:, 0])
    col0_result = np.abs(result_fft[:, 0])
    col0_ratio = ratio[:, 0]

    print("\nVertical frequency (kx=0) amplification:")
    print(f"  Input power (col 0): {col0_input.sum():.2f}")
    print(f"  Result power (col 0): {col0_result.sum():.2f}")
    print(f"  Mean amplification: {col0_ratio.mean():.2f}")
    print(f"  Max amplification: {col0_ratio.max():.2f} at ky={col0_ratio.argmax()}")

    # Save the ratio pattern
    output_dir = Path(__file__).parent / 'output'
    ratio_shifted = fftshift(ratio)
    ratio_log = np.log10(ratio_shifted + 1e-10)
    ratio_norm = (ratio_log - ratio_log.min()) / (ratio_log.max() - ratio_log.min())
    iio.imwrite(output_dir / 'debug_freq_amplification.png',
                (ratio_norm * 255).astype(np.uint8))
    print(f"\nSaved frequency amplification pattern to {output_dir}/debug_freq_amplification.png")


if __name__ == '__main__':
    analyze_noise_amplification()
    test_lower_lambda()
    test_with_synthetic_clean()
    check_frequency_structure()
