#!/usr/bin/env python3
"""Debug script to test FFT-based solve and understand banding issue."""

import numpy as np
from scipy.fft import fft2, ifft2
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


def solve_fft(Nomin1, Denom1, tau, lambda_val, f):
    """FFT-based solver."""
    x = (tau * 2 * lambda_val * Nomin1 + fft2(f)) / (tau * 2 * lambda_val * Denom1 + 1)
    return np.real(ifft2(x))


def analyze_otf(kernel, shape, name):
    """Analyze OTF structure."""
    otf = psf2otf(kernel, shape)

    print(f"\n=== {name} (kernel {kernel.shape}) ===")
    print(f"OTF shape: {otf.shape}")
    print(f"OTF |min|: {np.abs(otf).min():.6f}")
    print(f"OTF |max|: {np.abs(otf).max():.6f}")

    # Count near-zero frequencies
    abs_otf = np.abs(otf)
    threshold = 0.01 * abs_otf.max()
    near_zero = np.sum(abs_otf < threshold)
    print(f"Near-zero frequencies (<1% of max): {near_zero} / {otf.size} = {100*near_zero/otf.size:.1f}%")

    # Check for structure in near-zeros
    # Look at rows/cols with many near-zeros
    near_zero_mask = abs_otf < threshold
    row_zeros = near_zero_mask.sum(axis=1)
    col_zeros = near_zero_mask.sum(axis=0)

    print(f"Rows with most near-zeros: max={row_zeros.max()} at row {row_zeros.argmax()}")
    print(f"Cols with most near-zeros: max={col_zeros.max()} at col {col_zeros.argmax()}")

    # Check DC component
    print(f"DC component (OTF[0,0]): {otf[0,0]:.6f}")

    return otf


def test_simple_deconv():
    """Test simple deconvolution to isolate the banding issue."""
    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'

    # Create a simple test image (smooth gradient)
    h, w = 100, 150
    test_img = np.outer(np.linspace(0.2, 0.8, h), np.ones(w))
    print(f"Test image: {h}x{w}, range [{test_img.min():.3f}, {test_img.max():.3f}]")

    # Test with different kernel sizes
    for kernel_size in [15, 25, 43]:
        kernel = create_blur_kernel(kernel_path, kernel_size)

        # Analyze OTF
        shape = test_img.shape
        otf = analyze_otf(kernel, shape, f"Kernel {kernel_size}x{kernel_size}")

        # Prepare FFT solve inputs
        Nomin1 = np.conj(otf) * fft2(test_img)
        Denom1 = np.abs(otf) ** 2

        # Apply solve_fft
        tau = 0.1
        lambda_val = 100.0
        result = solve_fft(Nomin1, Denom1, tau, lambda_val, test_img)

        # Analyze result for banding
        row_means = result.mean(axis=1)
        row_stds = result.std(axis=1)
        row_diffs = np.diff(row_means)

        print(f"\nResult analysis:")
        print(f"  Range: [{result.min():.3f}, {result.max():.3f}]")
        print(f"  Row mean std: {row_means.std():.6f}")
        print(f"  Row-to-row jumps: max={np.abs(row_diffs).max():.6f}")
        print(f"  Overall std: {result.std():.6f}")

        # Check if result is smooth (no banding)
        if np.abs(row_diffs).max() > 0.01:
            print(f"  WARNING: Potential banding detected!")
            # Find the problematic rows
            problem_rows = np.where(np.abs(row_diffs) > 0.01)[0]
            if len(problem_rows) > 0:
                print(f"  Problem row transitions: {problem_rows[:10]}...")


def test_with_real_image():
    """Test with the actual demo image."""
    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'
    image_path = base_dir / 'images' / 'houses_big.jpg'

    # Load image
    I = iio.imread(image_path)
    I = resize(I, (int(I.shape[0] * 0.15), int(I.shape[1] * 0.15)), anti_aliasing=True)
    I = I.astype(np.float64)
    I = I / I.max()
    I = I ** 2.0  # Linearize

    print(f"\nReal image: {I.shape}")

    # Test channel 2 (largest kernel, worst banding)
    kernel_size = 43
    kernel = create_blur_kernel(kernel_path, kernel_size)

    # Get one channel
    img = I[:, :, 2]
    shape = img.shape

    # Analyze OTF
    otf = analyze_otf(kernel, shape, f"Real image kernel {kernel_size}x{kernel_size}")

    # Check OTF symmetry
    print("\nOTF symmetry check:")
    print(f"  |OTF| row 0 mean: {np.abs(otf[0, :]).mean():.6f}")
    print(f"  |OTF| col 0 mean: {np.abs(otf[:, 0]).mean():.6f}")

    # Visualize OTF magnitude pattern
    abs_otf = np.abs(otf)
    print(f"\nOTF magnitude at key locations:")
    print(f"  OTF[0,0] (DC): {abs_otf[0,0]:.6f}")
    print(f"  OTF[0,1]: {abs_otf[0,1]:.6f}")
    print(f"  OTF[1,0]: {abs_otf[1,0]:.6f}")
    print(f"  OTF[h//2,0] (Nyquist row): {abs_otf[shape[0]//2,0]:.6f}")
    print(f"  OTF[0,w//2] (Nyquist col): {abs_otf[0,shape[1]//2]:.6f}")

    # Test solve_fft
    Nomin1 = np.conj(otf) * fft2(img)
    Denom1 = np.abs(otf) ** 2

    tau = 0.1
    lambda_val = 750.0
    result = solve_fft(Nomin1, Denom1, tau, lambda_val, img)

    # Analyze for banding
    row_means = result.mean(axis=1)
    row_diffs = np.diff(row_means)

    print(f"\nResult analysis:")
    print(f"  Input row mean std: {img.mean(axis=1).std():.6f}")
    print(f"  Output row mean std: {row_means.std():.6f}")
    print(f"  Row-to-row max jump: {np.abs(row_diffs).max():.6f}")

    # Compare with input
    input_row_diffs = np.diff(img.mean(axis=1))
    print(f"  Input row-to-row max jump: {np.abs(input_row_diffs).max():.6f}")

    # Check frequency content difference
    fft_input = fft2(img)
    fft_output = fft2(result)

    print(f"\nFrequency comparison:")
    print(f"  Input |FFT| at row 0 sum: {np.abs(fft_input[0, :]).sum():.2f}")
    print(f"  Output |FFT| at row 0 sum: {np.abs(fft_output[0, :]).sum():.2f}")
    print(f"  Input |FFT| at col 0 sum: {np.abs(fft_input[:, 0]).sum():.2f}")
    print(f"  Output |FFT| at col 0 sum: {np.abs(fft_output[:, 0]).sum():.2f}")

    # Save result for visual inspection
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    result_clipped = np.clip(result, 0, 1)
    iio.imwrite(output_dir / 'debug_fft_result.png',
                (result_clipped * 255).astype(np.uint8))
    print(f"\nSaved debug result to {output_dir}/debug_fft_result.png")


def test_kernel_symmetry():
    """Check if kernel is symmetric."""
    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'

    print("\n=== Kernel Symmetry Check ===")

    for size in [15, 25, 43]:
        kernel = create_blur_kernel(kernel_path, size)

        # Check various symmetries
        horiz_flip = np.allclose(kernel, kernel[:, ::-1], rtol=0.01)
        vert_flip = np.allclose(kernel, kernel[::-1, :], rtol=0.01)
        transpose = np.allclose(kernel, kernel.T, rtol=0.01)

        print(f"\nKernel {size}x{size}:")
        print(f"  Horizontally symmetric: {horiz_flip}")
        print(f"  Vertically symmetric: {vert_flip}")
        print(f"  Transpose symmetric: {transpose}")

        # Check center of mass
        y_coords, x_coords = np.ogrid[:size, :size]
        center_y = (kernel * y_coords).sum() / kernel.sum()
        center_x = (kernel * x_coords).sum() / kernel.sum()
        expected_center = (size - 1) / 2

        print(f"  Center of mass: ({center_y:.2f}, {center_x:.2f})")
        print(f"  Expected center: ({expected_center:.2f}, {expected_center:.2f})")

        # Asymmetry measure
        h_asym = np.abs(kernel - kernel[:, ::-1]).max()
        v_asym = np.abs(kernel - kernel[::-1, :]).max()
        print(f"  Max horizontal asymmetry: {h_asym:.6f}")
        print(f"  Max vertical asymmetry: {v_asym:.6f}")


if __name__ == '__main__':
    test_kernel_symmetry()
    test_simple_deconv()
    test_with_real_image()
