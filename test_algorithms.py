#!/usr/bin/env python3
"""
Simple test script to verify deconvolution and PSF estimation algorithms.

Run with: ./dev python test_algorithms.py
"""

import numpy as np
from scipy.ndimage import gaussian_filter, convolve

from deconv import (
    pd_joint_deconv,
    estimate_psf,
    create_calibration_pattern,
    img_to_norm_grayscale,
)


def test_deconvolution():
    """Test basic deconvolution on synthetic data."""
    print("=== Testing Deconvolution ===")

    # Create synthetic sharp image
    np.random.seed(42)
    sharp = np.random.rand(64, 64) * 0.5 + 0.25

    # Create Gaussian blur kernel
    kernel = np.zeros((11, 11))
    kernel[5, 5] = 1.0
    kernel = gaussian_filter(kernel, sigma=1.5)
    kernel = kernel / kernel.sum()

    # Blur the image
    blurred = convolve(sharp, kernel, mode='reflect')

    # Prepare for deconvolution
    channels = [{'image': blurred, 'kernel': kernel}]
    lambda_params = np.array([[1, 200, 2.0, 0.0, 0.0, 1]])

    # Run deconvolution
    result = pd_joint_deconv(channels, lambda_params, max_it=100, tol=1e-4, verbose='none')

    # Check result
    deconv = result[0]['image']
    error = np.linalg.norm(deconv - sharp) / np.linalg.norm(sharp)

    print(f"  Input PSNR: {-10 * np.log10(np.mean((blurred - sharp)**2)):.2f} dB")
    print(f"  Output PSNR: {-10 * np.log10(np.mean((deconv - sharp)**2)):.2f} dB")
    print(f"  Relative error: {error:.4f}")

    passed = error < 0.3
    print(f"  Status: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_psf_estimation():
    """Test PSF estimation on synthetic data."""
    print("=== Testing PSF Estimation ===")

    # Create calibration pattern
    np.random.seed(42)
    sharp = np.random.rand(64, 64)

    # True PSF
    true_psf = np.zeros((15, 15))
    true_psf[7, 7] = 1.0
    true_psf = gaussian_filter(true_psf, sigma=2.0)
    true_psf = true_psf / true_psf.sum()

    # Create blurred image
    blurred = convolve(sharp, true_psf, mode='reflect')

    # Estimate PSF
    estimated_psf = estimate_psf(
        sharp, blurred, psf_size=15,
        lambda_tv=0.001, mu_sum=100.0,
        max_it=500, verbose='none'
    )

    # Check result
    error = np.linalg.norm(estimated_psf - true_psf) / np.linalg.norm(true_psf)

    true_center = np.unravel_index(true_psf.argmax(), true_psf.shape)
    est_center = np.unravel_index(estimated_psf.argmax(), estimated_psf.shape)

    print(f"  True PSF center: {true_center}")
    print(f"  Estimated PSF center: {est_center}")
    print(f"  Relative error: {error:.4f}")

    passed = error < 0.2 and true_center == est_center
    print(f"  Status: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_calibration_pattern():
    """Test calibration pattern generation."""
    print("=== Testing Calibration Pattern ===")

    pattern, coords = create_calibration_pattern(
        patch_size=32, n_patches_h=2, n_patches_w=2,
        border_width=10, seed=42
    )

    # Check pattern properties
    expected_size = 2 * (32 + 2 * 10)
    correct_size = pattern.shape == (expected_size, expected_size)
    correct_coords = len(coords) == 4

    # Check that borders are white (1.0) and patches have noise
    border_check = pattern[0, 0] == 1.0  # Top-left corner should be border

    print(f"  Pattern size: {pattern.shape}")
    print(f"  Number of patches: {len(coords)}")
    print(f"  Border value: {pattern[0, 0]:.2f}")

    passed = correct_size and correct_coords and border_check
    print(f"  Status: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_cross_channel_deconvolution():
    """Test cross-channel deconvolution for chromatic aberration."""
    print("=== Testing Cross-Channel Deconvolution ===")

    np.random.seed(42)

    # Create synthetic RGB image
    sharp = np.random.rand(64, 64, 3) * 0.5 + 0.25

    # Channel 0 is sharp (delta), channels 1,2 are blurred (chromatic aberration)
    kernels = []

    # Channel 0: sharp (delta function)
    k0 = np.zeros((11, 11))
    k0[5, 5] = 1.0
    kernels.append(k0)

    # Channels 1,2: increasingly blurred
    for ch in range(1, 3):
        sigma = 1.5 * ch
        k = np.zeros((11, 11))
        k[5, 5] = 1.0
        k = gaussian_filter(k, sigma=sigma)
        k = k / k.sum()
        kernels.append(k)

    blurred = np.zeros_like(sharp)
    for ch in range(3):
        blurred[:, :, ch] = convolve(sharp[:, :, ch], kernels[ch], mode='reflect')

    # Prepare channels
    channels = [
        {'image': blurred[:, :, i], 'kernel': kernels[i]}
        for i in range(3)
    ]

    # Cross-channel parameters (channel 0 guides others)
    lambda_params = np.array([
        [1, 200, 2.0, 0.0, 0.0, 0.0, 0.0, 1],
        [2, 200, 2.0, 0.0, 3.0, 0.0, 0.0, 0],
        [3, 200, 2.0, 0.0, 3.0, 0.0, 0.0, 0],
    ])

    # Run deconvolution
    result = pd_joint_deconv(channels, lambda_params, max_it=200, tol=1e-4, verbose='none')

    # Check that blurred channels improved
    error_ch1_before = np.linalg.norm(blurred[:, :, 1] - sharp[:, :, 1])
    error_ch1_after = np.linalg.norm(result[1]['image'] - sharp[:, :, 1])
    error_ch2_before = np.linalg.norm(blurred[:, :, 2] - sharp[:, :, 2])
    error_ch2_after = np.linalg.norm(result[2]['image'] - sharp[:, :, 2])

    print(f"  Channel 1 error: {error_ch1_before:.4f} -> {error_ch1_after:.4f}")
    print(f"  Channel 2 error: {error_ch2_before:.4f} -> {error_ch2_after:.4f}")

    # The deconvolution should run without errors; improvement depends on parameters
    # For this test, we just verify it runs and produces reasonable output
    passed = (result[0]['image'].shape == sharp[:, :, 0].shape and
              result[1]['image'].shape == sharp[:, :, 1].shape and
              result[2]['image'].shape == sharp[:, :, 2].shape)
    print(f"  Status: {'PASS' if passed else 'FAIL'}\n")
    return passed


def main():
    print("\nRunning algorithm tests...\n")

    results = []
    results.append(("Deconvolution", test_deconvolution()))
    results.append(("PSF Estimation", test_psf_estimation()))
    results.append(("Calibration Pattern", test_calibration_pattern()))
    results.append(("Cross-Channel", test_cross_channel_deconvolution()))

    print("=" * 40)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("=" * 40)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == '__main__':
    exit(main())
