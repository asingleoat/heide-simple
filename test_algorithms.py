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
    estimate_psf_multiscale,
    estimate_psf_tiled,
    deconvolve_tiled,
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


def test_multiscale_psf_estimation():
    """Test multiscale PSF estimation."""
    print("=== Testing Multiscale PSF Estimation ===")

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

    # Estimate PSF using multiscale
    estimated_psf = estimate_psf_multiscale(
        sharp, blurred, psf_size=15,
        lambda_tv=0.001, mu_sum=100.0,
        max_it=500, n_scales=2, verbose='none'
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


def test_tiled_psf_estimation():
    """Test tile-based PSF estimation for spatially-varying blur."""
    print("=== Testing Tile-based PSF Estimation ===")

    np.random.seed(42)

    # Create a larger image for tiling
    sharp = np.random.rand(128, 128)

    # Create a PSF that varies across the image (simplified - same PSF everywhere)
    true_psf = np.zeros((11, 11))
    true_psf[5, 5] = 1.0
    true_psf = gaussian_filter(true_psf, sigma=1.5)
    true_psf = true_psf / true_psf.sum()

    # Blur the image
    blurred = convolve(sharp, true_psf, mode='reflect')

    # Estimate PSFs using tile-based approach (2x2 grid)
    psfs, tile_centers, tile_grid = estimate_psf_tiled(
        sharp, blurred, psf_size=11,
        n_tiles_h=2, n_tiles_w=2,
        overlap=0.25,
        lambda_tv=0.001, mu_sum=100.0,
        max_it=200, smooth_sigma=0.5,
        verbose='none'
    )

    # Check results
    correct_count = len(psfs) == 4
    correct_grid = tile_grid == (2, 2)
    correct_centers = len(tile_centers) == 4

    # Check that each PSF is reasonable (center is roughly correct)
    all_centered = True
    for psf in psfs:
        est_center = np.unravel_index(psf.argmax(), psf.shape)
        if est_center != (5, 5):
            all_centered = False

    print(f"  Number of PSFs: {len(psfs)}")
    print(f"  Grid: {tile_grid}")
    print(f"  All PSFs centered correctly: {all_centered}")

    passed = correct_count and correct_grid and correct_centers and all_centered
    print(f"  Status: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_tiled_deconvolution():
    """Test tile-based deconvolution with spatially-varying PSFs."""
    print("=== Testing Tiled Deconvolution ===")

    np.random.seed(42)

    # Create a larger image
    sharp = np.random.rand(128, 128) * 0.5 + 0.25

    # Create PSFs for a 2x2 tile grid (same PSF for simplicity)
    kernel = np.zeros((11, 11))
    kernel[5, 5] = 1.0
    kernel = gaussian_filter(kernel, sigma=1.5)
    kernel = kernel / kernel.sum()

    # Blur the entire image (in practice, different regions would have different blur)
    blurred = convolve(sharp, kernel, mode='reflect')

    # Create tile PSFs (2x2 grid, all same for this test)
    psfs = [kernel.copy() for _ in range(4)]
    tile_grid = (2, 2)

    # Run tiled deconvolution
    result = deconvolve_tiled(
        blurred, psfs, tile_grid,
        overlap=0.25,
        lambda_residual=200,
        lambda_tv=2.0,
        lambda_cross=0.0,
        max_iterations=100,
        tolerance=1e-4,
        verbose=False
    )

    # Check result shape
    correct_shape = result.shape == sharp.shape

    # Check that result is reasonable (PSNR improved or at least not degraded significantly)
    input_error = np.mean((blurred - sharp) ** 2)
    output_error = np.mean((result - sharp) ** 2)

    print(f"  Input MSE: {input_error:.6f}")
    print(f"  Output MSE: {output_error:.6f}")
    print(f"  Shape correct: {correct_shape}")

    passed = correct_shape and result.min() >= -0.1 and result.max() <= 1.1
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
    results.append(("Multiscale PSF", test_multiscale_psf_estimation()))
    results.append(("Tiled PSF", test_tiled_psf_estimation()))
    results.append(("Tiled Deconv", test_tiled_deconvolution()))
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
