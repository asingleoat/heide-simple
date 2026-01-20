#!/usr/bin/env python3
"""Test with perfect reference to isolate the issue."""

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import convolve
from pathlib import Path
import imageio.v3 as iio
from skimage.transform import resize
from skimage.util import random_noise

from deconv import psf2otf, img_to_norm_grayscale, edgetaper
from deconv.utils import imconv
from deconv.operator_norm import compute_operator_norm


def create_blur_kernel(image_path, size):
    """Load and resize blur kernel from image."""
    kernel = iio.imread(image_path)
    kernel = img_to_norm_grayscale(kernel)
    kernel = resize(kernel, (size, size), order=3, anti_aliasing=True)
    kernel = kernel / kernel.sum()
    return kernel


def _Kmult_cross(f, adj_img, lambda_cross, lambda_tv):
    """Forward operator with cross-channel."""
    dxf = np.array([[-1, 1]])
    dyf = np.array([[-1], [1]])

    results = []

    # TV terms
    if lambda_tv > 1e-10:
        fx = imconv(f, dxf[:, ::-1][::-1, :], 'full')[:, 1:]
        fy = imconv(f, dyf[::-1, :], 'full')[1:, :]
        results.extend([lambda_tv * 0.5 * fx, lambda_tv * 0.5 * fy])

    # Cross-channel terms
    if abs(lambda_cross) > 1e-10:
        diag_x = imconv(adj_img, dxf[:, ::-1][::-1, :], 'full')[:, 1:] * f
        conv_x = imconv(f, dxf[:, ::-1][::-1, :], 'full')[:, 1:]
        Sxf = lambda_cross * 0.5 * (adj_img * conv_x - diag_x)

        diag_y = imconv(adj_img, dyf[::-1, :], 'full')[1:, :] * f
        conv_y = imconv(f, dyf[::-1, :], 'full')[1:, :]
        Syf = lambda_cross * 0.5 * (adj_img * conv_y - diag_y)

        results.extend([Sxf, Syf])

    return np.stack(results, axis=2)


def _KSmult_cross(g, adj_img, lambda_cross, lambda_tv):
    """Adjoint operator with cross-channel."""
    dxf = np.array([[-1, 1]])
    dyf = np.array([[-1], [1]])

    result = np.zeros((g.shape[0], g.shape[1]))
    i = 0

    # TV terms
    if lambda_tv > 1e-10:
        fx = imconv(lambda_tv * 0.5 * g[:, :, i], dxf, 'full')[:, :-1]
        result += fx
        i += 1

        fy = imconv(lambda_tv * 0.5 * g[:, :, i], dyf, 'full')[:-1, :]
        result += fy
        i += 1

    # Cross-channel terms
    if abs(lambda_cross) > 1e-10:
        # X direction
        f_i = lambda_cross * 0.5 * g[:, :, i]
        diag = imconv(adj_img, dxf[:, ::-1][::-1, :], 'full')[:, 1:] * f_i
        conv = imconv(adj_img * f_i, dxf, 'full')[:, :-1]
        result += conv - diag
        i += 1

        # Y direction
        f_i = lambda_cross * 0.5 * g[:, :, i]
        diag = imconv(adj_img, dyf[::-1, :], 'full')[1:, :] * f_i
        conv = imconv(adj_img * f_i, dyf, 'full')[:-1, :]
        result += conv - diag
        i += 1

    return result


def test_with_perfect_reference():
    """Test deconvolution with perfect (sharp) reference."""
    print("=== Testing with perfect reference channel ===")

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'
    image_path = base_dir / 'images' / 'houses_big.jpg'

    # Load image
    I = iio.imread(image_path)
    I = resize(I, (int(I.shape[0] * 0.15), int(I.shape[1] * 0.15)), anti_aliasing=True)
    I = I.astype(np.float64)
    I = I / I.max()
    I = I ** 2.0
    I_sharp = I.copy()

    # Create kernel for channel 2 (largest blur)
    blur_size = 15
    inc_blur_exp = 1.7
    curr_blur_size = round(blur_size * (inc_blur_exp ** 2))
    curr_blur_size = curr_blur_size + (1 - curr_blur_size % 2)
    kernel = create_blur_kernel(kernel_path, curr_blur_size)
    print(f"Kernel size: {kernel.shape}")

    # Create synthetic blurred image
    img_sharp = I_sharp[:, :, 2]
    img_blurred = convolve(img_sharp, kernel, mode='reflect')
    img_blurred = random_noise(img_blurred, mode='gaussian', var=0.005**2, clip=False)

    # Add noise to kernel
    kernel_noisy = kernel + np.random.randn(*kernel.shape) * np.sqrt(0.0000001 / 9)
    kernel_noisy = np.maximum(kernel_noisy, 0)
    kernel_noisy = kernel_noisy / kernel_noisy.sum()

    # Use perfect sharp reference (channel 0 = red channel, which is sharp)
    # In the MATLAB demo, channel 0 has a delta kernel, so it's essentially sharp
    ref_img = I_sharp[:, :, 0]  # Use sharp red channel as reference

    # Pad and edgetaper
    ks = kernel_noisy.shape[0]
    img_pad = np.pad(img_blurred, ks, mode='edge')
    ref_pad = np.pad(ref_img, ks, mode='edge')

    for _ in range(4):
        img_pad = edgetaper(img_pad, kernel_noisy)
        ref_pad = edgetaper(ref_pad, kernel_noisy)

    img_pad += 1.0
    ref_pad += 1.0

    shape = img_pad.shape

    # Setup
    otf = psf2otf(kernel_noisy, shape)
    Nomin1 = np.conj(otf) * fft2(img_pad)
    Denom1 = np.abs(otf) ** 2

    lambda_residual = 750
    lambda_tv = 0.5
    lambda_cross = 1.0

    def A(x):
        return _Kmult_cross(x, ref_pad, lambda_cross, lambda_tv)

    def AS(x):
        return _KSmult_cross(x, ref_pad, lambda_cross, lambda_tv)

    L = compute_operator_norm(A, AS, shape)
    print(f"Operator norm L: {L:.4f}")

    sigma = 1.0
    tau = 0.7 / (sigma * L ** 2)
    print(f"tau: {tau:.4f}")
    print(f"c = tau * 2 * lambda: {tau * 2 * lambda_residual:.1f}")

    def prox_fs(u, sigma):
        amplitude = np.sqrt(np.sum(u ** 2, axis=2, keepdims=True))
        return u / np.maximum(1.0, amplitude)

    def solve_fft(Nomin1, Denom1, tau, lambda_val, f):
        x = (tau * 2 * lambda_val * Nomin1 + fft2(f)) / (tau * 2 * lambda_val * Denom1 + 1)
        return np.real(ifft2(x))

    # Initialize
    f = img_pad.copy()
    g = A(f)
    f1 = f.copy()

    print("\nRunning PD iterations with PERFECT reference...")
    for i in range(50):
        f_old = f.copy()

        g = prox_fs(g + sigma * A(f1), sigma)
        f = solve_fft(Nomin1, Denom1, tau, lambda_residual, f - tau * AS(g))
        f1 = f + 1.0 * (f - f_old)

        rel_diff = np.linalg.norm((f - f_old).ravel()) / (np.linalg.norm(f.ravel()) + 1e-10)

        if i < 10 or i % 10 == 0:
            row_jump = np.abs(np.diff(f1.mean(axis=1))).max()
            print(f"  Iter {i+1}: diff={rel_diff:.5g}, row_jump={row_jump:.4f}")

        if rel_diff < 1e-4:
            print(f"  Converged at iteration {i+1}")
            break

    # Save result
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    result = np.maximum(f1, 1.0)
    result = result[ks:-ks, ks:-ks] - 1.0
    result_clipped = np.clip(result, 0, 1) ** 0.5
    iio.imwrite(output_dir / 'debug_perfect_ref.png',
                (result_clipped * 255).astype(np.uint8))
    print(f"\nSaved to {output_dir}/debug_perfect_ref.png")

    # Compare: test with NO cross-channel
    print("\n--- Testing WITHOUT cross-channel (TV only) ---")
    lambda_cross_zero = 0.0

    def A_tv(x):
        return _Kmult_cross(x, ref_pad, lambda_cross_zero, lambda_tv)

    def AS_tv(x):
        return _KSmult_cross(x, ref_pad, lambda_cross_zero, lambda_tv)

    L_tv = compute_operator_norm(A_tv, AS_tv, shape)
    tau_tv = 0.7 / (sigma * L_tv ** 2)
    print(f"Operator norm L (TV only): {L_tv:.4f}")
    print(f"tau (TV only): {tau_tv:.4f}")

    f = img_pad.copy()
    g = A_tv(f)
    f1 = f.copy()

    print("\nRunning PD iterations with TV only (no cross-channel)...")
    for i in range(50):
        f_old = f.copy()

        g = prox_fs(g + sigma * A_tv(f1), sigma)
        f = solve_fft(Nomin1, Denom1, tau_tv, lambda_residual, f - tau_tv * AS_tv(g))
        f1 = f + 1.0 * (f - f_old)

        rel_diff = np.linalg.norm((f - f_old).ravel()) / (np.linalg.norm(f.ravel()) + 1e-10)

        if i < 10 or i % 10 == 0:
            row_jump = np.abs(np.diff(f1.mean(axis=1))).max()
            print(f"  Iter {i+1}: diff={rel_diff:.5g}, row_jump={row_jump:.4f}")

        if rel_diff < 1e-4:
            break

    result_tv = np.maximum(f1, 1.0)
    result_tv = result_tv[ks:-ks, ks:-ks] - 1.0
    result_tv_clipped = np.clip(result_tv, 0, 1) ** 0.5
    iio.imwrite(output_dir / 'debug_tv_only.png',
                (result_tv_clipped * 255).astype(np.uint8))
    print(f"Saved to {output_dir}/debug_tv_only.png")

    # Test with lower lambda_residual
    print("\n--- Testing with LOWER lambda_residual (more regularization) ---")
    lambda_residual_low = 100

    f = img_pad.copy()
    g = A(f)
    f1 = f.copy()

    print(f"Using lambda_residual = {lambda_residual_low}")
    print(f"c = tau * 2 * lambda: {tau * 2 * lambda_residual_low:.1f}")

    for i in range(50):
        f_old = f.copy()

        g = prox_fs(g + sigma * A(f1), sigma)
        f = solve_fft(Nomin1, Denom1, tau, lambda_residual_low, f - tau * AS(g))
        f1 = f + 1.0 * (f - f_old)

        rel_diff = np.linalg.norm((f - f_old).ravel()) / (np.linalg.norm(f.ravel()) + 1e-10)

        if i < 10 or i % 10 == 0:
            row_jump = np.abs(np.diff(f1.mean(axis=1))).max()
            print(f"  Iter {i+1}: diff={rel_diff:.5g}, row_jump={row_jump:.4f}")

        if rel_diff < 1e-4:
            break

    result_low = np.maximum(f1, 1.0)
    result_low = result_low[ks:-ks, ks:-ks] - 1.0
    result_low_clipped = np.clip(result_low, 0, 1) ** 0.5
    iio.imwrite(output_dir / 'debug_low_lambda.png',
                (result_low_clipped * 255).astype(np.uint8))
    print(f"Saved to {output_dir}/debug_low_lambda.png")


if __name__ == '__main__':
    test_with_perfect_reference()
