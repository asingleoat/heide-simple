#!/usr/bin/env python3
"""Debug parameter settings to find working values."""

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


def _Kmult(f, adj_img, lambda_cross, lambda_tv):
    """Forward operator."""
    dxf = np.array([[-1, 1]])
    dyf = np.array([[-1], [1]])
    dxxf = np.array([[-1, 2, -1]])
    dyyf = np.array([[-1], [2], [-1]])
    dxyf = np.array([[-1, 1], [1, -1]])

    results = []

    # TV terms (first order)
    if lambda_tv > 1e-10:
        fx = imconv(f, dxf[:, ::-1][::-1, :], 'full')[:, 1:]
        fy = imconv(f, dyf[::-1, :], 'full')[1:, :]
        results.extend([lambda_tv * 0.5 * fx, lambda_tv * 0.5 * fy])

        # Second order
        sd_w = 0.15
        fxx = imconv(f, dxxf[:, ::-1], 'full')[:, 2:]
        fyy = imconv(f, dyyf[::-1, :], 'full')[2:, :]
        fxy = imconv(f, dxyf[::-1, ::-1], 'full')[1:, 1:]
        results.extend([lambda_tv * sd_w * fxx, lambda_tv * sd_w * fyy, lambda_tv * sd_w * fxy])

    # Cross-channel terms
    if abs(lambda_cross) > 1e-10 and adj_img is not None:
        diag_x = imconv(adj_img, dxf[:, ::-1][::-1, :], 'full')[:, 1:] * f
        conv_x = imconv(f, dxf[:, ::-1][::-1, :], 'full')[:, 1:]
        Sxf = lambda_cross * 0.5 * (adj_img * conv_x - diag_x)

        diag_y = imconv(adj_img, dyf[::-1, :], 'full')[1:, :] * f
        conv_y = imconv(f, dyf[::-1, :], 'full')[1:, :]
        Syf = lambda_cross * 0.5 * (adj_img * conv_y - diag_y)

        results.extend([Sxf, Syf])

    return np.stack(results, axis=2)


def _KSmult(g, adj_img, lambda_cross, lambda_tv):
    """Adjoint operator."""
    dxf = np.array([[-1, 1]])
    dyf = np.array([[-1], [1]])
    dxxf = np.array([[-1, 2, -1]])
    dyyf = np.array([[-1], [2], [-1]])
    dxyf = np.array([[-1, 1], [1, -1]])

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

        sd_w = 0.15
        fxx = imconv(lambda_tv * sd_w * g[:, :, i], dxxf, 'full')[:, :-2]
        result += fxx
        i += 1

        fyy = imconv(lambda_tv * sd_w * g[:, :, i], dyyf, 'full')[:-2, :]
        result += fyy
        i += 1

        fxy = imconv(lambda_tv * sd_w * g[:, :, i], dxyf, 'full')[:-1, :-1]
        result += fxy
        i += 1

    # Cross-channel terms
    if abs(lambda_cross) > 1e-10 and adj_img is not None:
        f_i = lambda_cross * 0.5 * g[:, :, i]
        diag = imconv(adj_img, dxf[:, ::-1][::-1, :], 'full')[:, 1:] * f_i
        conv = imconv(adj_img * f_i, dxf, 'full')[:, :-1]
        result += conv - diag
        i += 1

        f_i = lambda_cross * 0.5 * g[:, :, i]
        diag = imconv(adj_img, dyf[::-1, :], 'full')[1:, :] * f_i
        conv = imconv(adj_img * f_i, dyf, 'full')[:-1, :]
        result += conv - diag
        i += 1

    return result


def run_pd(img_pad, kernel, ref_pad, lambda_residual, lambda_tv, lambda_cross, max_it=50, verbose=True):
    """Run primal-dual optimization."""
    shape = img_pad.shape

    otf = psf2otf(kernel, shape)
    Nomin1 = np.conj(otf) * fft2(img_pad)
    Denom1 = np.abs(otf) ** 2

    def A(x):
        return _Kmult(x, ref_pad, lambda_cross, lambda_tv)

    def AS(x):
        return _KSmult(x, ref_pad, lambda_cross, lambda_tv)

    L = compute_operator_norm(A, AS, shape)

    sigma = 1.0
    tau = 0.7 / (sigma * L ** 2)

    if verbose:
        print(f"  L={L:.3f}, tau={tau:.4f}, c={tau*2*lambda_residual:.1f}")

    def prox_fs(u, sigma):
        amplitude = np.sqrt(np.sum(u ** 2, axis=2, keepdims=True))
        return u / np.maximum(1.0, amplitude)

    def solve_fft(Nomin1, Denom1, tau, lambda_val, f):
        x = (tau * 2 * lambda_val * Nomin1 + fft2(f)) / (tau * 2 * lambda_val * Denom1 + 1)
        return np.real(ifft2(x))

    f = img_pad.copy()
    g = A(f)
    f1 = f.copy()

    for i in range(max_it):
        f_old = f.copy()

        g = prox_fs(g + sigma * A(f1), sigma)
        f = solve_fft(Nomin1, Denom1, tau, lambda_residual, f - tau * AS(g))
        f1 = f + 1.0 * (f - f_old)

        rel_diff = np.linalg.norm((f - f_old).ravel()) / (np.linalg.norm(f.ravel()) + 1e-10)

        if rel_diff < 1e-4:
            break

    row_jump = np.abs(np.diff(f1.mean(axis=1))).max()
    return f1, row_jump, i+1


def sweep_parameters():
    """Sweep over different parameter combinations."""
    print("=== Parameter sweep ===")

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

    # Kernel
    blur_size = 15
    inc_blur_exp = 1.7
    curr_blur_size = round(blur_size * (inc_blur_exp ** 2))
    curr_blur_size = curr_blur_size + (1 - curr_blur_size % 2)
    kernel = create_blur_kernel(kernel_path, curr_blur_size)

    # Create blurred image
    img_sharp = I_sharp[:, :, 2]
    img_blurred = convolve(img_sharp, kernel, mode='reflect')
    img_blurred = random_noise(img_blurred, mode='gaussian', var=0.005**2, clip=False)

    kernel_noisy = kernel + np.random.randn(*kernel.shape) * np.sqrt(0.0000001 / 9)
    kernel_noisy = np.maximum(kernel_noisy, 0)
    kernel_noisy = kernel_noisy / kernel_noisy.sum()

    ref_img = I_sharp[:, :, 0]

    # Pad and edgetaper
    ks = kernel_noisy.shape[0]
    img_pad = np.pad(img_blurred, ks, mode='edge')
    ref_pad = np.pad(ref_img, ks, mode='edge')

    for _ in range(4):
        img_pad = edgetaper(img_pad, kernel_noisy)
        ref_pad = edgetaper(ref_pad, kernel_noisy)

    img_pad += 1.0
    ref_pad += 1.0

    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    # Test different parameter combinations
    print("\n--- Testing different lambda_residual with lambda_tv=0.5, lambda_cross=1.0 ---")
    for lambda_res in [50, 100, 200, 300, 500, 750]:
        result, row_jump, iters = run_pd(img_pad, kernel_noisy, ref_pad,
                                         lambda_res, 0.5, 1.0, max_it=50, verbose=False)
        print(f"lambda_res={lambda_res}: row_jump={row_jump:.4f}, iters={iters}")

    print("\n--- Testing different lambda_tv with lambda_residual=750, lambda_cross=1.0 ---")
    for lambda_tv in [0.1, 0.5, 1.0, 2.0, 5.0]:
        result, row_jump, iters = run_pd(img_pad, kernel_noisy, ref_pad,
                                         750, lambda_tv, 1.0, max_it=50, verbose=False)
        print(f"lambda_tv={lambda_tv}: row_jump={row_jump:.4f}, iters={iters}")

    print("\n--- Testing different lambda_cross with lambda_residual=750, lambda_tv=0.5 ---")
    for lambda_cross in [0.0, 0.5, 1.0, 2.0, 5.0]:
        result, row_jump, iters = run_pd(img_pad, kernel_noisy, ref_pad,
                                         750, 0.5, lambda_cross, max_it=50, verbose=False)
        print(f"lambda_cross={lambda_cross}: row_jump={row_jump:.4f}, iters={iters}")

    # Find best combination
    print("\n--- Finding best combination for low banding ---")
    best_params = None
    best_jump = float('inf')

    for lambda_res in [200, 300, 500]:
        for lambda_tv in [1.0, 2.0, 3.0]:
            for lambda_cross in [1.0, 2.0, 3.0]:
                result, row_jump, iters = run_pd(img_pad, kernel_noisy, ref_pad,
                                                 lambda_res, lambda_tv, lambda_cross,
                                                 max_it=50, verbose=False)
                if row_jump < best_jump:
                    best_jump = row_jump
                    best_params = (lambda_res, lambda_tv, lambda_cross)

    print(f"\nBest parameters: lambda_res={best_params[0]}, lambda_tv={best_params[1]}, lambda_cross={best_params[2]}")
    print(f"Best row_jump: {best_jump:.4f}")

    # Save result with best parameters
    result, row_jump, iters = run_pd(img_pad, kernel_noisy, ref_pad,
                                     best_params[0], best_params[1], best_params[2],
                                     max_it=100, verbose=True)
    result = np.maximum(result, 1.0)
    result = result[ks:-ks, ks:-ks] - 1.0
    result_clipped = np.clip(result, 0, 1) ** 0.5
    iio.imwrite(output_dir / 'debug_best_params.png',
                (result_clipped * 255).astype(np.uint8))
    print(f"Saved best result to {output_dir}/debug_best_params.png")

    # Also save with original MATLAB params for comparison
    print("\n--- Result with original MATLAB parameters ---")
    result, row_jump, iters = run_pd(img_pad, kernel_noisy, ref_pad,
                                     750, 0.5, 1.0, max_it=100, verbose=True)
    print(f"Final row_jump: {row_jump:.4f}")
    result = np.maximum(result, 1.0)
    result = result[ks:-ks, ks:-ks] - 1.0
    result_clipped = np.clip(result, 0, 1) ** 0.5
    iio.imwrite(output_dir / 'debug_matlab_params.png',
                (result_clipped * 255).astype(np.uint8))


if __name__ == '__main__':
    sweep_parameters()
