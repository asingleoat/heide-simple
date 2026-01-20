#!/usr/bin/env python3
"""Debug cross-channel coupling."""

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


def _Kmult_with_cross(f, ch, db_chs, lambda_cross_ch, lambda_tv):
    """Forward operator K with cross-channel terms."""
    dxf = np.array([[-1, 1]])
    dyf = np.array([[-1], [1]])

    results = []

    # TV terms
    if lambda_tv > 1e-10:
        fx = imconv(f, dxf[:, ::-1][::-1, :], 'full')
        fx = (lambda_tv * 0.5) * fx[:, 1:]

        fy = imconv(f, dyf[::-1, :], 'full')
        fy = (lambda_tv * 0.5) * fy[1:, :]

        results.extend([fx, fy])

    # Cross-channel terms
    if np.sum(np.abs(lambda_cross_ch)) > 1e-10:
        for adj_ch in range(len(db_chs)):
            if adj_ch == ch or db_chs[adj_ch]['kernel'] is None:
                continue

            lam = lambda_cross_ch[adj_ch] if adj_ch < len(lambda_cross_ch) else 0
            if abs(lam) < 1e-10:
                continue

            adj_img = db_chs[adj_ch]['image']

            # Cross-channel gradient coupling
            diag_term = imconv(adj_img, dxf[:, ::-1][::-1, :], 'full')
            diag_term = diag_term[:, 1:] * f
            conv_term = imconv(f, dxf[:, ::-1][::-1, :], 'full')
            Sxf = (lam * 0.5) * (adj_img * conv_term[:, 1:] - diag_term)

            diag_term = imconv(adj_img, dyf[::-1, :], 'full')
            diag_term = diag_term[1:, :] * f
            conv_term = imconv(f, dyf[::-1, :], 'full')
            Syf = (lam * 0.5) * (adj_img * conv_term[1:, :] - diag_term)

            results.extend([Sxf, Syf])

    if not results:
        return np.zeros((*f.shape, 1))

    return np.stack(results, axis=2)


def _KSmult_with_cross(g, ch, db_chs, lambda_cross_ch, lambda_tv):
    """Adjoint operator K* with cross-channel terms."""
    dxf = np.array([[-1, 1]])
    dyf = np.array([[-1], [1]])

    result = np.zeros((g.shape[0], g.shape[1]))

    i = 0

    # TV terms
    if lambda_tv > 1e-10:
        fx = imconv((lambda_tv * 0.5) * g[:, :, i], dxf, 'full')
        result += fx[:, :-1]
        i += 1

        fy = imconv((lambda_tv * 0.5) * g[:, :, i], dyf, 'full')
        result += fy[:-1, :]
        i += 1

    # Cross-channel terms
    if np.sum(np.abs(lambda_cross_ch)) > 1e-10:
        for adj_ch in range(len(db_chs)):
            if adj_ch == ch or db_chs[adj_ch]['kernel'] is None:
                continue

            lam = lambda_cross_ch[adj_ch] if adj_ch < len(lambda_cross_ch) else 0
            if abs(lam) < 1e-10:
                continue

            adj_img = db_chs[adj_ch]['image']

            # X direction adjoint
            f_i = (lam * 0.5) * g[:, :, i]
            diag_term = imconv(adj_img, dxf[:, ::-1][::-1, :], 'full')
            diag_term = diag_term[:, 1:] * f_i
            conv_term = imconv(adj_img * f_i, dxf, 'full')
            Sxtf = conv_term[:, :-1] - diag_term
            result += Sxtf
            i += 1

            # Y direction adjoint
            f_i = (lam * 0.5) * g[:, :, i]
            diag_term = imconv(adj_img, dyf[::-1, :], 'full')
            diag_term = diag_term[1:, :] * f_i
            conv_term = imconv(adj_img * f_i, dyf, 'full')
            Sytf = conv_term[:-1, :] - diag_term
            result += Sytf
            i += 1

    return result


def test_cross_channel_deconv():
    """Test deconvolution with cross-channel coupling."""
    print("=== Testing cross-channel deconvolution ===")

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

    print(f"Image shape: {I.shape}")

    # Create kernels
    blur_size = 15
    inc_blur_exp = 1.7
    n_channels = 3

    K_blur = []
    for ch in range(n_channels):
        curr_blur_size = round(blur_size * (inc_blur_exp ** ch))
        curr_blur_size = curr_blur_size + (1 - curr_blur_size % 2)
        kernel = create_blur_kernel(kernel_path, curr_blur_size)
        K_blur.append(kernel)

    # Set first channel to sharp (delta)
    center = blur_size // 2
    K_blur[0] = np.zeros((blur_size, blur_size))
    K_blur[0][center, center] = 1.0

    # Apply blur and noise
    noise_sd = 0.005
    I_blurred = np.zeros_like(I)
    K_blur_noisy = []

    for ch in range(n_channels):
        I_blurred[:, :, ch] = convolve(I_sharp[:, :, ch], K_blur[ch], mode='reflect')
        I_blurred[:, :, ch] = random_noise(I_blurred[:, :, ch], mode='gaussian',
                                           var=noise_sd ** 2, clip=False)
        kernel_var = 0.0000001
        K_noisy = K_blur[ch] + np.random.randn(*K_blur[ch].shape) * np.sqrt(kernel_var / (ch + 1) ** 2)
        K_noisy = np.maximum(K_noisy, 0)
        K_noisy = K_noisy / K_noisy.sum()
        K_blur_noisy.append(K_noisy)

    # Prepare channel data
    channels = []
    for ch in range(n_channels):
        channels.append({
            'image': I_blurred[:, :, ch].copy(),
            'kernel': K_blur_noisy[ch]
        })

    db_chs = [
        {'image': ch['image'].copy(), 'kernel': ch['kernel'].copy()}
        for ch in channels
    ]

    # Process channel 0 first (it's essentially sharp)
    print("\n--- Processing channel 0 (sharp reference) ---")
    ch = 0
    ks = channels[ch]['kernel'].shape[0]

    # Pad and edgetaper
    img_pad = np.pad(channels[ch]['image'], ks, mode='edge')
    for _ in range(4):
        img_pad = edgetaper(img_pad, channels[ch]['kernel'])
    img_pad = img_pad + 1.0

    # For channel 0, just set the result (it's already sharp)
    db_chs[ch]['image'] = img_pad[ks:-ks, ks:-ks] - 1.0

    print(f"Channel 0 deconvolved (sharp reference)")

    # Process channel 2 with cross-channel coupling
    print("\n--- Processing channel 2 with cross-channel coupling ---")
    ch = 2

    lambda_residual = 750
    lambda_tv = 0.5
    lambda_cross_ch = np.array([1.0, 0.0, 0.0])  # Coupled to channel 0

    # Pad and edgetaper
    img = channels[ch]['image'].copy()
    kernel = channels[ch]['kernel']

    ks = kernel.shape[0]
    img_pad = np.pad(img, ks, mode='edge')
    db_pad = np.pad(db_chs[ch]['image'], ks, mode='edge')

    # Also pad the reference channel (channel 0)
    ref_pad = np.pad(db_chs[0]['image'], ks, mode='edge')

    for _ in range(4):
        img_pad = edgetaper(img_pad, kernel)
        db_pad = edgetaper(db_pad, kernel)
        ref_pad = edgetaper(ref_pad, kernel)

    img_pad += 1.0
    db_pad += 1.0
    ref_pad += 1.0

    # Create padded channel structures
    db_chs_padded = [
        {'image': ref_pad, 'kernel': db_chs[0]['kernel']},
        {'image': db_pad, 'kernel': db_chs[1]['kernel']},
        {'image': db_pad, 'kernel': db_chs[2]['kernel']},
    ]

    shape = img_pad.shape

    # Setup FFT solve
    otf = psf2otf(kernel, shape)
    Nomin1 = np.conj(otf) * fft2(img_pad)
    Denom1 = np.abs(otf) ** 2

    # Compute operator norm with cross-channel
    def A(x):
        return _Kmult_with_cross(x, ch, db_chs_padded, lambda_cross_ch, lambda_tv)

    def AS(x):
        return _KSmult_with_cross(x, ch, db_chs_padded, lambda_cross_ch, lambda_tv)

    L = compute_operator_norm(A, AS, shape)
    print(f"Operator norm L: {L:.4f}")

    # Step sizes
    sigma = 1.0
    tau = 0.7 / (sigma * L ** 2)

    print(f"tau: {tau:.4f}")
    print(f"tau * 2 * lambda_residual: {tau * 2 * lambda_residual:.1f}")

    # Proximal operators
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

    # PD iterations
    max_it = 50
    tol = 1e-4

    print("\nRunning PD iterations with cross-channel coupling...")
    for i in range(max_it):
        f_old = f.copy()

        g = prox_fs(g + sigma * A(f1), sigma)
        f = solve_fft(Nomin1, Denom1, tau, lambda_residual, f - tau * AS(g))
        f1 = f + 1.0 * (f - f_old)

        diff = f - f_old
        rel_diff = np.linalg.norm(diff.ravel()) / (np.linalg.norm(f.ravel()) + 1e-10)

        if i < 5 or i % 10 == 0:
            row_jump = np.abs(np.diff(f1.mean(axis=1))).max()
            print(f"  Iter {i+1}: diff={rel_diff:.5g}, row_jump={row_jump:.4f}")

        if rel_diff < tol:
            print(f"  Converged at iteration {i+1}")
            break

    # Threshold and unpad
    result = np.maximum(f1, 1.0)
    result = result[ks:-ks, ks:-ks] - 1.0

    # Save result
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    result_clipped = np.clip(result, 0, 1)
    result_gamma = result_clipped ** 0.5  # Apply gamma for display
    iio.imwrite(output_dir / 'debug_cross_channel_ch2.png',
                (result_gamma * 255).astype(np.uint8))
    print(f"\nSaved to {output_dir}/debug_cross_channel_ch2.png")

    # Also test without cross-channel for comparison
    print("\n--- Testing channel 2 WITHOUT cross-channel coupling ---")
    lambda_cross_ch_zero = np.array([0.0, 0.0, 0.0])

    def A_no_cross(x):
        return _Kmult_with_cross(x, ch, db_chs_padded, lambda_cross_ch_zero, lambda_tv)

    def AS_no_cross(x):
        return _KSmult_with_cross(x, ch, db_chs_padded, lambda_cross_ch_zero, lambda_tv)

    L_no_cross = compute_operator_norm(A_no_cross, AS_no_cross, shape)
    print(f"Operator norm L (no cross): {L_no_cross:.4f}")

    tau_no_cross = 0.7 / (sigma * L_no_cross ** 2)

    f = img_pad.copy()
    g = A_no_cross(f)
    f1 = f.copy()

    print("\nRunning PD iterations WITHOUT cross-channel coupling...")
    for i in range(max_it):
        f_old = f.copy()

        g = prox_fs(g + sigma * A_no_cross(f1), sigma)
        f = solve_fft(Nomin1, Denom1, tau_no_cross, lambda_residual, f - tau_no_cross * AS_no_cross(g))
        f1 = f + 1.0 * (f - f_old)

        diff = f - f_old
        rel_diff = np.linalg.norm(diff.ravel()) / (np.linalg.norm(f.ravel()) + 1e-10)

        if i < 5 or i % 10 == 0:
            row_jump = np.abs(np.diff(f1.mean(axis=1))).max()
            print(f"  Iter {i+1}: diff={rel_diff:.5g}, row_jump={row_jump:.4f}")

        if rel_diff < tol:
            print(f"  Converged at iteration {i+1}")
            break

    result_no_cross = np.maximum(f1, 1.0)
    result_no_cross = result_no_cross[ks:-ks, ks:-ks] - 1.0

    result_no_cross_clipped = np.clip(result_no_cross, 0, 1)
    result_no_cross_gamma = result_no_cross_clipped ** 0.5
    iio.imwrite(output_dir / 'debug_no_cross_channel_ch2.png',
                (result_no_cross_gamma * 255).astype(np.uint8))
    print(f"Saved to {output_dir}/debug_no_cross_channel_ch2.png")


if __name__ == '__main__':
    test_cross_channel_deconv()
