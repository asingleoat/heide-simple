#!/usr/bin/env python3
"""Debug operator norm and step size computation."""

import numpy as np
from scipy.fft import fft2, ifft2
from pathlib import Path
import imageio.v3 as iio
from skimage.transform import resize

from deconv import psf2otf, img_to_norm_grayscale, compute_operator_norm
from deconv.utils import imconv


def create_blur_kernel(image_path, size):
    """Load and resize blur kernel from image."""
    kernel = iio.imread(image_path)
    kernel = img_to_norm_grayscale(kernel)
    kernel = resize(kernel, (size, size), order=3, anti_aliasing=True)
    kernel = kernel / kernel.sum()
    return kernel


def _Kmult(f, lambda_tv):
    """Forward operator K: computes TV gradient terms."""
    dxf = np.array([[-1, 1]])
    dyf = np.array([[-1], [1]])
    dxxf = np.array([[-1, 2, -1]])
    dyyf = np.array([[-1], [2], [-1]])
    dxyf = np.array([[-1, 1], [1, -1]])

    results = []

    if lambda_tv > 1e-10:
        fx = imconv(f, dxf[:, ::-1][::-1, :], 'full')
        fx = (lambda_tv * 0.5) * fx[:, 1:]

        fy = imconv(f, dyf[::-1, :], 'full')
        fy = (lambda_tv * 0.5) * fy[1:, :]

        sd_w = 0.15
        fxx = imconv(f, dxxf[:, ::-1], 'full')
        fxx = (lambda_tv * sd_w) * fxx[:, 2:]

        fyy = imconv(f, dyyf[::-1, :], 'full')
        fyy = (lambda_tv * sd_w) * fyy[2:, :]

        fxy = imconv(f, dxyf[::-1, ::-1], 'full')
        fxy = (lambda_tv * sd_w) * fxy[1:, 1:]

        results.extend([fx, fy, fxx, fyy, fxy])

    if not results:
        return np.zeros((*f.shape, 1))

    return np.stack(results, axis=2)


def _KSmult(f, lambda_tv):
    """Adjoint operator K*: transpose of the forward operator."""
    dxf = np.array([[-1, 1]])
    dyf = np.array([[-1], [1]])
    dxxf = np.array([[-1, 2, -1]])
    dyyf = np.array([[-1], [2], [-1]])
    dxyf = np.array([[-1, 1], [1, -1]])

    result = np.zeros((f.shape[0], f.shape[1]))

    i = 0
    if lambda_tv > 1e-10:
        fx = imconv((lambda_tv * 0.5) * f[:, :, i], dxf, 'full')
        result += fx[:, :-1]
        i += 1

        fy = imconv((lambda_tv * 0.5) * f[:, :, i], dyf, 'full')
        result += fy[:-1, :]
        i += 1

        sd_w = 0.15
        fxx = imconv((lambda_tv * sd_w) * f[:, :, i], dxxf, 'full')
        result += fxx[:, :-2]
        i += 1

        fyy = imconv((lambda_tv * sd_w) * f[:, :, i], dyyf, 'full')
        result += fyy[:-2, :]
        i += 1

        fxy = imconv((lambda_tv * sd_w) * f[:, :, i], dxyf, 'full')
        result += fxy[:-1, :-1]
        i += 1

    return result


def test_adjoint_property():
    """Test that K and K* satisfy adjoint property: <Kx, y> = <x, K*y>"""
    print("=== Testing adjoint property ===")

    lambda_tv = 1.0
    shape = (50, 60)

    # Random test vectors
    np.random.seed(42)
    x = np.random.randn(*shape)

    # Apply K to x
    Kx = _Kmult(x, lambda_tv)

    # Random y of same shape as Kx
    y = np.random.randn(*Kx.shape)

    # Apply K* to y
    KSy = _KSmult(y, lambda_tv)

    # Compute inner products
    inner1 = np.sum(Kx * y)
    inner2 = np.sum(x * KSy)

    print(f"<Kx, y>  = {inner1:.6f}")
    print(f"<x, K*y> = {inner2:.6f}")
    print(f"Relative error: {abs(inner1 - inner2) / abs(inner1):.2%}")

    return abs(inner1 - inner2) / abs(inner1) < 0.01


def test_operator_norm():
    """Test operator norm computation."""
    print("\n=== Testing operator norm ===")

    lambda_tv = 1.0
    shape = (50, 60)

    def A(x):
        return _Kmult(x, lambda_tv)

    def AS(x):
        return _KSmult(x, lambda_tv)

    L = compute_operator_norm(A, AS, shape)
    print(f"Computed operator norm L: {L:.6f}")

    # Verify by applying A^* A to a random vector multiple times (power iteration check)
    np.random.seed(42)
    x = np.random.randn(*shape)
    x = x / np.linalg.norm(x)

    for _ in range(20):
        y = AS(A(x))
        eigenvalue = np.dot(x.ravel(), y.ravel())
        norm_y = np.linalg.norm(y)
        x = y / norm_y

    print(f"Power iteration eigenvalue: {eigenvalue:.6f}")
    print(f"sqrt(eigenvalue) = {np.sqrt(eigenvalue):.6f}")
    print(f"Match: {abs(L - np.sqrt(eigenvalue)) < 0.1}")


def test_step_sizes():
    """Compute step sizes and compare with expected values."""
    print("\n=== Testing step sizes ===")

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'

    kernel = create_blur_kernel(kernel_path, 25)
    shape = (100, 150)

    lambda_tv = 0.5

    def A(x):
        return _Kmult(x, lambda_tv)

    def AS(x):
        return _KSmult(x, lambda_tv)

    L = compute_operator_norm(A, AS, shape)

    sigma = 1.0
    tau = 0.7 / (sigma * L ** 2)
    theta = 1.0

    print(f"Operator norm L: {L:.6f}")
    print(f"sigma: {sigma:.6f}")
    print(f"tau: {tau:.6f}")
    print(f"theta: {theta:.6f}")

    # The product sigma * tau * L^2 should be < 1 for convergence
    print(f"sigma * tau * L^2 = {sigma * tau * L**2:.6f} (should be < 1)")


def test_prox_operators():
    """Test proximal operators."""
    print("\n=== Testing proximal operators ===")

    # Test _prox_fs (L1 proximal)
    def _prox_fs(u, sigma):
        amplitude = np.sqrt(np.sum(u ** 2, axis=2, keepdims=True))
        return u / np.maximum(1.0, amplitude)

    # Random 3D array
    u = np.random.randn(10, 10, 5) * 2

    result = _prox_fs(u, 1.0)

    # Check that amplitude is <= 1
    amplitude = np.sqrt(np.sum(result ** 2, axis=2))
    print(f"Max amplitude after prox: {amplitude.max():.6f} (should be <= 1)")

    # Test _solve_fft
    def _solve_fft(Nomin1, Denom1, tau, lambda_val, f):
        x = (tau * 2 * lambda_val * Nomin1 + fft2(f)) / (tau * 2 * lambda_val * Denom1 + 1)
        return np.real(ifft2(x))

    # Simple test: identity OTF
    shape = (50, 60)
    otf = np.ones(shape, dtype=complex)
    f = np.random.randn(*shape)
    b = np.random.randn(*shape)  # observed image

    Nomin1 = np.conj(otf) * fft2(b)
    Denom1 = np.abs(otf) ** 2

    tau = 0.1
    lambda_val = 100

    result = _solve_fft(Nomin1, Denom1, tau, lambda_val, f)

    # For identity OTF, this should be a weighted average of b and f
    # x = (tau*2*lambda*b + f) / (tau*2*lambda + 1)
    c = tau * 2 * lambda_val
    expected = (c * b + f) / (c + 1)

    print(f"Solve FFT test (identity OTF):")
    print(f"  Max diff from expected: {np.abs(result - expected).max():.10f}")


def debug_first_iteration():
    """Debug the first iteration of primal-dual to see where banding starts."""
    print("\n=== Debugging first iteration ===")

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

    # Setup
    otf = psf2otf(kernel, shape)
    Nomin1 = np.conj(otf) * fft2(img)
    Denom1 = np.abs(otf) ** 2

    lambda_residual = 750
    lambda_tv = 0.5

    def A(x):
        return _Kmult(x, lambda_tv)

    def AS(x):
        return _KSmult(x, lambda_tv)

    L = compute_operator_norm(A, AS, shape)
    print(f"Operator norm L: {L:.6f}")

    sigma = 1.0
    tau = 0.7 / (sigma * L ** 2)
    print(f"tau: {tau:.6f}")

    # Initialize
    f = img.copy()
    g = A(f)
    f1 = f.copy()

    print(f"\nInitial state:")
    print(f"  f row mean std: {f.mean(axis=1).std():.6f}")
    print(f"  f1 row mean std: {f1.mean(axis=1).std():.6f}")

    # One iteration
    def _prox_fs(u, sigma):
        amplitude = np.sqrt(np.sum(u ** 2, axis=2, keepdims=True))
        return u / np.maximum(1.0, amplitude)

    def _solve_fft(Nomin1, Denom1, tau, lambda_val, f):
        x = (tau * 2 * lambda_val * Nomin1 + fft2(f)) / (tau * 2 * lambda_val * Denom1 + 1)
        return np.real(ifft2(x))

    f_old = f.copy()

    # Dual update
    Af1 = A(f1)
    g_before = g.copy()
    g = _prox_fs(g + sigma * Af1, sigma)
    print(f"\nAfter dual update:")
    print(f"  g change: {np.abs(g - g_before).max():.6f}")

    # Primal update
    ASg = AS(g)
    f_input = f - tau * ASg

    print(f"\nBefore solve_fft:")
    print(f"  f_input row mean std: {f_input.mean(axis=1).std():.6f}")
    print(f"  f_input row-to-row max jump: {np.abs(np.diff(f_input.mean(axis=1))).max():.6f}")

    f = _solve_fft(Nomin1, Denom1, tau, lambda_residual, f_input)

    print(f"\nAfter solve_fft:")
    print(f"  f row mean std: {f.mean(axis=1).std():.6f}")
    print(f"  f row-to-row max jump: {np.abs(np.diff(f.mean(axis=1))).max():.6f}")

    # Analyze what solve_fft does at specific frequencies
    print(f"\nAnalyzing solve_fft effect on frequencies:")

    # The solve_fft formula is:
    # x = (c * Nomin1 + fft2(f_input)) / (c * Denom1 + 1)
    c = tau * 2 * lambda_residual
    print(f"  c = tau * 2 * lambda = {c:.6f}")

    # At DC (0,0): Denom1 is 1.0
    print(f"  At DC: Denom1={Denom1[0,0]:.4f}, denominator={c*Denom1[0,0]+1:.4f}")

    # At low frequencies where OTF is still significant
    print(f"  At (1,0): Denom1={Denom1[1,0]:.4f}, denominator={c*Denom1[1,0]+1:.4f}")
    print(f"  At (0,1): Denom1={Denom1[0,1]:.4f}, denominator={c*Denom1[0,1]+1:.4f}")

    # At high frequencies where OTF is near zero
    mid_row = shape[0] // 2
    mid_col = shape[1] // 2
    print(f"  At ({mid_row},0): Denom1={Denom1[mid_row,0]:.6f}, denominator={c*Denom1[mid_row,0]+1:.4f}")
    print(f"  At (0,{mid_col}): Denom1={Denom1[0,mid_col]:.6f}, denominator={c*Denom1[0,mid_col]+1:.4f}")

    # Over-relaxation
    f1 = f + 1.0 * (f - f_old)

    print(f"\nAfter over-relaxation:")
    print(f"  f1 row mean std: {f1.mean(axis=1).std():.6f}")
    print(f"  f1 row-to-row max jump: {np.abs(np.diff(f1.mean(axis=1))).max():.6f}")


if __name__ == '__main__':
    adjoint_ok = test_adjoint_property()
    if not adjoint_ok:
        print("WARNING: Adjoint property not satisfied!")

    test_operator_norm()
    test_step_sizes()
    test_prox_operators()
    debug_first_iteration()
