#!/usr/bin/env python3
"""Further debug the adjoint issue."""

import numpy as np
from deconv.utils import imconv


def test_all_derivative_adjoints():
    """Test adjoint for all derivative filters."""
    print("=== Testing all derivative filter adjoints ===")

    np.random.seed(42)

    # Test with progressively larger images to see if boundary error matters
    for size in [(10, 15), (50, 60), (100, 150)]:
        H, W = size
        f = np.random.randn(H, W)

        print(f"\n--- Image size {H}x{W} ---")

        # Test fx (x-derivative)
        dxf = np.array([[-1, 1]])
        dxf_flipped = dxf[:, ::-1]

        # Forward
        fx_full = imconv(f, dxf_flipped, 'full')
        fx = fx_full[:, 1:]

        # Random g of matching shape
        g = np.random.randn(*fx.shape)

        # Adjoint
        fxt_full = imconv(g, dxf, 'full')
        fxt = fxt_full[:, :-1]

        inner1 = np.sum(fx * g)
        inner2 = np.sum(f * fxt)
        error_fx = abs(inner1 - inner2) / abs(inner1)
        print(f"  fx adjoint error: {error_fx:.2%}")

        # Test fy (y-derivative)
        dyf = np.array([[-1], [1]])
        dyf_flipped = dyf[::-1, :]

        fy_full = imconv(f, dyf_flipped, 'full')
        fy = fy_full[1:, :]

        g = np.random.randn(*fy.shape)

        fyt_full = imconv(g, dyf, 'full')
        fyt = fyt_full[:-1, :]

        inner1 = np.sum(fy * g)
        inner2 = np.sum(f * fyt)
        error_fy = abs(inner1 - inner2) / abs(inner1)
        print(f"  fy adjoint error: {error_fy:.2%}")

        # Test fxx (second x-derivative)
        dxxf = np.array([[-1, 2, -1]])
        dxxf_flipped = dxxf[:, ::-1]

        fxx_full = imconv(f, dxxf_flipped, 'full')
        fxx = fxx_full[:, 2:]

        g = np.random.randn(*fxx.shape)

        fxxt_full = imconv(g, dxxf, 'full')
        fxxt = fxxt_full[:, :-2]

        inner1 = np.sum(fxx * g)
        inner2 = np.sum(f * fxxt)
        error_fxx = abs(inner1 - inner2) / abs(inner1)
        print(f"  fxx adjoint error: {error_fxx:.2%}")

        # Test fyy (second y-derivative)
        dyyf = np.array([[-1], [2], [-1]])
        dyyf_flipped = dyyf[::-1, :]

        fyy_full = imconv(f, dyyf_flipped, 'full')
        fyy = fyy_full[2:, :]

        g = np.random.randn(*fyy.shape)

        fyyt_full = imconv(g, dyyf, 'full')
        fyyt = fyyt_full[:-2, :]

        inner1 = np.sum(fyy * g)
        inner2 = np.sum(f * fyyt)
        error_fyy = abs(inner1 - inner2) / abs(inner1)
        print(f"  fyy adjoint error: {error_fyy:.2%}")

        # Test fxy (cross derivative)
        dxyf = np.array([[-1, 1], [1, -1]])
        dxyf_flipped = dxyf[::-1, ::-1]

        fxy_full = imconv(f, dxyf_flipped, 'full')
        fxy = fxy_full[1:, 1:]

        g = np.random.randn(*fxy.shape)

        fxyt_full = imconv(g, dxyf, 'full')
        fxyt = fxyt_full[:-1, :-1]

        inner1 = np.sum(fxy * g)
        inner2 = np.sum(f * fxyt)
        error_fxy = abs(inner1 - inner2) / abs(inner1)
        print(f"  fxy adjoint error: {error_fxy:.2%}")


def compare_with_matlab_style():
    """Compare MATLAB-style boundary vs zero-pad boundary."""
    print("\n=== Comparing boundary styles ===")

    np.random.seed(42)
    H, W = 10, 15
    f = np.random.randn(H, W)
    g = np.random.randn(H, W)

    dxf = np.array([[-1, 1]])
    dxf_flipped = dxf[:, ::-1]

    print("Testing x-derivative with different boundary handling:")

    # Current implementation (replicate boundary)
    fx_full = imconv(f, dxf_flipped, 'full')
    fx = fx_full[:, 1:]

    fxt_full = imconv(g, dxf, 'full')
    fxt = fxt_full[:, :-1]

    inner1 = np.sum(fx * g)
    inner2_rep = np.sum(f * fxt)
    print(f"  Replicate boundary: <Df, g>={inner1:.4f}, <f, D*g>={inner2_rep:.4f}, error={abs(inner1-inner2_rep)/abs(inner1):.2%}")

    # Try zero-padding instead
    from scipy.signal import convolve2d

    # Forward with zero-pad
    fx_zp = convolve2d(f, dxf_flipped, mode='full', boundary='fill', fillvalue=0)
    fx_zp = fx_zp[:, 1:]

    # Adjoint with zero-pad
    fxt_zp = convolve2d(g, dxf, mode='full', boundary='fill', fillvalue=0)
    fxt_zp = fxt_zp[:, :-1]

    inner2_zp = np.sum(f * fxt_zp)
    inner1_zp = np.sum(fx_zp * g)
    print(f"  Zero-pad boundary: <Df, g>={inner1_zp:.4f}, <f, D*g>={inner2_zp:.4f}, error={abs(inner1_zp-inner2_zp)/abs(inner1_zp):.2%}")


def test_exact_adjoint():
    """Test with mathematically correct adjoint (direct matrix implementation)."""
    print("\n=== Testing exact adjoint ===")

    np.random.seed(42)
    H, W = 10, 15
    f = np.random.randn(H, W)

    dxf = np.array([[-1, 1]])
    dxf_flipped = dxf[:, ::-1]

    # Forward (same as before)
    fx_full = imconv(f, dxf_flipped, 'full')
    fx = fx_full[:, 1:]

    # Random g
    g = np.random.randn(*fx.shape)

    # Exact adjoint (derived from matrix transpose)
    # For forward y[j] = f[j+1] - f[j] (with j=0..W-2, y[W-1]=0)
    # Adjoint: x[j] = -g[j] + g[j-1] for j>0, x[0] = -g[0], x[W-1] = g[W-2]
    # Wait, let me re-derive...

    # Forward matrix M is W x W where:
    # M[j, j] = -1, M[j, j+1] = 1 for j < W-1
    # M[W-1, :] = 0 (last row all zeros)

    # M^T[j, j] = -1, M^T[j+1, j] = 1 for j < W-1
    # Which means:
    # (M^T @ g)[0] = M^T[0,0]*g[0] = -g[0]
    # (M^T @ g)[j] = M^T[j,j-1]*g[j-1] + M^T[j,j]*g[j] = g[j-1] - g[j] for 0 < j < W-1
    # (M^T @ g)[W-1] = M^T[W-1,W-2]*g[W-2] = g[W-2]

    def exact_adjoint_fx(g):
        H, W = g.shape
        result = np.zeros((H, W))
        result[:, 0] = -g[:, 0]
        for j in range(1, W - 1):
            result[:, j] = g[:, j - 1] - g[:, j]
        result[:, -1] = g[:, -2]
        return result

    fxt_exact = exact_adjoint_fx(g)

    inner1 = np.sum(fx * g)
    inner2_exact = np.sum(f * fxt_exact)
    print(f"Exact adjoint: <Df, g>={inner1:.4f}, <f, D*g>={inner2_exact:.4f}, error={abs(inner1-inner2_exact)/abs(inner1):.6%}")

    # Current implementation
    fxt_full = imconv(g, dxf, 'full')
    fxt_current = fxt_full[:, :-1]
    inner2_current = np.sum(f * fxt_current)
    print(f"Current adjoint: <f, D*g>={inner2_current:.4f}, error={abs(inner1-inner2_current)/abs(inner1):.2%}")

    # Compare the two
    print(f"\nDifference between exact and current adjoint:")
    diff = fxt_exact - fxt_current
    print(f"  Max abs diff: {np.abs(diff).max():.4f}")
    print(f"  Diff at boundaries:")
    print(f"    First column: exact={fxt_exact[0,:3]}, current={fxt_current[0,:3]}")
    print(f"    Last column: exact={fxt_exact[0,-3:]}, current={fxt_current[0,-3:]}")


def implement_correct_adjoint_operators():
    """Implement corrected adjoint operators."""
    print("\n=== Implementing correct adjoint ===")

    def forward_fx(f, lambd):
        """Forward x-derivative."""
        dxf_flipped = np.array([[1, -1]])
        fx = imconv(f, dxf_flipped, 'full')[:, 1:]
        return lambd * fx

    def adjoint_fx_correct(g, lambd):
        """Correct adjoint of x-derivative."""
        H, W = g.shape
        scaled_g = lambd * g
        result = np.zeros((H, W))
        result[:, 0] = -scaled_g[:, 0]
        for j in range(1, W - 1):
            result[:, j] = scaled_g[:, j - 1] - scaled_g[:, j]
        result[:, -1] = scaled_g[:, -2]
        return result

    def forward_fy(f, lambd):
        """Forward y-derivative."""
        dyf_flipped = np.array([[1], [-1]])
        fy = imconv(f, dyf_flipped, 'full')[1:, :]
        return lambd * fy

    def adjoint_fy_correct(g, lambd):
        """Correct adjoint of y-derivative."""
        H, W = g.shape
        scaled_g = lambd * g
        result = np.zeros((H, W))
        result[0, :] = -scaled_g[0, :]
        for i in range(1, H - 1):
            result[i, :] = scaled_g[i - 1, :] - scaled_g[i, :]
        result[-1, :] = scaled_g[-2, :]
        return result

    # Test
    np.random.seed(42)
    H, W = 50, 60
    f = np.random.randn(H, W)
    lambd = 0.5

    # Test fx
    Df = forward_fx(f, lambd)
    g = np.random.randn(*Df.shape)
    Dtg = adjoint_fx_correct(g, lambd)

    inner1 = np.sum(Df * g)
    inner2 = np.sum(f * Dtg)
    print(f"fx: <Df, g>={inner1:.4f}, <f, D*g>={inner2:.4f}, error={abs(inner1-inner2)/abs(inner1):.6%}")

    # Test fy
    Df = forward_fy(f, lambd)
    g = np.random.randn(*Df.shape)
    Dtg = adjoint_fy_correct(g, lambd)

    inner1 = np.sum(Df * g)
    inner2 = np.sum(f * Dtg)
    print(f"fy: <Df, g>={inner1:.4f}, <f, D*g>={inner2:.4f}, error={abs(inner1-inner2)/abs(inner1):.6%}")


def test_primal_dual_with_correct_adjoint():
    """Test if using correct adjoint fixes the banding."""
    print("\n=== Testing PD with correct adjoint ===")

    from pathlib import Path
    import imageio.v3 as iio
    from skimage.transform import resize
    from scipy.fft import fft2, ifft2

    from deconv import psf2otf, img_to_norm_grayscale

    base_dir = Path(__file__).parent / 'DeconvolutionColorPrior'
    kernel_path = base_dir / 'kernels' / 'fading.png'
    image_path = base_dir / 'images' / 'houses_big.jpg'

    # Load and prepare
    def create_blur_kernel(image_path, size):
        kernel = iio.imread(image_path)
        kernel = img_to_norm_grayscale(kernel)
        kernel = resize(kernel, (size, size), order=3, anti_aliasing=True)
        kernel = kernel / kernel.sum()
        return kernel

    I = iio.imread(image_path)
    I = resize(I, (int(I.shape[0] * 0.15), int(I.shape[1] * 0.15)), anti_aliasing=True)
    I = I.astype(np.float64)
    I = I / I.max()
    I = I ** 2.0

    img = I[:, :, 2]
    kernel = create_blur_kernel(kernel_path, 43)
    shape = img.shape

    # Simplified PD iteration with correct adjoint
    def forward_K(f, lambda_tv):
        """Forward operator with only fx, fy."""
        dxf_flipped = np.array([[1, -1]])
        dyf_flipped = np.array([[1], [-1]])

        fx = imconv(f, dxf_flipped, 'full')[:, 1:]
        fy = imconv(f, dyf_flipped, 'full')[1:, :]

        return np.stack([lambda_tv * 0.5 * fx, lambda_tv * 0.5 * fy], axis=2)

    def adjoint_K_correct(g, lambda_tv):
        """Correct adjoint."""
        H, W = g.shape[:2]

        # fx adjoint
        gx = lambda_tv * 0.5 * g[:, :, 0]
        result_x = np.zeros((H, W))
        result_x[:, 0] = -gx[:, 0]
        for j in range(1, W - 1):
            result_x[:, j] = gx[:, j - 1] - gx[:, j]
        result_x[:, -1] = gx[:, -2]

        # fy adjoint
        gy = lambda_tv * 0.5 * g[:, :, 1]
        result_y = np.zeros((H, W))
        result_y[0, :] = -gy[0, :]
        for i in range(1, H - 1):
            result_y[i, :] = gy[i - 1, :] - gy[i, :]
        result_y[-1, :] = gy[-2, :]

        return result_x + result_y

    def adjoint_K_current(g, lambda_tv):
        """Current (wrong) adjoint."""
        dxf = np.array([[-1, 1]])
        dyf = np.array([[-1], [1]])

        fx = imconv(lambda_tv * 0.5 * g[:, :, 0], dxf, 'full')[:, :-1]
        fy = imconv(lambda_tv * 0.5 * g[:, :, 1], dyf, 'full')[:-1, :]

        return fx + fy

    # Test adjoint property
    np.random.seed(42)
    f = np.random.randn(*shape)
    lambda_tv = 0.5

    Kf = forward_K(f, lambda_tv)
    g = np.random.randn(*Kf.shape)

    inner1 = np.sum(Kf * g)

    Ktg_correct = adjoint_K_correct(g, lambda_tv)
    inner2_correct = np.sum(f * Ktg_correct)
    print(f"Correct adjoint error: {abs(inner1-inner2_correct)/abs(inner1):.4%}")

    Ktg_current = adjoint_K_current(g, lambda_tv)
    inner2_current = np.sum(f * Ktg_current)
    print(f"Current adjoint error: {abs(inner1-inner2_current)/abs(inner1):.4%}")

    # Now run a few PD iterations with correct adjoint
    otf = psf2otf(kernel, shape)
    Nomin1 = np.conj(otf) * fft2(img)
    Denom1 = np.abs(otf) ** 2

    lambda_residual = 750
    lambda_tv = 0.5

    # Compute L with correct adjoint
    def matvec(x):
        return adjoint_K_correct(forward_K(x, lambda_tv), lambda_tv)

    x = np.random.randn(*shape)
    x = x / np.linalg.norm(x)
    for _ in range(20):
        y = matvec(x)
        eigenvalue = np.dot(x.ravel(), y.ravel())
        x = y / np.linalg.norm(y)
    L_correct = np.sqrt(eigenvalue)
    print(f"\nOperator norm with correct adjoint: {L_correct:.4f}")

    # Compare with current
    def matvec_current(x):
        return adjoint_K_current(forward_K(x, lambda_tv), lambda_tv)

    x = np.random.randn(*shape)
    x = x / np.linalg.norm(x)
    for _ in range(20):
        y = matvec_current(x)
        eigenvalue = np.dot(x.ravel(), y.ravel())
        x = y / np.linalg.norm(y)
    L_current = np.sqrt(eigenvalue)
    print(f"Operator norm with current adjoint: {L_current:.4f}")

    # Run iterations
    sigma = 1.0
    tau = 0.7 / (sigma * L_correct ** 2)

    f = img.copy()
    g = forward_K(f, lambda_tv)
    f1 = f.copy()

    def prox_fs(u, sigma):
        amplitude = np.sqrt(np.sum(u ** 2, axis=2, keepdims=True))
        return u / np.maximum(1.0, amplitude)

    def solve_fft(Nomin1, Denom1, tau, lambda_val, f):
        x = (tau * 2 * lambda_val * Nomin1 + fft2(f)) / (tau * 2 * lambda_val * Denom1 + 1)
        return np.real(ifft2(x))

    print(f"\nRunning PD iterations with correct adjoint...")
    for i in range(5):
        f_old = f.copy()
        g = prox_fs(g + sigma * forward_K(f1, lambda_tv), sigma)
        f = solve_fft(Nomin1, Denom1, tau, lambda_residual, f - tau * adjoint_K_correct(g, lambda_tv))
        f1 = f + 1.0 * (f - f_old)

        row_means = f1.mean(axis=1)
        row_jump = np.abs(np.diff(row_means)).max()
        print(f"  Iter {i+1}: row-to-row max jump = {row_jump:.4f}")

    # Save result
    output_dir = Path(__file__).parent / 'output'
    result = np.clip(f1, 0, 1)
    iio.imwrite(output_dir / 'debug_correct_adjoint.png', (result * 255).astype(np.uint8))
    print(f"Saved to {output_dir}/debug_correct_adjoint.png")


if __name__ == '__main__':
    test_all_derivative_adjoints()
    compare_with_matlab_style()
    test_exact_adjoint()
    implement_correct_adjoint_operators()
    test_primal_dual_with_correct_adjoint()
