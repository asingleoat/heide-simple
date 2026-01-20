#!/usr/bin/env python3
"""Debug the adjoint property for derivative operators."""

import numpy as np
from deconv.utils import imconv


def test_simple_derivative_adjoint():
    """Test adjoint of simple derivative operator."""
    print("=== Testing simple 1D derivative adjoint ===")

    # Test vectors
    np.random.seed(42)
    f = np.random.randn(5, 7)  # H=5, W=7
    g_full = np.random.randn(5, 8)  # Full output size for 'full' convolution

    dxf = np.array([[-1, 1]])
    dxf_flipped = dxf[:, ::-1]  # [[1, -1]]

    # Forward: convolve with flipped, 'full' mode
    Df_full = imconv(f, dxf_flipped, 'full')
    print(f"Input f shape: {f.shape}")
    print(f"Full convolution output shape: {Df_full.shape}")

    # For the adjoint test, g should have the same shape as Df_full
    g = g_full

    # Adjoint: convolve with unflipped
    Dtg_full = imconv(g, dxf, 'full')
    print(f"Adjoint full output shape: {Dtg_full.shape}")

    # Test adjoint property: <Df, g> = <f, D*g>
    # But shapes don't match! Df_full is 5x8, g is 5x8, so inner product works
    # f is 5x7, Dtg_full is 5x9, so we need to slice Dtg_full

    # The adjoint of 'full' convolution needs proper slicing
    # For conv(f, k, 'full') where f is (H,W) and k is (1,2):
    # output is (H, W+1)
    # The adjoint operation conv(g, k_reversed, 'full') gives (H, W+2)
    # We need to slice to get (H, W)

    # Let me verify by computing the matrix form
    # D_ij = contribution to output j from input i

    print("\n--- Testing with slicing as in MATLAB code ---")

    # Forward: conv(f, [1,-1], 'full')[:, 1:]
    # This drops the first column
    Df = imconv(f, dxf_flipped, 'full')[:, 1:]
    print(f"Forward output shape (after slice): {Df.shape}")

    # Adjoint: conv(g, [-1,1], 'full')[:, :-1]
    # This drops the last column
    Dtg = imconv(g[:, 1:], dxf, 'full')[:, :-1]
    print(f"Adjoint output shape (after slice): {Dtg.shape}")

    # Now test adjoint property
    inner1 = np.sum(Df * g[:, 1:])  # Need to use g[:, 1:] since Df shape matches that
    inner2 = np.sum(f * Dtg)
    print(f"\n<Df, g[1:]> = {inner1:.6f}")
    print(f"<f, D*g>    = {inner2:.6f}")
    print(f"Difference: {abs(inner1 - inner2):.6f}")
    print(f"Relative error: {abs(inner1 - inner2) / abs(inner1):.2%}")


def test_derivative_adjoint_explicit():
    """Test derivative adjoint with explicit matrix computation."""
    print("\n=== Explicit matrix test ===")

    H, W = 3, 4
    np.random.seed(42)
    f = np.random.randn(H, W)

    dxf = np.array([[-1, 1]])
    dxf_flipped = np.array([[1, -1]])

    # Build forward operator matrix (for x derivative)
    # Forward: conv(f, [1,-1], 'full')[:, 1:]

    # First, compute conv(f, [1,-1], 'full') which gives H x (W+1)
    # Then slice [:, 1:] to get H x W

    # The convolution with [1, -1] computes:
    # out[j] = 1*f_padded[j] + (-1)*f_padded[j+1]
    # With edge padding, f_padded = [f[0], f[0], f[1], ..., f[W-1], f[W-1]]

    # Let me compute the matrix form manually for one row
    # Input vector: [f0, f1, f2, f3]
    # Padded: [f0, f0, f1, f2, f3, f3]
    # Convolve with [1, -1]:
    # out[0] = f0 - f0 = 0
    # out[1] = f0 - f1
    # out[2] = f1 - f2
    # out[3] = f2 - f3
    # out[4] = f3 - f3 = 0

    # After [:, 1:] slice:
    # [f0-f1, f1-f2, f2-f3, 0]

    # As a matrix operating on [f0, f1, f2, f3]:
    D_forward = np.array([
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1],
        [0, 0, 0, 0],
    ])

    print(f"Forward operator matrix (for one row):\n{D_forward}")

    # Adjoint should be D^T:
    D_adjoint_expected = D_forward.T
    print(f"\nExpected adjoint matrix (D^T):\n{D_adjoint_expected}")

    # Now let's see what our adjoint code computes
    # Adjoint: conv(g, [-1, 1], 'full')[:, :-1]

    # For input g = [g0, g1, g2, g3]
    # Padded: [g0, g0, g1, g2, g3, g3]
    # Convolve with [-1, 1]:
    # out[0] = -g0 + g0 = 0
    # out[1] = -g0 + g1
    # out[2] = -g1 + g2
    # out[3] = -g2 + g3
    # out[4] = -g3 + g3 = 0

    # After [:, :-1] slice:
    # [0, -g0+g1, -g1+g2, -g2+g3]

    # As a matrix operating on [g0, g1, g2, g3]:
    D_adjoint_actual = np.array([
        [0, 0, 0, 0],
        [-1, 1, 0, 0],
        [0, -1, 1, 0],
        [0, 0, -1, 1],
    ])

    print(f"\nActual adjoint matrix (from code):\n{D_adjoint_actual}")

    print(f"\nDifference:\n{D_adjoint_expected - D_adjoint_actual}")

    # The matrices don't match!


def test_correct_adjoint():
    """Figure out the correct adjoint slicing."""
    print("\n=== Finding correct adjoint ===")

    H, W = 3, 4

    # Forward operator matrix (for one row)
    D_forward = np.array([
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1],
        [0, 0, 0, 0],
    ])

    # The adjoint should be D^T
    D_adjoint = D_forward.T
    print(f"Correct adjoint matrix (D^T):\n{D_adjoint}")

    # D^T is:
    # [1  0  0  0]
    # [-1 1  0  0]
    # [0 -1  1  0]
    # [0  0 -1  0]

    # This corresponds to:
    # out[0] = g0
    # out[1] = -g0 + g1
    # out[2] = -g1 + g2
    # out[3] = -g2

    # So the correct adjoint operation is:
    # 1. Prepend 0 to g
    # 2. Convolve with [-1, 1]
    # 3. Slice appropriately

    # Actually, let's verify this by checking the convolution more carefully
    # g = [g0, g1, g2, g3]
    # We want output [g0, g1-g0, g2-g1, -g2]

    # Alternative interpretation:
    # The forward takes f (W elements) and produces:
    # [f0-f1, f1-f2, f2-f3, 0]

    # Actually, the last element being 0 means the forward operator effectively
    # maps W elements to W-1 useful elements (plus a zero).

    # The adjoint of dropping a dimension should add back that dimension.

    # Let me think about this differently.
    # The forward operation is:
    # y = D @ x where D has shape (W, W) but last row is all zeros

    # So effectively D: R^W -> R^(W-1) (ignoring the zero row)
    # The adjoint D*: R^(W-1) -> R^W

    # If we only consider the non-zero part:
    D_reduced = np.array([
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1],
    ])

    D_reduced_adjoint = D_reduced.T
    print(f"\nReduced forward matrix (3x4):\n{D_reduced}")
    print(f"Reduced adjoint (4x3):\n{D_reduced_adjoint}")

    # D_reduced.T is (4x3):
    # [1   0   0]
    # [-1  1   0]
    # [0  -1   1]
    # [0   0  -1]

    # So for input g = [g0, g1, g2] (3 elements):
    # out[0] = g0
    # out[1] = -g0 + g1
    # out[2] = -g1 + g2
    # out[3] = -g2


def test_adjoint_with_different_sizes():
    """Test when forward output has fewer columns than input."""
    print("\n=== Testing with correct dimensions ===")

    np.random.seed(42)
    H, W = 5, 7

    f = np.random.randn(H, W)

    dxf = np.array([[-1, 1]])
    dxf_flipped = np.array([[1, -1]])

    # Forward operation that MATLAB uses:
    # conv(f, [1,-1], 'full')[:, 1:]
    # This gives H x W output, but last column is always 0

    Df = imconv(f, dxf_flipped, 'full')[:, 1:]
    print(f"Forward output shape: {Df.shape}")
    print(f"Last column of Df: {Df[:, -1]}")  # Should be near zero

    # The "useful" output is really Df[:, :-1] which has W-1 columns
    Df_useful = Df[:, :-1]
    print(f"Useful forward output shape: {Df_useful.shape}")

    # For this reduced forward, what should the adjoint be?
    # We want D*: R^(H x (W-1)) -> R^(H x W)

    # Let's test with g having W-1 columns
    g = np.random.randn(H, W - 1)

    # Based on the matrix analysis, the adjoint should be:
    # out[:, 0] = g[:, 0]
    # out[:, j] = g[:, j] - g[:, j-1] for 1 <= j <= W-2
    # out[:, W-1] = -g[:, W-2]

    Dtg_correct = np.zeros((H, W))
    Dtg_correct[:, 0] = g[:, 0]
    for j in range(1, W - 1):
        Dtg_correct[:, j] = g[:, j] - g[:, j - 1]
    Dtg_correct[:, W - 1] = -g[:, W - 2]

    # Test adjoint property
    inner1 = np.sum(Df_useful * g)
    inner2 = np.sum(f * Dtg_correct)
    print(f"\n<Df_useful, g> = {inner1:.6f}")
    print(f"<f, D*g_correct> = {inner2:.6f}")
    print(f"Relative error: {abs(inner1 - inner2) / abs(inner1):.2%}")

    # Now, can we express Dtg_correct using convolution?
    # Dtg_correct[:, 0] = g[:, 0]
    # Dtg_correct[:, j] = g[:, j] - g[:, j-1] for 1 <= j <= W-2
    # Dtg_correct[:, W-1] = -g[:, W-2]

    # This is almost like convolving g with [-1, 1] but with special boundary handling

    # Let's try: pad g on the left with zeros, convolve with [-1, 1], then handle boundary
    g_padded = np.pad(g, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    Dtg_via_conv = imconv(g_padded, dxf, 'same')  # Hmm, doesn't quite work

    # Actually, let me try constructing it differently
    # If we use 'full' convolution and slice appropriately...

    # Actually the MATLAB code does it differently. Let me look at the dimensions again.
    # In MATLAB:
    # Forward: fx = imconv(f, fliplr(flipud(dxf)), 'full')[:, 2:end]
    # Adjoint: fx = imconv(scaled_g, dxf, 'full')[:, 1:end-1]

    # In MATLAB 1-indexing:
    # Forward slices as (:, 2:end) = columns 2 to end, which is W columns from W+1 total
    # Adjoint slices as (:, 1:end-1) = columns 1 to end-1, which is also W columns from W+1 total

    # But wait, in adjoint the input g has W columns (same as forward output)
    # So adjoint input is W cols, 'full' gives W+1 cols, slice to W cols

    # Let me re-verify with the full-size g
    print("\n--- With full-size g (matching forward output shape) ---")
    g_full = np.random.randn(H, W)

    # Current adjoint code: conv(g, [-1,1], 'full')[:, :-1]
    Dtg_current = imconv(g_full, dxf, 'full')[:, :-1]
    print(f"Current adjoint output shape: {Dtg_current.shape}")

    # Test adjoint property with full g
    inner1 = np.sum(Df * g_full)
    inner2 = np.sum(f * Dtg_current)
    print(f"<Df, g_full> = {inner1:.6f}")
    print(f"<f, D*g_current> = {inner2:.6f}")
    print(f"Relative error: {abs(inner1 - inner2) / abs(inner1):.2%}")

    # Hmm, still wrong. Let me trace through the exact matrices.


def analyze_convolution_adjoint():
    """Analyze exactly what the adjoint of conv+slice should be."""
    print("\n=== Analyzing convolution adjoint ===")

    # Forward: y = slice(conv(x, k, 'full'))
    # where x is (H, W), k is (1, 2), conv gives (H, W+1), slice gives (H, W)

    # Let's work out the matrix form for W=4
    # x = [x0, x1, x2, x3]
    # k = [1, -1] (flipped dxf)

    # With edge padding: x_padded = [x0, x0, x1, x2, x3, x3]
    # conv output (before slice):
    # y_full[0] = x0 - x0 = 0
    # y_full[1] = x0 - x1
    # y_full[2] = x1 - x2
    # y_full[3] = x2 - x3
    # y_full[4] = x3 - x3 = 0

    # After slice [:, 1:]:
    # y[0] = x0 - x1
    # y[1] = x1 - x2
    # y[2] = x2 - x3
    # y[3] = 0

    # Matrix form (y = M @ x):
    M = np.array([
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1],
        [0, 0, 0, 0],
    ])

    print(f"Forward matrix M:\n{M}")
    print(f"\nAdjoint M^T:\n{M.T}")

    # M^T:
    # [1  0  0  0]
    # [-1 1  0  0]
    # [0 -1  1  0]
    # [0  0 -1  0]

    # For g = [g0, g1, g2, g3]:
    # (M^T @ g)[0] = g0
    # (M^T @ g)[1] = -g0 + g1
    # (M^T @ g)[2] = -g1 + g2
    # (M^T @ g)[3] = -g2

    # Now, the current adjoint code does:
    # conv(g, [-1, 1], 'full')[:, :-1]

    # With edge padding: g_padded = [g0, g0, g1, g2, g3, g3]
    # conv with [-1, 1]:
    # out[0] = -g0 + g0 = 0
    # out[1] = -g0 + g1
    # out[2] = -g1 + g2
    # out[3] = -g2 + g3
    # out[4] = -g3 + g3 = 0

    # After [:, :-1]:
    # [0, -g0+g1, -g1+g2, -g2+g3]

    # This is NOT the same as M^T @ g = [g0, -g0+g1, -g1+g2, -g2]

    print("\n--- The boundary handling is the issue! ---")
    print("Forward operator uses edge-pad BEFORE conv")
    print("The adjoint needs different boundary treatment")

    # The correct adjoint:
    # result[0] = g[0]
    # result[1] = g[1] - g[0]
    # result[2] = g[2] - g[1]
    # result[3] = -g[2]

    # This is like:
    # 1. Convolve g with [1, -1] (NOT [-1, 1]) with ZERO padding
    # 2. Handle boundaries specially

    # Actually, let me think about this more carefully.
    # The adjoint of (pad + conv + slice) is (slice_adjoint + conv_adjoint + pad_adjoint)

    # - slice_adjoint of [:, 1:] is: prepend a zero column
    # - conv_adjoint of conv(x, [1,-1]) is conv(x, [-1,1]) (flip the kernel back)
    # - pad_adjoint of edge-pad is: sum the contributions that went to the padding

    # This is getting complicated. Let me just implement the correct adjoint directly.


def implement_correct_adjoint():
    """Implement the correct adjoint operator."""
    print("\n=== Implementing correct adjoint ===")

    def forward_x(f, lambd):
        """Forward x-derivative: conv(f, [1,-1], 'full')[:, 1:] * lambda"""
        dxf_flipped = np.array([[1, -1]])
        result = imconv(f, dxf_flipped, 'full')
        return lambd * result[:, 1:]

    def adjoint_x_wrong(g, lambd):
        """Current (wrong) adjoint."""
        dxf = np.array([[-1, 1]])
        result = imconv(lambd * g, dxf, 'full')
        return result[:, :-1]

    def adjoint_x_correct(g, lambd):
        """Correct adjoint based on matrix analysis."""
        H, W = g.shape
        result = np.zeros((H, W))

        # M^T @ g where M is the forward matrix
        # result[0] = g[0]
        # result[j] = g[j] - g[j-1] for 1 <= j < W-1
        # result[W-1] = -g[W-2]

        scaled_g = lambd * g
        result[:, 0] = scaled_g[:, 0]
        for j in range(1, W - 1):
            result[:, j] = scaled_g[:, j] - scaled_g[:, j - 1]
        result[:, -1] = -scaled_g[:, -2]

        return result

    # Test
    np.random.seed(42)
    H, W = 5, 7
    f = np.random.randn(H, W)
    g = np.random.randn(H, W)
    lambd = 0.5

    Df = forward_x(f, lambd)
    Dtg_wrong = adjoint_x_wrong(g, lambd)
    Dtg_correct = adjoint_x_correct(g, lambd)

    inner1 = np.sum(Df * g)
    inner2_wrong = np.sum(f * Dtg_wrong)
    inner2_correct = np.sum(f * Dtg_correct)

    print(f"<Df, g>          = {inner1:.6f}")
    print(f"<f, D*g_wrong>   = {inner2_wrong:.6f}  (error: {abs(inner1-inner2_wrong)/abs(inner1):.2%})")
    print(f"<f, D*g_correct> = {inner2_correct:.6f}  (error: {abs(inner1-inner2_correct)/abs(inner1):.2%})")

    # Now let's see if we can express the correct adjoint using convolution
    # result[:, 0] = g[:, 0]
    # result[:, j] = g[:, j] - g[:, j-1]  for 1 <= j < W-1
    # result[:, W-1] = -g[:, W-2]

    # The middle part (j=1 to W-2) is conv(g, [-1, 1]) with zero padding on left
    # The boundary terms are special

    # Can we use: conv(g_zero_padded, [-1,1], ...) and then fix boundaries?
    print("\n--- Trying to express correct adjoint using convolution ---")

    g_zpad = np.pad(g[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=0)
    # g_zpad = [0, g0, g1, ..., g[W-2]]

    dxf = np.array([[-1, 1]])
    # conv(g_zpad, [-1,1], 'same') at position j:
    # = -g_zpad[j] + g_zpad[j+1] with edge extension
    # We want: g[0] at j=0, g[j]-g[j-1] at j=1..W-2, -g[W-2] at j=W-1

    # Hmm, this doesn't quite work out with standard convolution modes.

    # Let me try a different approach: use conv with 'full' and slice appropriately
    # conv(g, [-1, 1], 'full') with ZERO padding would give:
    # out[0] = -0 + g[0] = g[0]
    # out[j] = -g[j-1] + g[j] for 1 <= j < W
    # out[W] = -g[W-1] + 0 = -g[W-1]

    # That's almost right! We get g[0], g[j]-g[j-1], -g[W-1]
    # But we want: g[0], g[j]-g[j-1] (for j<W-1), -g[W-2]

    # The issue is the last element. We want -g[W-2], not -g[W-1].

    # Actually wait, let me re-check the forward operator...


def recheck_forward_operator():
    """Re-check what the forward operator actually computes."""
    print("\n=== Re-checking forward operator ===")

    W = 4
    # x = [x0, x1, x2, x3]
    # With edge padding for 'full' convolution: [x0, x0, x1, x2, x3, x3]

    # Wait, the padding for 'full' mode should add (kernel_size - 1) = 1 element
    # on each side. Let me check my imconv implementation.

    x = np.array([[1.0, 2.0, 3.0, 4.0]])  # 1xW
    k = np.array([[1, -1]])  # flipped dxf

    result = imconv(x, k, 'full')
    print(f"Input: {x}")
    print(f"Kernel: {k}")
    print(f"imconv 'full' output: {result}")

    # What should this be?
    # For k = [1, -1], 'full' mode with replicate boundary:
    # The optimized path in imconv is:
    # f_left_pad = [1, 1, 2, 3, 4]
    # f_right_pad = [1, 2, 3, 4, 4]
    # result = k[0,1] * f_left_pad + k[0,0] * f_right_pad
    #        = -1 * [1,1,2,3,4] + 1 * [1,2,3,4,4]
    #        = [0, 1, 1, 1, 0]

    print(f"Expected: [0, 1, 1, 1, 0]")

    # After [:, 1:] slice:
    # [1, 1, 1, 0]
    # This represents: x1-x0, x2-x1, x3-x2, 0

    result_sliced = result[:, 1:]
    print(f"After [:, 1:] slice: {result_sliced}")

    # Now the adjoint of this operation...
    # The forward matrix M (4x4) where y = Mx:
    # y[0] = x1 - x0  -> row [[-1, 1, 0, 0]]
    # y[1] = x2 - x1  -> row [[0, -1, 1, 0]]
    # y[2] = x3 - x2  -> row [[0, 0, -1, 1]]
    # y[3] = 0        -> row [[0, 0, 0, 0]]

    M = np.array([
        [-1, 1, 0, 0],
        [0, -1, 1, 0],
        [0, 0, -1, 1],
        [0, 0, 0, 0],
    ])

    print(f"\nForward matrix M:\n{M}")
    print(f"M @ x = {M @ x.flatten()}")

    # Adjoint is M^T:
    print(f"\nAdjoint M^T:\n{M.T}")

    # M^T:
    # [[-1  0  0  0]
    #  [ 1 -1  0  0]
    #  [ 0  1 -1  0]
    #  [ 0  0  1  0]]

    # For g = [g0, g1, g2, g3]:
    # (M^T @ g)[0] = -g0
    # (M^T @ g)[1] = g0 - g1
    # (M^T @ g)[2] = g1 - g2
    # (M^T @ g)[3] = g2

    g = np.array([1.0, 2.0, 3.0, 4.0])
    expected_adjoint = M.T @ g
    print(f"\nFor g = {g}")
    print(f"Expected adjoint M^T @ g = {expected_adjoint}")

    # What does current code compute?
    # conv(g, [-1, 1], 'full')[:, :-1]
    g_2d = g.reshape(1, -1)
    adjoint_current = imconv(g_2d, np.array([[-1, 1]]), 'full')[:, :-1]
    print(f"Current adjoint code gives: {adjoint_current.flatten()}")

    # They're different!


if __name__ == '__main__':
    test_simple_derivative_adjoint()
    test_derivative_adjoint_explicit()
    test_correct_adjoint()
    analyze_convolution_adjoint()
    implement_correct_adjoint()
    recheck_forward_operator()
