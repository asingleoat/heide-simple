"""
PSF Estimation using Primal-Dual Optimization.

Based on Section 6 of:
F. Heide, M. Rouf, M. Hullin, B. Labitzke, W. Heidrich, A. Kolb.
"High-Quality Computational Imaging Through Simple Lenses."
ACM ToG 2013

The PSF estimation problem is posed as:
    b_opt = argmin_b ||Ib - s·j||²₂ + λ||∇b||₁ + μ||1ᵀb - 1||²₂

Where:
    - I: sharp pinhole-aperture image (acts as convolution operator)
    - j: blurred wide-aperture image
    - b: PSF to estimate
    - s: exposure scale factor
    - λ: TV regularization weight
    - μ: energy conservation weight (PSF sums to 1)
"""

import numpy as np
from scipy.fft import fft2, ifft2

from .utils import psf2otf
from .operator_norm import compute_operator_norm


def estimate_psf(sharp_patch, blurred_patch, psf_size, lambda_tv=0.001,
                 mu_sum=50.0, max_it=500, tol=1e-5, verbose='brief'):
    """
    Estimate PSF from sharp and blurred image patches.

    The estimation solves (Eq. 18 from paper):
        b_opt = argmin_b ||Ib - s·j||² + λ||∇b||₁ + μ||1ᵀb - 1||²

    where I is the sharp image (as convolution operator), j is the blurred
    image, b is the PSF, and s is an exposure scale factor.

    Uses the primal-dual algorithm from Section 4 with operators (Eq. 19):
        K(x) = ∇x
        F(y) = ||y||₁
        G(x) = (1/λ)||Ib - s·j||² + (μ/λ)||1ᵀb - 1||²

    Parameters
    ----------
    sharp_patch : ndarray
        Sharp reference image (captured with pinhole aperture or known sharp)
    blurred_patch : ndarray
        Blurred image (captured with wide aperture, same scene)
    psf_size : int or tuple
        Size of PSF to estimate. If int, assumes square PSF.
    lambda_tv : float
        TV regularization weight on PSF gradients (default: 0.001)
    mu_sum : float
        Weight for sum-to-one constraint (default: 50.0)
    max_it : int
        Maximum iterations (default: 500)
    tol : float
        Convergence tolerance (default: 1e-5)
    verbose : str
        'none', 'brief', or 'all'

    Returns
    -------
    psf : ndarray
        Estimated PSF, normalized to sum to 1
    """
    sharp_patch = np.asarray(sharp_patch, dtype=np.float64)
    blurred_patch = np.asarray(blurred_patch, dtype=np.float64)

    if isinstance(psf_size, int):
        psf_size = (psf_size, psf_size)

    # Compute exposure scale factor: s = sum(I) / sum(J) (Eq. 18)
    s = sharp_patch.sum() / (blurred_patch.sum() + 1e-10)

    # Work in the image domain for FFT
    # The model is: blurred = sharp ⊗ psf
    # In Fourier: F(blurred) = F(sharp) · F(psf)
    work_shape = blurred_patch.shape

    # FFT of sharp image (I in paper) - NOT psf2otf since it's an image
    fft_I = fft2(sharp_patch)
    fft_I_conj = np.conj(fft_I)
    fft_I_sq = np.abs(fft_I) ** 2

    # FFT of blurred image (j in paper)
    fft_j = fft2(blurred_patch)

    # For sum constraint: O is "convolution matrix of ones"
    # F(O) for a ones matrix of size psf_size is N at DC
    N_psf = np.prod(psf_size)

    # Initialize PSF with delta at center
    b = np.zeros(psf_size, dtype=np.float64)
    b[psf_size[0] // 2, psf_size[1] // 2] = 1.0

    # Compute operator norm for gradient K = ∇
    def K(x):
        return _gradient_forward(x)

    def KS(x):
        return _gradient_adjoint(x)

    L = compute_operator_norm(K, KS, psf_size)

    # Primal-dual parameters (Section 5.4: σ=10, τ=0.9/(σL²), θ=1)
    sigma = 10.0
    tau = 0.9 / (sigma * L ** 2)
    theta = 1.0

    # Initialize dual variable
    g = K(b)
    b1 = b.copy()

    # Main primal-dual loop (Algorithm 1)
    for i in range(max_it):
        b_old = b.copy()

        # Dual update: y^{n+1} = prox_{σF*}(y^n + σKx̄^n) (Eq. 20a)
        # For F(y) = ||y||₁, prox_{σF*}(ỹ) = ỹ / max(1, |ỹ|)
        g = _prox_l1_dual(g + sigma * K(b1))

        # Primal update: x^{n+1} = prox_{τG}(x^n - τK*y^{n+1}) (Eq. 21)
        rhs = b - tau * KS(g)
        b = _solve_psf_eq21(fft_I, fft_I_conj, fft_I_sq, fft_j, s,
                            tau, lambda_tv, mu_sum, N_psf, rhs,
                            psf_size, work_shape)

        # Over-relaxation: x̄^{n+1} = x^{n+1} + θ(x^{n+1} - x^n)
        b1 = b + theta * (b - b_old)

        # Check convergence
        diff = np.linalg.norm(b - b_old) / (np.linalg.norm(b) + 1e-10)

        if verbose == 'all' or (verbose == 'brief' and i % 50 == 0):
            print(f"PSF estimation iter {i + 1}, diff {diff:.6g}")

        if diff < tol:
            if verbose in ('brief', 'all'):
                print(f"PSF estimation converged at iteration {i + 1}")
            break

    # Post-process: enforce non-negativity and normalize
    b = np.maximum(b, 0)
    b = b / (b.sum() + 1e-10)

    return b


def _gradient_forward(f):
    """
    Forward gradient operator K(x) = ∇x.
    Returns stacked horizontal and vertical gradients.
    """
    # Horizontal gradient (forward difference)
    dx = np.diff(f, axis=1, append=f[:, -1:])
    # Vertical gradient (forward difference)
    dy = np.diff(f, axis=0, append=f[-1:, :])

    return np.stack([dx, dy], axis=2)


def _gradient_adjoint(g):
    """
    Adjoint of gradient operator K*(y) = -div(y).
    """
    dx = g[:, :, 0]
    dy = g[:, :, 1]

    # Adjoint of forward difference is backward difference with negation
    # div_x: subtract shifted version
    div_x = np.zeros_like(dx)
    div_x[:, 0] = dx[:, 0]
    div_x[:, 1:-1] = dx[:, 1:-1] - dx[:, :-2]
    div_x[:, -1] = -dx[:, -2]

    # div_y: subtract shifted version
    div_y = np.zeros_like(dy)
    div_y[0, :] = dy[0, :]
    div_y[1:-1, :] = dy[1:-1, :] - dy[:-2, :]
    div_y[-1, :] = -dy[-2, :]

    return -(div_x + div_y)


def _prox_l1_dual(u):
    """
    Proximal operator for dual of L1 norm (Eq. 20a from paper).

    prox_{σF*}(ỹ) = ỹ_i / max(1, |ỹ_i|)

    For isotropic 2D gradient, we use the L2 norm of the gradient vector.
    """
    # For 2D gradient, compute magnitude at each pixel
    magnitude = np.sqrt(np.sum(u ** 2, axis=2, keepdims=True))
    return u / np.maximum(1.0, magnitude)


def _solve_psf_eq21(fft_I, fft_I_conj, fft_I_sq, fft_j, s,
                    tau, lambda_tv, mu_sum, N_psf, rhs, psf_size, work_shape):
    """
    Solve proximal operator for G using Eq. 21 from paper.

    From paper Eq. 21:
        u_opt = F⁻¹((τs·F(I)*·F(j) + (λ/2)·F(ũ) + τμ·F(1)) /
                    (τ|F(I)|² + λ/2 + τμ·F(O)))

    Where:
        - I is the sharp image
        - j is the blurred image
        - s is exposure scale factor
        - λ is the TV weight (appears as λ/2 from G definition)
        - μ is sum constraint weight
        - O is convolution matrix of ones (F(O) = N at DC)
        - ũ is the input (rhs)
    """
    # Pad rhs to work_shape and use psf2otf-style centering
    rhs_padded = np.zeros(work_shape)
    rhs_padded[:psf_size[0], :psf_size[1]] = rhs
    # Circular shift so PSF center is at (0,0)
    shift = [-(sz // 2) for sz in psf_size]
    rhs_padded = np.roll(rhs_padded, shift, axis=(0, 1))

    # FFT of rhs (ũ in the paper)
    fft_rhs = fft2(rhs_padded)

    # Build numerator: τs·F(I)*·F(j) + (λ/2)·F(ũ) + τμ·F(1)
    numer = tau * s * fft_I_conj * fft_j + (lambda_tv / 2) * fft_rhs

    # Build denominator: τ|F(I)|² + λ/2 + τμ·F(O)
    denom = tau * fft_I_sq + (lambda_tv / 2)

    # Add sum constraint (affects DC component)
    # The sum constraint ||sum(b)-1||² adds terms at DC
    numer[0, 0] += tau * mu_sum * N_psf  # pulls toward sum=1
    denom[0, 0] += tau * mu_sum * N_psf

    # Solve
    result_fft = numer / (denom + 1e-10)
    result = np.real(ifft2(result_fft))

    # Shift back to standard PSF layout and extract
    result = np.roll(result, [-sz for sz in shift], axis=(0, 1))
    b = result[:psf_size[0], :psf_size[1]]

    return b


def estimate_psf_from_patches(sharp_patches, blurred_patches, psf_size,
                              lambda_tv=0.001, mu_sum=50.0, max_it=500,
                              tol=1e-5, verbose='brief'):
    """
    Estimate PSF from multiple sharp/blurred patch pairs.

    Averages estimates from multiple patches for more robust results.

    Parameters
    ----------
    sharp_patches : list of ndarray
        List of sharp reference patches
    blurred_patches : list of ndarray
        List of corresponding blurred patches
    psf_size : int or tuple
        Size of PSF to estimate
    lambda_tv : float
        TV regularization weight
    mu_sum : float
        Sum-to-one constraint weight
    max_it : int
        Maximum iterations per patch
    tol : float
        Convergence tolerance
    verbose : str
        Verbosity level

    Returns
    -------
    psf : ndarray
        Averaged estimated PSF
    """
    if len(sharp_patches) != len(blurred_patches):
        raise ValueError("Number of sharp and blurred patches must match")

    psfs = []
    for i, (sharp, blurred) in enumerate(zip(sharp_patches, blurred_patches)):
        if verbose in ('brief', 'all'):
            print(f"\nEstimating PSF from patch {i + 1}/{len(sharp_patches)}")

        psf = estimate_psf(sharp, blurred, psf_size,
                          lambda_tv=lambda_tv, mu_sum=mu_sum,
                          max_it=max_it, tol=tol, verbose=verbose)
        psfs.append(psf)

    # Average PSFs
    avg_psf = np.mean(psfs, axis=0)
    avg_psf = np.maximum(avg_psf, 0)
    avg_psf = avg_psf / avg_psf.sum()

    return avg_psf


def create_calibration_pattern(patch_size=128, n_patches_h=4, n_patches_w=4,
                               border_width=20, seed=None):
    """
    Create a white noise calibration pattern for PSF estimation.

    The pattern consists of a grid of white noise patches separated by
    white borders. The borders help suppress boundary effects during
    PSF estimation.

    Parameters
    ----------
    patch_size : int
        Size of each noise patch (square)
    n_patches_h : int
        Number of patches vertically
    n_patches_w : int
        Number of patches horizontally
    border_width : int
        Width of white border around each patch (should be >= blur radius)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pattern : ndarray
        Calibration pattern image
    patch_coords : list of tuple
        List of (y_start, x_start, y_end, x_end) for each noise patch
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate total size
    cell_size = patch_size + 2 * border_width
    total_h = n_patches_h * cell_size
    total_w = n_patches_w * cell_size

    # Create white background
    pattern = np.ones((total_h, total_w), dtype=np.float64)

    # Track patch coordinates
    patch_coords = []

    # Fill in noise patches
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y_start = i * cell_size + border_width
            x_start = j * cell_size + border_width
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            # Generate white noise patch
            noise = np.random.rand(patch_size, patch_size)
            pattern[y_start:y_end, x_start:x_end] = noise

            patch_coords.append((y_start, x_start, y_end, x_end))

    return pattern, patch_coords


def extract_patches_from_images(sharp_image, blurred_image, patch_coords,
                                crop_border=0):
    """
    Extract corresponding patches from sharp and blurred images.

    Parameters
    ----------
    sharp_image : ndarray
        Sharp calibration image
    blurred_image : ndarray
        Blurred calibration image
    patch_coords : list of tuple
        Patch coordinates from create_calibration_pattern
    crop_border : int
        Additional border to crop from each patch (to avoid edge effects)

    Returns
    -------
    sharp_patches : list of ndarray
        Extracted sharp patches
    blurred_patches : list of ndarray
        Extracted blurred patches
    """
    sharp_patches = []
    blurred_patches = []

    for y_start, x_start, y_end, x_end in patch_coords:
        # Apply additional cropping
        y_start += crop_border
        x_start += crop_border
        y_end -= crop_border
        x_end -= crop_border

        if y_end <= y_start or x_end <= x_start:
            continue

        sharp_patches.append(sharp_image[y_start:y_end, x_start:x_end].copy())
        blurred_patches.append(blurred_image[y_start:y_end, x_start:x_end].copy())

    return sharp_patches, blurred_patches


def smooth_psf_spatially(psfs, positions, sigma=1.0):
    """
    Spatially smooth a grid of PSFs using weighted averaging.

    As recommended in Section 6.3 of the paper, neighboring PSFs are
    averaged to reduce noise while preserving spatially varying features.

    Parameters
    ----------
    psfs : list of ndarray
        List of estimated PSFs
    positions : list of tuple
        (row, col) position of each PSF in the grid
    sigma : float
        Gaussian smoothing sigma in grid units

    Returns
    -------
    smoothed_psfs : list of ndarray
        Spatially smoothed PSFs
    """
    if len(psfs) != len(positions):
        raise ValueError("Number of PSFs must match number of positions")

    positions = np.array(positions)
    smoothed_psfs = []

    for i, (psf, pos) in enumerate(zip(psfs, positions)):
        # Compute distances to all other PSFs
        distances = np.sqrt(np.sum((positions - pos) ** 2, axis=1))

        # Gaussian weights
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
        weights = weights / weights.sum()

        # Weighted average
        avg_psf = np.zeros_like(psf)
        for j, (other_psf, w) in enumerate(zip(psfs, weights)):
            avg_psf += w * other_psf

        # Normalize
        avg_psf = np.maximum(avg_psf, 0)
        avg_psf = avg_psf / (avg_psf.sum() + 1e-10)

        smoothed_psfs.append(avg_psf)

    return smoothed_psfs
