"""
Primal-Dual Cross-Channel Deconvolution.

Python port of MATLAB code from:
F. Heide, M. Rouf, M. Hullin, B. Labitzke, W. Heidrich, A. Kolb.
"High-Quality Computational Imaging Through Simple Lenses."
ACM ToG 2013
"""

import numpy as np
from scipy.fft import fft2, ifft2

from .utils import psf2otf, edgetaper, imconv
from .operator_norm import compute_operator_norm
from .tracing import trace


def pd_joint_deconv(channels, lambda_params, max_it=200, tol=1e-4, verbose='brief'):
    """
    Primal-dual cross-channel deconvolution.

    Parameters
    ----------
    channels : list of dict
        List of channel data, each dict contains:
        - 'image': 2D array, the blurred input image for this channel
        - 'kernel': 2D array, the blur kernel (PSF) for this channel
    lambda_params : ndarray
        Parameter matrix where each row specifies optimization for one channel:
        [ch_idx, lambda_residual, lambda_tv, lambda_black, lambda_cross_ch..., n_detail_layers]
        - ch_idx: 1-based channel index to optimize
        - lambda_residual: weight for data fidelity term
        - lambda_tv: weight for total variation regularization
        - lambda_black: weight for black level prior (usually 0)
        - lambda_cross_ch: weights for cross-channel terms (one per channel)
        - n_detail_layers: number of residual detail layers (0 for single pass)
    max_it : int
        Maximum iterations per optimization
    tol : float
        Convergence tolerance
    verbose : str
        'none', 'brief', or 'all'

    Returns
    -------
    result : list of dict
        Deconvolved channels in same format as input
    """
    n_channels = len(channels)

    if n_channels < 1:
        raise ValueError("No valid channels found for deconvolution.")

    # Initialize output with input
    db_chs = [
        {'image': ch['image'].copy(), 'kernel': ch['kernel'].copy()}
        for ch in channels
    ]

    # Process each row of lambda_params (startup iterations)
    for s in range(lambda_params.shape[0]):
        if verbose in ('brief', 'all'):
            print(f"\n### Startup iteration {s + 1} ###")

        # Parse parameters for this iteration
        row = lambda_params[s]
        ch_opt = int(row[0]) - 1  # Convert to 0-based index
        w_res = row[1]
        w_tv = row[2]

        # Cross-channel weights (variable length)
        n_cross = n_channels
        w_cross = row[3:3 + n_cross]
        res_iter = int(row[-1])

        # Validate
        if res_iter > 0 and np.any(w_cross != 0):
            raise ValueError("Residual iteration with cross channel terms is not supported")

        # Get kernel size for padding
        ks = channels[ch_opt]['kernel'].shape[0]

        # Pad and edgetaper all channels
        with trace("pad_and_edgetaper"):
            channels_padded = []
            db_chs_padded = []
            for ch_idx in range(n_channels):
                # Pad with replicate boundary
                with trace("pad"):
                    img_pad = np.pad(channels[ch_idx]['image'], ks, mode='edge')
                    db_pad = np.pad(db_chs[ch_idx]['image'], ks, mode='edge')

                # Edgetaper
                with trace("edgetaper"):
                    kernel = channels[ch_opt]['kernel']
                    for _ in range(4):
                        img_pad = edgetaper(img_pad, kernel)
                        db_pad = edgetaper(db_pad, kernel)

                # Add offset (MATLAB code adds 1.0)
                img_pad = img_pad + 1.0
                db_pad = db_pad + 1.0

                channels_padded.append({
                    'image': img_pad,
                    'kernel': channels[ch_idx]['kernel']
                })
                db_chs_padded.append({
                    'image': db_pad,
                    'kernel': db_chs[ch_idx]['kernel']
                })

        # Run residual PD deconvolution
        with trace("residual_pd_deconv"):
            result = _residual_pd_deconv(
                channels_padded, db_chs_padded, ch_opt,
                w_res, w_tv, w_cross,
                res_iter, tol, max_it, verbose
            )

        # Remove padding and offset
        for ch_idx in range(n_channels):
            channels[ch_idx]['image'] = channels_padded[ch_idx]['image'][ks:-ks, ks:-ks] - 1.0
            db_chs[ch_idx]['image'] = db_chs_padded[ch_idx]['image'][ks:-ks, ks:-ks] - 1.0

        db_chs[ch_opt]['image'] = result[ks:-ks, ks:-ks] - 1.0

    return db_chs


def _residual_pd_deconv(channels, db_chs, ch, w_res, w_tv, w_cross,
                        res_iter, tol, max_it, verbose):
    """
    Residual deconvolution with optional detail layers.
    """
    detail_tol = tol

    for d in range(res_iter + 1):
        if d == 0:
            # First iteration: use original blurred image
            channels_res = [
                {'image': c['image'].copy(), 'kernel': c['kernel']}
                for c in channels
            ]
            x_0 = db_chs[ch]['image'].copy()
            tol_offset = np.zeros_like(db_chs[ch]['image'])
        else:
            # Residual iterations: compute residual blur
            channels_res = [
                {'image': c['image'].copy(), 'kernel': c['kernel']}
                for c in channels
            ]

            # Residual = observed - (current_estimate * kernel)
            blurred_estimate = imconv(db_chs[ch]['image'], db_chs[ch]['kernel'], 'same')
            channels_res[ch]['image'] = channels[ch]['image'] - blurred_estimate + 1.0

            x_0 = channels_res[ch]['image'].copy()

            # Increase regularization for detail layers
            w_res = w_res * 3.0

            tol_offset = db_chs[ch]['image'] - 1.0

        # Run primal-dual optimization
        x = _pd_channel_deconv(
            channels_res, ch, x_0, db_chs,
            w_res, w_cross, w_tv,
            max_it, detail_tol, tol_offset, verbose
        )

        # Threshold negative values
        x = np.maximum(x, 1.0)

        # Update result
        if d == 0:
            db_chs[ch]['image'] = x
        else:
            db_chs[ch]['image'] = db_chs[ch]['image'] + (x - 1.0)
            db_chs[ch]['image'] = np.maximum(db_chs[ch]['image'], 0)

    return db_chs[ch]['image']


def _pd_channel_deconv(channels, ch, x_0, db_chs,
                       lambda_residual, lambda_cross_ch, lambda_tv,
                       max_it, tol, tol_offset, verbose):
    """
    Primal-dual deconvolution for a single channel.
    """
    # Prepare FFT-based solver components
    with trace("fft_setup"):
        sizey = channels[ch]['image'].shape
        otfk = psf2otf(channels[ch]['kernel'], sizey)
        Nomin1 = np.conj(otfk) * fft2(channels[ch]['image'])
        Denom1 = np.abs(otfk) ** 2

    # Compute operator norm for step size selection
    def A(x):
        return _Kmult(x, ch, db_chs, lambda_cross_ch, lambda_tv)

    def AS(x):
        return _KSmult(x, ch, db_chs, lambda_cross_ch, lambda_tv)

    with trace("operator_norm"):
        L = compute_operator_norm(A, AS, sizey)

    # Primal-dual step sizes
    sigma = 1.0
    tau = 0.7 / (sigma * L ** 2)
    theta = 1.0

    # Initialize
    f = x_0.copy() if x_0 is not None else channels[ch]['image'].copy()
    g = A(f)
    f1 = f.copy()

    # Primal-dual iterations
    with trace("pd_iterations"):
        for i in range(max_it):
            f_old = f.copy()

            # Dual update: g = prox_sigma_F*(g + sigma * K * f1)
            with trace("forward_op"):
                Af1 = A(f1)
            with trace("prox_dual"):
                g = _prox_fs(g + sigma * Af1, sigma)

            # Primal update: f = prox_tau_G(f - tau * K* * g)
            with trace("adjoint_op"):
                ASg = AS(g)
            with trace("fft_solve"):
                f = _solve_fft(Nomin1, Denom1, tau, lambda_residual, f - tau * ASg)

            # Over-relaxation
            f1 = f + theta * (f - f_old)

            # Check convergence
            diff = (f + tol_offset) - (f_old + tol_offset)
            f_comp = f + tol_offset
            rel_diff = np.linalg.norm(diff.ravel()) / (np.linalg.norm(f_comp.ravel()) + 1e-10)

            if verbose in ('brief', 'all'):
                print(f"Ch: {ch + 1}, iter {i + 1}, diff {rel_diff:.5g}")

            if rel_diff < tol:
                break

    return f1


def _prox_fs(u, sigma):
    """
    Proximal operator for the dual variable (L1 norm).

    prox_F*(u) = u / max(1, |u|)
    """
    amplitude = np.sqrt(np.sum(u ** 2, axis=2, keepdims=True))
    return u / np.maximum(1.0, amplitude)


def _solve_fft(Nomin1, Denom1, tau, lambda_val, f):
    """
    FFT-based solver for the primal update.

    Solves: (tau * 2 * lambda * K'K + I) x = tau * 2 * lambda * K'B + f
    """
    x = (tau * 2 * lambda_val * Nomin1 + fft2(f)) / (tau * 2 * lambda_val * Denom1 + 1)
    return np.real(ifft2(x))


def _Kmult(f, ch, db_chs, lambda_cross_ch, lambda_tv):
    """
    Forward operator K: computes TV and cross-channel gradient terms.

    Returns a 3D array where each slice is a different gradient term.
    """
    # Derivative filters
    dxf = np.array([[-1, 1]])
    dyf = np.array([[-1], [1]])
    dxxf = np.array([[-1, 2, -1]])
    dyyf = np.array([[-1], [2], [-1]])
    dxyf = np.array([[-1, 1], [1, -1]])

    results = []

    # TV terms
    if lambda_tv > 1e-10:
        # First derivatives
        fx = imconv(f, dxf[:, ::-1][::-1, :], 'full')
        fx = (lambda_tv * 0.5) * fx[:, 1:]

        fy = imconv(f, dyf[::-1, :], 'full')
        fy = (lambda_tv * 0.5) * fy[1:, :]

        # Second derivatives (with smaller weight)
        sd_w = 0.15
        fxx = imconv(f, dxxf[:, ::-1], 'full')
        fxx = (lambda_tv * sd_w) * fxx[:, 2:]

        fyy = imconv(f, dyyf[::-1, :], 'full')
        fyy = (lambda_tv * sd_w) * fyy[2:, :]

        fxy = imconv(f, dxyf[::-1, ::-1], 'full')
        fxy = (lambda_tv * sd_w) * fxy[1:, 1:]

        results.extend([fx, fy, fxx, fyy, fxy])

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
        # Return zeros if no regularization
        return np.zeros((*f.shape, 1))

    return np.stack(results, axis=2)


def _KSmult(f, ch, db_chs, lambda_cross_ch, lambda_tv):
    """
    Adjoint operator K*: transpose of the forward operator.
    """
    # Derivative filters
    dxf = np.array([[-1, 1]])
    dyf = np.array([[-1], [1]])
    dxxf = np.array([[-1, 2, -1]])
    dyyf = np.array([[-1], [2], [-1]])
    dxyf = np.array([[-1, 1], [1, -1]])

    result = np.zeros((f.shape[0], f.shape[1]))

    i = 0  # Index into the stacked gradient terms

    # TV terms
    if lambda_tv > 1e-10:
        # First derivatives (adjoint)
        fx = imconv((lambda_tv * 0.5) * f[:, :, i], dxf, 'full')
        result += fx[:, :-1]
        i += 1

        fy = imconv((lambda_tv * 0.5) * f[:, :, i], dyf, 'full')
        result += fy[:-1, :]
        i += 1

        # Second derivatives
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

    # Cross-channel terms (adjoint)
    if np.sum(np.abs(lambda_cross_ch)) > 1e-10:
        for adj_ch in range(len(db_chs)):
            if adj_ch == ch or db_chs[adj_ch]['kernel'] is None:
                continue

            lam = lambda_cross_ch[adj_ch] if adj_ch < len(lambda_cross_ch) else 0
            if abs(lam) < 1e-10:
                continue

            adj_img = db_chs[adj_ch]['image']

            # X direction
            f_i = (lam * 0.5) * f[:, :, i]
            diag_term = imconv(adj_img, dxf[:, ::-1][::-1, :], 'full')
            diag_term = diag_term[:, 1:] * f_i
            conv_term = imconv(adj_img * f_i, dxf, 'full')
            Sxtf = conv_term[:, :-1] - diag_term
            result += Sxtf
            i += 1

            # Y direction
            f_i = (lam * 0.5) * f[:, :, i]
            diag_term = imconv(adj_img, dyf[::-1, :], 'full')
            diag_term = diag_term[1:, :] * f_i
            conv_term = imconv(adj_img * f_i, dyf, 'full')
            Sytf = conv_term[:-1, :] - diag_term
            result += Sytf
            i += 1

    return result
