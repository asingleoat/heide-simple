"""
Utility functions for image deconvolution.
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft2


def psf2otf(psf, shape):
    """
    Convert point spread function to optical transfer function.

    Equivalent to MATLAB's psf2otf: zero-pads the PSF to the given shape,
    circularly shifts it so the center is at (0,0), then computes FFT.

    Parameters
    ----------
    psf : ndarray
        Point spread function (kernel)
    shape : tuple
        Output shape (height, width)

    Returns
    -------
    otf : ndarray
        Optical transfer function (complex)
    """
    psf = np.asarray(psf)

    # Pad PSF to output shape
    padded = np.zeros(shape, dtype=psf.dtype)
    psf_shape = psf.shape
    padded[:psf_shape[0], :psf_shape[1]] = psf

    # Circularly shift so center of PSF is at (0,0)
    shift = [-(s // 2) for s in psf_shape]
    padded = np.roll(padded, shift, axis=(0, 1))

    return fft2(padded)


def _create_taper_weights(psf, shape):
    """Create edge tapering weights from a PSF."""
    # Sum PSF along each axis and compute autocorrelation
    psf_proj_h = psf.sum(axis=1)
    psf_proj_w = psf.sum(axis=0)

    # Autocorrelation via convolution with flipped version
    autocorr_h = np.convolve(psf_proj_h, psf_proj_h[::-1], mode='full')
    autocorr_w = np.convolve(psf_proj_w, psf_proj_w[::-1], mode='full')

    # Normalize to [0, 1]
    autocorr_h = autocorr_h / autocorr_h.max()
    autocorr_w = autocorr_w / autocorr_w.max()

    # Create 2D weighting function
    h, w = shape[:2]

    # Build vertical weights
    weight_h = np.ones(h)
    taper_len_h = len(autocorr_h) // 2
    if taper_len_h > 0 and taper_len_h < h // 2:
        # Top edge
        weight_h[:taper_len_h] = autocorr_h[taper_len_h:2*taper_len_h]
        # Bottom edge
        weight_h[-taper_len_h:] = autocorr_h[taper_len_h:2*taper_len_h][::-1]

    # Build horizontal weights
    weight_w = np.ones(w)
    taper_len_w = len(autocorr_w) // 2
    if taper_len_w > 0 and taper_len_w < w // 2:
        # Left edge
        weight_w[:taper_len_w] = autocorr_w[taper_len_w:2*taper_len_w]
        # Right edge
        weight_w[-taper_len_w:] = autocorr_w[taper_len_w:2*taper_len_w][::-1]

    # Combine into 2D weight (outer product)
    return np.outer(weight_h, weight_w)


def edgetaper(img, psf, n_iterations=1):
    """
    Taper image edges to reduce boundary artifacts in deconvolution.

    This is a simplified version of MATLAB's edgetaper that blends
    the image edges with a blurred version to reduce ringing.

    Parameters
    ----------
    img : ndarray
        Input image (2D)
    psf : ndarray
        Point spread function used for blurring
    n_iterations : int
        Number of tapering iterations

    Returns
    -------
    tapered : ndarray
        Edge-tapered image
    """
    img = np.asarray(img, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)

    # Normalize PSF
    psf = psf / psf.sum()

    # Create tapering weights
    weight = _create_taper_weights(psf, img.shape)

    result = img.copy()
    for _ in range(n_iterations):
        # Blur the current result
        blurred = ndimage.convolve(result, psf, mode='reflect')
        # Blend: result = weight * result + (1 - weight) * blurred
        result = weight * result + (1 - weight) * blurred

    return result


def _imconv_horizontal_optimized(f, k):
    """Optimized convolution for horizontal 2-element kernel."""
    # MATLAB: F_filt = K(1,2)*F(:,[1 1:end],:) + K(1,1)*F(:,[1:end end],:)
    # K(1,2) is second element (Python k[0,1]), multiplies left-padded array
    # K(1,1) is first element (Python k[0,0]), multiplies right-padded array
    f_left_pad = np.pad(f, ((0, 0), (1, 0)), mode='edge')   # prepend first col
    f_right_pad = np.pad(f, ((0, 0), (0, 1)), mode='edge')  # append last col
    return k[0, 1] * f_left_pad + k[0, 0] * f_right_pad


def _imconv_vertical_optimized(f, k):
    """Optimized convolution for vertical 2-element kernel."""
    # MATLAB: F_filt = K(2,1)*F([1 1:end],:,:) + K(1,1)*F([1:end end],:,:)
    f_top_pad = np.pad(f, ((1, 0), (0, 0)), mode='edge')    # prepend first row
    f_bot_pad = np.pad(f, ((0, 1), (0, 0)), mode='edge')    # append last row
    return k[1, 0] * f_top_pad + k[0, 0] * f_bot_pad


def imconv(f, k, mode='same'):
    """
    Convolution with replicate boundary conditions.

    Mimics MATLAB's imfilter(F, K, mode, 'conv', 'replicate').

    Parameters
    ----------
    f : ndarray
        Input image
    k : ndarray
        Convolution kernel
    mode : str
        'same' or 'full'

    Returns
    -------
    result : ndarray
        Convolved image
    """
    f = np.asarray(f, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    # Optimized paths for small gradient kernels
    if k.shape == (1, 2) and mode == 'full':
        return _imconv_horizontal_optimized(f, k)
    elif k.shape == (2, 1) and mode == 'full':
        return _imconv_vertical_optimized(f, k)
    else:
        # General case using scipy.signal.convolve2d
        from scipy.signal import convolve2d
        kh, kw = k.shape
        if mode == 'full':
            # Pad for 'full' convolution with replicate boundary
            f_padded = np.pad(f, ((kh - 1, kh - 1), (kw - 1, kw - 1)), mode='edge')
            return convolve2d(f_padded, k, mode='valid')
        else: # 'same' mode
            pad_h = kh // 2
            pad_w = kw // 2
            f_padded = np.pad(f, ((pad_h, kh - 1 - pad_h), (pad_w, kw - 1 - pad_w)), mode='edge')
            return convolve2d(f_padded, k, mode='valid')


def img_to_norm_grayscale(img):
    """
    Convert image to normalized grayscale in range [0, 1].

    Parameters
    ----------
    img : ndarray
        Input image (can be color or grayscale, any dtype)

    Returns
    -------
    gray : ndarray
        Grayscale image normalized to [0, 1]
    """
    img = np.asarray(img)

    # Convert to grayscale if color
    if img.ndim == 3 and img.shape[2] >= 3:
        # RGB to grayscale using standard weights
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.squeeze()

    # Convert to float
    gray = gray.astype(np.float64)

    # Normalize based on dtype
    if np.issubdtype(img.dtype, np.integer):
        # Integer type - normalize by dtype range
        info = np.iinfo(img.dtype)
        gray = (gray - info.min) / (info.max - info.min)
    else:
        # Float type - normalize by actual min/max
        vmin, vmax = gray.min(), gray.max()
        if vmax > vmin:
            gray = (gray - vmin) / (vmax - vmin)
        else:
            gray = np.zeros_like(gray)

    return gray
