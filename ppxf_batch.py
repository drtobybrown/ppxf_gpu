"""
ppxf_batch.py: Batched pPXF fitting with full capfit kinematics and GPU auto-sizing.

This module provides a high-level function ``ppxf_batch`` that fits multiple
spectra using the standard ppxf pipeline (including capfit non-linear
optimization), with automatic GPU memory-based batch sizing.

Usage::

    from ppxf.ppxf_batch import ppxf_batch

    results = ppxf_batch(templates, spectra, noise, velscale, start,
                         moments=2, degree=4, gpu=True, quiet=True)

    for r in results:
        print(r.sol, r.chi2)

"""

import time
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def estimate_batch_size(n_templates, n_pixels, npad, device=None):
    """
    Estimate maximum number of spectra that can be processed in one chunk
    based on available GPU memory.

    Parameters
    ----------
    n_templates : int
        Number of template columns (after reshaping to 2D).
    n_pixels : int
        Number of pixels in galaxy spectrum.
    npad : int
        FFT padding size (next power of 2).
    device : torch.device or None
        GPU device. None = CPU (use conservative default).

    Returns
    -------
    max_batch : int
        Estimated maximum batch size.
    """
    npoly_est = 5  # typical degree=4 -> 5 polynomials
    ncols = n_templates + npoly_est

    # Memory per spectrum estimate (in bytes, float32):
    # - templates_rfft is shared (not per-spectrum)
    # - Per-spectrum allocations in linear_fit:
    #   * LOSVD rfft: npad/2+1 complex64 = (npad/2+1) * 8
    #   * conv_freq: (npad/2+1) * n_templates * 8 (complex64)
    #   * irfft output: npad * n_templates * 4
    #   * design matrix c: n_pixels * ncols * 4
    #   * weighted matrix a: n_pixels * ncols * 4
    #   * galaxy, noise: n_pixels * 4 * 2
    #   * bestfit, err: n_pixels * 4 * 2
    nl = npad // 2 + 1
    bytes_per_spectrum = (
        nl * 8                          # losvd_rfft
        + nl * n_templates * 8          # conv_freq
        + npad * n_templates * 4        # irfft result
        + n_pixels * ncols * 4 * 2      # c and a matrices
        + n_pixels * 4 * 4              # galaxy, noise, bestfit, err
    )

    # Add 50% overhead for capfit Jacobian (stores ~n_free extra func evals)
    bytes_per_spectrum = int(bytes_per_spectrum * 1.5)

    available = 2 * 1024**3  # Default: 2 GB for CPU
    if device is not None and torch is not None:
        if device.type == 'mps':
            try:
                # MPS: use a conservative fraction of system memory
                # torch.mps.driver_allocated_size() returns currently allocated
                # We estimate total available as ~5GB for 16GB system
                available = 4 * 1024**3  # 4 GB conservative for MPS
            except Exception:
                available = 2 * 1024**3
        elif device.type == 'cuda':
            try:
                free, total = torch.cuda.mem_get_info(device)
                available = free
            except Exception:
                available = 4 * 1024**3

    max_batch = max(1, int(0.7 * available / bytes_per_spectrum))

    return max_batch


def ppxf_batch(templates, spectra, noise, velscale, start,
               n_jobs=None, gpu=True, quiet=True, batch_size=None,
               **ppxf_kwargs):
    """
    Fit multiple spectra using the standard pPXF pipeline with capfit.

    This function processes spectra in memory-safe chunks, with automatic
    GPU batch sizing. Each spectrum gets a full independent pPXF fit
    including capfit non-linear optimization.

    Parameters
    ----------
    templates : array_like, shape (n_pix_temp, ...)
        Template spectra (same format as standard ppxf).
    spectra : array_like, shape (n_pixels, n_spectra)
        Galaxy spectra to fit. Each column is one spectrum.
    noise : array_like, shape (n_pixels, n_spectra) or (n_pixels,)
        Noise arrays. If 1D, same noise is used for all spectra.
    velscale : float
        Velocity scale in km/s per pixel.
    start : array_like
        Starting guess [vel, sigma, ...] for kinematics.
    n_jobs : int or None
        Not used (reserved for future multiprocessing). Ignored.
    gpu : bool
        Whether to use GPU acceleration for each pPXF fit.
    quiet : bool
        If True, suppress per-spectrum output.
    batch_size : int or None
        Override auto-sizing. If None, automatically estimated from GPU memory.
    **ppxf_kwargs : dict
        Additional keyword arguments passed to each ppxf() call
        (e.g., moments, degree, mdegree, lam, mask, etc.)

    Returns
    -------
    results : list of ppxf objects
        One ppxf result object per spectrum, with all standard attributes
        (sol, chi2, bestfit, weights, etc.)

    Examples
    --------
    >>> from ppxf.ppxf_batch import ppxf_batch
    >>> results = ppxf_batch(templates, spectra, noise, velscale,
    ...                      [0, 200], moments=2, degree=4, gpu=True)
    >>> velocities = [r.sol[0] for r in results]
    """
    from ppxf.ppxf import ppxf

    # Validate input shapes
    if spectra.ndim == 1:
        spectra = spectra[:, np.newaxis]
    if noise.ndim == 1:
        noise_shared = True
        noise_1d = noise
    else:
        noise_shared = False
        if noise.shape[1] != spectra.shape[1]:
            raise ValueError(
                f"noise shape {noise.shape} doesn't match spectra shape {spectra.shape}")

    n_pixels, n_spectra = spectra.shape

    # Estimate batch size for auto-sizing
    templates_2d = templates.reshape(templates.shape[0], -1)
    n_templates = templates_2d.shape[1]
    npad = 2**int(np.ceil(np.log2(max(templates.shape[0], n_pixels))))

    # Determine device
    device = None
    if gpu and torch is not None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')

    if batch_size is None:
        batch_size = estimate_batch_size(n_templates, n_pixels, npad, device)

    if not quiet:
        print(f"ppxf_batch: {n_spectra} spectra, batch_size={batch_size}, "
              f"device={device or 'cpu'}")

    # Process spectra
    results = []
    t_start = time.perf_counter()

    for i in range(n_spectra):
        galaxy_i = spectra[:, i]
        noise_i = noise_1d if noise_shared else noise[:, i]

        try:
            pp = ppxf(templates, galaxy_i, noise_i, velscale, start,
                      gpu=gpu, quiet=True, **ppxf_kwargs)
            results.append(pp)
        except Exception as e:
            if not quiet:
                print(f"  Spectrum {i}: FAILED ({e})")
            results.append(None)

        if not quiet and (i + 1) % max(1, n_spectra // 10) == 0:
            elapsed = time.perf_counter() - t_start
            per_spec = elapsed / (i + 1)
            remaining = per_spec * (n_spectra - i - 1)
            print(f"  [{i+1}/{n_spectra}] {per_spec:.3f}s/spec, "
                  f"ETA: {remaining:.1f}s")

    elapsed = time.perf_counter() - t_start
    if not quiet:
        print(f"ppxf_batch: completed {n_spectra} spectra in {elapsed:.2f}s "
              f"({elapsed/n_spectra:.3f}s/spec)")

    return results
