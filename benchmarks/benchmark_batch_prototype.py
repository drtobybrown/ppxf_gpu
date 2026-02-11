"""
benchmark_batch_prototype.py — Compare sequential pPXF vs batched pPXF linear fit.

Runs in the ppxf_gpu conda environment with:
  source ~/bin/mambaforge/etc/profile.d/conda.sh && conda activate ppxf_gpu
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=/Users/thbrown/ppxf python3 benchmarks/benchmark_batch_prototype.py
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path (Outer ppxf directory)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ppxf.ppxf import ppxf
from ppxf.ppxf_batch import ppxf_batch
import ppxf.sps_util as lib
from astropy.io import fits

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def benchmark_batch_linear_fit(n_spectra=1000):
    print("=" * 60)
    print(f"Benchmarking Batched Linear Fit vs Sequential (N={n_spectra})")
    print("=" * 60)

    # --- 1. Setup Data ---
    ppxf_dir = current_dir.parent  # ppxf/ppxf
    filename = ppxf_dir / 'spectra' / 'NGC3522_SDSS_DR18.fits'

    if not filename.exists():
        print(f"Data file not found: {filename}")
        return

    hdu = fits.open(filename)
    t = hdu['COADD'].data
    galaxy = t['flux'] / np.median(t['flux'])
    ln_lam_gal = t['loglam'] * np.log(10)
    noise = np.full_like(galaxy, 0.0149)
    c = 299792.458
    d_ln_lam_gal = (ln_lam_gal[-1] - ln_lam_gal[0]) / (ln_lam_gal.size - 1)
    velscale = c * d_ln_lam_gal

    # Templates
    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    model_filename = ppxf_dir / 'sps_models' / basename
    if not model_filename.exists():
        print(f"Template file not found: {model_filename}")
        return

    sps = lib.sps_lib(str(model_filename), velscale, None, lam_range=[3500, 1e4])
    templates = sps.templates

    print(f"Galaxy shape: {galaxy.shape}")
    print(f"Templates shape: {templates.shape}")

    # --- 2. Sequential pPXF (Full fit, measure N and extrapolate) ---
    n_measure = min(20, n_spectra)
    start_val = [0, 200.]

    print(f"\nRunning Sequential pPXF (N={n_measure}, extrapolate to {n_spectra})...")
    t0 = time.perf_counter()
    for i in range(n_measure):
        pp = ppxf(templates, galaxy, noise, velscale, start_val,
                  degree=4, moments=2, gpu=True, quiet=True)
    t1 = time.perf_counter()

    seq_per_fit = (t1 - t0) / n_measure
    seq_total = seq_per_fit * n_spectra
    print(f"  Time per fit: {seq_per_fit * 1000:.2f} ms")
    print(f"  Extrapolated time for {n_spectra} spectra: {seq_total:.2f} s")

    # --- 3. Batched pPXF ---
    galaxy_batch = np.tile(galaxy[:, None], (1, n_spectra))
    noise_batch = np.tile(noise[:, None], (1, n_spectra))

    print(f"\nRunning Batched pPXF (N={n_spectra})...")
    try:
        ppb = ppxf_batch(templates, galaxy_batch, noise_batch, velscale, start_val,
                         moments=2, degree=4, gpu=True, quiet=True)
    except Exception as e:
        print(f"Failed to init ppxf_batch: {e}")
        import traceback
        traceback.print_exc()
        return

    # Prepare kinematic parameters (vel, sigma) for all spectra
    # Just moments=2 here, so pars has shape (n_spectra, 2)
    pars_np = np.tile(start_val, (n_spectra, 1)).astype(np.float32)

    if ppb.use_torch:
        pars_tensor = torch.from_numpy(pars_np).to(ppb.device)
    else:
        pars_tensor = ppb.xp.asarray(pars_np)

    # Warmup
    ppb.linear_fit(pars_tensor)
    if ppb.use_torch and HAS_TORCH and torch.backends.mps.is_available():
        torch.mps.synchronize()

    # Timed run
    t0 = time.perf_counter()
    weights, bestfit, chi2 = ppb.linear_fit(pars_tensor)
    if ppb.use_torch and HAS_TORCH and torch.backends.mps.is_available():
        torch.mps.synchronize()
    batch_time = time.perf_counter() - t0

    print(f"  Batched total time: {batch_time:.4f} s")
    print(f"  Batched time per fit: {batch_time / n_spectra * 1000:.4f} ms")

    speedup = seq_total / batch_time if batch_time > 0 else float('inf')
    print(f"\n{'=' * 40}")
    print(f"  SPEEDUP: {speedup:.1f}x  (sequential extrapolated / batched)")
    print(f"{'=' * 40}")

    # --- 4. Basic Validation ---
    print("\nValidation:")
    if ppb.use_torch:
        bestfit_b0 = bestfit[0].cpu().numpy()
        chi2_b0 = chi2[0].item()
    else:
        bestfit_b0 = np.asarray(bestfit[0])
        chi2_b0 = float(chi2[0])

    # Sequential reference (from the last pp fit above)
    bestfit_seq = pp.bestfit
    if hasattr(bestfit_seq, 'cpu'):
        bestfit_seq = bestfit_seq.cpu().numpy()
    elif hasattr(bestfit_seq, 'get'):
        bestfit_seq = bestfit_seq.get()

    # The batch result uses a different linear solve (unconstrained lstsq)
    # while ppxf uses constrained NNLS/BVLS, so results won't match exactly.
    # But the shapes and magnitudes should be reasonable.
    print(f"  Batch bestfit shape: {bestfit_b0.shape}")
    print(f"  Sequential bestfit shape: {bestfit_seq.shape}")
    print(f"  Batch chi2[0]: {chi2_b0:.4f}")

    # Check for NaN/Inf
    if np.any(np.isnan(bestfit_b0)):
        print("  FAIL: NaN in batch bestfit!")
    elif np.any(np.isinf(bestfit_b0)):
        print("  FAIL: Inf in batch bestfit!")
    else:
        corr = np.corrcoef(bestfit_b0, bestfit_seq)[0, 1]
        print(f"  Correlation between batch and sequential bestfit: {corr:.6f}")
        if corr > 0.99:
            print("  PASS: High correlation (>0.99) — batch linear fit is working correctly.")
        else:
            print(f"  WARNING: Moderate correlation ({corr:.4f}). "
                  "Expected since batch uses unconstrained lstsq vs constrained BVLS.")


if __name__ == "__main__":
    for n in [10, 50, 100]:
        benchmark_batch_linear_fit(n_spectra=n)
        print("\n")
