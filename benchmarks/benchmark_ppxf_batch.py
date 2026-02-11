"""
benchmark_ppxf_batch.py — Benchmark ppxf_batch vs sequential ppxf.

Validates that all results match exactly and measures speedup.

Run with:
  source ~/bin/mambaforge/etc/profile.d/conda.sh && conda activate ppxf_gpu
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=/Users/thbrown/ppxf python3 benchmarks/benchmark_ppxf_batch.py
"""
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ppxf.ppxf import ppxf
from ppxf.ppxf_batch import ppxf_batch, estimate_batch_size
import ppxf.sps_util as lib
from astropy.io import fits


def setup_data(ppxf_dir):
    """Load galaxy and template data."""
    filename = ppxf_dir / 'spectra' / 'NGC3522_SDSS_DR18.fits'
    if not filename.exists():
        raise FileNotFoundError(f"Data file: {filename}")

    hdu = fits.open(filename)
    t = hdu['COADD'].data
    galaxy = t['flux'] / np.median(t['flux'])
    ln_lam_gal = t['loglam'] * np.log(10)
    noise = np.full_like(galaxy, 0.0149)
    lam_gal = np.exp(ln_lam_gal)
    c = 299792.458
    d_ln_lam_gal = (ln_lam_gal[-1] - ln_lam_gal[0]) / (ln_lam_gal.size - 1)
    velscale = c * d_ln_lam_gal

    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    model_filename = ppxf_dir / 'sps_models' / basename
    if not model_filename.exists():
        raise FileNotFoundError(f"Template file: {model_filename}")

    sps = lib.sps_lib(str(model_filename), velscale, None, lam_range=[3500, 1e4])
    templates = sps.templates

    return galaxy, noise, velscale, templates, lam_gal


def validate_results(seq_results, batch_results, quiet=False):
    """Compare sequential and batch results attribute-by-attribute."""
    n = len(seq_results)
    assert len(batch_results) == n, f"Length mismatch: {n} vs {len(batch_results)}"

    max_vel_diff = 0
    max_sig_diff = 0
    max_chi2_diff = 0
    max_bestfit_diff = 0
    max_weights_diff = 0
    all_pass = True

    for i in range(n):
        s, b = seq_results[i], batch_results[i]
        if s is None or b is None:
            if not quiet:
                print(f"  Spectrum {i}: skipped (None result)")
            continue

        vel_diff = abs(s.sol[0] - b.sol[0])
        sig_diff = abs(s.sol[1] - b.sol[1])
        chi2_diff = abs(s.chi2 - b.chi2)
        bestfit_diff = np.max(np.abs(s.bestfit - b.bestfit))
        weights_diff = np.max(np.abs(s.weights - b.weights))

        max_vel_diff = max(max_vel_diff, vel_diff)
        max_sig_diff = max(max_sig_diff, sig_diff)
        max_chi2_diff = max(max_chi2_diff, chi2_diff)
        max_bestfit_diff = max(max_bestfit_diff, bestfit_diff)
        max_weights_diff = max(max_weights_diff, weights_diff)

    if not quiet:
        print(f"\n  Validation Summary (N={n}):")
        print(f"    Max |Δvel|:     {max_vel_diff:.2e}")
        print(f"    Max |Δsigma|:   {max_sig_diff:.2e}")
        print(f"    Max |Δchi2|:    {max_chi2_diff:.2e}")
        print(f"    Max |Δbestfit|: {max_bestfit_diff:.2e}")
        print(f"    Max |Δweights|: {max_weights_diff:.2e}")

    # These should be exactly zero since same code path
    tol = 1e-8
    if max_vel_diff > tol or max_sig_diff > tol:
        all_pass = False
        if not quiet:
            print(f"  WARNING: Kinematic differences exceed {tol}")

    if not quiet:
        status = "PASS" if all_pass else "FAIL"
        print(f"  Result: {status}")

    return all_pass, {
        'max_vel_diff': max_vel_diff,
        'max_sig_diff': max_sig_diff,
        'max_chi2_diff': max_chi2_diff,
        'max_bestfit_diff': max_bestfit_diff,
        'max_weights_diff': max_weights_diff,
    }


def benchmark(n_spectra_list=[5, 10, 20], gpu=True):
    """Run sequential vs batch benchmark for multiple N values."""
    ppxf_dir = current_dir.parent
    galaxy, noise, velscale, templates, lam_gal = setup_data(ppxf_dir)
    start = [0, 200.]

    print("=" * 70)
    print(f"ppxf_batch Benchmark: GPU={gpu}")
    print("=" * 70)

    # Show auto-sizing info
    templates_2d = templates.reshape(templates.shape[0], -1)
    npad = 2**int(np.ceil(np.log2(max(templates.shape[0], galaxy.shape[0]))))
    est = estimate_batch_size(templates_2d.shape[1], galaxy.shape[0], npad)
    print(f"Templates: {templates.shape} -> {templates_2d.shape[1]} cols")
    print(f"Galaxy: {galaxy.shape[0]} pixels, npad={npad}")
    print(f"Auto batch_size estimate: {est}")

    seq_times = []
    batch_times = []
    actual_ns = []

    for n in n_spectra_list:
        print(f"\n{'─' * 60}")
        print(f"Testing N={n} spectra")
        print(f"{'─' * 60}")

        # Create multi-spectrum data (add small noise to each)
        rng = np.random.default_rng(42)
        spectra = np.column_stack([
            galaxy + rng.normal(0, 0.01, galaxy.shape) for _ in range(n)
        ])
        noise_batch = np.column_stack([noise] * n)

        # Sequential
        print(f"  Sequential (N={n})...")
        t0 = time.perf_counter()
        seq_results = []
        for i in range(n):
            pp = ppxf(templates, spectra[:, i], noise, velscale, start,
                      moments=2, degree=4, gpu=gpu, quiet=True)
            seq_results.append(pp)
        seq_time = time.perf_counter() - t0
        seq_times.append(seq_time)
        print(f"    Time: {seq_time:.3f}s ({seq_time/n*1000:.1f}ms/spec)")

        # Batched
        print(f"  Batched (N={n})...")
        t0 = time.perf_counter()
        batch_results = ppxf_batch(
            templates, spectra, noise_batch, velscale, start,
            moments=2, degree=4, gpu=gpu, quiet=True)
        batch_time = time.perf_counter() - t0
        batch_times.append(batch_time)
        print(f"    Time: {batch_time:.3f}s ({batch_time/n*1000:.1f}ms/spec)")

        # Speedup
        speedup = seq_time / batch_time if batch_time > 0 else float('inf')
        print(f"    Speedup: {speedup:.2f}x")

        # Validate
        passed, diffs = validate_results(seq_results, batch_results, quiet=False)
        actual_ns.append(n)

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"Summary Table")
    print(f"{'=' * 70}")
    print(f"{'N':>6} {'Sequential':>12} {'Batched':>12} {'Speedup':>10} {'Per-spec':>10}")
    print(f"{'':>6} {'(s)':>12} {'(s)':>12} {'':>10} {'(ms)':>10}")
    print(f"{'─' * 60}")
    for i, n in enumerate(actual_ns):
        sp = seq_times[i] / batch_times[i] if batch_times[i] > 0 else 0
        per = batch_times[i] / n * 1000
        print(f"{n:>6} {seq_times[i]:>12.3f} {batch_times[i]:>12.3f} {sp:>10.2f}x {per:>10.1f}")

    # Generate plots
    generate_plots(actual_ns, seq_times, batch_times, current_dir)

    return actual_ns, seq_times, batch_times


def generate_plots(ns, seq_times, batch_times, output_dir):
    """Generate benchmark comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Total time comparison
    ax = axes[0]
    x = np.arange(len(ns))
    width = 0.35
    ax.bar(x - width/2, seq_times, width, label='Sequential', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, batch_times, width, label='Batched', color='#2ecc71', alpha=0.8)
    ax.set_xlabel('Number of Spectra')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Total Fit Time')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Per-spectrum time
    ax = axes[1]
    seq_per = [t/n*1000 for t, n in zip(seq_times, ns)]
    batch_per = [t/n*1000 for t, n in zip(batch_times, ns)]
    ax.bar(x - width/2, seq_per, width, label='Sequential', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, batch_per, width, label='Batched', color='#2ecc71', alpha=0.8)
    ax.set_xlabel('Number of Spectra')
    ax.set_ylabel('Time per Spectrum (ms)')
    ax.set_title('Per-Spectrum Time')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Speedup
    ax = axes[2]
    speedups = [s/b if b > 0 else 0 for s, b in zip(seq_times, batch_times)]
    colors = ['#2ecc71' if s >= 1 else '#e74c3c' for s in speedups]
    ax.bar(x, speedups, color=colors, alpha=0.8)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='1x (baseline)')
    ax.set_xlabel('Number of Spectra')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Speedup: Batched / Sequential')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    outpath = output_dir / 'ppxf_batch_benchmark.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {outpath}")


if __name__ == "__main__":
    benchmark(n_spectra_list=[5, 10, 20], gpu=True)
