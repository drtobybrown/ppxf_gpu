
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as clock
from importlib import resources
from urllib import request
from astropy.io import fits
import os
import sys
from pathlib import Path

# Add project root to path to allow importing ppxf as a package
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ppxf.ppxf import ppxf
    import ppxf.ppxf_util as util
    import ppxf.sps_util as lib
except ImportError:
    import ppxf as ppxf_package
    from ppxf import ppxf
    import ppxf_util as util
    import sps_util as lib

def verify_ppxf():
    print("--------------------------------------------------------------------------------")
    print("pPXF GPU Verification Script")
    print("--------------------------------------------------------------------------------")

    # --- Setup Data (from ppxf_example_kinematics_sdss.py) ---
    print("Loading data...")
    ppxf_dir = resources.files('ppxf')
    filename = ppxf_dir / 'spectra/NGC3522_SDSS_DR18.fits'
    # Ensure file exists
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    hdu = fits.open(filename)
    t = hdu['COADD'].data
    redshift_0 = hdu['SPECOBJ'].data['z'].item()
    galaxy = t['flux']/np.median(t['flux'])
    ln_lam_gal = t['loglam']*np.log(10)
    lam_gal = np.exp(ln_lam_gal)
    lam_gal *= np.median(util.vac_to_air(lam_gal)/lam_gal)
    noise = np.full_like(galaxy, 0.0149)
    c = 299792.458
    d_ln_lam_gal = (ln_lam_gal[-1] - ln_lam_gal[0])/(ln_lam_gal.size - 1)
    velscale = c*d_ln_lam_gal
    dlam_gal = np.gradient(lam_gal)
    wdisp = t['wdisp']
    fwhm_gal = 2.355*wdisp*dlam_gal
    lam_gal = lam_gal/(1 + redshift_0)
    fwhm_gal = fwhm_gal/(1 + redshift_0)

    # --- Setup Templates ---
    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    filename_sps = ppxf_dir / 'sps_models' / basename
    if not os.path.exists(filename_sps):
         print(f"SPS models not found at {filename_sps}. Downloading...")
         url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
         request.urlretrieve(url, filename_sps)

    lam_range_temp = [3500, 1e4]
    fwhm_gal_dict = {"lam": lam_gal, "fwhm": fwhm_gal}
    # To save time for repeated runs, we can cache sps? 
    # But initialization is fast enough usually.
    print("Initializing SPS library...")
    sps = lib.sps_lib(filename_sps, velscale, fwhm_gal_dict, lam_range=lam_range_temp)
    goodpixels = util.determine_goodpixels(np.log(lam_gal), lam_range_temp)

    # --- Run CPU Fit ---
    print("Running pPXF on CPU...")
    vel = 0
    start = [vel, 200.]
    t_cpu_start = clock()
    pp_cpu = ppxf(sps.templates, galaxy, noise, velscale, start,
              goodpixels=goodpixels, plot=False, moments=4, trig=1,
              degree=20, lam=lam_gal, lam_temp=sps.lam_temp, gpu=False)
    t_cpu = clock() - t_cpu_start
    print(f"CPU Time: {t_cpu:.4f} s")

    # --- Run GPU Fit ---
    print("Running pPXF on GPU...")
    t_gpu_start = clock()
    pp_gpu = ppxf(sps.templates, galaxy, noise, velscale, start,
              goodpixels=goodpixels, plot=False, moments=4, trig=1,
              degree=20, lam=lam_gal, lam_temp=sps.lam_temp, gpu=True)
    t_gpu = clock() - t_gpu_start
    print(f"GPU Time: {t_gpu:.4f} s")
    print(f"GPU used: {pp_gpu.gpu}")
    if pp_gpu.gpu:
         print(f"Backend: {pp_gpu.xp.__name__ if hasattr(pp_gpu.xp, '__name__') else type(pp_gpu.xp)}")

    # --- Compare Results ---
    print("Comparing results...")
    
    # 1. Bestfit
    diff_bestfit = np.abs(pp_cpu.bestfit - pp_gpu.bestfit)
    max_diff_bestfit = np.max(diff_bestfit)
    print(f"Max abs diff (bestfit): {max_diff_bestfit:.2e}")
    
    # 2. Chi2
    diff_chi2 = np.abs(pp_cpu.chi2 - pp_gpu.chi2)
    print(f"Diff Chi2: {diff_chi2:.2e}")
    
    # 3. Kinematic Solution
    diff_sol = np.abs(pp_cpu.sol - pp_gpu.sol)
    print(f"Diff Sol: {diff_sol}")
    
    TOLERANCE = 1e-4
    if max_diff_bestfit < TOLERANCE:
        print("PASS: Results match within tolerance.")
    else:
        print("FAIL: Results differ significantly!")

    # --- Plotting ---
    print("Generating plots...")
    
    # Plot 1: Bestfit Comparison & Residuals
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.plot(pp_cpu.lam, pp_cpu.galaxy, 'k-', label='Galaxy', alpha=0.5)
    plt.plot(pp_cpu.lam, pp_cpu.bestfit, 'r-', label='CPU Bestfit')
    plt.plot(pp_gpu.lam, pp_gpu.bestfit, 'b--', label='GPU Bestfit')
    plt.legend()
    plt.title("Fit Comparison")
    plt.ylabel("Flux")
    
    plt.subplot(212)
    plt.plot(pp_cpu.lam, pp_cpu.galaxy - pp_cpu.bestfit, 'r-', label='CPU Residual', alpha=0.5)
    plt.plot(pp_gpu.lam, pp_gpu.galaxy - pp_gpu.bestfit, 'b--', label='GPU Residual', alpha=0.5)
    plt.plot(pp_cpu.lam, pp_cpu.bestfit - pp_gpu.bestfit, 'g-', label='CPU - GPU Diff')
    plt.legend()
    plt.title(f"Residuals (Max Diff: {max_diff_bestfit:.2e})")
    plt.xlabel("Wavelength")
    
    plt.tight_layout()
    plt.savefig('verification_report.png')
    print("Saved verification_report.png")
    
    # Plot 2: Performance
    plt.figure(figsize=(8, 6))
    methods = ['CPU', 'GPU']
    times = [t_cpu, t_gpu]
    
    plt.bar(methods, times, color=['red', 'blue'])
    plt.ylabel("Time (s)")
    plt.title(f"Performance Comparison\nSpeedup: {t_cpu/t_gpu:.2f}x")
    for i, v in enumerate(times):
        plt.text(i, v + 0.05, f"{v:.2f}s", ha='center')
        
    plt.savefig('performance_comparison.png')
    print("Saved performance_comparison.png")

if __name__ == "__main__":
    verify_ppxf()
