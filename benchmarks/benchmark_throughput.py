
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from astropy.io import fits
from urllib import request
import os

try:
    from ppxf.ppxf import ppxf
    import ppxf.ppxf_util as util
    import ppxf.sps_util as lib
except ImportError:
    import ppxf as ppxf_package
    from ppxf import ppxf
    import ppxf_util as util
    import sps_util as lib

def benchmark_throughput(n_spectra=10000, n_measure=50, use_gpu=True):
    """
    Benchmarks throughput by simulating fitting `n_spectra`.
    To save time, we run `n_measure` actual fits and extrapolate.
    """
    
    print("="*60)
    print(f"pPXF Throughput Benchmark (Target: {n_spectra} spectra)")
    print(f"Configuration: GPU={use_gpu}, Measuring first {n_measure} iterations")
    print("="*60)

    # --- 1. Setup Data (SDSS Single Spectrum) ---
    ppxf_dir = Path(__file__).resolve().parent.parent
    filename = ppxf_dir / 'spectra/NGC3522_SDSS_DR18.fits'
    
    if not filename.exists():
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
    lam_gal = lam_gal/(1 + redshift_0)
    
    # --- 2. Setup Templates ---
    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    model_filename = ppxf_dir / 'sps_models' / basename
    
    if not model_filename.exists():
        print(f"Downloading {basename}...")
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, model_filename)

    lam_range_temp = [3500, 1e4]
    fwhm_gal = None # Simplified for throughput test
    sps = lib.sps_lib(model_filename, velscale, fwhm_gal, lam_range=lam_range_temp)
    goodpixels = util.determine_goodpixels(np.log(lam_gal), lam_range_temp)
    
    # --- 3. Benchmark Loop ---
    print(f"Running {n_measure} iterations to estimate total time...")
    
    # Warmup
    start = [0, 200.]
    _ = ppxf(sps.templates, galaxy, noise, velscale, start,
             goodpixels=goodpixels, plot=False, moments=4, trig=1,
             degree=20, lam=lam_gal, lam_temp=sps.lam_temp, gpu=use_gpu, quiet=True)
             
    start_time = time.perf_counter()
    
    for i in range(n_measure):
        pp = ppxf(sps.templates, galaxy, noise, velscale, start,
                  goodpixels=goodpixels, plot=False, moments=4, trig=1,
                  degree=20, lam=lam_gal, lam_temp=sps.lam_temp, gpu=use_gpu, quiet=True)
        # In a real scenario, we'd loop over different galaxies.
        # Here we just check throughput.
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    avg_per_fit = duration / n_measure
    
    projected_total = avg_per_fit * n_spectra
    
    print("-" * 40)
    print(f"Measured {n_measure} fits in {duration:.4f}s")
    print(f"Average time per fit: {avg_per_fit:.4f}s")
    print("-" * 40)
    print(f"Projected time for {n_spectra} spectra: {projected_total:.2f}s ({projected_total/60:.2f} min)")
    print("="*60)
    
    return avg_per_fit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--n", type=int, default=10000, help="Total spectra to simulate")
    parser.add_argument("--measure", type=int, default=50, help="Number of iterations to measure")
    args = parser.parse_args()
    
    use_gpu = not args.cpu
    benchmark_throughput(n_spectra=args.n, n_measure=args.measure, use_gpu=use_gpu)
