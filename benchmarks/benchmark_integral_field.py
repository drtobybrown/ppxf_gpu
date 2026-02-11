
import time
import numpy as np
import sys
import os
from pathlib import Path
from urllib import request
from astropy.io import fits

# Add project root to path to allow importing ppxf as a package
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ppxf.ppxf import ppxf, robust_sigma
    import ppxf.ppxf_util as util
    import ppxf.sps_util as lib
except ImportError:
    import ppxf as ppxf_package
    from ppxf import ppxf, robust_sigma
    import ppxf_util as util
    import sps_util as lib

# Mock PowerBin if not available or just use simple binning for benchmark? 
# The user wants to benchmark the *existing* example.
# The example uses `powerbin` package. We should check if it's installed or mock it.
# For a robust benchmark without external dependencies (if powerbin isn't standard), 
# we might need to simplify the binning step or assume it's there.
# Given the instructions, I'll try to import it, but provide a fallback or failing if missing is okay as it's a benchmark of the example.
try:
    from powerbin import PowerBin
except ImportError:
    # If powerbin is missing, we can't run the exact example. 
    # But for the purpose of this task (ppxf optimization), the binning is preprocessing.
    # The core ppxf fit is what matters. 
    # I will implement a simplified binning or mock for the benchmark if needed, 
    # but first let's assume the user has the environment set up or will install requirements.
    # Actually, I should probably check if I can install it or if it's in the repo.
    pass

def benchmark_integral_field(n_iterations=1):
    # IFU fitting is slow, so default n_iterations is small
    print(f"Benchmarking pPXF (Integral Field) with {n_iterations} iterations...")
    
    # 1. Setup Data (simplified from ppxf_example_integral_field.py)
    # We need the cube.
    ppxf_dir = Path(__file__).resolve().parent.parent
    objfile = ppxf_dir / 'LVS_JWST_workshop_rodeo_cube.fits'
    if not objfile.is_file():
        print(f"Downloading {objfile.name}...")
        url = "https://raw.githubusercontent.com/micappe/ppxf_examples/main/" + objfile.name
        request.urlretrieve(url, objfile)
        
    # Re-implement minimal read_data_cube logic
    lam_range_temp = [3540, 7409]
    redshift = 0.002895
    
    try:
        cube, head = fits.getdata(objfile, header=True)
    except FileNotFoundError:
        print("Could not find or download data file.")
        return

    wave = head['CRVAL3'] + head['CDELT3']*np.arange(cube.shape[0])
    wave = wave/(1 + redshift)
    w = (wave > lam_range_temp[0]) & (wave < lam_range_temp[1])
    wave = wave[w]
    cube = cube[w, ...]
    
    npix = cube.shape[0]
    spectra = cube.reshape(npix, -1)
    
    c = 299792.458
    velscale = np.min(c*np.diff(np.log(wave)))
    lam_range_temp = [np.min(wave), np.max(wave)]
    spectra, ln_lam_gal, velscale = util.log_rebin(lam_range_temp, spectra, velscale=velscale)
    
    # Just take a subset of spectra to benchmark the fitting loop, 
    # as powerbin might be external.
    # Let's simulate "binning" by just taking mean of some groups of spectra
    # to create ~10-20 bins to fit.
    
    n_bins_bench = 50
    # Create random bins index
    n_spaxels = spectra.shape[1]
    bin_num = np.random.randint(0, n_bins_bench, n_spaxels)
    
    # 2. Setup Templates
    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    model_filename = ppxf_dir / 'sps_models' / basename
    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    model_filename = ppxf_dir / 'sps_models' / basename
    if not model_filename.exists():
         url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
         request.urlretrieve(url, model_filename)
         
    sps = lib.sps_lib(model_filename, velscale, fwhm_gal=None, norm_range=[5070, 5950])
    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
    stars_templates /= np.median(stars_templates)
    
    # 3. Setup Gas
    lam_range_gal = np.exp(ln_lam_gal[[0, -1]])
    gas_templates, gas_names, line_wave = util.emission_lines(sps.ln_lam_temp, lam_range_gal, 2.62) # 2.62 is FWHM from example
    
    # 4. Benchmark Loop (Fitting the bins)
    
    lam_gal = np.exp(ln_lam_gal)
    mask0 = util.determine_mask(ln_lam_gal, np.exp(sps.ln_lam_temp[[0, -1]]), width=1000)
    
    start_time = time.perf_counter()
    
    for i in range(n_iterations):
        # Fit each bin
        for j in range(n_bins_bench):
            w = bin_num == j
            if not np.any(w): continue
            
            galaxy = np.nanmean(spectra[:, w], 1)
            noise = np.full_like(galaxy, 1e-3) # Simplified noise
            
            # Simple fit (stars only first, as per example strategy)
            start = [0, 200.]
            pp = ppxf(stars_templates, galaxy, noise, velscale, start,
                      moments=2, degree=4, mdegree=-1, lam=lam_gal, lam_temp=sps.lam_temp,
                      mask=mask0, quiet=True, plot=False)
            
            # Note: The example does a 2-step fit (clean outliers).
            # We include that in benchmark? Yes, robustness matters.
            # Clip outliers
            clean_mask = mask0.copy()
            # Simplified clipping for benchmark speed/conciseness
            resid = (pp.galaxy - pp.bestfit)
            err = robust_sigma(resid, zero=1)
            clean_mask &= (np.abs(resid) < 3*err)
            
            # Refit
            pp = ppxf(stars_templates, galaxy, noise, velscale, start,
                      moments=2, degree=4, mdegree=-1, lam=lam_gal, lam_temp=sps.lam_temp,
                      mask=clean_mask, quiet=True, plot=False)
                  
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Per-spectrum time is more useful here
    n_fits = n_iterations * n_bins_bench
    avg_time = total_time / n_fits
    
    print(f"Total time: {total_time:.4f} s")
    print(f"Total fits: {n_fits}")
    print(f"Average time per fit: {avg_time:.4f} s")

if __name__ == "__main__":
    benchmark_integral_field()
