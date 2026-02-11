
import time
import numpy as np
import sys
from pathlib import Path

# Add project root to path to allow importing ppxf as a package
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from astropy.io import fits
try:
    from ppxf.ppxf import ppxf
    import ppxf.ppxf_util as util
    import ppxf.sps_util as lib
except ImportError:
    import ppxf as ppxf_package
    from ppxf import ppxf
    import ppxf_util as util
    import sps_util as lib
from importlib import resources
from urllib import request
import os

def benchmark_sdss(n_iterations=10):
    print(f"Benchmarking pPXF with {n_iterations} iterations...")
    
    # Setup data (copied from ppxf_example_kinematics_sdss.py)
    # Setup data (copied from ppxf_example_kinematics_sdss.py)
    ppxf_dir = Path(__file__).resolve().parent.parent
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
    
    # Setup templates
    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    model_filename = ppxf_dir / 'sps_models' / basename
    
    if not os.path.exists(model_filename):
        print(f"Downloading {basename}...")
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, model_filename)

    lam_range_temp = [3500, 1e4]
    fwhm_gal_dict = {"lam": lam_gal, "fwhm": fwhm_gal}
    
    sps = lib.sps_lib(model_filename, velscale, fwhm_gal_dict, lam_range=lam_range_temp)
    goodpixels = util.determine_goodpixels(np.log(lam_gal), lam_range_temp)
    
    # Benchmark loop
    try:
        import torch
        HAS_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
    except ImportError:
        HAS_GPU = False

    if not HAS_GPU:
        print("WARNING: No GPU detected. Benchmarking in CPU mode.")

    start_time = time.perf_counter()
    
    for i in range(n_iterations):
        vel = 0
        start = [vel, 200.]
        pp_start = time.perf_counter()
        pp = ppxf(sps.templates, galaxy, noise, velscale, start,
                  goodpixels=goodpixels, plot=False, moments=4, trig=1,
                  degree=20, lam=lam_gal, lam_temp=sps.lam_temp, gpu=HAS_GPU)
                  
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / n_iterations
    
    print(f"Total time: {total_time:.4f} s")
    print(f"Average time per fit: {avg_time:.4f} s")
    
if __name__ == "__main__":
    benchmark_sdss()
