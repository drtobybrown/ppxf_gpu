
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
from urllib import request
from importlib import resources

try:
    from ppxf.ppxf import ppxf
    import ppxf.ppxf_util as util
    import ppxf.sps_util as lib
except ImportError:
    import ppxf as ppxf_package
    from ppxf import ppxf
    import ppxf_util as util
    import sps_util as lib
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

def benchmark_gas_sdss_tied(n_iterations=5):
    print(f"Benchmarking pPXF (Gas SDSS Tied) with {n_iterations} iterations...")
    
    # Setup data (based on ppxf_example_gas_sdss_tied.py)
    # Setup data (based on ppxf_example_gas_sdss_tied.py)
    ppxf_dir = Path(__file__).resolve().parent.parent
    filename = ppxf_dir / 'spectra/NGC3073_SDSS_DR18.fits'
    
    if not filename.exists():
        print(f"File not found: {filename}")
        return

    hdu = fits.open(filename)
    t = hdu['COADD'].data
    redshift = hdu['SPECOBJ'].data['z'].item()
    
    galaxy = t['flux']/np.median(t['flux'])
    ln_lam_gal = t['loglam']*np.log(10)
    lam_gal = np.exp(ln_lam_gal)
    lam_gal *= np.median(util.vac_to_air(lam_gal)/lam_gal)
    
    # Assume constant noise per pixel
    noise = np.full_like(galaxy, 0.01635)
    
    c = 299792.458
    d_ln_lam_gal = (ln_lam_gal[-1] - ln_lam_gal[0])/(ln_lam_gal.size - 1)
    velscale = c*d_ln_lam_gal
    
    dlam_gal = np.gradient(lam_gal)
    wdisp = t['wdisp']
    fwhm_gal = 2.355*wdisp*dlam_gal
    
    # Setup templates
    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    sps_name = 'emiles'
    basename = f"spectra_{sps_name}_9.0.npz"
    model_filename = ppxf_dir / 'sps_models' / basename
    
    if not model_filename.exists():
        print(f"Downloading {basename}...")
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, model_filename)

    fwhm_gal_dic = {"lam": lam_gal, "fwhm": fwhm_gal}
    # Note: ssp_lib arguments might vary slightly depending on ppxf version, 
    # but based on example it takes filename, velscale, fwhm_gal_dic
    sps = lib.sps_lib(model_filename, velscale, fwhm_gal_dic)
    
    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
    
    lam_range_gal = np.array([np.min(lam_gal), np.max(lam_gal)])/(1 + redshift)
    gas_templates, gas_names, gas_wave = \
        util.emission_lines(sps.ln_lam_temp, lam_range_gal, fwhm_gal_dic)
        
    templates = np.column_stack([stars_templates, gas_templates])
    n_temps = stars_templates.shape[1]
    
    # Setup components and constraints
    vel0 = c*np.log(1 + redshift)
    sol = [vel0, 200]
    
    component = [0]*n_temps
    component += [1]*8
    component += [2, 2, 3, 3, 4, 5, 6, 7, 8]
    component = np.array(component)
    
    moments = [2]*9
    start = [sol for j in range(len(moments))]
    
    tied = [['', ''] for j in range(len(moments))]
    for j in range(3, len(moments)):
        tied[j][0] = 'p[4]'
        
    A_ineq = [[0, -2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    b_ineq = [0.0] * 8
    constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}
    
    degree = -1
    mdegree = 10
    
    # Benchmark loop
    start_time = time.perf_counter()
    
    for i in range(n_iterations):
        pp = ppxf(templates, galaxy, noise, velscale, start, plot=False,
                  moments=moments, degree=degree, mdegree=mdegree, 
                  lam=lam_gal, component=component, tied=tied, 
                  gas_component=component > 0, gas_names=gas_names,
                  constr_kinem=constr_kinem, lam_temp=sps.lam_temp)
                  
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / n_iterations
    
    print(f"Total time: {total_time:.4f} s")
    print(f"Average time per fit: {avg_time:.4f} s")
    
if __name__ == "__main__":
    benchmark_gas_sdss_tied()
