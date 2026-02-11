
import sys
from pathlib import Path

# Add project root to path to allow importing ppxf as a package
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
try:
    from ppxf.ppxf import ppxf
    import ppxf.ppxf_util as util
    import ppxf.sps_util as lib
except ImportError:
    import ppxf as ppxf_package
    from ppxf import ppxf
    import ppxf_util as util
    import sps_util as lib

def verify_gpu_fallback():
    print("Verifying GPU fallback behavior...")
    
    # Create minimal mock data
    npix = 100
    galaxy = np.random.rand(npix)
    noise = np.ones(npix) * 0.1
    velscale = 1.0
    start = [0, 100.]
    
    # Mock templates
    ntemp = 2
    templates = np.random.rand(npix, ntemp)
    
    # Run pPXF with gpu=True
    try:
        print("Initializing pPXF with gpu=True...")
        pp = ppxf(templates, galaxy, noise, velscale, start, gpu=True, quiet=True)
        
        if pp.gpu is False:
            print("SUCCESS: pPXF fell back to CPU (pp.gpu is False).")
        else:
            backend = pp.xp.__name__ if hasattr(pp.xp, '__name__') else str(pp.xp)
            if 'cupy' in backend:
                print("SUCCESS: pPXF is using CuPy.")
            elif 'TorchWrapper' in backend or 'torch' in backend:
                print("SUCCESS: pPXF is using PyTorch/MPS.")
            else:
                print(f"WARNING: pPXF claims GPU usage with backend: {backend}")
            
    except Exception as e:
        print(f"FAILURE: pPXF crashed with gpu=True: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_gpu_fallback()

    # DIAGNOSTIC: Test TorchWrapper directly
    print("\n--- TorchWrapper Diagnostic ---")
    try:
        try:
            from ppxf.torch_wrapper import TorchWrapper
        except ImportError:
            from torch_wrapper import TorchWrapper
            
        tw = TorchWrapper()
        print(f"TorchWrapper device: {tw.device}")
        
        z = tw.zeros((10, 10))
        print(f"tw.zeros((10,10)) dtype: {z.dtype}")
        
        z_float = tw.zeros((10, 10), dtype=float)
        print(f"tw.zeros((10,10), dtype=float) dtype: {z_float.dtype}")
        
        if z.dtype == torch.float32 and tw.device.type != 'mps':
            print("CRITICAL IF: TorchWrapper defaulting to float32 on non-MPS device!")
            
    except Exception as e:
        print(f"Diagnostic failed: {e}")
