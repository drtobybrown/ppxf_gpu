
import sys
try:
    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Check for Metal/MPS
    metal_devices = [d for d in jax.devices() if 'METAL' in str(d).upper() or 'MPS' in str(d).upper()]
    if metal_devices:
        print(f"Metal GPU detected: {metal_devices}")
    else:
        print("No Metal GPU detected by JAX. Is jax-metal installed?")
        
except ImportError:
    print("JAX is not installed.")
