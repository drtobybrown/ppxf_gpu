# pPXF with GPU Support and Batch Processing

This is a **hard fork** of the [pPXF](https://pypi.org/project/ppxf/) package (originally by Michele Cappellari) designed for high-performance spectral fitting.

## Key Features

### 1. GPU Acceleration (PyTorch)
- **Multi-Backend Support**: Works seamlessly on **macOS (MPS/Metal)** and **NVIDIA (CUDA)**.
- **Native PyTorch Implementation**: Replaced `numpy`/`scipy` linear algebra with `torch` equivalents (`fft`, `lstsq`).
- **Automatic Fallback**: Gracefully falls back to CPU if no GPU is detected.

### 2. Batched Processing (`ppxf_batch`)
- **Massive Speedup**: Achieves **~20-30x speedup** over sequential execution for large datasets.
- **Full Capfit Integration**: Runs the complete `capfit` non-linear optimization for *every* spectrum (kinematics are NOT tied).
- **Auto-Sizing**: Automatically estimates the optimal batch size based on your GPU's available memory.
- **Memory Safe**: Processes huge datasets (e.g., 10k+ spectra) in efficient chunks to prevent OOM errors.

## Installation

We recommend using `conda` for environment management.

### 1. Create the environment
```bash
conda create -n ppxf_gpu python=3.10 numpy scipy matplotlib astropy
conda activate ppxf_gpu
```

### 2. Install PyTorch
Follow the [official instructions](https://pytorch.org/get-started/locally/) for your hardware.

**For Mac (M1/M2/M3):**
```bash
pip install torch torchvision
```

**For NVIDIA GPU (Linux/Windows):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118  # or your CUDA version
```

### 3. Install this package
```bash
pip install -e .
```

## Usage

### Standard Single-Spectrum Fit (with GPU)
Simply add `gpu=True` to your `ppxf()` call.

```python
from ppxf.ppxf import ppxf

pp = ppxf(templates, galaxy, noise, velscale, start,
          moments=2, degree=4, gpu=True)  # <--- Enable GPU
```

### Batched Processing (Multiple Spectra)
Use the `ppxf_batch` wrapper to fit N spectra in parallel on the GPU.

```python
from ppxf.ppxf_batch import ppxf_batch

# spectra: (n_pixels, n_spectra) array
# noise:   (n_pixels, n_spectra) array OR (n_pixels,) vector

results = ppxf_batch(
    templates, 
    spectra, 
    noise, 
    velscale, 
    start,
    moments=2, 
    degree=4, 
    gpu=True
)

# results is a list of ppxf objects (one per spectrum)
for i, pp in enumerate(results):
    print(f"Spectrum {i}: Vel={pp.sol[0]:.1f}, Sig={pp.sol[1]:.1f}")
```

## Performance

| Scenario | Sequential (CPU) | Batched (GPU) | Speedup |
| :--- | :---: | :---: | :---: |
| 100 Spectra | ~20 s | ~1.0 s | **~20x** |
| 10,000 Spectra | ~1.7 hours | ~4 mins | **~25x** |

*Benchmarks run on Apple M1 Pro (16-core GPU).*

## How It Works
- **Linear Step (GPU)**: The expensive convolution and least-squares solving (for weights) are batched and executed on the GPU.
- **Non-Linear Step (CPU+GPU)**: The `capfit` optimizer runs independently for each spectrum (CPU), but offloads the heavy `linear_fit` evaluations to the GPU in batches.
- **Precision**: Results match the standard CPU version **bit-for-bit** (within float precision).

## License
This project retains the original [pPXF License](LICENSE.txt).
