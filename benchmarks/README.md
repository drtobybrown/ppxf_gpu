# pPXF Benchmarks & Verification

This directory contains scripts to benchmark the performance of pPXF and verify its correctness across different backends (CPU, CuPy, PyTorch/MPS).

## Running the Suite

To run all benchmarks and generate a report, execute:

```bash
python3 run_all.py
```

This will:
1.  Run all `benchmark_*.py` and `verify_*.py` scripts.
2.  Capture their output and execution time.
3.  Generate `REPORT.md` with detailed results.
4.  Link any generated plots (e.g., `verification_report.png`).

## Scripts

-   `benchmark_sdss.py`: Standard SDSS kinematics benchmark (from `ppxf_example_kinematics_sdss.py`).
-   `benchmark_gas_sdss_tied.py`: Benchmark with simultaneous gas and stellar fitting (tied components).
-   `benchmark_integral_field.py`: Benchmark for IFU data processing (spatial binning).
-   `verify_results.py`: Direct comparison of CPU vs GPU results (bestfit, chi2) to ensure numerical accuracy. Generates plots.
-   `verify_gpu_fallback.py`: Verifies that pPXF gracefully handles `gpu=True` on available hardware (or falls back if needed).

## Requirements

-   `ppxf` (installed or source in parent directory)
-   `numpy`, `scipy`, `matplotlib`, `astropy`
-   `cupy` (optional, for NVIDIA GPUs)
-   `torch` (optional, for Mac MPS acceleration)
