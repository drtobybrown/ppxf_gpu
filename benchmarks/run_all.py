
import subprocess
import sys
import os
import time
from pathlib import Path
import re

def run_script(script_name, args=[]):
    print(f"Running {script_name}...")
    start = time.perf_counter()
    result = subprocess.run([sys.executable, script_name] + args, capture_output=True, text=True)
    duration = time.perf_counter() - start
    
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"  Status: {status} ({duration:.2f}s)")
    
    if status == "FAIL":
        print("  Error Output:")
        print(result.stderr)
        
    return {
        "name": script_name,
        "status": status,
        "duration": duration,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

def extract_metrics(output, name):
    metrics = []
    
    # regex for average time
    avg_match = re.search(r"Average time per fit: ([\d\.]+) s", output)
    if avg_match:
        metrics.append(f"Avg: {avg_match.group(1)}s")
        
    # regex for GPU time in verify_results
    gpu_match = re.search(r"GPU Time: ([\d\.]+) s", output)
    if gpu_match:
        metrics.append(f"GPU: {gpu_match.group(1)}s")
        
    cpu_match = re.search(r"CPU Time: ([\d\.]+) s", output)
    if cpu_match:
        metrics.append(f"CPU: {cpu_match.group(1)}s")
        
    # regex for total time
    total_match = re.search(r"Total time: ([\d\.]+) s", output)
    if total_match and "Average" not in output: # if average not found, use total
        metrics.append(f"Total: {total_match.group(1)}s")

    # regex for speedup
    if gpu_match and cpu_match:
        try:
             speedup = float(cpu_match.group(1)) / float(gpu_match.group(1))
             metrics.append(f"**Speedup: {speedup:.2f}x**")
        except:
             pass

    # fallback check logic
    if "verify_gpu_fallback" in name:
        if "SUCCESS" in output:
             metrics.append("Fallback: OK")
        else:
             metrics.append("Fallback: FAIL")

    return ", ".join(metrics)

def main():
    # Ensure current directory is benchmarks
    benchmarks_dir = Path(__file__).resolve().parent
    os.chdir(benchmarks_dir)
    
    scripts = [
        "benchmark_sdss.py",
        "benchmark_gas_sdss_tied.py",
        "benchmark_integral_field.py",
        "verify_results.py",
        "verify_gpu_fallback.py"
    ]
    
    results = []
    
    print("Starting Benchmark Suite...")
    print("="*60)
    
    for script in scripts:
        if not (benchmarks_dir / script).exists():
            print(f"Warning: {script} not found, skipping.")
            continue
            
        res = run_script(script)
        results.append(res)
        
    print("="*60)
    print("Generating Report...")
    
    report_file = "REPORT.md"
    with open(report_file, "w") as f:
        f.write("# pPXF Benchmark & Verification Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Script | Status | Duration (s) | Key Metrics |\n")
        f.write("| :--- | :---: | :---: | :--- |\n")
        
        for res in results:
            metrics = extract_metrics(res['stdout'], res['name'])
            f.write(f"| `{res['name']}` | {res['status']} | {res['duration']:.2f} | {metrics} |\n")
            
        f.write("\n## Detailed Results\n\n")
        
        for res in results:
            f.write(f"### {res['name']}\n\n")
            f.write(f"- **Status**: {res['status']}\n")
            f.write(f"- **Duration**: {res['duration']:.2f}s\n")
            
            # Embed images only if they exist and are relevant to this script
            if "verify_results.py" in res['name'] and res['status'] == "PASS":
                if os.path.exists("verification_report.png"):
                    f.write("\n#### Verification Plots\n")
                    f.write("![Verification Report](verification_report.png)\n")
                if os.path.exists("performance_comparison.png"):
                    f.write("![Performance Comparison](performance_comparison.png)\n")
            
            f.write("\n<details>\n<summary>Output Log</summary>\n\n")
            f.write("```\n")
            # Limit output length?
            out = res['stdout']
            if len(out) > 5000:
                out = out[:2500] + "\n... [truncated] ...\n" + out[-2500:]
            f.write(out)
            
            if res['stderr']:
                f.write("\nERROR OUTPUT:\n")
                f.write(res['stderr'])
            f.write("\n```\n")
            f.write("</details>\n\n")
            
    print(f"Report generated: {benchmarks_dir / report_file}")

if __name__ == "__main__":
    main()
