#!/usr/bin/env python3
"""
FusionML Tri-Compute Benchmark Suite
=====================================
Comprehensive benchmarking of matmul (and other operations) across all
backend configurations on Apple Silicon.

Configurations benchmarked:
- CPU only (NumPy / Accelerate BLAS)
- GPU only (MLX)
- ANE only (CoreML)
- GPU + CPU (current FusionML dual)
- GPU + ANE
- CPU + ANE
- GPU + CPU + ANE (FusionML tri-compute)
- Pure MLX baseline
- Pure PyTorch MPS baseline (if available)

Output: JSON results + console table + optional plots
"""

import numpy as np
import time
import json
import os
import sys
import platform
import subprocess
from typing import Dict, List, Optional
from datetime import datetime

# Add parent path for fusionml imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

# Backend availability checks
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    from fusionml._metal.ane_backend import ane_matmul, HAS_COREML
except ImportError:
    HAS_COREML = False

try:
    from fusionml._metal.tri_scheduler import TriComputeScheduler
    HAS_TRI = True
except ImportError:
    HAS_TRI = False

try:
    import torch
    HAS_PYTORCH = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    HAS_PYTORCH = False

import concurrent.futures


# ============================================================================
# SYSTEM INFO
# ============================================================================

def get_system_info() -> dict:
    """Gather system hardware information."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "date": datetime.now().isoformat(),
    }
    
    # CPU info
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        )
        info["cpu_brand"] = result.stdout.strip()
    except Exception:
        info["cpu_brand"] = "Unknown"
    
    # RAM
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True,
        )
        info["ram_gb"] = int(result.stdout.strip()) // (1024**3)
    except Exception:
        info["ram_gb"] = 0
    
    # GPU (same as CPU on Apple Silicon)
    info["gpu_name"] = info["cpu_brand"]
    
    # CPU cores
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True, text=True,
        )
        info["cpu_cores"] = int(result.stdout.strip())
    except Exception:
        info["cpu_cores"] = os.cpu_count()
    
    # Backend versions
    info["backends"] = {
        "mlx": mx.__version__ if HAS_MLX else None,
        "coremltools": None,
        "pytorch": None,
    }
    try:
        import coremltools as ct
        info["backends"]["coremltools"] = ct.__version__
    except ImportError:
        pass
    if HAS_PYTORCH:
        info["backends"]["pytorch"] = torch.__version__
    
    return info


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def bench_cpu_matmul(a: np.ndarray, b: np.ndarray, iterations: int) -> Dict:
    """CPU-only matmul via NumPy (Accelerate BLAS)."""
    # Warmup
    for _ in range(3):
        _ = np.matmul(a, b)
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = np.matmul(a, b)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return {"mean_ms": np.mean(times), "median_ms": np.median(times),
            "min_ms": np.min(times), "max_ms": np.max(times), "std_ms": np.std(times)}


def bench_gpu_matmul(a: np.ndarray, b: np.ndarray, iterations: int) -> Optional[Dict]:
    """GPU-only matmul via MLX."""
    if not HAS_MLX:
        return None
    
    a_mlx = mx.array(a)
    b_mlx = mx.array(b)
    
    # Warmup
    for _ in range(3):
        c = a_mlx @ b_mlx
        mx.eval(c)
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        c = a_mlx @ b_mlx
        mx.eval(c)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return {"mean_ms": np.mean(times), "median_ms": np.median(times),
            "min_ms": np.min(times), "max_ms": np.max(times), "std_ms": np.std(times)}


def bench_ane_matmul(a: np.ndarray, b: np.ndarray, iterations: int) -> Optional[Dict]:
    """ANE-only matmul via CoreML."""
    if not HAS_COREML:
        return None
    
    # Warmup (includes model compilation)
    _ = ane_matmul(a, b, compute_units="CPU_AND_NE")
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = ane_matmul(a, b, compute_units="CPU_AND_NE")
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return {"mean_ms": np.mean(times), "median_ms": np.median(times),
            "min_ms": np.min(times), "max_ms": np.max(times), "std_ms": np.std(times)}


def bench_dual_gpu_cpu(a: np.ndarray, b: np.ndarray, iterations: int, 
                       gpu_ratio: float = 0.7) -> Optional[Dict]:
    """GPU+CPU parallel matmul (current FusionML approach)."""
    if not HAS_MLX:
        return None
    
    M = a.shape[0]
    
    def _dual_matmul(a, b):
        split = int(M * gpu_ratio)
        a_gpu, a_cpu = a[:split], a[split:]
        
        def gpu_work():
            c = mx.array(a_gpu) @ mx.array(b)
            mx.eval(c)
            return np.array(c)
        
        def cpu_work():
            return np.matmul(a_cpu, b)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(gpu_work)
            f2 = ex.submit(cpu_work)
            r1, r2 = f1.result(), f2.result()
        
        result = np.zeros((M, b.shape[1]), dtype=np.float32)
        result[:split] = r1
        result[split:] = r2
        return result
    
    # Warmup
    for _ in range(2):
        _ = _dual_matmul(a, b)
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = _dual_matmul(a, b)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return {"mean_ms": np.mean(times), "median_ms": np.median(times),
            "min_ms": np.min(times), "max_ms": np.max(times), "std_ms": np.std(times),
            "gpu_ratio": gpu_ratio}


def bench_tri_compute(a: np.ndarray, b: np.ndarray, iterations: int) -> Optional[Dict]:
    """GPU+CPU+ANE tri-compute matmul (FusionML innovation)."""
    if not HAS_TRI:
        return None
    
    scheduler = TriComputeScheduler()
    size = min(a.shape[0], a.shape[1])
    scheduler.calibrate(sizes=[size], iterations=3, verbose=False)
    
    # Warmup
    for _ in range(2):
        _ = scheduler.tri_matmul(a, b)
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = scheduler.tri_matmul(a, b)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    ratios = scheduler.get_ratios(size)
    
    return {"mean_ms": np.mean(times), "median_ms": np.median(times),
            "min_ms": np.min(times), "max_ms": np.max(times), "std_ms": np.std(times),
            "ratios": ratios}


def bench_pytorch_mps(a: np.ndarray, b: np.ndarray, iterations: int) -> Optional[Dict]:
    """PyTorch MPS baseline."""
    if not HAS_PYTORCH:
        return None
    
    a_t = torch.from_numpy(a).to("mps")
    b_t = torch.from_numpy(b).to("mps")
    
    # Warmup
    for _ in range(3):
        c = a_t @ b_t
        torch.mps.synchronize()
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        c = a_t @ b_t
        torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return {"mean_ms": np.mean(times), "median_ms": np.median(times),
            "min_ms": np.min(times), "max_ms": np.max(times), "std_ms": np.std(times)}


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_matmul_benchmarks(
    sizes: List[int] = None,
    iterations: int = 20,
    output_dir: str = None,
    verbose: bool = True,
) -> Dict:
    """
    Run comprehensive matmul benchmarks across all configurations.
    
    Args:
        sizes: Matrix sizes to benchmark
        iterations: Number of timed iterations per config
        output_dir: Directory to save results JSON
        verbose: Print progress
    
    Returns:
        Complete benchmark results dict
    """
    if sizes is None:
        sizes = [256, 512, 1024, 2048, 4096]
    
    system_info = get_system_info()
    
    if verbose:
        print("=" * 70)
        print("🔥 FusionML Tri-Compute Benchmark Suite")
        print("=" * 70)
        print(f"  CPU: {system_info['cpu_brand']}")
        print(f"  RAM: {system_info['ram_gb']}GB")
        print(f"  Backends: MLX={HAS_MLX}, CoreML={HAS_COREML}, PyTorch={HAS_PYTORCH}")
        print(f"  Sizes: {sizes}")
        print(f"  Iterations: {iterations}")
        print("=" * 70)
    
    all_results = {
        "benchmark": "tri_compute_matmul",
        "system_info": system_info,
        "date": datetime.now().isoformat(),
        "config": {"sizes": sizes, "iterations": iterations},
        "results": {},
    }
    
    for size in sizes:
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  MatMul {size}×{size}")
            print(f"{'─' * 50}")
        
        # Generate test data
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        size_results = {}
        
        # 1. CPU only
        if verbose:
            print(f"    CPU only...", end="", flush=True)
        r = bench_cpu_matmul(a, b, iterations)
        size_results["cpu"] = r
        if verbose:
            print(f" {r['median_ms']:.3f}ms")
        
        # 2. GPU only
        if HAS_MLX:
            if verbose:
                print(f"    GPU only...", end="", flush=True)
            r = bench_gpu_matmul(a, b, iterations)
            size_results["gpu"] = r
            if verbose:
                print(f" {r['median_ms']:.3f}ms")
        
        # 3. ANE only
        if HAS_COREML:
            if verbose:
                print(f"    ANE only...", end="", flush=True)
            r = bench_ane_matmul(a, b, iterations)
            size_results["ane"] = r
            if verbose:
                print(f" {r['median_ms']:.3f}ms")
        
        # 4. GPU+CPU (dual)
        if HAS_MLX:
            if verbose:
                print(f"    GPU+CPU...", end="", flush=True)
            r = bench_dual_gpu_cpu(a, b, iterations)
            size_results["gpu_cpu"] = r
            if verbose:
                print(f" {r['median_ms']:.3f}ms")
        
        # 5. Tri-compute (GPU+CPU+ANE)
        if HAS_TRI and HAS_MLX and HAS_COREML:
            if verbose:
                print(f"    TRI (GPU+CPU+ANE)...", end="", flush=True)
            r = bench_tri_compute(a, b, iterations)
            size_results["tri_compute"] = r
            if verbose:
                ratios = r.get("ratios", {})
                ratio_str = " | ".join(f"{k}={v:.0%}" for k, v in ratios.items())
                print(f" {r['median_ms']:.3f}ms [{ratio_str}]")
        
        # 6. PyTorch MPS
        if HAS_PYTORCH:
            if verbose:
                print(f"    PyTorch MPS...", end="", flush=True)
            r = bench_pytorch_mps(a, b, iterations)
            size_results["pytorch_mps"] = r
            if verbose:
                print(f" {r['median_ms']:.3f}ms")
        
        # Compute speedups
        cpu_time = size_results["cpu"]["median_ms"]
        speedups = {}
        for config, result in size_results.items():
            if config != "cpu" and result is not None:
                speedups[f"vs_cpu_{config}"] = cpu_time / result["median_ms"]
        
        best_single = cpu_time
        if "gpu" in size_results:
            best_single = min(best_single, size_results["gpu"]["median_ms"])
        if "ane" in size_results:
            best_single = min(best_single, size_results["ane"]["median_ms"])
        
        if "tri_compute" in size_results:
            speedups["tri_vs_best_single"] = best_single / size_results["tri_compute"]["median_ms"]
        if "gpu_cpu" in size_results:
            speedups["dual_vs_best_single"] = best_single / size_results["gpu_cpu"]["median_ms"]
        
        size_results["speedups"] = speedups
        all_results["results"][str(size)] = size_results
        
        if verbose and speedups:
            print(f"    Speedups: ", end="")
            parts = [f"{k}={v:.2f}x" for k, v in speedups.items()]
            print(", ".join(parts))
    
    # Summary table
    if verbose:
        print(f"\n{'=' * 70}")
        print("📊 SUMMARY (median ms)")
        print(f"{'=' * 70}")
        
        configs = ["cpu", "gpu", "ane", "gpu_cpu", "tri_compute", "pytorch_mps"]
        config_names = ["CPU", "GPU", "ANE", "GPU+CPU", "TRI", "PyTorch"]
        
        # Header
        header = f"{'Size':>6}"
        for name in config_names:
            header += f" | {name:>10}"
        print(header)
        print("-" * len(header))
        
        for size in sizes:
            row = f"{size:>6}"
            sr = all_results["results"][str(size)]
            for config in configs:
                if config in sr and sr[config] is not None:
                    row += f" | {sr[config]['median_ms']:>10.3f}"
                else:
                    row += f" | {'--':>10}"
            print(row)
        
        print(f"\n{'=' * 70}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create device-specific folder name
        cpu = system_info.get("cpu_brand", "Unknown").replace(" ", "_")
        ram = system_info.get("ram_gb", 0)
        device_dir = os.path.join(output_dir, f"{cpu}_{ram}GB")
        os.makedirs(device_dir, exist_ok=True)
        
        output_path = os.path.join(device_dir, "tri_compute.json")
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=convert)
        
        if verbose:
            print(f"\n💾 Results saved: {output_path}")
    
    return all_results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FusionML Tri-Compute Benchmarks")
    parser.add_argument("--sizes", type=str, default="256,512,1024,2048,4096",
                        help="Comma-separated matrix sizes")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Timed iterations per config")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")
    
    args = parser.parse_args()
    
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    
    results = run_matmul_benchmarks(
        sizes=sizes,
        iterations=args.iterations,
        output_dir=args.output,
        verbose=not args.quiet,
    )
