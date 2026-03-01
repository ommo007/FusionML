#!/usr/bin/env python3
"""
FusionML Ablation Study — NeurIPS 2026
========================================
Measures the contribution of each compute unit (GPU, CPU, ANE)
and validates the adaptive scheduler against baselines.

Configurations tested:
  1. CPU-only          — NumPy / Accelerate BLAS
  2. GPU-only          — MLX on Apple GPU
  3. GPU + CPU         — Dual-compute, adaptive split
  4. GPU + CPU + ANE   — Full Tri-Compute (proposed)
  5. Random routing    — Tri-Compute with random ratios (ablation baseline)
  6. Equal split       — Fixed 33/33/33 ratio

Output: JSON results + console table for each matrix size.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

from fusionml._metal.tri_scheduler import TriComputeScheduler, HAS_MLX, HAS_COREML

# Suppress NumPy warnings and force unbuffered stdout
np.seterr(all='ignore')

def log(msg=""):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZES = [256, 512, 1024, 2048, 4096]
ITERATIONS = 20
WARMUP = 5

CONFIGS = {
    "CPU-only": dict(enable_gpu=False, enable_cpu=True, enable_ane=False,
                     random_routing=False),
    "GPU-only": dict(enable_gpu=True, enable_cpu=False, enable_ane=False,
                     random_routing=False),
    "GPU+CPU": dict(enable_gpu=True, enable_cpu=True, enable_ane=False,
                    random_routing=False),
    "Tri-Compute": dict(enable_gpu=True, enable_cpu=True, enable_ane=True,
                        random_routing=False),
    "Random-Route": dict(enable_gpu=True, enable_cpu=True, enable_ane=True,
                         random_routing=True),
}

# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def time_matmul(scheduler: TriComputeScheduler, size: int,
                iterations: int = ITERATIONS, warmup: int = WARMUP) -> Dict:
    """Time a square matmul through the scheduler."""
    a = np.random.randn(size, size).astype(np.float32) * 0.02
    b = np.random.randn(size, size).astype(np.float32) * 0.02

    # Warmup
    for _ in range(warmup):
        scheduler.tri_matmul(a, b)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        scheduler.tri_matmul(a, b)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "std_ms": float(np.std(times)),
    }


def run_ablation() -> Dict:
    """Run all ablation configurations across all sizes."""
    log("=" * 70)
    log("  FusionML Ablation Study — NeurIPS 2026")
    log("=" * 70)
    log(f"  Backends available: CPU=True, GPU(MLX)={HAS_MLX}, ANE(CoreML)={HAS_COREML}")
    log(f"  Sizes: {SIZES}")
    log(f"  Iterations: {ITERATIONS}  |  Warmup: {WARMUP}")
    log("=" * 70)

    # Warmup ANE (CoreML compilation is slow on first call)
    if HAS_COREML:
        log("\n  🧠 Warming up ANE (CoreML compilation, may take 30-60s)...")
        from fusionml._metal.ane_backend import warmup_ane
        warmup_ane(sizes=[256, 512, 1024])
        log("  ANE warmup complete.")

    all_results = {}

    for size in SIZES:
        log(f"\n{'─' * 70}")
        log(f"  Matrix Size: {size}×{size}")
        log(f"{'─' * 70}")
        log(f"  {'Config':<16} | {'Median (ms)':>12} | {'Std (ms)':>10} | {'Speedup':>8}")
        log(f"  {'-'*16}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")

        size_results = {}
        cpu_baseline = None

        for config_name, flags in CONFIGS.items():
            # Skip configs that require unavailable backends
            if flags["enable_gpu"] and not HAS_MLX:
                if not flags["enable_cpu"]:
                    log(f"  {config_name:<16} | {'SKIPPED (no MLX)':>35}")
                    continue
            if flags["enable_ane"] and not HAS_COREML:
                # Still run but ANE will be disabled internally
                pass

            log(f"  {config_name:<16} | benchmarking...", )

            scheduler = TriComputeScheduler(
                auto_calibrate=True,
                **flags
            )
            # Calibrate with all enabled backends
            scheduler.calibrate(sizes=[size], verbose=False)

            stats = time_matmul(scheduler, size)
            size_results[config_name] = stats

            if config_name == "CPU-only":
                cpu_baseline = stats["median_ms"]

            speedup = cpu_baseline / stats["median_ms"] if cpu_baseline else 1.0
            # Overwrite the "benchmarking..." line
            print(f"\r  {config_name:<16} | {stats['median_ms']:>10.3f}ms | "
                  f"{stats['std_ms']:>8.3f}ms | {speedup:>6.2f}×", flush=True)

        all_results[str(size)] = size_results

        # Print winner
        if size_results:
            winner = min(size_results.items(), key=lambda kv: kv[1]["median_ms"])
            log(f"\n  🏆 Winner: {winner[0]} ({winner[1]['median_ms']:.3f}ms)")

    return all_results


def print_summary(results: Dict):
    """Print a final summary table."""
    print(f"\n\n{'=' * 70}")
    print("  ABLATION SUMMARY — Median Latency (ms)")
    print(f"{'=' * 70}")

    configs = list(CONFIGS.keys())
    header = f"  {'Size':>6}"
    for c in configs:
        header += f" | {c:>14}"
    print(header)
    print(f"  {'-' * 6}" + "".join(f"-+-{'-' * 14}" for _ in configs))

    for size in SIZES:
        row = f"  {size:>6}"
        size_key = str(size)
        for c in configs:
            if size_key in results and c in results[size_key]:
                val = results[size_key][c]["median_ms"]
                row += f" | {val:>12.3f}ms"
            else:
                row += f" | {'N/A':>14}"
        print(row)


if __name__ == "__main__":
    results = run_ablation()
    print_summary(results)

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "ablation")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"ablation_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "sizes": SIZES,
            "iterations": ITERATIONS,
            "has_mlx": HAS_MLX,
            "has_coreml": HAS_COREML,
            "results": results,
        }, f, indent=2)

    print(f"\n💾 Results saved to {out_path}")
