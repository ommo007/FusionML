#!/usr/bin/env python3
"""
FusionML MatMul Benchmark
Compares CPU, GPU, and intelligent routing performance
"""

import sys
import time
import json
import platform
from datetime import datetime

sys.path.insert(0, '../../python')
import fusionml as fml

def benchmark_matmul(size: int, iterations: int = 10) -> dict:
    """Benchmark matrix multiplication for a given size"""
    
    # Create random matrices
    a = fml.rand(size, size)
    b = fml.rand(size, size)
    
    # Warmup
    for _ in range(3):
        _ = a @ b
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = a @ b
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    return {
        "size": size,
        "iterations": iterations,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times)
    }


def main():
    print("=" * 60)
    print("🔥 FusionML MatMul Benchmark")
    print("=" * 60)
    
    fml.init()
    
    sizes = [256, 512, 1024, 2048, 4096]
    results = {"benchmarks": {}}
    
    print(f"\n{'Size':<10} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 46)
    
    for size in sizes:
        result = benchmark_matmul(size)
        results["benchmarks"][f"matmul_{size}"] = result
        print(f"{size}x{size:<6} {result['mean_ms']:<12.3f} {result['min_ms']:<12.3f} {result['max_ms']:<12.3f}")
    
    # Device info
    results["device"] = platform.processor() or "Apple Silicon"
    results["platform"] = platform.platform()
    results["date"] = datetime.now().isoformat()
    
    print("\n" + "=" * 60)
    print("📊 Results saved to benchmark_results.json")
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()
