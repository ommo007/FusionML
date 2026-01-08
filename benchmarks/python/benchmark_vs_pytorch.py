#!/usr/bin/env python3
"""
FusionML vs PyTorch Comparison Benchmark
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '../../python')

def benchmark_fusionml(sizes, iterations=10):
    """Benchmark FusionML"""
    import fusionml as fml
    fml.init()
    
    results = {}
    for size in sizes:
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
            times.append((end - start) * 1000)
        
        results[size] = sum(times) / len(times)
    
    return results


def benchmark_pytorch(sizes, iterations=10):
    """Benchmark PyTorch"""
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("   Using PyTorch MPS (Metal)")
        else:
            device = torch.device('cpu')
            print("   Using PyTorch CPU")
    except ImportError:
        print("   PyTorch not installed, skipping...")
        return None
    
    results = {}
    for size in sizes:
        a = torch.rand(size, size, device=device)
        b = torch.rand(size, size, device=device)
        
        # Warmup
        for _ in range(3):
            _ = torch.mm(a, b)
            if device.type == 'mps':
                torch.mps.synchronize()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = torch.mm(a, b)
            if device.type == 'mps':
                torch.mps.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        results[size] = sum(times) / len(times)
    
    return results


def main():
    print("=" * 60)
    print("🔥 FusionML vs PyTorch Benchmark")
    print("=" * 60)
    
    sizes = [256, 512, 1024, 2048]
    
    print("\n📊 FusionML:")
    fusionml_results = benchmark_fusionml(sizes)
    
    print("\n📊 PyTorch:")
    pytorch_results = benchmark_pytorch(sizes)
    
    print("\n" + "=" * 60)
    print(f"{'Size':<10} {'FusionML (ms)':<15} {'PyTorch (ms)':<15} {'Diff':<10}")
    print("-" * 50)
    
    comparison = {}
    for size in sizes:
        fml_time = fusionml_results[size]
        pt_time = pytorch_results[size] if pytorch_results else 0
        
        if pt_time > 0:
            diff = ((pt_time - fml_time) / pt_time) * 100
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "N/A"
        
        print(f"{size}x{size:<6} {fml_time:<15.3f} {pt_time:<15.3f} {diff_str}")
        
        comparison[size] = {
            "fusionml_ms": fml_time,
            "pytorch_ms": pt_time
        }
    
    # Save results
    results = {
        "date": datetime.now().isoformat(),
        "comparison": comparison
    }
    
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n📊 Results saved to comparison_results.json")


if __name__ == "__main__":
    main()
