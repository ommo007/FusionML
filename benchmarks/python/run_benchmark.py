#!/usr/bin/env python3
"""
FusionML Benchmark - Main entry point
Automatically detects system specs and names result files accordingly
"""

import sys
import os
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '../../python')

# Ensure results directory exists
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_system_info() -> dict:
    """Get detailed system information"""
    info = {
        "processor": platform.processor() or "Unknown",
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "date": datetime.now().isoformat(),
    }
    
    # Try to get macOS specific info
    try:
        # Get chip name
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info["cpu_brand"] = result.stdout.strip()
        
        # Get RAM
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            ram_bytes = int(result.stdout.strip())
            info["ram_gb"] = ram_bytes // (1024**3)
        
        # Get GPU cores (Apple Silicon)
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_data = json.loads(result.stdout)
            displays = gpu_data.get("SPDisplaysDataType", [])
            if displays:
                info["gpu_name"] = displays[0].get("sppci_model", "Unknown GPU")
        
        # Get CPU cores
        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info["cpu_cores"] = int(result.stdout.strip())
        
        # Get storage
        result = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 2:
                    info["storage_total"] = parts[1]
                    
    except Exception as e:
        print(f"   Warning: Could not get all system info: {e}")
    
    return info


def generate_filename(system_info: dict) -> str:
    """Generate a descriptive filename from system info"""
    parts = []
    
    # Processor name (sanitized)
    proc = system_info.get("cpu_brand", system_info.get("processor", "unknown"))
    proc = proc.replace(" ", "_").replace("(", "").replace(")", "")
    proc = proc.replace("@", "at").replace(",", "")[:30]
    parts.append(proc)
    
    # RAM
    if "ram_gb" in system_info:
        parts.append(f"{system_info['ram_gb']}GB")
    
    # Cores
    if "cpu_cores" in system_info:
        parts.append(f"{system_info['cpu_cores']}cores")
    
    # Date
    date_str = datetime.now().strftime("%Y%m%d")
    parts.append(date_str)
    
    return "_".join(parts) + ".json"


def benchmark_matmul(sizes, iterations=30):
    """Benchmark matrix multiplication"""
    import fusionml as fml
    
    results = {}
    for size in sizes:
        a = fml.rand(size, size)
        b = fml.rand(size, size)
        a.eval()
        b.eval()
        
        # Warmup
        for _ in range(10):
            c = a @ b
            c.eval()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            c = a @ b
            c.eval()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        results[size] = {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times)
        }
    
    return results


def benchmark_training(batch_size=32, hidden=256, epochs=100):
    """Benchmark training"""
    import fusionml as fml
    
    model = fml.nn.Sequential([
        fml.nn.Linear(784, hidden),
        fml.nn.ReLU(),
        fml.nn.Linear(hidden, 10)
    ])
    
    optimizer = fml.optim.Adam(model.parameters(), lr=0.001)
    
    # Warmup
    for _ in range(5):
        x = fml.rand(batch_size, 784)
        x.requires_grad = True
        y = fml.Tensor([i % 10 for i in range(batch_size)])
        out = model(x)
        loss = fml.nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Benchmark
    times = []
    for _ in range(epochs):
        x = fml.rand(batch_size, 784)
        x.requires_grad = True
        x.eval()
        y = fml.Tensor([i % 10 for i in range(batch_size)])
        
        start = time.perf_counter()
        out = model(x)
        loss = fml.nn.functional.cross_entropy(out, y)
        loss.eval()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end = time.perf_counter()
        
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "throughput_samples_per_sec": batch_size * 1000 / (sum(times) / len(times))
    }


def main():
    print("=" * 70)
    print("🔥 FusionML Comprehensive Benchmark")
    print("=" * 70)
    
    # Get system info
    print("\n📱 Detecting system...")
    system_info = get_system_info()
    
    print(f"   Processor: {system_info.get('cpu_brand', 'Unknown')}")
    print(f"   RAM: {system_info.get('ram_gb', '?')} GB")
    print(f"   CPU Cores: {system_info.get('cpu_cores', '?')}")
    print(f"   GPU: {system_info.get('gpu_name', 'Unknown')}")
    
    # Initialize FusionML
    import fusionml as fml
    fml.init()
    
    # Run benchmarks
    print("\n📊 Running Matrix Multiplication Benchmark...")
    sizes = [256, 512, 1024, 2048, 4096]
    matmul_results = {}
    
    for size in sizes:
        result = benchmark_matmul([size])
        matmul_results[str(size)] = result[size]
        print(f"   {size}x{size}: {result[size]['mean_ms']:.2f} ms")
    
    print("\n📊 Running Training Benchmark...")
    training_results = benchmark_training()
    print(f"   MLP Training: {training_results['mean_ms']:.2f} ms/batch")
    print(f"   Throughput: {training_results['throughput_samples_per_sec']:.0f} samples/sec")
    
    # Compile results
    results = {
        "system_info": system_info,
        "matmul": matmul_results,
        "training": training_results,
        "fusionml_version": fml.__version__
    }
    
    # Save results
    filename = generate_filename(system_info)
    filepath = RESULTS_DIR / filename
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {filepath}")
    print("\n" + "=" * 70)
    
    # Generate plots
    print("📈 Generating comparison plots...")
    try:
        from plot_results import generate_plots
        generate_plots()
        print("   Plots saved to benchmarks/results/")
    except Exception as e:
        print(f"   Could not generate plots: {e}")
        print("   Run 'python plot_results.py' manually")
    
    return results


if __name__ == "__main__":
    main()
