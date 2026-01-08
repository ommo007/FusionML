#!/usr/bin/env python3
"""
FusionML Benchmark - Main entry point
Creates device-specific folder with all benchmark results
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

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_system_info() -> dict:
    """Get detailed system information"""
    info = {
        "processor": platform.processor() or "Unknown",
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "date": datetime.now().isoformat(),
    }
    
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
        
        # Get GPU info
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


def get_device_folder_name(system_info: dict) -> str:
    """Generate a descriptive folder name from system info"""
    parts = []
    
    # Processor name (sanitized)
    proc = system_info.get("cpu_brand", system_info.get("processor", "unknown"))
    # Simplify Apple Silicon names
    if "Apple" in proc:
        for chip in ["M1 Ultra", "M1 Max", "M1 Pro", "M1", "M2 Ultra", "M2 Max", "M2 Pro", "M2", "M3 Max", "M3 Pro", "M3"]:
            if chip in proc:
                proc = chip.replace(" ", "_")
                break
    else:
        proc = proc.replace(" ", "_").replace("(", "").replace(")", "")[:20]
    parts.append(proc)
    
    # RAM
    if "ram_gb" in system_info:
        parts.append(f"{system_info['ram_gb']}GB")
    
    # Cores
    if "cpu_cores" in system_info:
        parts.append(f"{system_info['cpu_cores']}cores")
    
    # Storage
    if "storage_total" in system_info:
        storage = system_info["storage_total"].replace(" ", "")
        parts.append(storage)
    
    return "_".join(parts)


def save_result(device_folder: Path, benchmark_name: str, data: dict, system_info: dict):
    """Save benchmark result to device folder"""
    result = {
        "benchmark": benchmark_name,
        "system_info": system_info,
        "date": datetime.now().isoformat(),
        "results": data
    }
    
    filepath = device_folder / f"{benchmark_name}.json"
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"   ✓ Saved: {filepath.name}")


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
        
        results[str(size)] = {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times)
        }
    
    return results


def benchmark_training(batch_size=32, hidden=256, epochs=100):
    """Benchmark training - compares FusionML, MLX, and PyTorch"""
    results = {"fusionml": {}, "mlx": {}, "pytorch": {}}
    
    # FusionML Training
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
    
    results["fusionml"] = {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "throughput_samples_per_sec": batch_size * 1000 / (sum(times) / len(times))
    }
    
    # MLX Training
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, hidden)
                self.fc2 = nn.Linear(hidden, 10)
            
            def __call__(self, x):
                x = nn.relu(self.fc1(x))
                return self.fc2(x)
        
        mlx_model = MLP()
        mlx_optimizer = optim.Adam(learning_rate=0.001)
        
        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))
        
        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)
        
        # Warmup
        for _ in range(5):
            x = mx.random.uniform(shape=(batch_size, 784))
            y = mx.array([i % 10 for i in range(batch_size)])
            loss, grads = loss_and_grad(mlx_model, x, y)
            mlx_optimizer.update(mlx_model, grads)
            mx.eval(mlx_model.parameters())
        
        # Benchmark
        times = []
        for _ in range(epochs):
            x = mx.random.uniform(shape=(batch_size, 784))
            y = mx.array([i % 10 for i in range(batch_size)])
            
            start = time.perf_counter()
            loss, grads = loss_and_grad(mlx_model, x, y)
            mlx_optimizer.update(mlx_model, grads)
            mx.eval(mlx_model.parameters())
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
        
        results["mlx"] = {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "throughput_samples_per_sec": batch_size * 1000 / (sum(times) / len(times))
        }
        results["mlx_version"] = mx.__version__ if hasattr(mx, '__version__') else "installed"
    except ImportError:
        results["mlx"] = None
        results["mlx_error"] = "MLX not installed"
    
    # PyTorch Training
    try:
        import torch
        import torch.nn as ptnn
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            results["pytorch_device"] = "mps"
        else:
            device = torch.device('cpu')
            results["pytorch_device"] = "cpu"
        
        pt_model = ptnn.Sequential(
            ptnn.Linear(784, hidden),
            ptnn.ReLU(),
            ptnn.Linear(hidden, 10)
        ).to(device)
        
        pt_optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.001)
        pt_loss_fn = ptnn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(5):
            x = torch.rand(batch_size, 784, device=device)
            y = torch.tensor([i % 10 for i in range(batch_size)], device=device)
            pt_optimizer.zero_grad()
            out = pt_model(x)
            loss = pt_loss_fn(out, y)
            loss.backward()
            pt_optimizer.step()
            if device.type == 'mps':
                torch.mps.synchronize()
        
        # Benchmark
        times = []
        for _ in range(epochs):
            x = torch.rand(batch_size, 784, device=device)
            y = torch.tensor([i % 10 for i in range(batch_size)], device=device)
            
            start = time.perf_counter()
            pt_optimizer.zero_grad()
            out = pt_model(x)
            loss = pt_loss_fn(out, y)
            loss.backward()
            pt_optimizer.step()
            if device.type == 'mps':
                torch.mps.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
        
        results["pytorch"] = {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "throughput_samples_per_sec": batch_size * 1000 / (sum(times) / len(times))
        }
        results["pytorch_version"] = torch.__version__
    except ImportError:
        results["pytorch"] = None
        results["pytorch_error"] = "PyTorch not installed"
    
    results["config"] = {
        "batch_size": batch_size,
        "hidden_size": hidden,
        "epochs": epochs,
        "model": "MLP 784→256→10"
    }
    
    return results


def benchmark_vs_mlx(sizes, iterations=30):
    """Compare FusionML vs MLX"""
    import fusionml as fml
    
    results = {"fusionml": {}, "mlx": {}}
    
    # FusionML
    for size in sizes:
        a = fml.rand(size, size)
        b = fml.rand(size, size)
        a.eval()
        b.eval()
        
        for _ in range(10):
            c = a @ b
            c.eval()
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            c = a @ b
            c.eval()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        results["fusionml"][str(size)] = sum(times) / len(times)
    
    # MLX
    try:
        import mlx.core as mx
        
        for size in sizes:
            a = mx.random.uniform(shape=(size, size))
            b = mx.random.uniform(shape=(size, size))
            mx.eval(a, b)
            
            for _ in range(10):
                c = a @ b
                mx.eval(c)
            
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                c = a @ b
                mx.eval(c)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            results["mlx"][str(size)] = sum(times) / len(times)
        
        results["mlx_version"] = mx.__version__ if hasattr(mx, '__version__') else "installed"
    except ImportError:
        results["mlx"] = None
        results["mlx_error"] = "MLX not installed"
    
    return results


def benchmark_vs_pytorch(sizes, iterations=30):
    """Compare FusionML vs PyTorch"""
    import fusionml as fml
    
    results = {"fusionml": {}, "pytorch": {}}
    
    # FusionML
    for size in sizes:
        a = fml.rand(size, size)
        b = fml.rand(size, size)
        a.eval()
        b.eval()
        
        for _ in range(10):
            c = a @ b
            c.eval()
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            c = a @ b
            c.eval()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        results["fusionml"][str(size)] = sum(times) / len(times)
    
    # PyTorch
    try:
        import torch
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            results["pytorch_device"] = "mps"
        else:
            device = torch.device('cpu')
            results["pytorch_device"] = "cpu"
        
        for size in sizes:
            a = torch.rand(size, size, device=device)
            b = torch.rand(size, size, device=device)
            
            for _ in range(10):
                c = torch.mm(a, b)
                if device.type == 'mps':
                    torch.mps.synchronize()
            
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                c = torch.mm(a, b)
                if device.type == 'mps':
                    torch.mps.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            results["pytorch"][str(size)] = sum(times) / len(times)
        
        results["pytorch_version"] = torch.__version__
    except ImportError:
        results["pytorch"] = None
        results["pytorch_error"] = "PyTorch not installed"
    
    return results


def main():
    print("=" * 70)
    print("🔥 FusionML Comprehensive Benchmark Suite")
    print("=" * 70)
    
    # Get system info
    print("\n📱 Detecting system...")
    system_info = get_system_info()
    
    print(f"   Processor: {system_info.get('cpu_brand', 'Unknown')}")
    print(f"   RAM: {system_info.get('ram_gb', '?')} GB")
    print(f"   CPU Cores: {system_info.get('cpu_cores', '?')}")
    print(f"   GPU: {system_info.get('gpu_name', 'Unknown')}")
    print(f"   Storage: {system_info.get('storage_total', 'Unknown')}")
    
    # Create device-specific folder
    folder_name = get_device_folder_name(system_info)
    device_folder = RESULTS_DIR / folder_name
    device_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Results folder: results/{folder_name}/")
    
    # Initialize FusionML
    import fusionml as fml
    fml.init()
    
    sizes = [256, 512, 1024, 2048, 4096]
    
    # 1. MatMul Benchmark
    print("\n" + "=" * 70)
    print("📊 1/4: Matrix Multiplication Benchmark")
    print("=" * 70)
    
    matmul_results = benchmark_matmul(sizes)
    for size in sizes:
        print(f"   {size}x{size}: {matmul_results[str(size)]['mean_ms']:.2f} ms")
    save_result(device_folder, "matmul", matmul_results, system_info)
    
    # 2. Training Benchmark
    print("\n" + "=" * 70)
    print("📊 2/4: Training Benchmark (MLP 784→256→10)")
    print("=" * 70)
    
    training_results = benchmark_training()
    
    print(f"   {'Framework':<12} {'ms/batch':<12} {'samples/sec':<15}")
    print("   " + "-" * 40)
    
    fml = training_results.get("fusionml", {})
    if fml:
        print(f"   {'FusionML':<12} {fml['mean_ms']:<12.2f} {fml['throughput_samples_per_sec']:<15.0f}")
    
    mlx = training_results.get("mlx", {})
    if mlx and isinstance(mlx, dict) and "mean_ms" in mlx:
        print(f"   {'MLX':<12} {mlx['mean_ms']:<12.2f} {mlx['throughput_samples_per_sec']:<15.0f}")
    
    pt = training_results.get("pytorch", {})
    if pt and isinstance(pt, dict) and "mean_ms" in pt:
        print(f"   {'PyTorch':<12} {pt['mean_ms']:<12.2f} {pt['throughput_samples_per_sec']:<15.0f}")
    
    save_result(device_folder, "training", training_results, system_info)
    
    # 3. MLX Comparison
    print("\n" + "=" * 70)
    print("📊 3/4: FusionML vs MLX Comparison")
    print("=" * 70)
    
    mlx_results = benchmark_vs_mlx(sizes)
    if mlx_results.get("mlx"):
        for size in sizes:
            fml_t = mlx_results["fusionml"][str(size)]
            mlx_t = mlx_results["mlx"][str(size)]
            diff = ((mlx_t - fml_t) / mlx_t) * 100
            winner = "✅ FusionML" if diff > 0 else "MLX"
            print(f"   {size}x{size}: FusionML={fml_t:.2f}ms, MLX={mlx_t:.2f}ms ({winner})")
    else:
        print("   ⚠️  MLX not installed")
    save_result(device_folder, "vs_mlx", mlx_results, system_info)
    
    # 4. PyTorch Comparison
    print("\n" + "=" * 70)
    print("📊 4/4: FusionML vs PyTorch Comparison")
    print("=" * 70)
    
    pytorch_results = benchmark_vs_pytorch(sizes)
    if pytorch_results.get("pytorch"):
        for size in sizes:
            fml_t = pytorch_results["fusionml"][str(size)]
            pt_t = pytorch_results["pytorch"][str(size)]
            diff = ((pt_t - fml_t) / pt_t) * 100
            winner = "✅ FusionML" if diff > 0 else "PyTorch"
            print(f"   {size}x{size}: FusionML={fml_t:.2f}ms, PyTorch={pt_t:.2f}ms ({winner})")
    else:
        print("   ⚠️  PyTorch not installed")
    save_result(device_folder, "vs_pytorch", pytorch_results, system_info)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL BENCHMARKS COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved in: results/{folder_name}/")
    print(f"   ├── matmul.json")
    print(f"   ├── training.json")
    print(f"   ├── vs_mlx.json")
    print(f"   └── vs_pytorch.json")
    print("\nTo submit: Create a PR with your results folder!")
    
    return device_folder


if __name__ == "__main__":
    main()
