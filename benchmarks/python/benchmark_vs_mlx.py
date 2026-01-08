#!/usr/bin/env python3
"""
FusionML vs MLX Benchmark
Compares performance against Apple's MLX framework
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '../../python')


def benchmark_fusionml(sizes, iterations=30):
    """Benchmark FusionML matmul"""
    import fusionml as fml
    fml.init()
    
    results = {}
    for size in sizes:
        a = fml.rand(size, size)
        b = fml.rand(size, size)
        
        # Force tensors to be ready
        a.eval()
        b.eval()
        
        # Extended warmup
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


def benchmark_mlx(sizes, iterations=30):
    """Benchmark MLX matmul"""
    try:
        import mlx.core as mx
        print("   MLX version:", mx.__version__ if hasattr(mx, '__version__') else "installed")
    except ImportError:
        print("   MLX not installed. Install with: pip install mlx")
        return None
    
    results = {}
    for size in sizes:
        a = mx.random.uniform(shape=(size, size))
        b = mx.random.uniform(shape=(size, size))
        mx.eval(a, b)
        
        # Extended warmup
        for _ in range(10):
            c = a @ b
            mx.eval(c)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            c = a @ b
            mx.eval(c)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        results[size] = {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times)
        }
    
    return results


def benchmark_training_fusionml(batch_size=32, hidden=256, epochs=100):
    """Benchmark FusionML training"""
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
        loss.eval()  # Force computation
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


def benchmark_training_mlx(batch_size=32, hidden=256, epochs=100):
    """Benchmark MLX training"""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError:
        return None
    
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, hidden)
            self.fc2 = nn.Linear(hidden, 10)
        
        def __call__(self, x):
            x = nn.relu(self.fc1(x))
            return self.fc2(x)
    
    model = MLP()
    optimizer = optim.Adam(learning_rate=0.001)
    
    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(logits, y))
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    # Warmup
    for _ in range(5):
        x = mx.random.uniform(shape=(batch_size, 784))
        y = mx.array([i % 10 for i in range(batch_size)])
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
    
    # Benchmark
    times = []
    for _ in range(epochs):
        x = mx.random.uniform(shape=(batch_size, 784))
        y = mx.array([i % 10 for i in range(batch_size)])
        
        start = time.perf_counter()
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        end = time.perf_counter()
        
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "throughput_samples_per_sec": batch_size * 1000 / (sum(times) / len(times))
    }


def main():
    print("=" * 70)
    print("🔥 FusionML vs MLX Benchmark")
    print("=" * 70)
    
    sizes = [256, 512, 1024, 2048]
    
    # MatMul benchmark
    print("\n📊 Matrix Multiplication Benchmark")
    print("-" * 50)
    
    print("   Running FusionML...")
    fml_results = benchmark_fusionml(sizes)
    
    print("   Running MLX...")
    mlx_results = benchmark_mlx(sizes)
    
    print(f"\n{'Size':<10} {'FusionML (ms)':<15} {'MLX (ms)':<15} {'Diff':<10}")
    print("-" * 50)
    
    comparison = {}
    for size in sizes:
        fml_time = fml_results[size]["mean_ms"]
        mlx_time = mlx_results[size]["mean_ms"] if mlx_results else 0
        
        if mlx_time > 0:
            if fml_time < mlx_time:
                diff = f"+{((mlx_time - fml_time) / mlx_time) * 100:.1f}%"
            else:
                diff = f"-{((fml_time - mlx_time) / fml_time) * 100:.1f}%"
        else:
            diff = "N/A"
        
        print(f"{size}x{size:<6} {fml_time:<15.3f} {mlx_time:<15.3f} {diff}")
        
        comparison[f"matmul_{size}"] = {
            "fusionml_ms": fml_time,
            "mlx_ms": mlx_time
        }
    
    # Training benchmark
    print("\n📊 Training Benchmark (MLP 784→256→10, batch=32)")
    print("-" * 50)
    
    print("   Running FusionML training...")
    fml_train = benchmark_training_fusionml()
    
    print("   Running MLX training...")
    mlx_train = benchmark_training_mlx()
    
    print(f"\n{'Framework':<15} {'ms/batch':<12} {'samples/sec':<15}")
    print("-" * 42)
    print(f"{'FusionML':<15} {fml_train['mean_ms']:<12.2f} {fml_train['throughput_samples_per_sec']:<15.0f}")
    if mlx_train:
        print(f"{'MLX':<15} {mlx_train['mean_ms']:<12.2f} {mlx_train['throughput_samples_per_sec']:<15.0f}")
    
    comparison["training"] = {
        "fusionml": fml_train,
        "mlx": mlx_train
    }
    
    # Save results
    results = {
        "date": datetime.now().isoformat(),
        "comparison": comparison
    }
    
    with open("mlx_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("📊 Results saved to mlx_comparison_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
