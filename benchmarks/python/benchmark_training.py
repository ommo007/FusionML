#!/usr/bin/env python3
"""
FusionML Training Benchmark
Measures training performance for neural networks
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '../../python')
import fusionml as fml

def benchmark_training(input_size=784, hidden_size=256, output_size=10,
                       batch_size=32, num_batches=100) -> dict:
    """Benchmark neural network training"""
    
    # Build model
    model = fml.nn.Sequential([
        fml.nn.Linear(input_size, hidden_size),
        fml.nn.ReLU(),
        fml.nn.Linear(hidden_size, hidden_size),
        fml.nn.ReLU(),
        fml.nn.Linear(hidden_size, output_size)
    ])
    
    optimizer = fml.optim.Adam(model.parameters(), lr=0.001)
    
    # Warmup
    for _ in range(5):
        x = fml.rand(batch_size, input_size)
        x.requires_grad = True
        y = fml.Tensor([i % output_size for i in range(batch_size)])
        
        output = model(x)
        loss = fml.nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Benchmark
    times = []
    for _ in range(num_batches):
        x = fml.rand(batch_size, input_size)
        x.requires_grad = True
        y = fml.Tensor([i % output_size for i in range(batch_size)])
        
        start = time.perf_counter()
        
        output = model(x)
        loss = fml.nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "mean_ms_per_batch": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "throughput_samples_per_sec": batch_size * 1000 / (sum(times) / len(times))
    }


def main():
    print("=" * 60)
    print("🔥 FusionML Training Benchmark")
    print("=" * 60)
    
    fml.init()
    
    configs = [
        {"input_size": 784, "hidden_size": 128, "output_size": 10, "batch_size": 32},
        {"input_size": 784, "hidden_size": 256, "output_size": 10, "batch_size": 32},
        {"input_size": 784, "hidden_size": 512, "output_size": 10, "batch_size": 32},
        {"input_size": 1024, "hidden_size": 512, "output_size": 100, "batch_size": 64},
    ]
    
    results = {"benchmarks": {}}
    
    for i, config in enumerate(configs):
        print(f"\n📊 Config {i+1}: {config['input_size']}→{config['hidden_size']}→{config['output_size']}, batch={config['batch_size']}")
        result = benchmark_training(**config)
        results["benchmarks"][f"config_{i+1}"] = result
        print(f"   Mean: {result['mean_ms_per_batch']:.2f} ms/batch")
        print(f"   Throughput: {result['throughput_samples_per_sec']:.0f} samples/sec")
    
    results["date"] = datetime.now().isoformat()
    
    print("\n" + "=" * 60)
    print("📊 Results saved to training_results.json")
    
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
