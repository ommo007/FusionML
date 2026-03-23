#!/usr/bin/env python3
"""
FusionML End-to-End Pipeline Benchmark
======================================
Benchmarks real models (ResNet-50 and BERT-base) across different
architectural scheduling strategies.

Measures throughput (images/sec or tokens/sec) and latency for:
1. GPU only (MLX execution of the pipeline)
2. CPU only (NumPy / Torch CPU execution)
3. ANE only (CoreML static execution)
4. FusionML Auto-Pipeline (Heterogeneous overlap)
"""

import sys
import os
import time
import json
import numpy as np

# Add parent path for fusionml imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from fusionml.models.resnet50 import build_fusionml_resnet50_pipeline
from fusionml.models.bert_base import build_fusionml_bert_pipeline
from fusionml._metal.pipeline_scheduler import PipelineScheduler

def get_system_info():
    import platform, subprocess
    info = {"machine": platform.machine()}
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        info["cpu_brand"] = r.stdout.strip()
    except:
        info["cpu_brand"] = "Unknown"
    return info

def run_schedule(sched: PipelineScheduler, backend_policy: str, inputs: list) -> float:
    """
    Run the pipeline with a specific backend policy.
    backend_policy: "gpu", "ane", "cpu", or "auto"
    Returns the elapsed time for the batch in ms.
    """
    n_layers = len(sched.layers)
    
    # Auto uses the profiled best backends plus overlapping
    if backend_policy != "auto":
        for entry in sched.schedule:
            entry.backend = backend_policy
            
    # Warmup
    _ = sched.run_pipelined_batch(inputs[:min(2, len(inputs))])
    
    # Timing
    t0 = time.perf_counter()
    if backend_policy == "auto":
        # True pipeline utilizes threading overlap
        _ = sched.run_pipelined_batch(inputs)
    else:
        # Pipelined batch without overlapping hardware (since it's all on same backend)
        _ = sched.run_pipelined_batch(inputs)
    t1 = time.perf_counter()
    
    return (t1 - t0) * 1000.0


def bench_model(model_name: str, batch_count: int = 10):
    print(f"\n{'='*60}")
    print(f" End-to-End Benchmark: {model_name}")
    print(f"{'='*60}\n")
    
    if model_name == "resnet50":
        sched = build_fusionml_resnet50_pipeline()
        input_shape = (1, 3, 224, 224)
        inputs = [np.random.randn(*input_shape).astype(np.float32) for _ in range(batch_count)]
        metric = "images/sec"
    elif model_name == "bert_base":
        seq_len = 128
        sched = build_fusionml_bert_pipeline(seq_len=seq_len)
        input_shape = (1, seq_len)
        inputs = [np.random.randint(0, 30000, input_shape).astype(np.int32) for _ in range(batch_count)]
        metric = "iterations/sec"
    else:
        raise ValueError()

    print(f"Compiling profiles for {len(sched.layers)} layers...")
    sched.compile(profile_iters=2)
    
    results = {}
    
    # Run the backends
    policies = ["gpu", "ane", "cpu", "auto"]
    
    for policy in policies:
        print(f"Running {policy.upper()} policy...", end="", flush=True)
        try:
            total_ms = run_schedule(sched, policy, inputs)
            ms_per_item = total_ms / batch_count
            throughput = 1000.0 / ms_per_item
            print(f" {ms_per_item:.1f}ms per item -> {throughput:.1f} {metric}")
            
            # Record assignment
            if policy == "auto":
                counts = {"gpu": 0, "ane": 0, "cpu": 0}
                for e in sched.schedule:
                    counts[e.backend] += 1
            else:
                counts = {policy: len(sched.layers)}
                
            results[policy] = {
                "ms_per_item": ms_per_item,
                "throughput": throughput,
                "layer_assignment": counts
            }
        except Exception as e:
            print(f" FAILED: {e}")
            results[policy] = None

    return results


def main():
    info = get_system_info()
    print("System:", info["cpu_brand"])
    
    all_res = {"system": info, "benchmarks": {}}
    
    # 1. ResNet-50
    try:
        res = bench_model("resnet50", batch_count=20)
        all_res["benchmarks"]["resnet50"] = res
    except Exception as e:
        print("ResNet50 Failed:", e)

    # 2. BERT-base
    try:
        res = bench_model("bert_base", batch_count=20)
        all_res["benchmarks"]["bert_base"] = res
    except Exception as e:
        print("BERT Failed:", e)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"e2e_results_{info['cpu_brand'].replace(' ', '_')}.json")
    
    with open(out_file, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nSaved E2E results to {out_file}")

if __name__ == "__main__":
    main()
