"""
BERT-Base Inference Benchmark: GPU vs ANE vs Pipeline
====================================================
Tests 12 transformer encoder layers. Each layer has 4 matmul ops,
all with (B*L, D) -> (B*L, D) shape (attention done inline on GPU).
"""
import numpy as np
import time
import sys
import json

sys.path.insert(0, "python")
import warnings; warnings.filterwarnings("ignore")

from fusionml._metal.pipeline_scheduler import (
    PipelineScheduler, LayerConfig
)
from fusionml._metal.ane_backend import ANECompiledLayer, HAS_COREML
import mlx.core as mx

log = lambda m: print(m, flush=True)

ITERS = 12
WARMUP = 5

# BERT-base config
B = 1; L = 128; D = 768; H = 12; D_FF = 3072

log("=" * 65)
log("  BERT-BASE BENCHMARK: 12 Transformer Layers")
log(f"  Config: B={B}, L={L}, D={D}, H={H}, D_FF={D_FF}")
log("=" * 65)

# For pipeline scheduling, we treat each matmul as an independent
# (B*L, D) -> (B*L, D) op. In reality BERT has attention between
# QKV and out_proj, but for benchmarking matmul throughput this
# accurately measures the compute-dominant operations.
#
# Each transformer layer = 4 matmuls:
#   1. QKV: (128, 768) @ (768, 768) -> (128, 768) [x3, fused for benchmark]
#   2. Attn output: (128, 768) @ (768, 768) -> (128, 768)
#   3. FFN up: (128, 768) @ (768, 3072) -> (128, 3072)
#   4. FFN down: (128, 3072) @ (3072, 768) -> (128, 768)

sched = PipelineScheduler(verbose=True)
all_layers = []

log("\nBuilding 12 transformer layers (48 matmul ops)...")

for layer_idx in range(12):
    # 3 matmuls for Q, K, V (same shape)
    for qkv_name in ["q", "k", "v"]:
        w = np.random.randn(D, D).astype(np.float32) * 0.02
        all_layers.append(LayerConfig(
            name=f"L{layer_idx}_{qkv_name}_proj",
            op_type="matmul",
            input_shape=(B * L, D),
            weights={"weight": w},
        ))

    # Output projection
    w_out = np.random.randn(D, D).astype(np.float32) * 0.02
    all_layers.append(LayerConfig(
        name=f"L{layer_idx}_out_proj",
        op_type="matmul",
        input_shape=(B * L, D),
        weights={"weight": w_out},
    ))

    # FFN up (D -> D_FF)
    w_ff1 = np.random.randn(D, D_FF).astype(np.float32) * 0.02
    all_layers.append(LayerConfig(
        name=f"L{layer_idx}_ffn_up",
        op_type="matmul",
        input_shape=(B * L, D),
        weights={"weight": w_ff1},
    ))

    # FFN down (D_FF -> D)
    w_ff2 = np.random.randn(D_FF, D).astype(np.float32) * 0.02
    all_layers.append(LayerConfig(
        name=f"L{layer_idx}_ffn_down",
        op_type="matmul",
        input_shape=(B * L, D_FF),
        weights={"weight": w_ff2},
    ))

log(f"Total ops: {len(all_layers)} matmuls")

for layer in all_layers:
    sched.add_layer(layer)

log("\nCompiling and profiling...")
t_start = time.time()
sched.compile(profile_iters=5)
compile_s = time.time() - t_start
log(f"Compile time: {compile_s:.1f}s")

# ================================================================
# BENCHMARK — Run all ops sequentially (as independent matmuls)
# ================================================================

x_d = np.random.randn(B * L, D).astype(np.float32) * 0.02
x_ff = np.random.randn(B * L, D_FF).astype(np.float32) * 0.02

def bench_all(executor_list, label):
    """Benchmark running all ops with correct input shapes."""
    def run():
        for i, exe in enumerate(executor_list):
            layer = all_layers[i]
            # Select correct input shape
            if layer.input_shape[-1] == D_FF:
                exe.run(x_ff)
            else:
                exe.run(x_d)
    for _ in range(WARMUP):
        run()
    t = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run()
        t.append((time.perf_counter() - t0) * 1000)
    ms = float(np.median(t))
    log(f"  {label:<25} {ms:>8.2f}ms")
    return ms

log("\nBenchmarking total throughput...")

# All GPU
gpu_ms = bench_all(sched._gpu_execs, "All GPU (MLX)")

# All ANE
ane_ms = bench_all(sched._ane_execs, "All ANE (CoreML)")

# Pipeline (scheduled backends)
def run_scheduled():
    for i, entry in enumerate(sched.schedule):
        layer = all_layers[i]
        exe = sched._get_executor(entry)
        if layer.input_shape[-1] == D_FF:
            exe.run(x_ff)
        else:
            exe.run(x_d)

for _ in range(WARMUP): run_scheduled()
t = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    run_scheduled()
    t.append((time.perf_counter() - t0) * 1000)
pipe_ms = float(np.median(t))
log(f"  {'Pipeline (scheduled)':<25} {pipe_ms:>8.2f}ms")

# All CPU
cpu_ms = bench_all(sched._cpu_execs, "All CPU")

# ================================================================
# RESULTS
# ================================================================

log("")
log("=" * 65)
log(f"  BERT-BASE RESULTS (B={B}, L={L}, 12 layers, 72 matmul ops)")
log("=" * 65)
s = sched.summary()
log(f"  Schedule: GPU={s['gpu_layers']}, ANE={s['ane_layers']}, CPU={s['cpu_layers']}")
log("")
log(f"  All CPU:      {cpu_ms:>8.2f}ms")
log(f"  All GPU:      {gpu_ms:>8.2f}ms  ({(cpu_ms-gpu_ms)/cpu_ms*100:>+.1f}% vs CPU)")
log(f"  All ANE:      {ane_ms:>8.2f}ms  ({(cpu_ms-ane_ms)/cpu_ms*100:>+.1f}% vs CPU)")
log(f"  Pipeline:     {pipe_ms:>8.2f}ms  ({(gpu_ms-pipe_ms)/gpu_ms*100:>+.1f}% vs GPU)")

vals = [("CPU", cpu_ms), ("GPU", gpu_ms), ("ANE", ane_ms), ("Pipeline", pipe_ms)]
best = min(vals, key=lambda v: v[1])
log(f"\n  WINNER: {best[0]} ({best[1]:.2f}ms)")

# Save
results = {
    "model": "BERT-base",
    "config": {"B": B, "L": L, "D": D, "H": H, "D_FF": D_FF},
    "total_ops": len(all_layers),
    "schedule": {"gpu": s["gpu_layers"], "ane": s["ane_layers"], "cpu": s["cpu_layers"]},
    "latency_ms": {"cpu": cpu_ms, "gpu": gpu_ms, "ane": ane_ms, "pipeline": pipe_ms},
    "speedup_vs_gpu": {
        "ane": (gpu_ms - ane_ms) / gpu_ms * 100,
        "pipeline": (gpu_ms - pipe_ms) / gpu_ms * 100,
    },
    "winner": best[0],
}

import os
out = "benchmarks/results/bert_pipeline.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults saved to {out}")
