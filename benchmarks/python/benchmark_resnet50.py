"""
Full ResNet-50 Benchmark: GPU vs ANE vs Pipeline
================================================
Tests all 16 residual blocks of ResNet-50 across GPU, ANE, CPU,
and the pipeline scheduler.
"""
import numpy as np
import time
import sys
import json

sys.path.insert(0, "python")
import warnings; warnings.filterwarnings("ignore")

from fusionml._metal.pipeline_scheduler import (
    PipelineScheduler, LayerConfig, build_resnet_block
)
import mlx.core as mx

log = lambda m: print(m, flush=True)

ITERS = 12
WARMUP = 5

log("=" * 65)
log("  RESNET-50 BENCHMARK: Full Model (16 Residual Blocks)")
log("=" * 65)

# ResNet-50 architecture:
# Stage 1: 3 blocks, 64->256, 56x56  (using basic blocks for clarity)
# Stage 2: 4 blocks, 128->512, 28x28
# Stage 3: 6 blocks, 256->1024, 14x14
# Stage 4: 3 blocks, 512->2048, 7x7
#
# Simplified to BasicBlocks (2 convs each) at each stage:
# Stage 1: 3 blocks, 64 ch, 56x56
# Stage 2: 4 blocks, 128 ch, 28x28
# Stage 3: 6 blocks, 256 ch, 14x14
# Stage 4: 3 blocks, 512 ch, 7x7

sched = PipelineScheduler(verbose=True)
all_layers = []

log("\nBuilding ResNet-50 layers...")

# Stage 1: 3 blocks at 64ch 56x56
for i in range(3):
    layers = build_resnet_block(
        block_idx=i, in_channels=64, out_channels=64,
        height=56, width=56, batch=1
    )
    all_layers.extend(layers)

# Stage 2: 4 blocks at 128ch 28x28 (first block downsamples)
layers = build_resnet_block(
    block_idx=3, in_channels=64, out_channels=128,
    height=56, width=56, batch=1, downsample=True
)
all_layers.extend(layers)
for i in range(3):
    layers = build_resnet_block(
        block_idx=4+i, in_channels=128, out_channels=128,
        height=28, width=28, batch=1
    )
    all_layers.extend(layers)

# Stage 3: 6 blocks at 256ch 14x14
layers = build_resnet_block(
    block_idx=7, in_channels=128, out_channels=256,
    height=28, width=28, batch=1, downsample=True
)
all_layers.extend(layers)
for i in range(5):
    layers = build_resnet_block(
        block_idx=8+i, in_channels=256, out_channels=256,
        height=14, width=14, batch=1
    )
    all_layers.extend(layers)

# Stage 4: 3 blocks at 512ch 7x7
layers = build_resnet_block(
    block_idx=13, in_channels=256, out_channels=512,
    height=14, width=14, batch=1, downsample=True
)
all_layers.extend(layers)
for i in range(2):
    layers = build_resnet_block(
        block_idx=14+i, in_channels=512, out_channels=512,
        height=7, width=7, batch=1
    )
    all_layers.extend(layers)

log(f"Total layers: {len(all_layers)}")

for layer in all_layers:
    sched.add_layer(layer)

log("\nCompiling and profiling all layers...")
t_compile_start = time.time()
sched.compile(profile_iters=5)
compile_time = time.time() - t_compile_start
log(f"\nTotal compile time: {compile_time:.1f}s")

# ================================================================
# BENCHMARK
# ================================================================

x = np.random.randn(1, 64, 56, 56).astype(np.float32) * 0.02

def bench(fn, label):
    for _ in range(WARMUP):
        fn(x)
    t = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) * 1000)
    ms = float(np.median(t))
    log(f"  {label:<25} {ms:>8.2f}ms")
    return ms

log("\nBenchmarking...")

# Pipeline scheduler
pipe_ms = bench(lambda xi: sched.run(xi), "Pipeline (scheduled)")

# All GPU
def run_all_gpu(xi):
    r = xi
    for exe in sched._gpu_execs:
        r = exe.run(r)
    return r
gpu_ms = bench(run_all_gpu, "All GPU (MLX)")

# All ANE
def run_all_ane(xi):
    r = xi
    for exe in sched._ane_execs:
        r = exe.run(r)
    return r
ane_ms = bench(run_all_ane, "All ANE (CoreML)")

# All CPU
def run_all_cpu(xi):
    r = xi
    for exe in sched._cpu_execs:
        r = exe.run(r)
    return r
cpu_ms = bench(run_all_cpu, "All CPU")

# ================================================================
# RESULTS
# ================================================================

log("")
log("=" * 65)
log("  RESNET-50 RESULTS")
log("=" * 65)
log(f"  Total layers: {len(all_layers)}")
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

# Per-layer breakdown
log("\n  Per-layer schedule:")
for i, entry in enumerate(sched.schedule):
    layer = sched.layers[i]
    p = sched.profiles[i]
    assigned_t = getattr(p, f"{entry.backend}_ms")
    log(f"    [{i:2d}] {layer.name:<30} -> {entry.backend.upper():<4} "
        f"({assigned_t:.2f}ms)  "
        f"[GPU={p.gpu_ms:.2f} ANE={p.ane_ms:.2f} CPU={p.cpu_ms:.2f}]")

# Save results
results = {
    "model": "ResNet-50 (BasicBlock)",
    "total_layers": len(all_layers),
    "schedule": {
        "gpu_layers": s["gpu_layers"],
        "ane_layers": s["ane_layers"],
        "cpu_layers": s["cpu_layers"],
    },
    "latency_ms": {
        "cpu": cpu_ms,
        "gpu": gpu_ms,
        "ane": ane_ms,
        "pipeline": pipe_ms,
    },
    "speedup_vs_gpu": {
        "ane": (gpu_ms - ane_ms) / gpu_ms * 100,
        "pipeline": (gpu_ms - pipe_ms) / gpu_ms * 100,
    },
    "winner": best[0],
    "per_layer": s["profiles"],
}

out_path = "benchmarks/results/resnet50_pipeline.json"
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults saved to {out_path}")
