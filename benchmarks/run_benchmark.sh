#!/bin/bash
# ============================================================
# FusionML Cross-Device Benchmark
# ============================================================
# Run this on any Apple Silicon Mac to contribute benchmark data
# for the NeurIPS 2026 paper.
#
# Usage:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh
#
# Results will be saved to: ~/fusionml_results_<chip>.json
# Please share this file with Om.
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║     FusionML Tri-Compute Benchmark                   ║"
echo "║     NeurIPS 2026 Cross-Device Study                  ß║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Detect hardware ──────────────────────────────────────
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
CORES_PERF=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo "?")
CORES_EFF=$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo "?")
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
RAM_GB=$((RAM_BYTES / 1073741824))
OS_VER=$(sw_vers -productVersion 2>/dev/null || echo "Unknown")

echo -e "${GREEN}Hardware detected:${NC}"
echo "  Chip:     $CHIP"
echo "  P-cores:  $CORES_PERF, E-cores: $CORES_EFF"
echo "  RAM:      ${RAM_GB}GB unified memory"
echo "  macOS:    $OS_VER"
echo ""

# ── 2. Setup environment ────────────────────────────────────
WORK_DIR=$(mktemp -d)

echo -e "${YELLOW}Setting up in $WORK_DIR ...${NC}"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}Error: python3 not found. Install Python 3.9+${NC}"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python:   $PY_VER"

# Clone repo
echo -e "${YELLOW}Cloning FusionML...${NC}"
cd "$WORK_DIR"
git clone --depth 1 --branch benchmarks https://github.com/ommo007/FusionML.git 2>/dev/null
cd FusionML

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip3 install --quiet numpy mlx coremltools 2>/dev/null || {
    echo -e "${YELLOW}Trying with --user flag...${NC}"
    pip3 install --quiet --user numpy mlx coremltools 2>/dev/null || {
        echo -e "${RED}Failed to install deps. Try: pip3 install numpy mlx coremltools${NC}"
        exit 1
    }
}

# ── 3. Run benchmark ────────────────────────────────────────
echo ""
echo -e "${CYAN}Running benchmarks (this takes ~2-3 minutes)...${NC}"
echo ""

python3 -u -W ignore 2>/dev/null - <<'PYEOF'
import numpy as np
import time
import json
import sys
import os
import platform
import subprocess
import warnings
warnings.filterwarnings("ignore")

# Suppress CoreML stderr noise
import io, contextlib

sys.path.insert(0, "python")

log = lambda m: print(m, flush=True)

# ── Detect hardware ──
def get_chip():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode().strip()
    except: return platform.processor()

def get_gpu_cores():
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True, timeout=5
        )
        for line in out.split("\n"):
            if "Total Number of Cores" in line:
                return int(line.split(":")[-1].strip())
    except: pass
    return -1

chip = get_chip()
ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024**3)
gpu_cores = get_gpu_cores()

device_info = {
    "chip": chip, "ram_gb": ram_gb, "gpu_cores": gpu_cores,
    "macos": platform.mac_ver()[0], "python": platform.python_version(),
}

log(f"Device: {chip} ({ram_gb}GB RAM, {gpu_cores} GPU cores)")

import mlx.core as mx
from fusionml._metal.pipeline_scheduler import PipelineScheduler, build_resnet_block, LayerConfig
from fusionml._metal.ane_backend import ANECompiledLayer, HAS_COREML

log(f"MLX: {mx.__version__}, CoreML: {HAS_COREML}")

ITERS = 15
WARMUP = 5

def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup): fn()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(t)), "min_ms": float(np.min(t)),
            "max_ms": float(np.max(t)), "std_ms": float(np.std(t))}

results = {"device": device_info, "benchmarks": {}}

# ════════════════════════════════════════════════════════════
# BENCHMARK 1: ResNet-50 (32 conv+bn layers)
# ════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("  BENCHMARK 1: ResNet-50 (32 conv+bn layers)")
log("="*60)

sched = PipelineScheduler(verbose=False)
all_layers = []

# Build full ResNet-50
for i in range(3):
    all_layers += build_resnet_block(i, 64, 64, 56, 56)
all_layers += build_resnet_block(3, 64, 128, 56, 56, downsample=True)
for i in range(3):
    all_layers += build_resnet_block(4+i, 128, 128, 28, 28)
all_layers += build_resnet_block(7, 128, 256, 28, 28, downsample=True)
for i in range(5):
    all_layers += build_resnet_block(8+i, 256, 256, 14, 14)
all_layers += build_resnet_block(13, 256, 512, 14, 14, downsample=True)
for i in range(2):
    all_layers += build_resnet_block(14+i, 512, 512, 7, 7)

for layer in all_layers:
    sched.add_layer(layer)

log(f"  Compiling {len(all_layers)} layers (please wait)...")
t0 = time.time()

# Suppress CoreML noise during compilation
with contextlib.redirect_stderr(io.StringIO()):
    sched.compile(profile_iters=5)

compile_s = time.time() - t0
log(f"  Compiled in {compile_s:.1f}s")

# Pre-generate correct dummy inputs for each layer
layer_dummies = []
for layer in all_layers:
    layer_dummies.append(np.random.randn(*layer.input_shape).astype(np.float32) * 0.02)

# All GPU — each layer gets its own correct-shape input
def run_all_gpu():
    for i, exe in enumerate(sched._gpu_execs):
        exe.run(layer_dummies[i])
log("  Benchmarking GPU...")
gpu_r = bench(run_all_gpu)
log(f"    All GPU:  {gpu_r['median_ms']:.2f}ms")

# All ANE
def run_all_ane():
    for i, exe in enumerate(sched._ane_execs):
        exe.run(layer_dummies[i])
log("  Benchmarking ANE...")
ane_r = bench(run_all_ane)
log(f"    All ANE:  {ane_r['median_ms']:.2f}ms")

# Pipeline (uses sched.run which chains properly)
x_init = np.random.randn(1, 64, 56, 56).astype(np.float32) * 0.02
log("  Benchmarking Pipeline...")
pipe_r = bench(lambda: sched.run(x_init))
log(f"    Pipeline: {pipe_r['median_ms']:.2f}ms")

# All CPU
def run_all_cpu():
    for i, exe in enumerate(sched._cpu_execs):
        exe.run(layer_dummies[i])
log("  Benchmarking CPU...")
cpu_r = bench(run_all_cpu)
log(f"    All CPU:  {cpu_r['median_ms']:.2f}ms")

s = sched.summary()
results["benchmarks"]["resnet50"] = {
    "model": "ResNet-50", "layers": len(all_layers),
    "compile_seconds": compile_s,
    "schedule": {"gpu": s["gpu_layers"], "ane": s["ane_layers"], "cpu": s["cpu_layers"]},
    "gpu": gpu_r, "ane": ane_r, "pipeline": pipe_r, "cpu": cpu_r,
    "per_layer": s["profiles"],
}

# ════════════════════════════════════════════════════════════
# BENCHMARK 2: BERT-base (72 matmul ops)
# ════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("  BENCHMARK 2: BERT-base (72 matmul ops)")
log("="*60)

B, L, D, H, D_FF = 1, 128, 768, 12, 3072
sched2 = PipelineScheduler(verbose=False)
bert_layers = []

for li in range(12):
    for name in ["q", "k", "v"]:
        w = np.random.randn(D, D).astype(np.float32) * 0.02
        bert_layers.append(LayerConfig(f"L{li}_{name}", "matmul", (B*L, D), {"weight": w}))
    w = np.random.randn(D, D).astype(np.float32) * 0.02
    bert_layers.append(LayerConfig(f"L{li}_out", "matmul", (B*L, D), {"weight": w}))
    w1 = np.random.randn(D, D_FF).astype(np.float32) * 0.02
    bert_layers.append(LayerConfig(f"L{li}_ff1", "matmul", (B*L, D), {"weight": w1}))
    w2 = np.random.randn(D_FF, D).astype(np.float32) * 0.02
    bert_layers.append(LayerConfig(f"L{li}_ff2", "matmul", (B*L, D_FF), {"weight": w2}))

for layer in bert_layers:
    sched2.add_layer(layer)

log(f"  Compiling {len(bert_layers)} ops (please wait)...")
t0 = time.time()
with contextlib.redirect_stderr(io.StringIO()):
    sched2.compile(profile_iters=5)
compile_s2 = time.time() - t0
log(f"  Compiled in {compile_s2:.1f}s")

# Per-layer dummy inputs
bert_dummies = []
for layer in bert_layers:
    bert_dummies.append(np.random.randn(*layer.input_shape).astype(np.float32) * 0.02)

def run_bert_all(execs):
    for i, exe in enumerate(execs):
        exe.run(bert_dummies[i])

log("  Benchmarking GPU...")
gpu_r2 = bench(lambda: run_bert_all(sched2._gpu_execs))
log(f"    All GPU:  {gpu_r2['median_ms']:.2f}ms")

log("  Benchmarking ANE...")
ane_r2 = bench(lambda: run_bert_all(sched2._ane_execs))
log(f"    All ANE:  {ane_r2['median_ms']:.2f}ms")

def run_bert_pipe():
    for i, entry in enumerate(sched2.schedule):
        exe = sched2._get_executor(entry)
        exe.run(bert_dummies[i])
log("  Benchmarking Pipeline...")
pipe_r2 = bench(run_bert_pipe)
log(f"    Pipeline: {pipe_r2['median_ms']:.2f}ms")

log("  Benchmarking CPU...")
cpu_r2 = bench(lambda: run_bert_all(sched2._cpu_execs))
log(f"    All CPU:  {cpu_r2['median_ms']:.2f}ms")

s2 = sched2.summary()
results["benchmarks"]["bert_base"] = {
    "model": "BERT-base", "ops": len(bert_layers),
    "config": {"B": B, "L": L, "D": D, "H": H, "D_FF": D_FF},
    "compile_seconds": compile_s2,
    "schedule": {"gpu": s2["gpu_layers"], "ane": s2["ane_layers"], "cpu": s2["cpu_layers"]},
    "gpu": gpu_r2, "ane": ane_r2, "pipeline": pipe_r2, "cpu": cpu_r2,
}

# ════════════════════════════════════════════════════════════
# BENCHMARK 3: Matmul Scaling
# ════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("  BENCHMARK 3: Matmul Scaling")
log("="*60)

sizes = [128, 256, 512, 1024, 2048]
matmul_results = {}

for N in sizes:
    a = np.random.randn(N, N).astype(np.float32) * 0.02
    b_np = np.random.randn(N, N).astype(np.float32) * 0.02
    a_mx = mx.array(a); b_mx = mx.array(b_np)

    def gpu_mm(): r = a_mx @ b_mx; mx.eval(r)
    g = bench(gpu_mm)

    if HAS_COREML:
        with contextlib.redirect_stderr(io.StringIO()):
            mm_layer = ANECompiledLayer.matmul(M=N, K=N, weight=b_np)
        an = bench(lambda: mm_layer(a))
    else:
        an = {"median_ms": -1, "min_ms": -1, "max_ms": -1, "std_ms": -1}

    c = bench(lambda: np.matmul(a, b_np))

    matmul_results[str(N)] = {"gpu": g, "ane": an, "cpu": c}
    log(f"  {N:>5}x{N}: GPU={g['median_ms']:.2f}ms  ANE={an['median_ms']:.2f}ms  CPU={c['median_ms']:.2f}ms")

results["benchmarks"]["matmul_scaling"] = matmul_results

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

chip_clean = chip.replace(" ", "_").replace("(", "").replace(")", "")
out_name = f"fusionml_results_{chip_clean}.json"
results_dir = os.path.join(os.path.dirname(os.path.abspath("python")), "benchmarks", "results")
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, out_name)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log("\n" + "="*60)
log("  SUMMARY")
log("="*60)
log(f"  Device:     {chip}")
log(f"  ResNet-50:  GPU={gpu_r['median_ms']:.2f}ms  ANE={ane_r['median_ms']:.2f}ms  Pipeline={pipe_r['median_ms']:.2f}ms")
log(f"  BERT-base:  GPU={gpu_r2['median_ms']:.2f}ms  ANE={ane_r2['median_ms']:.2f}ms  Pipeline={pipe_r2['median_ms']:.2f}ms")

ane_r_pct = (gpu_r['median_ms'] - ane_r['median_ms']) / gpu_r['median_ms'] * 100
ane_b_pct = (gpu_r2['median_ms'] - ane_r2['median_ms']) / gpu_r2['median_ms'] * 100
log(f"  ANE vs GPU: ResNet={ane_r_pct:+.1f}%, BERT={ane_b_pct:+.1f}%")
log(f"\n  Results saved to: {out_path}")
log(f"  Please share this file with Om! 🙏")
PYEOF

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗"
echo -e "║  Done! Results saved to your home directory.         ║"
echo -e "║  Please share the JSON file with Om.                 ║"
echo -e "╚══════════════════════════════════════════════════════╝${NC}"

# Cleanup temp dir
rm -rf "$WORK_DIR"
