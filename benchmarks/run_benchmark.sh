#!/bin/bash
# ============================================================
# FusionML Cross-Device Benchmark
# ============================================================
# Run this on any Apple Silicon Mac to contribute benchmark data.
#
# Usage:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh
#
# Results saved to: ~/fusionml_results_<chip>.json
#   + benchmarks/results/ if running from repo
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║     FusionML Tri-Compute Benchmark                  ║"
echo "║     NeurIPS 2026 Cross-Device Study                 ║"
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
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "$RESULTS_DIR" 2>/dev/null || true
export RESULTS_DIR

WORK_DIR=$(mktemp -d)

echo -e "${YELLOW}Setting up in $WORK_DIR ...${NC}"

if ! command -v python3 &>/dev/null; then
    echo -e "${RED}Error: python3 not found. Install Python 3.9+${NC}"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python:   $PY_VER"

echo -e "${YELLOW}Cloning FusionML...${NC}"
cd "$WORK_DIR"
git clone --depth 1 --branch benchmarks https://github.com/ommo007/FusionML.git 2>/dev/null
cd FusionML

echo -e "${YELLOW}Installing dependencies...${NC}"
pip3 install --quiet numpy mlx coremltools 2>/dev/null || \
pip3 install --quiet --user numpy mlx coremltools 2>/dev/null || \
pip3 install --quiet --break-system-packages numpy mlx coremltools 2>/dev/null || {
    echo -e "${RED}Failed to install deps. Try creating a venv:${NC}"
    echo "  python3 -m venv ~/bench_env && source ~/bench_env/bin/activate"
    echo "  pip install numpy mlx coremltools"
    echo "  Then re-run this script."
    exit 1
}

# ── 3. Run benchmark ────────────────────────────────────────
echo ""
echo -e "${CYAN}Running benchmarks (this takes ~2-3 minutes)...${NC}"
echo ""

python3 -u -W ignore - <<'PYEOF'
import numpy as np, time, json, sys, os, platform, subprocess, warnings, gc
import io, contextlib
warnings.filterwarnings("ignore")
sys.path.insert(0, "python")

log = lambda m: print(m, flush=True)

# ── Hardware detection ──
def get_chip():
    try: return subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"]).decode().strip()
    except: return platform.processor()

def get_gpu_cores():
    try:
        out = subprocess.check_output(["system_profiler","SPDisplaysDataType"],text=True,timeout=5)
        for l in out.split("\n"):
            if "Total Number of Cores" in l: return int(l.split(":")[-1].strip())
    except: pass
    return -1

chip = get_chip()
ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024**3)
gpu_cores = get_gpu_cores()
device_info = {"chip": chip, "ram_gb": ram_gb, "gpu_cores": gpu_cores,
               "macos": platform.mac_ver()[0], "python": platform.python_version()}

log(f"Device: {chip} ({ram_gb}GB RAM, {gpu_cores} GPU cores)")

import mlx.core as mx
from fusionml._metal.pipeline_scheduler import PipelineScheduler, build_resnet_block, LayerConfig
from fusionml._metal.ane_backend import ANECompiledLayer, HAS_COREML

log(f"MLX: {mx.__version__}, CoreML: {HAS_COREML}")

ITERS = 15; WARMUP = 5

def bench(fn):
    for _ in range(WARMUP): fn()
    t = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter()-t0)*1000)
    return {"median_ms": round(float(np.median(t)),2), "min_ms": round(float(np.min(t)),2),
            "max_ms": round(float(np.max(t)),2), "std_ms": round(float(np.std(t)),2)}

def save_results(results):
    """Save results to home dir + RESULTS_DIR."""
    chip_clean = chip.replace(" ","_").replace("(","").replace(")","")
    name = f"fusionml_results_{chip_clean}.json"
    # Always home dir
    home = os.path.join(os.path.expanduser("~"), name)
    with open(home, "w") as f: json.dump(results, f, indent=2)
    saved = home
    # Also RESULTS_DIR if persistent
    rd = os.environ.get("RESULTS_DIR", "")
    if rd and "/tmp" not in rd and "/var/folders" not in rd:
        os.makedirs(rd, exist_ok=True)
        alt = os.path.join(rd, name)
        with open(alt, "w") as f: json.dump(results, f, indent=2)
        saved = alt
    return saved

results = {"device": device_info, "benchmarks": {}, "timestamp": ""}

# ════════════════════════════════════════════════════════════
# BENCHMARK 1: ResNet-50
# ════════════════════════════════════════════════════════════
try:
    log("\n" + "="*60)
    log("  BENCHMARK 1: ResNet-50 (32 conv+bn layers)")
    log("="*60)

    sched = PipelineScheduler(verbose=False)
    all_layers = []
    for i in range(3): all_layers += build_resnet_block(i, 64, 64, 56, 56)
    all_layers += build_resnet_block(3, 64, 128, 56, 56, downsample=True)
    for i in range(3): all_layers += build_resnet_block(4+i, 128, 128, 28, 28)
    all_layers += build_resnet_block(7, 128, 256, 28, 28, downsample=True)
    for i in range(5): all_layers += build_resnet_block(8+i, 256, 256, 14, 14)
    all_layers += build_resnet_block(13, 256, 512, 14, 14, downsample=True)
    for i in range(2): all_layers += build_resnet_block(14+i, 512, 512, 7, 7)

    for l in all_layers: sched.add_layer(l)

    log(f"  Compiling {len(all_layers)} layers...")
    t0 = time.time()
    with contextlib.redirect_stderr(io.StringIO()): sched.compile(profile_iters=5)
    log(f"  Compiled in {time.time()-t0:.1f}s")

    dummies = [np.random.randn(*l.input_shape).astype(np.float32)*0.02 for l in all_layers]

    log("  Benchmarking GPU...")
    gpu_r = bench(lambda: [sched._gpu_execs[i].run(dummies[i]) for i in range(len(dummies))])
    log(f"    All GPU:  {gpu_r['median_ms']:.2f}ms")

    log("  Benchmarking ANE...")
    ane_r = bench(lambda: [sched._ane_execs[i].run(dummies[i]) for i in range(len(dummies))])
    log(f"    All ANE:  {ane_r['median_ms']:.2f}ms")

    x0 = np.random.randn(1,64,56,56).astype(np.float32)*0.02
    log("  Benchmarking Pipeline...")
    pipe_r = bench(lambda: sched.run(x0))
    log(f"    Pipeline: {pipe_r['median_ms']:.2f}ms")

    log("  Benchmarking CPU...")
    cpu_r = bench(lambda: [sched._cpu_execs[i].run(dummies[i]) for i in range(len(dummies))])
    log(f"    All CPU:  {cpu_r['median_ms']:.2f}ms")

    s = sched.summary()
    results["benchmarks"]["resnet50"] = {
        "model":"ResNet-50","layers":len(all_layers),"compile_s":round(time.time()-t0,1),
        "schedule":{"gpu":s["gpu_layers"],"ane":s["ane_layers"],"cpu":s["cpu_layers"]},
        "gpu":gpu_r,"ane":ane_r,"pipeline":pipe_r,"cpu":cpu_r,"per_layer":s["profiles"],
    }
    # Free memory before next benchmark
    del sched, all_layers, dummies, x0; gc.collect()

except Exception as e:
    log(f"  ❌ ResNet-50 failed: {e}")

# Save partial results
results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
save_results(results)

# ════════════════════════════════════════════════════════════
# BENCHMARK 2: BERT-base
# ════════════════════════════════════════════════════════════
try:
    log("\n" + "="*60)
    log("  BENCHMARK 2: BERT-base (72 matmul ops)")
    log("="*60)

    B,L,D,H,D_FF = 1,128,768,12,3072
    sched2 = PipelineScheduler(verbose=False)
    bert_layers = []

    for li in range(12):
        for n in ["q","k","v"]:
            w = np.random.randn(D,D).astype(np.float32)*0.02
            bert_layers.append(LayerConfig(f"L{li}_{n}","matmul",(B*L,D),{"weight":w}))
        w = np.random.randn(D,D).astype(np.float32)*0.02
        bert_layers.append(LayerConfig(f"L{li}_out","matmul",(B*L,D),{"weight":w}))
        w1 = np.random.randn(D,D_FF).astype(np.float32)*0.02
        bert_layers.append(LayerConfig(f"L{li}_ff1","matmul",(B*L,D),{"weight":w1}))
        w2 = np.random.randn(D_FF,D).astype(np.float32)*0.02
        bert_layers.append(LayerConfig(f"L{li}_ff2","matmul",(B*L,D_FF),{"weight":w2}))

    for l in bert_layers: sched2.add_layer(l)

    log(f"  Compiling {len(bert_layers)} ops...")
    t0 = time.time()
    with contextlib.redirect_stderr(io.StringIO()): sched2.compile(profile_iters=3)
    log(f"  Compiled in {time.time()-t0:.1f}s")

    bdummies = [np.random.randn(*l.input_shape).astype(np.float32)*0.02 for l in bert_layers]

    log("  Benchmarking GPU...")
    gpu_r2 = bench(lambda: [sched2._gpu_execs[i].run(bdummies[i]) for i in range(len(bdummies))])
    log(f"    All GPU:  {gpu_r2['median_ms']:.2f}ms")

    log("  Benchmarking ANE...")
    ane_r2 = bench(lambda: [sched2._ane_execs[i].run(bdummies[i]) for i in range(len(bdummies))])
    log(f"    All ANE:  {ane_r2['median_ms']:.2f}ms")

    def run_bert_pipe():
        for i,e in enumerate(sched2.schedule):
            sched2._get_executor(e).run(bdummies[i])
    log("  Benchmarking Pipeline...")
    pipe_r2 = bench(run_bert_pipe)
    log(f"    Pipeline: {pipe_r2['median_ms']:.2f}ms")

    log("  Benchmarking CPU...")
    cpu_r2 = bench(lambda: [sched2._cpu_execs[i].run(bdummies[i]) for i in range(len(bdummies))])
    log(f"    All CPU:  {cpu_r2['median_ms']:.2f}ms")

    s2 = sched2.summary()
    results["benchmarks"]["bert_base"] = {
        "model":"BERT-base","ops":len(bert_layers),
        "config":{"B":B,"L":L,"D":D,"H":H,"D_FF":D_FF},
        "compile_s":round(time.time()-t0,1),
        "schedule":{"gpu":s2["gpu_layers"],"ane":s2["ane_layers"],"cpu":s2["cpu_layers"]},
        "gpu":gpu_r2,"ane":ane_r2,"pipeline":pipe_r2,"cpu":cpu_r2,
    }
    del sched2, bert_layers, bdummies; gc.collect()

except Exception as e:
    log(f"  ❌ BERT-base failed: {e}")

# Save partial results
results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
save_results(results)

# ════════════════════════════════════════════════════════════
# BENCHMARK 3: Matmul Scaling
# ════════════════════════════════════════════════════════════
try:
    log("\n" + "="*60)
    log("  BENCHMARK 3: Matmul Scaling")
    log("="*60)

    matmul_results = {}
    for N in [128, 256, 512, 1024, 2048]:
        a = np.random.randn(N,N).astype(np.float32)*0.02
        b_np = np.random.randn(N,N).astype(np.float32)*0.02
        a_mx = mx.array(a); b_mx = mx.array(b_np)

        def gpu_mm(): r = a_mx @ b_mx; mx.eval(r)
        g = bench(gpu_mm)

        if HAS_COREML:
            with contextlib.redirect_stderr(io.StringIO()):
                mm = ANECompiledLayer.matmul(M=N, K=N, weight=b_np)
            an = bench(lambda: mm(a))
            del mm
        else:
            an = {"median_ms":-1,"min_ms":-1,"max_ms":-1,"std_ms":-1}

        c = bench(lambda: np.matmul(a, b_np))
        matmul_results[str(N)] = {"gpu":g,"ane":an,"cpu":c}
        log(f"  {N:>5}x{N}: GPU={g['median_ms']:.2f}ms  ANE={an['median_ms']:.2f}ms  CPU={c['median_ms']:.2f}ms")
        gc.collect()

    results["benchmarks"]["matmul_scaling"] = matmul_results

except Exception as e:
    log(f"  ❌ Matmul scaling failed: {e}")

# ════════════════════════════════════════════════════════════
# FINAL SAVE
# ════════════════════════════════════════════════════════════
results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
saved = save_results(results)

log("\n" + "="*60)
log("  SUMMARY")
log("="*60)
log(f"  Device: {chip} ({ram_gb}GB, {gpu_cores} GPU cores)")

if "resnet50" in results["benchmarks"]:
    r = results["benchmarks"]["resnet50"]
    pct = (r["gpu"]["median_ms"] - r["ane"]["median_ms"]) / r["gpu"]["median_ms"] * 100
    log(f"  ResNet-50:  GPU={r['gpu']['median_ms']:.2f}ms  ANE={r['ane']['median_ms']:.2f}ms  ({pct:+.1f}%)")

if "bert_base" in results["benchmarks"]:
    r = results["benchmarks"]["bert_base"]
    pct = (r["gpu"]["median_ms"] - r["ane"]["median_ms"]) / r["gpu"]["median_ms"] * 100
    log(f"  BERT-base:  GPU={r['gpu']['median_ms']:.2f}ms  ANE={r['ane']['median_ms']:.2f}ms  ({pct:+.1f}%)")

log(f"\n  Results saved to: {saved}")
log(f"  Please share this file with Om! 🙏")
PYEOF

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗"
echo -e "║  Done! Results saved to ~/                           ║"
echo -e "║  Please share the JSON file with Om.                 ║"
echo -e "╚══════════════════════════════════════════════════════╝${NC}"

# Cleanup temp dir
rm -rf "$WORK_DIR"
