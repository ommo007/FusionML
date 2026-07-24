#!/usr/bin/env python3
"""
full_depth_prefill_benchmark.py — does the block-level per-layer split win
survive full model depth? 32 stacked Llama-3-8B-geometry decoder blocks
(~15.6 GB FP16 weights), prefill at seq 2048/4096(/8192), split vs fair
eager MLX-FP16 baseline.

Answers the "single synthetic block" reviewer objection on the controlled
side (real-checkpoint side: mlxlm_ttft_benchmark.py).

Requires ≥20 GB unified memory (M4/M4 Pro 24GB machines) — exits early
otherwise. Arms run in separate subprocesses, strictly sequential.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

N_BLOCKS = 32
D, F = 4096, 14336
SEQ_LENS = [2048, 4096, 8192]
WARMUPS = 3
RUNS = 10
RATIO_CANDIDATES = [0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
CALIB_RUNS = 5
MIN_MEM_GB = 20


def build_blocks():
    import mlx.core as mx
    np.random.seed(42)
    # Small init scale keeps activations bounded through 32 blocks in fp16
    scale = 0.02 / (N_BLOCKS ** 0.5)

    def _a(shape):
        return mx.array((np.random.randn(*shape) * scale).astype(np.float32)).astype(mx.float16)

    blocks = []
    for i in range(N_BLOCKS):
        b = {'w_q': _a((D, D)), 'w_k': _a((D, D)), 'w_v': _a((D, D)), 'w_o': _a((D, D)),
             'w_gate': _a((D, F)), 'w_up': _a((D, F)), 'w_down': _a((F, D)),
             'ln1_g': mx.ones((D,), dtype=mx.float16), 'ln1_b': mx.zeros((D,), dtype=mx.float16),
             'ln2_g': mx.ones((D,), dtype=mx.float16), 'ln2_b': mx.zeros((D,), dtype=mx.float16)}
        mx.eval(*b.values())
        blocks.append(b)
        if (i + 1) % 8 == 0:
            print(f"    built {i+1}/{N_BLOCKS} blocks", file=sys.stderr)
    return blocks


def calibrate_ratios(L):
    """Same contention-aware per-shape search as prefill_scale_benchmark."""
    import mlx.core as mx
    np.random.seed(7)
    ratios = {}
    for (M, K, N) in [(L, D, D), (L, D, F), (L, F, D)]:
        a = mx.array((np.random.randn(M, K) * 0.02).astype(np.float32)).astype(mx.float16)
        w = mx.array((np.random.randn(K, N) * 0.02).astype(np.float32)).astype(mx.float16)
        mx.eval(a, w)
        best, best_t = 0.0, float("inf")
        for r in RATIO_CANDIDATES:
            cpu_rows = int(M * r)

            def once():
                if cpu_rows == 0:
                    mx.eval(a @ w)
                else:
                    c = mx.matmul(a[:cpu_rows], w, stream=mx.cpu)
                    g = mx.matmul(a[cpu_rows:], w, stream=mx.gpu)
                    mx.eval(mx.concatenate([c, g], axis=0))

            once(); once()
            ts = []
            for _ in range(CALIB_RUNS):
                t0 = time.perf_counter(); once(); ts.append(time.perf_counter() - t0)
            med = float(np.median(ts))
            if med < best_t:
                best, best_t = r, med
        ratios[(M, K, N)] = best
        del a, w
        mx.clear_cache()
    return ratios


def block_forward(x, w, linear):
    import mlx.core as mx
    import mlx.core.fast as mxf
    import mlx.nn as mlx_nn
    h = mxf.layer_norm(x, w['ln1_g'], w['ln1_b'], 1e-5)
    q, k, v = linear(h, w['w_q']), linear(h, w['w_k']), linear(h, w['w_v'])
    scores = (q @ mx.transpose(k)) * (1.0 / (D ** 0.5))
    attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
    h1 = x + linear(attn @ v, w['w_o'])
    h2 = mxf.layer_norm(h1, w['ln2_g'], w['ln2_b'], 1e-5)
    out = linear(mlx_nn.silu(linear(h2, w['w_gate'])) * linear(h2, w['w_up']), w['w_down'])
    return h1 + out


def run_worker(L, arm):
    import mlx.core as mx
    import resource

    gpu_linear = lambda h, w: h @ w
    if arm == "fusion_split":
        print("  calibrating ratios...", file=sys.stderr)
        ratios = calibrate_ratios(L)

        def split_linear(h, w):
            M, K = h.shape[0], h.shape[1]
            r = ratios.get((M, K, w.shape[1]), 0.0)
            if r < 0.02:
                return h @ w
            mx.eval(h)  # eager boundary (see prefill_scale_benchmark.py)
            cpu_rows = int(M * r)
            c = mx.matmul(h[:cpu_rows], w, stream=mx.cpu)
            g = mx.matmul(h[cpu_rows:], w, stream=mx.gpu)
            return mx.concatenate([c, g], axis=0)

        linear = split_linear
        calib = {f"{k[0]}x{k[1]}x{k[2]}": v for k, v in ratios.items()}
    else:
        linear = gpu_linear
        calib = None

    blocks = build_blocks()
    np.random.seed(1)
    x0 = mx.array((np.random.randn(L, D) * 0.02).astype(np.float32)).astype(mx.float16)
    mx.eval(x0)

    def fwd():
        x = x0
        for b in blocks:
            x = block_forward(x, b, linear)
            if arm == "fusion_split":
                mx.eval(x)  # per-block boundary keeps intermediates freed
        mx.eval(x)
        return x

    # Correctness at full depth (fp16 accumulates over 32 blocks — expect
    # larger rel err than single block; report, don't assert)
    rel_err = None
    if arm == "fusion_split":
        out_s = fwd()
        x = x0
        for b in blocks:
            x = block_forward(x, b, gpu_linear)
        mx.eval(x)
        a = np.array(out_s.astype(mx.float32), copy=False).astype(np.float64)
        r = np.array(x.astype(mx.float32), copy=False).astype(np.float64)
        rel_err = float(np.max(np.abs(a - r)) / (np.max(np.abs(r)) + 1e-8))
        del out_s, x
        mx.clear_cache()

    for _ in range(WARMUPS):
        fwd()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fwd()
        times.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    out = {"mean": float(np.mean(times)), "std": float(np.std(times)),
           "median": float(np.median(times)),
           "ci95": float(1.96 * np.std(times) / np.sqrt(len(times))),
           "n_runs": len(times), "n_blocks": N_BLOCKS, "peak_mem_mb": peak_mb,
           "tokens_per_sec_prefill": float(L * 1000.0 / np.mean(times))}
    if arm == "fusion_split":
        out["calibration"] = calib
        out["max_rel_err_vs_gpu_only_full_depth"] = rel_err
    return out


def run_sub(L, arm):
    cmd = [sys.executable, __file__, "--sub", "--seq", str(L), "--arm", arm]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(__file__)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    if res.returncode != 0:
        print(f"\n  ⚠ L={L} {arm}: rc={res.returncode} {res.stderr[-600:]}", file=sys.stderr)
        return None
    for line in reversed(res.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", action="store_true")
    parser.add_argument("--seq", type=int)
    parser.add_argument("--arm", choices=["mlx_fp16", "fusion_split"])
    parser.add_argument("--seqs", default=",".join(str(s) for s in SEQ_LENS))
    args = parser.parse_args()

    if args.sub:
        print(json.dumps(run_worker(args.seq, args.arm)))
        return

    mem_gb = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                text=True).stdout.strip()) // (1024 ** 3)
    if mem_gb < MIN_MEM_GB:
        print(f"✗ Needs ≥{MIN_MEM_GB}GB unified memory (this machine: {mem_gb}GB). "
              f"Run on the M4 / M4 Pro 24GB machines.")
        sys.exit(1)

    from bench_hw import get_system_info, get_power_state
    free_gb = get_power_state().get("free_ram_gb")
    if free_gb is not None and free_gb < 17.0:
        print(f"✗ Only {free_gb}GB RAM free — the 15.6GB model would swap and every "
              f"number would be garbage (this exact failure invalidated a prior run). "
              f"Reboot or quit other apps, then re-run.")
        sys.exit(1)
    slug = get_system_info()["cpu_slug"]
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results", slug))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "full_depth_prefill_benchmark.json")

    print("=" * 78)
    print(f"  Full-Depth Prefill — {N_BLOCKS} stacked Llama-3-8B blocks (~15.6GB fp16)")
    print(f"  Machine: {slug}  Runs: {RUNS}  Warmup: {WARMUPS}")
    print("=" * 78)

    results = {}
    for L in [int(s) for s in args.seqs.split(",")]:
        print(f"\n▶ seq_len={L}")
        results[str(L)] = {}
        for arm in ["mlx_fp16", "fusion_split"]:
            print(f"   {arm:13s} ... ", end="", flush=True)
            time.sleep(20.0)
            s = run_sub(L, arm)
            results[str(L)][arm] = s
            if s:
                extra = f"  rel_err={s['max_rel_err_vs_gpu_only_full_depth']:.2e}" if arm == "fusion_split" else ""
                print(f"{s['median']:9.1f} ms  ±{s['ci95']:.1f}{extra}")
            else:
                print("FAILED (likely OOM at this seq — acceptable, noted)")
        r = results[str(L)]
        if r.get("mlx_fp16") and r.get("fusion_split"):
            print(f"   → full-depth split vs MLX-FP16: "
                  f"{r['mlx_fp16']['median'] / r['fusion_split']['median']:.3f}x")
        with open(out_path, "w") as f:
            json.dump({"n_blocks": N_BLOCKS, "runs": RUNS, "warmups": WARMUPS,
                       "environment": get_power_state(), "results": results}, f, indent=2)

    print(f"\n💾 Saved to {out_path}")


if __name__ == "__main__":
    main()
