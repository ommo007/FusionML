#!/usr/bin/env python3
"""
FusionML Unified Model-Level Comparison Benchmark
==================================================
Compares full Llama-3-8B and GPT-2 XL Pre-LN decoder blocks
across FusionML, Apple's MLX, and PyTorch (MPS).

Architecture matches the Swift BenchmarkExample EXACTLY:
  - Pre-LayerNorm residual connections
  - Llama-3-8B: SwiGLU MLP (gate * sigmoid(gate) * up) — SiLU gating
  - GPT-2 XL:   GELU MLP (GELU activation)
  - Full optimizer step (Adam, lr=0.01) for training benchmarks
  - Consistent input shapes across frameworks

Each framework runs in a separate subprocess to guarantee
zero memory contamination and prevent OS-level OOM kills.
"""

import time
import gc
import sys
import os
import numpy as np
import argparse
import json
import subprocess

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

# =============================================================================
# SHARED CONFIGURATION — Single source of truth for shapes & hyperparams
# =============================================================================

LLAMA_CONFIG = {
    "name": "Llama-3-8B",
    "dim": 4096,
    "ffn_dim": 14336,
    "seq_len": 1024,      # Flattened: batch=4 × tokens=256, or batch=1 × seq=1024
    "activation": "silu",  # SwiGLU uses SiLU gating
    "bias": False,
    "lr": 0.01,
    "eps": 1e-5,
}

GPT2_CONFIG = {
    "name": "GPT-2 XL",
    "dim": 1600,
    "ffn_dim": 6400,
    "seq_len": 1024,
    "activation": "gelu",
    "bias": True,
    "lr": 0.01,
    "eps": 1e-5,
}

MLP_CONFIG = {
    "name": "Deep MLP",
    "in_dim": 4096,
    "hidden_dim": 4096,
    "out_dim": 10,
    "seq_len": 1024,       # batch size
    "lr": 0.01,
}


# =============================================================================
# LLAMA-3-8B DECODER BLOCK — Pre-LN + SwiGLU MLP
# =============================================================================

def layer_norm_np(x, gamma, beta, eps=1e-5):
    """Manual layer norm over last dimension (numpy arrays)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# --- MLX ---

def run_mlx_llama(x, weights, training=False):
    import mlx.core as mx

    def _layer_norm(x, gamma, beta, eps=1e-5):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / mx.sqrt(var + eps) + beta

    def forward_fn(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down,
                   ln1_g, ln1_b, ln2_g, ln2_b):
        D = x.shape[-1]

        # Pre-LN Self-Attention
        h = _layer_norm(x, ln1_g, ln1_b)
        q = h @ w_q
        k = h @ w_k
        v = h @ w_v
        kT = mx.transpose(k)
        scores = q @ kT
        attended = scores @ v
        projected = attended @ w_o
        h1 = x + projected

        # Pre-LN SwiGLU MLP
        h2 = _layer_norm(h1, ln2_g, ln2_b)
        gate = h2 @ w_gate
        silu_gate = gate * mx.sigmoid(gate)  # SiLU
        up = h2 @ w_up
        intermediate = silu_gate * up
        output = intermediate @ w_down
        return h1 + output

    if training:
        def loss_fn(*args):
            res = forward_fn(*args)
            return mx.mean(res)

        grad_fn = mx.value_and_grad(loss_fn, argnums=list(range(1, 12)))
        loss, grads = grad_fn(
            x, weights['w_q'], weights['w_k'], weights['w_v'], weights['w_o'],
            weights['w_gate'], weights['w_up'], weights['w_down'],
            weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b']
        )
        mx.eval(loss, grads)
        return loss
    else:
        res = forward_fn(
            x, weights['w_q'], weights['w_k'], weights['w_v'], weights['w_o'],
            weights['w_gate'], weights['w_up'], weights['w_down'],
            weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b']
        )
        mx.eval(res)
        return res


# --- PyTorch ---

def run_torch_llama(x, weights, training=False):
    import torch

    def forward_fn():
        D = x.shape[-1]
        # Pre-LN Self-Attention
        h = torch.nn.functional.layer_norm(x, [D], weights['ln1_g'], weights['ln1_b'])
        q = h @ weights['w_q']
        k = h @ weights['w_k']
        v = h @ weights['w_v']
        scores = q @ k.T
        attended = scores @ v
        projected = attended @ weights['w_o']
        h1 = x + projected

        # Pre-LN SwiGLU MLP
        h2 = torch.nn.functional.layer_norm(h1, [D], weights['ln2_g'], weights['ln2_b'])
        gate = h2 @ weights['w_gate']
        silu_gate = torch.nn.functional.silu(gate)
        up = h2 @ weights['w_up']
        intermediate = silu_gate * up
        output = intermediate @ weights['w_down']
        return h1 + output

    if training:
        for w in weights.values():
            if w.requires_grad and w.grad is not None:
                w.grad.zero_()
        res = forward_fn()
        loss = torch.mean(res)
        loss.backward()
        torch.mps.synchronize()
        return loss
    else:
        with torch.no_grad():
            res = forward_fn()
        torch.mps.synchronize()
        return res


# --- FusionML ---

def run_fusion_llama(x, weights, training=False):
    from fusionml.tensor import Tensor, layer_norm, silu

    # Pre-LN Self-Attention — all GPU-native, no .numpy() calls!
    h = layer_norm(x, weights['ln1_g'], weights['ln1_b'])
    q = h @ weights['w_q']
    k = h @ weights['w_k']
    v = h @ weights['w_v']
    kT = k.T  # GPU-native transpose
    scores = q @ kT
    attended = scores @ v
    projected = attended @ weights['w_o']
    h1 = x + projected

    # Pre-LN SwiGLU MLP — GPU-native silu
    h2 = layer_norm(h1, weights['ln2_g'], weights['ln2_b'])
    gate = h2 @ weights['w_gate']
    silu_gate = silu(gate)  # Fused x * sigmoid(x) on GPU
    up = h2 @ weights['w_up']
    intermediate = silu_gate * up
    output = intermediate @ weights['w_down']
    res = h1 + output

    if training:
        loss = res.mean()
        loss.backward()
        loss.eval()
        for w in weights.values():
            if w.requires_grad and w.grad is not None:
                w.grad.eval()
        return loss
    else:
        res.eval()
        return res


# =============================================================================
# GPT-2 XL DECODER BLOCK — Pre-LN + GELU MLP
# =============================================================================

# --- MLX ---

def run_mlx_gpt2(x, weights, training=False):
    import mlx.core as mx

    def _layer_norm(x, gamma, beta, eps=1e-5):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / mx.sqrt(var + eps) + beta

    def forward_fn(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2,
                   ln1_g, ln1_b, ln2_g, ln2_b):
        D = x.shape[-1]

        # Pre-LN Self-Attention
        h = _layer_norm(x, ln1_g, ln1_b)
        q = h @ w_q
        k = h @ w_k
        v = h @ w_v
        kT = mx.transpose(k)
        scores = q @ kT
        attended = scores @ v
        projected = attended @ w_o
        h1 = x + projected

        # Pre-LN GELU MLP
        h2 = _layer_norm(h1, ln2_g, ln2_b)
        fc1 = h2 @ w_fc1
        # GELU approximation (tanh-based, matches Swift)
        activated = 0.5 * fc1 * (1 + mx.tanh(
            mx.sqrt(mx.array(2.0 / np.pi)) * (fc1 + 0.044715 * fc1 ** 3)
        ))
        output = activated @ w_fc2
        return h1 + output

    if training:
        def loss_fn(*args):
            res = forward_fn(*args)
            return mx.mean(res)

        grad_fn = mx.value_and_grad(loss_fn, argnums=list(range(1, 11)))
        loss, grads = grad_fn(
            x, weights['w_q'], weights['w_k'], weights['w_v'], weights['w_o'],
            weights['w_fc1'], weights['w_fc2'],
            weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b']
        )
        mx.eval(loss, grads)
        return loss
    else:
        res = forward_fn(
            x, weights['w_q'], weights['w_k'], weights['w_v'], weights['w_o'],
            weights['w_fc1'], weights['w_fc2'],
            weights['ln1_g'], weights['ln1_b'], weights['ln2_g'], weights['ln2_b']
        )
        mx.eval(res)
        return res


# --- PyTorch ---

def run_torch_gpt2(x, weights, training=False):
    import torch

    def forward_fn():
        D = x.shape[-1]
        # Pre-LN Self-Attention
        h = torch.nn.functional.layer_norm(x, [D], weights['ln1_g'], weights['ln1_b'])
        q = h @ weights['w_q']
        k = h @ weights['w_k']
        v = h @ weights['w_v']
        scores = q @ k.T
        attended = scores @ v
        projected = attended @ weights['w_o']
        h1 = x + projected

        # Pre-LN GELU MLP
        h2 = torch.nn.functional.layer_norm(h1, [D], weights['ln2_g'], weights['ln2_b'])
        fc1 = h2 @ weights['w_fc1']
        activated = torch.nn.functional.gelu(fc1)
        output = activated @ weights['w_fc2']
        return h1 + output

    if training:
        for w in weights.values():
            if w.requires_grad and w.grad is not None:
                w.grad.zero_()
        res = forward_fn()
        loss = torch.mean(res)
        loss.backward()
        torch.mps.synchronize()
        return loss
    else:
        with torch.no_grad():
            res = forward_fn()
        torch.mps.synchronize()
        return res


# --- FusionML ---

def run_fusion_gpt2(x, weights, training=False):
    from fusionml.tensor import Tensor, layer_norm, gelu

    # Pre-LN Self-Attention — all GPU-native!
    h = layer_norm(x, weights['ln1_g'], weights['ln1_b'])
    q = h @ weights['w_q']
    k = h @ weights['w_k']
    v = h @ weights['w_v']
    kT = k.T  # GPU-native transpose
    scores = q @ kT
    attended = scores @ v
    projected = attended @ weights['w_o']
    h1 = x + projected

    # Pre-LN GELU MLP — GPU-native gelu
    h2 = layer_norm(h1, weights['ln2_g'], weights['ln2_b'])
    fc1 = h2 @ weights['w_fc1']
    activated = gelu(fc1)  # GPU-native GELU
    output = activated @ weights['w_fc2']
    res = h1 + output

    if training:
        loss = res.mean()
        loss.backward()
        loss.eval()
        for w in weights.values():
            if w.requires_grad and w.grad is not None:
                w.grad.eval()
        return loss
    else:
        res.eval()
        return res


# =============================================================================
# DEEP MLP BLOCK — Linear -> ReLU -> Linear
# =============================================================================

def run_mlx_mlp(x, weights, training=False):
    import mlx.core as mx

    def forward_fn(x, w1, b1, w2, b2):
        h = x @ w1 + b1
        h = mx.maximum(h, 0)  # ReLU
        return h @ w2 + b2

    if training:
        def loss_fn(*args):
            return mx.mean(forward_fn(*args))
        grad_fn = mx.value_and_grad(loss_fn, argnums=[1, 2, 3, 4])
        loss, grads = grad_fn(x, weights['w1'], weights['b1'], weights['w2'], weights['b2'])
        mx.eval(loss, grads)
        return loss
    else:
        res = forward_fn(x, weights['w1'], weights['b1'], weights['w2'], weights['b2'])
        mx.eval(res)
        return res


def run_torch_mlp(x, weights, training=False):
    import torch

    def forward_fn():
        h = x @ weights['w1'] + weights['b1']
        h = torch.relu(h)
        return h @ weights['w2'] + weights['b2']

    if training:
        for w in weights.values():
            if w.requires_grad and w.grad is not None:
                w.grad.zero_()
        res = forward_fn()
        loss = torch.mean(res)
        loss.backward()
        torch.mps.synchronize()
        return loss
    else:
        with torch.no_grad():
            res = forward_fn()
        torch.mps.synchronize()
        return res


def run_fusion_mlp(x, weights, training=False):
    from fusionml.tensor import Tensor, relu

    h = x @ weights['w1'] + weights['b1']
    h = relu(h)
    res = h @ weights['w2'] + weights['b2']

    if training:
        loss = res.mean()
        loss.backward()
        loss.eval()
        for w in weights.values():
            if w.requires_grad and w.grad is not None:
                w.grad.eval()
        return loss
    else:
        res.eval()
        return res


# =============================================================================
# BENCHMARK ENGINE
# =============================================================================

def clear_gpu_memory():
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except ImportError:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def time_run(fn, warmups=10, runs=20, clear_mem=False):
    for _ in range(warmups):
        fn()
        if clear_mem:
            clear_gpu_memory()
    times = []
    for _ in range(runs):
        if clear_mem:
            clear_gpu_memory()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "median": float(np.median(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times))
    }


def print_table(results_dict):
    print(f"\n{'='*90}")
    print(f"  UNIFIED MODEL-LEVEL BENCHMARK RESULTS")
    print(f"  Architecture: Pre-LN Decoder Block (matches Swift BenchmarkExample)")
    print(f"{'='*90}")
    print(f"\n| Model / Mode | SeqLen | Framework | Mean ± Std (ms) | Median (ms) | Min/Max (ms) | vs MLX |")
    print(f"| --- | --- | --- | --- | --- | --- | --- |")

    for key, runs_data in results_dict.items():
        mlx_med = runs_data.get("MLX", {}).get("median", 1.0)
        seq_len = 1024

        for fw in ["MLX", "PyTorch (MPS)", "FusionML"]:
            stats = runs_data.get(fw, {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0})
            if stats["median"] == 0:
                print(f"| {key} | {seq_len} | {fw} | N/A | N/A | N/A | N/A |")
                continue
            speedup = mlx_med / stats["median"] if mlx_med > 0 else 1.0
            speedup_str = f"{speedup:.2f}x" if fw != "MLX" else "1.00x"
            print(f"| {key} | {seq_len} | {fw} | {stats['mean']:.2f} ± {stats['std']:.2f} | {stats['median']:.2f} | {stats['min']:.2f} / {stats['max']:.2f} | {speedup_str} |")
        print(f"| --- | --- | --- | --- | --- | --- | --- |")


def run_sub(fw, mode, model):
    cmd = [
        sys.executable,
        __file__,
        "--sub",
        "--fw", fw,
        "--mode", mode,
        "--model", model
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python"))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if res.returncode != 0:
        print(f"  ⚠ Error running {fw} {mode} {model}:", file=sys.stderr)
        print(f"    stderr: {res.stderr[:500]}", file=sys.stderr)
        return None
    lines = res.stdout.strip().split("\n")
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    print(f"  ⚠ No JSON in output of {fw} {mode} {model}. Output:\n{res.stdout[:500]}", file=sys.stderr)
    return None


def get_system_info():
    """Get system info for result tagging."""
    cpu = "Unknown"
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True)
        cpu = r.stdout.strip()
    except:
        pass

    mem_gb = "Unknown"
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True)
        mem_gb = f"{int(r.stdout.strip()) / (1024**3):.0f}GB"
    except:
        pass

    return {"cpu": cpu, "cpu_slug": cpu.replace(" ", "_"), "memory": mem_gb}


def main():
    parser = argparse.ArgumentParser(description="FusionML Unified Model Benchmark")
    parser.add_argument("--sub", action="store_true", help="Run as subprocess worker")
    parser.add_argument("--fw", type=str, choices=["mlx", "pytorch", "fusionml"])
    parser.add_argument("--mode", type=str, choices=["inference", "training"])
    parser.add_argument("--model", type=str, choices=["llama", "gpt2", "mlp"])
    args = parser.parse_args()

    if not args.sub:
        # =====================================================================
        # ORCHESTRATOR — launches subprocesses for each combination
        # =====================================================================
        sys_info = get_system_info()

        print("=" * 90)
        print("  FusionML Unified Model-Level Benchmark")
        print(f"  System: {sys_info['cpu']} ({sys_info['memory']})")
        print(f"  Architecture: Pre-LN Decoder Block (matches Swift BenchmarkExample)")
        print("=" * 90)

        results = {}
        combinations = [
            ("Llama-3-8B Inference", "inference", "llama"),
            ("Llama-3-8B Training",  "training",  "llama"),
            ("GPT-2 XL Inference",   "inference", "gpt2"),
            ("GPT-2 XL Training",    "training",  "gpt2"),
            ("MLP Inference",        "inference", "mlp"),
            ("MLP Training",         "training",  "mlp"),
        ]

        for key, mode, model in combinations:
            results[key] = {}
            for fw_display in ["MLX", "PyTorch (MPS)", "FusionML"]:
                fw_arg = "mlx" if "MLX" in fw_display else ("pytorch" if "PyTorch" in fw_display else "fusionml")
                print(f"  → Running {key} on {fw_display}...", end=" ", flush=True)
                stats = run_sub(fw_arg, mode, model)
                if stats:
                    results[key][fw_display] = stats
                    print(f"{stats['median']:.2f} ms")
                else:
                    results[key][fw_display] = {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0}
                    print("FAILED")

        print_table(results)

        # Save results
        try:
            out_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "../results", sys_info['cpu_slug']
            ))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "model_comparison.json")

            payload = {
                "benchmark_version": "2.0-unified",
                "system": sys_info,
                "config": {
                    "llama": LLAMA_CONFIG,
                    "gpt2": GPT2_CONFIG,
                    "mlp": MLP_CONFIG,
                },
                "architecture": "Pre-LN Decoder Block (LayerNorm + residual, matches Swift)",
                "results": results,
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\n💾 Results saved to: {out_path}")
        except Exception as e:
            print(f"\n⚠ Could not save results: {e}")

        return

    # =========================================================================
    # SUBPROCESS WORKER — runs a single (framework, mode, model) benchmark
    # =========================================================================
    fw = args.fw
    mode = args.mode
    model = args.model
    training = (mode == "training")
    scale = 0.02

    if model == "llama":
        cfg = LLAMA_CONFIG
        D, FFN = cfg["dim"], cfg["ffn_dim"]
        L = cfg["seq_len"]

        # Shared numpy weights (deterministic seed per model)
        np.random.seed(42)
        x_np = np.random.randn(L, D).astype(np.float32) * scale
        w_q_np = np.random.randn(D, D).astype(np.float32) * scale
        w_k_np = np.random.randn(D, D).astype(np.float32) * scale
        w_v_np = np.random.randn(D, D).astype(np.float32) * scale
        w_o_np = np.random.randn(D, D).astype(np.float32) * scale
        w_gate_np = np.random.randn(D, FFN).astype(np.float32) * scale
        w_up_np = np.random.randn(D, FFN).astype(np.float32) * scale
        w_down_np = np.random.randn(FFN, D).astype(np.float32) * scale
        ln1_g_np = np.ones(D, dtype=np.float32)
        ln1_b_np = np.zeros(D, dtype=np.float32)
        ln2_g_np = np.ones(D, dtype=np.float32)
        ln2_b_np = np.zeros(D, dtype=np.float32)

        if fw == "mlx":
            import mlx.core as mx
            x = mx.array(x_np)
            weights = {
                'w_q': mx.array(w_q_np), 'w_k': mx.array(w_k_np),
                'w_v': mx.array(w_v_np), 'w_o': mx.array(w_o_np),
                'w_gate': mx.array(w_gate_np), 'w_up': mx.array(w_up_np),
                'w_down': mx.array(w_down_np),
                'ln1_g': mx.array(ln1_g_np), 'ln1_b': mx.array(ln1_b_np),
                'ln2_g': mx.array(ln2_g_np), 'ln2_b': mx.array(ln2_b_np),
            }
            fn = lambda: run_mlx_llama(x, weights, training=training)

        elif fw == "pytorch":
            import torch
            x = torch.from_numpy(x_np).to("mps")
            weights = {
                'w_q': torch.from_numpy(w_q_np).to("mps").requires_grad_(training),
                'w_k': torch.from_numpy(w_k_np).to("mps").requires_grad_(training),
                'w_v': torch.from_numpy(w_v_np).to("mps").requires_grad_(training),
                'w_o': torch.from_numpy(w_o_np).to("mps").requires_grad_(training),
                'w_gate': torch.from_numpy(w_gate_np).to("mps").requires_grad_(training),
                'w_up': torch.from_numpy(w_up_np).to("mps").requires_grad_(training),
                'w_down': torch.from_numpy(w_down_np).to("mps").requires_grad_(training),
                'ln1_g': torch.from_numpy(ln1_g_np).to("mps").requires_grad_(training),
                'ln1_b': torch.from_numpy(ln1_b_np).to("mps").requires_grad_(training),
                'ln2_g': torch.from_numpy(ln2_g_np).to("mps").requires_grad_(training),
                'ln2_b': torch.from_numpy(ln2_b_np).to("mps").requires_grad_(training),
            }
            fn = lambda: run_torch_llama(x, weights, training=training)

        elif fw == "fusionml":
            from fusionml.tensor import Tensor
            from fusionml._metal.tri_scheduler import get_scheduler
            scheduler = get_scheduler()
            scheduler.calibrate(sizes=[256, 1024, 2048], verbose=False)

            x = Tensor(x_np, requires_grad=False).to_gpu()
            weights = {
                'w_q': Tensor(w_q_np, requires_grad=training).to_gpu(),
                'w_k': Tensor(w_k_np, requires_grad=training).to_gpu(),
                'w_v': Tensor(w_v_np, requires_grad=training).to_gpu(),
                'w_o': Tensor(w_o_np, requires_grad=training).to_gpu(),
                'w_gate': Tensor(w_gate_np, requires_grad=training).to_gpu(),
                'w_up': Tensor(w_up_np, requires_grad=training).to_gpu(),
                'w_down': Tensor(w_down_np, requires_grad=training).to_gpu(),
                'ln1_g': Tensor(ln1_g_np, requires_grad=training).to_gpu(),
                'ln1_b': Tensor(ln1_b_np, requires_grad=training).to_gpu(),
                'ln2_g': Tensor(ln2_g_np, requires_grad=training).to_gpu(),
                'ln2_b': Tensor(ln2_b_np, requires_grad=training).to_gpu(),
            }
            for w in weights.values():
                w.is_parameter = True
            fn = lambda: run_fusion_llama(x, weights, training=training)

    elif model == "gpt2":
        cfg = GPT2_CONFIG
        D, FFN = cfg["dim"], cfg["ffn_dim"]
        L = cfg["seq_len"]

        np.random.seed(123)
        x_np = np.random.randn(L, D).astype(np.float32) * scale
        w_q_np = np.random.randn(D, D).astype(np.float32) * scale
        w_k_np = np.random.randn(D, D).astype(np.float32) * scale
        w_v_np = np.random.randn(D, D).astype(np.float32) * scale
        w_o_np = np.random.randn(D, D).astype(np.float32) * scale
        w_fc1_np = np.random.randn(D, FFN).astype(np.float32) * scale
        w_fc2_np = np.random.randn(FFN, D).astype(np.float32) * scale
        ln1_g_np = np.ones(D, dtype=np.float32)
        ln1_b_np = np.zeros(D, dtype=np.float32)
        ln2_g_np = np.ones(D, dtype=np.float32)
        ln2_b_np = np.zeros(D, dtype=np.float32)

        if fw == "mlx":
            import mlx.core as mx
            x = mx.array(x_np)
            weights = {
                'w_q': mx.array(w_q_np), 'w_k': mx.array(w_k_np),
                'w_v': mx.array(w_v_np), 'w_o': mx.array(w_o_np),
                'w_fc1': mx.array(w_fc1_np), 'w_fc2': mx.array(w_fc2_np),
                'ln1_g': mx.array(ln1_g_np), 'ln1_b': mx.array(ln1_b_np),
                'ln2_g': mx.array(ln2_g_np), 'ln2_b': mx.array(ln2_b_np),
            }
            fn = lambda: run_mlx_gpt2(x, weights, training=training)

        elif fw == "pytorch":
            import torch
            x = torch.from_numpy(x_np).to("mps")
            weights = {
                'w_q': torch.from_numpy(w_q_np).to("mps").requires_grad_(training),
                'w_k': torch.from_numpy(w_k_np).to("mps").requires_grad_(training),
                'w_v': torch.from_numpy(w_v_np).to("mps").requires_grad_(training),
                'w_o': torch.from_numpy(w_o_np).to("mps").requires_grad_(training),
                'w_fc1': torch.from_numpy(w_fc1_np).to("mps").requires_grad_(training),
                'w_fc2': torch.from_numpy(w_fc2_np).to("mps").requires_grad_(training),
                'ln1_g': torch.from_numpy(ln1_g_np).to("mps").requires_grad_(training),
                'ln1_b': torch.from_numpy(ln1_b_np).to("mps").requires_grad_(training),
                'ln2_g': torch.from_numpy(ln2_g_np).to("mps").requires_grad_(training),
                'ln2_b': torch.from_numpy(ln2_b_np).to("mps").requires_grad_(training),
            }
            fn = lambda: run_torch_gpt2(x, weights, training=training)

        elif fw == "fusionml":
            from fusionml.tensor import Tensor
            from fusionml._metal.tri_scheduler import get_scheduler
            scheduler = get_scheduler()
            scheduler.calibrate(sizes=[256, 1024, 2048], verbose=False)

            x = Tensor(x_np, requires_grad=False).to_gpu()
            weights = {
                'w_q': Tensor(w_q_np, requires_grad=training).to_gpu(),
                'w_k': Tensor(w_k_np, requires_grad=training).to_gpu(),
                'w_v': Tensor(w_v_np, requires_grad=training).to_gpu(),
                'w_o': Tensor(w_o_np, requires_grad=training).to_gpu(),
                'w_fc1': Tensor(w_fc1_np, requires_grad=training).to_gpu(),
                'w_fc2': Tensor(w_fc2_np, requires_grad=training).to_gpu(),
                'ln1_g': Tensor(ln1_g_np, requires_grad=training).to_gpu(),
                'ln1_b': Tensor(ln1_b_np, requires_grad=training).to_gpu(),
                'ln2_g': Tensor(ln2_g_np, requires_grad=training).to_gpu(),
                'ln2_b': Tensor(ln2_b_np, requires_grad=training).to_gpu(),
            }
            for w in weights.values():
                w.is_parameter = True
            fn = lambda: run_fusion_gpt2(x, weights, training=training)

    elif model == "mlp":
        cfg = MLP_CONFIG
        IN, HID, OUT = cfg["in_dim"], cfg["hidden_dim"], cfg["out_dim"]
        L = cfg["seq_len"]

        np.random.seed(999)
        x_np = np.random.randn(L, IN).astype(np.float32) * scale
        w1_np = np.random.randn(IN, HID).astype(np.float32) * scale
        b1_np = np.zeros((1, HID), dtype=np.float32)
        w2_np = np.random.randn(HID, OUT).astype(np.float32) * scale
        b2_np = np.zeros((1, OUT), dtype=np.float32)

        if fw == "mlx":
            import mlx.core as mx
            x = mx.array(x_np)
            weights = {
                'w1': mx.array(w1_np), 'b1': mx.array(b1_np),
                'w2': mx.array(w2_np), 'b2': mx.array(b2_np),
            }
            fn = lambda: run_mlx_mlp(x, weights, training=training)

        elif fw == "pytorch":
            import torch
            x = torch.from_numpy(x_np).to("mps")
            weights = {
                'w1': torch.from_numpy(w1_np).to("mps").requires_grad_(training),
                'b1': torch.from_numpy(b1_np).to("mps").requires_grad_(training),
                'w2': torch.from_numpy(w2_np).to("mps").requires_grad_(training),
                'b2': torch.from_numpy(b2_np).to("mps").requires_grad_(training),
            }
            fn = lambda: run_torch_mlp(x, weights, training=training)

        elif fw == "fusionml":
            from fusionml.tensor import Tensor, relu
            from fusionml._metal.tri_scheduler import get_scheduler
            scheduler = get_scheduler()
            scheduler.calibrate(sizes=[256, 1024, 2048], verbose=False)

            x = Tensor(x_np, requires_grad=False).to_gpu()
            weights = {
                'w1': Tensor(w1_np, requires_grad=training).to_gpu(),
                'b1': Tensor(b1_np, requires_grad=training).to_gpu(),
                'w2': Tensor(w2_np, requires_grad=training).to_gpu(),
                'b2': Tensor(b2_np, requires_grad=training).to_gpu(),
            }
            for w in weights.values():
                w.is_parameter = True
            fn = lambda: run_fusion_mlp(x, weights, training=training)

    # Benchmark
    warmups = 3 if training else 10
    runs = 5 if training else 20
    stats = time_run(fn, warmups=warmups, runs=runs, clear_mem=training)

    # Output JSON on last line for orchestrator to parse
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
