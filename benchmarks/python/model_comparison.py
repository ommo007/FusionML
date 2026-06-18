#!/usr/bin/env python3
"""
FusionML Model-Level Comparison Benchmark
=========================================
Compares full Llama-3-8B and GPT-2 XL decoder blocks directly in Python
across FusionML, Apple's MLX, and PyTorch (MPS).

Runs each framework configuration in a separate subprocess to guarantee
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

# -----------------------------------------------------------------------------
# LLAMA-3-8B DECODER LAYER (PRE-CONVERTED)
# -----------------------------------------------------------------------------

def run_mlx_llama(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, training=False):
    import mlx.core as mx
    if training:
        def loss_fn(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down):
            L, D = x.shape[1], x.shape[2]
            x_flat = x.reshape(L, D)
            q = x_flat @ w_q
            k = x_flat @ w_k
            v = x_flat @ w_v
            scores = (q @ k.T) * (1.0 / np.sqrt(D))
            attn = mx.softmax(scores, axis=-1)
            out = attn @ v
            attn_out = out @ w_o
            h1 = x_flat + attn_out
            gate = h1 @ w_gate
            silu_gate = gate * mx.sigmoid(gate)
            up = h1 @ w_up
            mlp_out = (silu_gate * up) @ w_down
            res = h1 + mlp_out
            return mx.mean(res)
            
        grad_fn = mx.value_and_grad(loss_fn, argnums=[1, 2, 3, 4, 5, 6, 7])
        loss, grads = grad_fn(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down)
        mx.eval(loss, grads)
        return loss
    else:
        L, D = x.shape[1], x.shape[2]
        x_flat = x.reshape(L, D)
        q = x_flat @ w_q
        k = x_flat @ w_k
        v = x_flat @ w_v
        scores = (q @ k.T) * (1.0 / np.sqrt(D))
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        attn_out = out @ w_o
        h1 = x_flat + attn_out
        gate = h1 @ w_gate
        silu_gate = gate * mx.sigmoid(gate)
        up = h1 @ w_up
        mlp_out = (silu_gate * up) @ w_down
        res = h1 + mlp_out
        res = res.reshape(1, L, D)
        mx.eval(res)
        return res

def run_torch_llama(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, training=False):
    import torch
    L, D = x.shape[1], x.shape[2]
    x_flat = x.reshape(L, D)
    q = x_flat @ w_q
    k = x_flat @ w_k
    v = x_flat @ w_v
    scores = (q @ k.T) * (1.0 / np.sqrt(D))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    attn_out = out @ w_o
    h1 = x_flat + attn_out
    gate = h1 @ w_gate
    silu_gate = gate * torch.sigmoid(gate)
    up = h1 @ w_up
    mlp_out = (silu_gate * up) @ w_down
    res = h1 + mlp_out
    res = res.reshape(1, L, D)
    
    if training:
        for w in [w_q, w_k, w_v, w_o, w_gate, w_up, w_down]:
            if w.grad is not None:
                w.grad.zero_()
        loss = torch.mean(res)
        loss.backward()
        torch.mps.synchronize()
        return loss
    else:
        torch.mps.synchronize()
        return res

def run_fusion_llama(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, training=False):
    from fusionml.tensor import Tensor, softmax, sigmoid
    L, D = x.shape[1], x.shape[2]
    x_flat = x.reshape(L, D)
    q = x_flat @ w_q
    k = x_flat @ w_k
    v = x_flat @ w_v
    scores = (q @ k.T) * (1.0 / np.sqrt(D))
    attn = softmax(scores, axis=-1)
    out = attn @ v
    attn_out = out @ w_o
    h1 = x_flat + attn_out
    gate = h1 @ w_gate
    silu_gate = gate * sigmoid(gate)
    up = h1 @ w_up
    mlp_out = (silu_gate * up) @ w_down
    res = h1 + mlp_out
    res = res.reshape(1, L, D)
    
    if training:
        for w in [w_q, w_k, w_v, w_o, w_gate, w_up, w_down]:
            w.grad = None
        loss = res.mean()
        loss.backward()
        loss.eval()
        for w in [w_q, w_k, w_v, w_o, w_gate, w_up, w_down]:
            if w.grad is not None:
                w.grad.eval()
        return loss
    else:
        res.eval()
        return res

# -----------------------------------------------------------------------------
# GPT-2 XL DECODER LAYER (PRE-CONVERTED)
# -----------------------------------------------------------------------------

def run_mlx_gpt2(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, training=False):
    import mlx.core as mx
    if training:
        def loss_fn(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2):
            L, D = x.shape[1], x.shape[2]
            x_flat = x.reshape(L, D)
            q = x_flat @ w_q
            k = x_flat @ w_k
            v = x_flat @ w_v
            scores = (q @ k.T) * (1.0 / np.sqrt(D))
            attn = mx.softmax(scores, axis=-1)
            out = attn @ v
            attn_out = out @ w_o
            h1 = x_flat + attn_out
            fc1 = h1 @ w_fc1
            gelu_fc1 = fc1 * mx.sigmoid(fc1 * 1.702)
            mlp_out = gelu_fc1 @ w_fc2
            res = h1 + mlp_out
            return mx.mean(res)
            
        grad_fn = mx.value_and_grad(loss_fn, argnums=[1, 2, 3, 4, 5, 6])
        loss, grads = grad_fn(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2)
        mx.eval(loss, grads)
        return loss
    else:
        L, D = x.shape[1], x.shape[2]
        x_flat = x.reshape(L, D)
        q = x_flat @ w_q
        k = x_flat @ w_k
        v = x_flat @ w_v
        scores = (q @ k.T) * (1.0 / np.sqrt(D))
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        attn_out = out @ w_o
        h1 = x_flat + attn_out
        fc1 = h1 @ w_fc1
        gelu_fc1 = fc1 * mx.sigmoid(fc1 * 1.702)
        mlp_out = gelu_fc1 @ w_fc2
        res = h1 + mlp_out
        res = res.reshape(1, L, D)
        mx.eval(res)
        return res

def run_torch_gpt2(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, training=False):
    import torch
    L, D = x.shape[1], x.shape[2]
    x_flat = x.reshape(L, D)
    q = x_flat @ w_q
    k = x_flat @ w_k
    v = x_flat @ w_v
    scores = (q @ k.T) * (1.0 / np.sqrt(D))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    attn_out = out @ w_o
    h1 = x_flat + attn_out
    fc1 = h1 @ w_fc1
    gelu_fc1 = fc1 * torch.sigmoid(fc1 * 1.702)
    mlp_out = gelu_fc1 @ w_fc2
    res = h1 + mlp_out
    res = res.reshape(1, L, D)
    
    if training:
        for w in [w_q, w_k, w_v, w_o, w_fc1, w_fc2]:
            if w.grad is not None:
                w.grad.zero_()
        loss = torch.mean(res)
        loss.backward()
        torch.mps.synchronize()
        return loss
    else:
        torch.mps.synchronize()
        return res

def run_fusion_gpt2(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, training=False):
    from fusionml.tensor import Tensor, softmax, sigmoid
    L, D = x.shape[1], x.shape[2]
    x_flat = x.reshape(L, D)
    q = x_flat @ w_q
    k = x_flat @ w_k
    v = x_flat @ w_v
    scores = (q @ k.T) * (1.0 / np.sqrt(D))
    attn = softmax(scores, axis=-1)
    out = attn @ v
    attn_out = out @ w_o
    h1 = x_flat + attn_out
    fc1 = h1 @ w_fc1
    gelu_fc1 = fc1 * sigmoid(fc1 * 1.702)
    mlp_out = gelu_fc1 @ w_fc2
    res = h1 + mlp_out
    res = res.reshape(1, L, D)
    
    if training:
        for w in [w_q, w_k, w_v, w_o, w_fc1, w_fc2]:
            w.grad = None
        loss = res.mean()
        loss.backward()
        loss.eval()
        for w in [w_q, w_k, w_v, w_o, w_fc1, w_fc2]:
            if w.grad is not None:
                w.grad.eval()
        return loss
    else:
        res.eval()
        return res

# -----------------------------------------------------------------------------
# BENCHMARK ENGINE
# -----------------------------------------------------------------------------

def clear_gpu_memory():
    """Clear memory caches across frameworks to prevent OOM."""
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
        times.append((t1 - t0) * 1000.0) # ms
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "median": np.median(times),
        "min": np.min(times),
        "max": np.max(times)
    }

def print_table(results_dict, B):
    print(f"\n### Benchmark Results (Batch={B})")
    print(f"| Model / Mode | SeqLen | Framework | Mean ± Std (ms) | Median (ms) | Min/Max (ms) | Speedup vs MLX |")
    print(f"| --- | --- | --- | --- | --- | --- | --- |")
    
    for key, runs_data in results_dict.items():
        mlx_med = runs_data.get("MLX", {}).get("median", 1.0)
        seq_len = 1024 if "Inference" in key else 256
        
        for fw in ["MLX", "PyTorch (MPS)", "FusionML"]:
            stats = runs_data.get(fw, {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0})
            if stats["median"] == 0:
                print(f"| {key} | {seq_len} | {fw} | N/A | N/A | N/A | N/A |")
                continue
            speedup = mlx_med / stats["median"] if mlx_med > 0 else 1.0
            speedup_str = f"{speedup:.2f}x" if fw != "MLX" else "1.00x"
            print(f"| {key} | {seq_len} | {fw} | {stats['mean']:.2f} ± {stats['std']:.2f} | {stats['median']:.2f} | {stats['min']:.2f} / {stats['max']:.2f} | {speedup_str} |")

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
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"Error running subprocess {fw} {mode} {model}:", res.stderr)
        return None
    # Parse JSON from last line of output
    lines = res.stdout.strip().split("\n")
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    print(f"No JSON found in output of {fw} {mode} {model}. Output was:\n{res.stdout}")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", action="store_true", help="Run as a subprocess")
    parser.add_argument("--fw", type=str, choices=["mlx", "pytorch", "fusionml"], help="Framework")
    parser.add_argument("--mode", type=str, choices=["inference", "training"], help="Mode")
    parser.add_argument("--model", type=str, choices=["llama", "gpt2"], help="Model")
    args = parser.parse_args()

    if not args.sub:
        print("==========================================================================")
        print("      FusionML Model-Level and Layer-Level Comparative Sweep")
        print("==========================================================================")
        
        results = {}
        combinations = [
            ("Llama-3-8B Inference", "inference", "llama"),
            ("Llama-3-8B Training", "training", "llama"),
            ("GPT-2 XL Inference", "inference", "gpt2"),
            ("GPT-2 XL Training", "training", "gpt2"),
        ]
        
        for key, mode, model in combinations:
            results[key] = {}
            for fw in ["MLX", "PyTorch (MPS)", "FusionML"]:
                fw_arg = "mlx" if "MLX" in fw else ("pytorch" if "PyTorch" in fw else "fusionml")
                print(f"Running {key} on {fw}...")
                stats = run_sub(fw_arg, mode, model)
                if stats:
                    results[key][fw] = stats
                else:
                    results[key][fw] = {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0}
                    
        # print final table
        print_table(results, 1)
        return

    # Subprocess execution logic
    fw = args.fw
    mode = args.mode
    model = args.model
    training = (mode == "training")

    # Dimensions
    B = 1
    L = 1024 if mode == "inference" else 256
    scale = 0.02

    # Load required backend
    if fw == "mlx":
        import mlx.core as mx
    elif fw == "pytorch":
        import torch
    elif fw == "fusionml":
        from fusionml.tensor import Tensor
        from fusionml._metal.tri_scheduler import get_scheduler
        # Calibrate for Llama/GPT2 shapes
        scheduler = get_scheduler()
        scheduler.calibrate(sizes=[256, 1024, 2048], verbose=False)

    # Inputs & weight templates
    x_np = np.random.randn(B, L, 4096 if model == "llama" else 1600).astype(np.float32)
    w_q_np = np.random.randn(4096 if model == "llama" else 1600, 4096 if model == "llama" else 1600).astype(np.float32) * scale
    w_k_np = np.random.randn(4096 if model == "llama" else 1600, 4096 if model == "llama" else 1600).astype(np.float32) * scale
    w_v_np = np.random.randn(4096 if model == "llama" else 1600, 4096 if model == "llama" else 1600).astype(np.float32) * scale
    w_o_np = np.random.randn(4096 if model == "llama" else 1600, 4096 if model == "llama" else 1600).astype(np.float32) * scale
    
    if model == "llama":
        w_gate_np = np.random.randn(4096, 14336).astype(np.float32) * scale
        w_up_np = np.random.randn(4096, 14336).astype(np.float32) * scale
        w_down_np = np.random.randn(14336, 4096).astype(np.float32) * scale
    else:
        w_fc1_np = np.random.randn(1600, 6400).astype(np.float32) * scale
        w_fc2_np = np.random.randn(6400, 1600).astype(np.float32) * scale

    # Setup weights/inputs for framework
    if fw == "mlx":
        x = mx.array(x_np)
        w_q = mx.array(w_q_np)
        w_k = mx.array(w_k_np)
        w_v = mx.array(w_v_np)
        w_o = mx.array(w_o_np)
        if model == "llama":
            w_gate = mx.array(w_gate_np)
            w_up = mx.array(w_up_np)
            w_down = mx.array(w_down_np)
            fn = lambda: run_mlx_llama(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, training=training)
        else:
            w_fc1 = mx.array(w_fc1_np)
            w_fc2 = mx.array(w_fc2_np)
            fn = lambda: run_mlx_gpt2(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, training=training)
            
    elif fw == "pytorch":
        x = torch.from_numpy(x_np).to("mps")
        w_q = torch.from_numpy(w_q_np).to("mps").requires_grad_(training)
        w_k = torch.from_numpy(w_k_np).to("mps").requires_grad_(training)
        w_v = torch.from_numpy(w_v_np).to("mps").requires_grad_(training)
        w_o = torch.from_numpy(w_o_np).to("mps").requires_grad_(training)
        if model == "llama":
            w_gate = torch.from_numpy(w_gate_np).to("mps").requires_grad_(training)
            w_up = torch.from_numpy(w_up_np).to("mps").requires_grad_(training)
            w_down = torch.from_numpy(w_down_np).to("mps").requires_grad_(training)
            fn = lambda: run_torch_llama(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, training=training)
        else:
            w_fc1 = torch.from_numpy(w_fc1_np).to("mps").requires_grad_(training)
            w_fc2 = torch.from_numpy(w_fc2_np).to("mps").requires_grad_(training)
            fn = lambda: run_torch_gpt2(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, training=training)
            
    elif fw == "fusionml":
        x = Tensor(x_np).to_gpu()
        w_q = Tensor(w_q_np, requires_grad=training).to_gpu()
        w_k = Tensor(w_k_np, requires_grad=training).to_gpu()
        w_v = Tensor(w_v_np, requires_grad=training).to_gpu()
        w_o = Tensor(w_o_np, requires_grad=training).to_gpu()
        if model == "llama":
            w_gate = Tensor(w_gate_np, requires_grad=training).to_gpu()
            w_up = Tensor(w_up_np, requires_grad=training).to_gpu()
            w_down = Tensor(w_down_np, requires_grad=training).to_gpu()
            fn = lambda: run_fusion_llama(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, training=training)
        else:
            w_fc1 = Tensor(w_fc1_np, requires_grad=training).to_gpu()
            w_fc2 = Tensor(w_fc2_np, requires_grad=training).to_gpu()
            fn = lambda: run_fusion_gpt2(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, training=training)

    # Benchmark execution
    warmups = 2 if training else 10
    runs = 3 if training else 20
    stats = time_run(fn, warmups=warmups, runs=runs, clear_mem=training)
    
    # print final JSON output
    print(json.dumps(stats))

if __name__ == "__main__":
    main()
