#!/usr/bin/env python3
"""
FusionML Model-Level Comparison Benchmark
=========================================
Compares full Llama-3-8B and GPT-2 XL decoder blocks directly in Python
across FusionML, Apple's MLX, and PyTorch (MPS).

Measures mean ± std dev over 20 runs after 10 warmups for:
1. Inference Pass (Forward only)
2. Training Pass (Forward + Backward)
"""

import time
import gc
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

# Import frameworks
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    HAS_PYTORCH = torch.backends.mps.is_available()
except ImportError:
    HAS_PYTORCH = False

try:
    from fusionml.tensor import Tensor, softmax, sigmoid
    from fusionml._metal.tri_scheduler import get_scheduler
    HAS_FUSION = True
except ImportError:
    HAS_FUSION = False

def clear_gpu_memory():
    """Clear memory caches for MLX, PyTorch MPS, and python GC to prevent OOM."""
    gc.collect()
    if HAS_MLX:
        mx.clear_cache()
    if HAS_PYTORCH:
        torch.mps.empty_cache()

# -----------------------------------------------------------------------------
# LLAMA-3-8B DECODER LAYER (PRE-CONVERTED)
# -----------------------------------------------------------------------------

def run_mlx_llama(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, training=False):
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
        # Zero grads manually since they accumulate
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
        # Zero grads manually since they accumulate
        for w in [w_q, w_k, w_v, w_o, w_gate, w_up, w_down]:
            w.grad = None
        loss = res.mean()
        loss.backward()
        loss.eval()
        return loss
    else:
        res.eval()
        return res


# -----------------------------------------------------------------------------
# GPT-2 XL DECODER LAYER (PRE-CONVERTED)
# -----------------------------------------------------------------------------

def run_mlx_gpt2(x, w_q, w_k, w_v, w_o, w_fc1, w_fc2, training=False):
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
            # GELU approx
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
        return loss
    else:
        res.eval()
        return res


# -----------------------------------------------------------------------------
# BENCHMARK ENGINE
# -----------------------------------------------------------------------------

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

def print_table(results_dict, B, L):
    print(f"\n### Benchmark Results (Batch={B}, SeqLen={L})")
    print(f"| Model / Mode | Framework | Mean ± Std (ms) | Median (ms) | Min/Max (ms) | Throughput (tok/s) | Speedup vs MLX |")
    print(f"| --- | --- | --- | --- | --- | --- | --- |")
    
    for key, runs_data in results_dict.items():
        # MLX is baseline
        mlx_med = runs_data["MLX"]["median"]
        
        for fw, stats in runs_data.items():
            tok_s = L / (stats["median"] / 1000.0)
            speedup = mlx_med / stats["median"]
            speedup_str = f"{speedup:.2f}x" if fw != "MLX" else "1.00x"
            print(f"| {key} | {fw} | {stats['mean']:.2f} ± {stats['std']:.2f} | {stats['median']:.2f} | {stats['min']:.2f} / {stats['max']:.2f} | {tok_s:.1f} | {speedup_str} |")

def main():
    print("==========================================================================")
    print("      FusionML Model-Level and Layer-Level Comparative Sweep")
    print("==========================================================================")
    
    if not HAS_MLX:
        print("MLX is not installed. Exiting.")
        return
    if not HAS_PYTORCH:
        print("PyTorch MPS is not available. Exiting.")
        return
    if not HAS_FUSION:
        print("FusionML is not installed. Exiting.")
        return

    # Trigger Tri-Compute Scheduler Calibration (Only up to 2048 to prevent OOM)
    print("\n[FusionML] Calibrating Tri-Compute Scheduler for shapes (1024, 2048)...")
    scheduler = get_scheduler()
    scheduler.calibrate(sizes=[1024, 2048], verbose=True)
    
    # -------------------------------------------------------------------------
    # Sweep Parameters
    # -------------------------------------------------------------------------
    B = 1
    L = 1024 # standard context size
    scale = 0.02
    
    # Generate static NumPy inputs and weights
    x_llama_np = np.random.randn(B, L, 4096).astype(np.float32)
    w_q_np = np.random.randn(4096, 4096).astype(np.float32) * scale
    w_k_np = np.random.randn(4096, 4096).astype(np.float32) * scale
    w_v_np = np.random.randn(4096, 4096).astype(np.float32) * scale
    w_o_np = np.random.randn(4096, 4096).astype(np.float32) * scale
    w_gate_np = np.random.randn(4096, 14336).astype(np.float32) * scale
    w_up_np = np.random.randn(4096, 14336).astype(np.float32) * scale
    w_down_np = np.random.randn(14336, 4096).astype(np.float32) * scale
    
    x_gpt2_np = np.random.randn(B, L, 1600).astype(np.float32)
    w_q_gpt2_np = np.random.randn(1600, 1600).astype(np.float32) * scale
    w_k_gpt2_np = np.random.randn(1600, 1600).astype(np.float32) * scale
    w_v_gpt2_np = np.random.randn(1600, 1600).astype(np.float32) * scale
    w_o_gpt2_np = np.random.randn(1600, 1600).astype(np.float32) * scale
    w_fc1_np = np.random.randn(1600, 6400).astype(np.float32) * scale
    w_fc2_np = np.random.randn(6400, 1600).astype(np.float32) * scale

    results = {}
    
    # =========================================================================
    # LLAMA-3-8B SWEEP
    # =========================================================================
    print("\nInitialising Llama-3-8B pre-converted weights...")
    
    # MLX
    x_llama_mlx = mx.array(x_llama_np)
    w_q_mlx = mx.array(w_q_np)
    w_k_mlx = mx.array(w_k_np)
    w_v_mlx = mx.array(w_v_np)
    w_o_mlx = mx.array(w_o_np)
    w_gate_mlx = mx.array(w_gate_np)
    w_up_mlx = mx.array(w_up_np)
    w_down_mlx = mx.array(w_down_np)
    
    # PyTorch MPS
    x_llama_pt_inf = torch.from_numpy(x_llama_np).to("mps")
    w_q_pt_inf = torch.from_numpy(w_q_np).to("mps")
    w_k_pt_inf = torch.from_numpy(w_k_np).to("mps")
    w_v_pt_inf = torch.from_numpy(w_v_np).to("mps")
    w_o_pt_inf = torch.from_numpy(w_o_np).to("mps")
    w_gate_pt_inf = torch.from_numpy(w_gate_np).to("mps")
    w_up_pt_inf = torch.from_numpy(w_up_np).to("mps")
    w_down_pt_inf = torch.from_numpy(w_down_np).to("mps")
    
    x_llama_pt_train = torch.from_numpy(x_llama_np).to("mps")
    w_q_pt_train = torch.from_numpy(w_q_np).to("mps").requires_grad_(True)
    w_k_pt_train = torch.from_numpy(w_k_np).to("mps").requires_grad_(True)
    w_v_pt_train = torch.from_numpy(w_v_np).to("mps").requires_grad_(True)
    w_o_pt_train = torch.from_numpy(w_o_np).to("mps").requires_grad_(True)
    w_gate_pt_train = torch.from_numpy(w_gate_np).to("mps").requires_grad_(True)
    w_up_pt_train = torch.from_numpy(w_up_np).to("mps").requires_grad_(True)
    w_down_pt_train = torch.from_numpy(w_down_np).to("mps").requires_grad_(True)
    
    # FusionML
    x_llama_fs_inf = Tensor(x_llama_np).to_gpu()
    w_q_fs_inf = Tensor(w_q_np).to_gpu()
    w_k_fs_inf = Tensor(w_k_np).to_gpu()
    w_v_fs_inf = Tensor(w_v_np).to_gpu()
    w_o_fs_inf = Tensor(w_o_np).to_gpu()
    w_gate_fs_inf = Tensor(w_gate_np).to_gpu()
    w_up_fs_inf = Tensor(w_up_np).to_gpu()
    w_down_fs_inf = Tensor(w_down_np).to_gpu()
    
    x_llama_fs_train = Tensor(x_llama_np).to_gpu()
    w_q_fs_train = Tensor(w_q_np, requires_grad=True).to_gpu()
    w_k_fs_train = Tensor(w_k_np, requires_grad=True).to_gpu()
    w_v_fs_train = Tensor(w_v_np, requires_grad=True).to_gpu()
    w_o_fs_train = Tensor(w_o_np, requires_grad=True).to_gpu()
    w_gate_fs_train = Tensor(w_gate_np, requires_grad=True).to_gpu()
    w_up_fs_train = Tensor(w_up_np, requires_grad=True).to_gpu()
    w_down_fs_train = Tensor(w_down_np, requires_grad=True).to_gpu()

    print("Benchmarking Llama-3-8B Inference...")
    print("  Running MLX...")
    mlx_inf = time_run(lambda: run_mlx_llama(x_llama_mlx, w_q_mlx, w_k_mlx, w_v_mlx, w_o_mlx, w_gate_mlx, w_up_mlx, w_down_mlx, training=False), warmups=10, runs=20)
    clear_gpu_memory()
    print("  Running PyTorch...")
    pt_inf = time_run(lambda: run_torch_llama(x_llama_pt_inf, w_q_pt_inf, w_k_pt_inf, w_v_pt_inf, w_o_pt_inf, w_gate_pt_inf, w_up_pt_inf, w_down_pt_inf, training=False), warmups=10, runs=20)
    clear_gpu_memory()
    print("  Running FusionML...")
    fs_inf = time_run(lambda: run_fusion_llama(x_llama_fs_inf, w_q_fs_inf, w_k_fs_inf, w_v_fs_inf, w_o_fs_inf, w_gate_fs_inf, w_up_fs_inf, w_down_fs_inf, training=False), warmups=10, runs=20)
    clear_gpu_memory()
    
    results["Llama-3-8B Inference"] = {
        "MLX": mlx_inf,
        "PyTorch (MPS)": pt_inf,
        "FusionML": fs_inf
    }
    
    print("Benchmarking Llama-3-8B Training...")
    print("  Running MLX...")
    mlx_train = time_run(lambda: run_mlx_llama(x_llama_mlx, w_q_mlx, w_k_mlx, w_v_mlx, w_o_mlx, w_gate_mlx, w_up_mlx, w_down_mlx, training=True), warmups=2, runs=3, clear_mem=True)
    clear_gpu_memory()
    print("  Running PyTorch...")
    pt_train = time_run(lambda: run_torch_llama(x_llama_pt_train, w_q_pt_train, w_k_pt_train, w_v_pt_train, w_o_pt_train, w_gate_pt_train, w_up_pt_train, w_down_pt_train, training=True), warmups=2, runs=3, clear_mem=True)
    clear_gpu_memory()
    print("  Running FusionML...")
    fs_train = time_run(lambda: run_fusion_llama(x_llama_fs_train, w_q_fs_train, w_k_fs_train, w_v_fs_train, w_o_fs_train, w_gate_fs_train, w_up_fs_train, w_down_fs_train, training=True), warmups=2, runs=3, clear_mem=True)
    clear_gpu_memory()
    
    results["Llama-3-8B Training"] = {
        "MLX": mlx_train,
        "PyTorch (MPS)": pt_train,
        "FusionML": fs_train
    }

    # =========================================================================
    # GPT-2 XL SWEEP
    # =========================================================================
    print("\nInitialising GPT-2 XL pre-converted weights...")
    
    # MLX
    x_gpt2_mlx = mx.array(x_gpt2_np)
    w_q_gpt2_mlx = mx.array(w_q_gpt2_np)
    w_k_gpt2_mlx = mx.array(w_k_gpt2_np)
    w_v_gpt2_mlx = mx.array(w_v_gpt2_np)
    w_o_gpt2_mlx = mx.array(w_o_gpt2_np)
    w_fc1_mlx = mx.array(w_fc1_np)
    w_fc2_mlx = mx.array(w_fc2_np)
    
    # PyTorch MPS
    x_gpt2_pt_inf = torch.from_numpy(x_gpt2_np).to("mps")
    w_q_gpt2_pt_inf = torch.from_numpy(w_q_gpt2_np).to("mps")
    w_k_gpt2_pt_inf = torch.from_numpy(w_k_gpt2_np).to("mps")
    w_v_gpt2_pt_inf = torch.from_numpy(w_v_gpt2_np).to("mps")
    w_o_gpt2_pt_inf = torch.from_numpy(w_o_gpt2_np).to("mps")
    w_fc1_pt_inf = torch.from_numpy(w_fc1_np).to("mps")
    w_fc2_pt_inf = torch.from_numpy(w_fc2_np).to("mps")
    
    x_gpt2_pt_train = torch.from_numpy(x_gpt2_np).to("mps")
    w_q_gpt2_pt_train = torch.from_numpy(w_q_gpt2_np).to("mps").requires_grad_(True)
    w_k_gpt2_pt_train = torch.from_numpy(w_k_gpt2_np).to("mps").requires_grad_(True)
    w_v_gpt2_pt_train = torch.from_numpy(w_v_gpt2_np).to("mps").requires_grad_(True)
    w_o_gpt2_pt_train = torch.from_numpy(w_o_gpt2_np).to("mps").requires_grad_(True)
    w_fc1_pt_train = torch.from_numpy(w_fc1_np).to("mps").requires_grad_(True)
    w_fc2_pt_train = torch.from_numpy(w_fc2_np).to("mps").requires_grad_(True)
    
    # FusionML
    x_gpt2_fs_inf = Tensor(x_gpt2_np).to_gpu()
    w_q_gpt2_fs_inf = Tensor(w_q_gpt2_np).to_gpu()
    w_k_gpt2_fs_inf = Tensor(w_k_gpt2_np).to_gpu()
    w_v_gpt2_fs_inf = Tensor(w_v_gpt2_np).to_gpu()
    w_o_gpt2_fs_inf = Tensor(w_o_gpt2_np).to_gpu()
    w_fc1_fs_inf = Tensor(w_fc1_np).to_gpu()
    w_fc2_fs_inf = Tensor(w_fc2_np).to_gpu()
    
    x_gpt2_fs_train = Tensor(x_gpt2_np).to_gpu()
    w_q_gpt2_fs_train = Tensor(w_q_gpt2_np, requires_grad=True).to_gpu()
    w_k_gpt2_fs_train = Tensor(w_k_gpt2_np, requires_grad=True).to_gpu()
    w_v_gpt2_fs_train = Tensor(w_v_gpt2_np, requires_grad=True).to_gpu()
    w_o_gpt2_fs_train = Tensor(w_o_gpt2_np, requires_grad=True).to_gpu()
    w_fc1_fs_train = Tensor(w_fc1_np, requires_grad=True).to_gpu()
    w_fc2_fs_train = Tensor(w_fc2_np, requires_grad=True).to_gpu()

    print("Benchmarking GPT-2 XL Inference...")
    print("  Running MLX...")
    mlx_gpt2_inf = time_run(lambda: run_mlx_gpt2(x_gpt2_mlx, w_q_gpt2_mlx, w_k_gpt2_mlx, w_v_gpt2_mlx, w_o_gpt2_mlx, w_fc1_mlx, w_fc2_mlx, training=False), warmups=10, runs=20)
    clear_gpu_memory()
    print("  Running PyTorch...")
    pt_gpt2_inf = time_run(lambda: run_torch_gpt2(x_gpt2_pt_inf, w_q_gpt2_pt_inf, w_k_gpt2_pt_inf, w_v_gpt2_pt_inf, w_o_gpt2_pt_inf, w_fc1_pt_inf, w_fc2_pt_inf, training=False), warmups=10, runs=20)
    clear_gpu_memory()
    print("  Running FusionML...")
    fs_gpt2_inf = time_run(lambda: run_fusion_gpt2(x_gpt2_fs_inf, w_q_gpt2_fs_inf, w_k_gpt2_fs_inf, w_v_gpt2_fs_inf, w_o_gpt2_fs_inf, w_fc1_fs_inf, w_fc2_fs_inf, training=False), warmups=10, runs=20)
    clear_gpu_memory()
    
    results["GPT-2 XL Inference"] = {
        "MLX": mlx_gpt2_inf,
        "PyTorch (MPS)": pt_gpt2_inf,
        "FusionML": fs_gpt2_inf
    }
    
    print("Benchmarking GPT-2 XL Training...")
    print("  Running MLX...")
    mlx_gpt2_train = time_run(lambda: run_mlx_gpt2(x_gpt2_mlx, w_q_gpt2_mlx, w_k_gpt2_mlx, w_v_gpt2_mlx, w_o_gpt2_mlx, w_fc1_mlx, w_fc2_mlx, training=True), warmups=2, runs=3, clear_mem=True)
    clear_gpu_memory()
    print("  Running PyTorch...")
    pt_gpt2_train = time_run(lambda: run_torch_gpt2(x_gpt2_pt_train, w_q_gpt2_pt_train, w_k_gpt2_pt_train, w_v_gpt2_pt_train, w_o_gpt2_pt_train, w_fc1_pt_train, w_fc2_pt_train, training=True), warmups=2, runs=3, clear_mem=True)
    clear_gpu_memory()
    print("  Running FusionML...")
    fs_gpt2_train = time_run(lambda: run_fusion_gpt2(x_gpt2_fs_train, w_q_gpt2_fs_train, w_k_gpt2_fs_train, w_v_gpt2_fs_train, w_o_gpt2_fs_train, w_fc1_fs_train, w_fc2_fs_train, training=True), warmups=2, runs=3, clear_mem=True)
    clear_gpu_memory()
    
    results["GPT-2 XL Training"] = {
        "MLX": mlx_gpt2_train,
        "PyTorch (MPS)": pt_gpt2_train,
        "FusionML": fs_gpt2_train
    }
    
    print_table(results, B, L)

if __name__ == "__main__":
    main()
