#!/usr/bin/env python3
"""
FusionML Transformer Benchmark
==============================
Benchmarks a full Transformer Encoder Block (Attention + MLP) across frameworks.
Measures End-to-End (E2E) latency: NumPy Input -> Process -> NumPy Output.

Competitors:
1. FusionML (Adaptive/Pipeline)
2. MLX (Apple Silicon Native)
3. PyTorch (MPS Backend)
4. NumPy (CPU Baseline)
"""

import time
import numpy as np
import argparse
import sys
import os

# Try imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

try:
    from fusionml._metal.fusion_engine import FusionEngine
    HAS_FUSION = True
except ImportError:
    HAS_FUSION = False

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------

def time_fn(fn, iterations=10, warmup=5):
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Measure
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0) # ms
    
    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "min_ms": np.min(times),
        "std_ms": np.std(times)
    }

# -----------------------------------------------------------------------------
# IMPLEMENTATIONS
# -----------------------------------------------------------------------------

def benchmark_transformer_block(B=1, L=128, D=512, Heads=8, iter=20):
    print(f"\nRunning Benchmark: B={B}, L={L}, D={D}, Heads={Heads}")
    results = {}
    
    # Inputs (NumPy)
    x = np.random.randn(B, L, D).astype(np.float32)
    
    # Init Weights (NumPy) - scaled to avoid overflow
    scale = 0.02
    w_q = np.random.randn(D, D).astype(np.float32) * scale
    w_k = np.random.randn(D, D).astype(np.float32) * scale
    w_v = np.random.randn(D, D).astype(np.float32) * scale
    w_o = np.random.randn(D, D).astype(np.float32) * scale
    w_ff1 = np.random.randn(D, 4*D).astype(np.float32) * scale
    w_ff2 = np.random.randn(4*D, D).astype(np.float32) * scale
    
    # 1. NumPy Baseline
    def numpy_attn_mlp():
        # Attention
        Q = x @ w_q
        K = x @ w_k
        V = x @ w_v
        
        d_head = D // Heads
        Q = Q.reshape(B, L, Heads, d_head).transpose(0, 2, 1, 3)
        K = K.reshape(B, L, Heads, d_head).transpose(0, 2, 1, 3)
        V = V.reshape(B, L, Heads, d_head).transpose(0, 2, 1, 3)
        
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_head)
        # Softmax
        exps = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exps / np.sum(exps, axis=-1, keepdims=True)
        
        out = attn @ V
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        out = (out @ w_o) + x
        
        # MLP
        ff = out @ w_ff1
        ff = np.maximum(ff, 0) # ReLU
        ff = ff @ w_ff2
        return ff + out

    results["NumPy"] = time_fn(numpy_attn_mlp, iterations=iter)
    
    # 2. MLX (End-to-End)
    if HAS_MLX:
        def mlx_e2e():
            # Conversion
            x_m = mx.array(x)
            q_m, k_m, v_m = mx.array(w_q), mx.array(w_k), mx.array(w_v)
            o_m = mx.array(w_o)
            ff1_m, ff2_m = mx.array(w_ff1), mx.array(w_ff2)
            
            # Attention
            Q = x_m @ q_m
            K = x_m @ k_m
            V = x_m @ v_m
            
            d_head = D // Heads
            Q = Q.reshape(B, L, Heads, d_head).transpose(0, 2, 1, 3)
            K = K.reshape(B, L, Heads, d_head).transpose(0, 2, 1, 3)
            V = V.reshape(B, L, Heads, d_head).transpose(0, 2, 1, 3)
            
            scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_head)
            attn = mx.softmax(scores, axis=-1)
            out = attn @ V
            
            out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
            out = (out @ o_m) + x_m
            
            # MLP
            ff = out @ ff1_m
            ff = mx.maximum(ff, mx.array(0.0))
            ff = ff @ ff2_m
            res = ff + out
            
            mx.eval(res)
            return np.array(res)
            
        results["MLX (E2E)"] = time_fn(mlx_e2e, iterations=iter)

    # 3. PyTorch (MPS End-to-End)
    if HAS_PYTORCH:
        def pt_e2e():
            x_p = torch.from_numpy(x).to("mps")
            q_p = torch.from_numpy(w_q).to("mps")
            k_p = torch.from_numpy(w_k).to("mps")
            v_p = torch.from_numpy(w_v).to("mps")
            o_p = torch.from_numpy(w_o).to("mps")
            ff1_p = torch.from_numpy(w_ff1).to("mps")
            ff2_p = torch.from_numpy(w_ff2).to("mps")
            
            # Attention
            Q = x_p @ q_p
            K = x_p @ k_p
            V = x_p @ v_p
            
            d_head = D // Heads
            Q = Q.reshape(B, L, Heads, d_head).permute(0, 2, 1, 3)
            K = K.reshape(B, L, Heads, d_head).permute(0, 2, 1, 3)
            V = V.reshape(B, L, Heads, d_head).permute(0, 2, 1, 3)
            
            scores = (Q @ K.transpose(-2, -1)) / np.sqrt(d_head)
            attn = torch.softmax(scores, dim=-1)
            out = attn @ V
            
            out = out.permute(0, 2, 1, 3).reshape(B, L, D)
            out = (out @ o_p) + x_p
            
            # MLP
            ff = out @ ff1_p
            ff = torch.relu(ff)
            ff = ff @ ff2_p
            res = ff + out
            
            torch.mps.synchronize()
            return res.cpu().numpy()
            
        results["PyTorch (E2E)"] = time_fn(pt_e2e, iterations=iter)

    # 4. FusionML
    if HAS_FUSION:
        engine = FusionEngine()
        # Warmup engine if logical for internal pools
        engine.transformer_encoder_block(x, w_q, w_k, w_v, w_o, w_ff1, w_ff2, Heads)
        
        def fusion_run():
            return engine.transformer_encoder_block(x, w_q, w_k, w_v, w_o, w_ff1, w_ff2, Heads)
            
        results["FusionML"] = time_fn(fusion_run, iterations=iter)

    # Print Results
    print(f"{'Framework':<15} | {'Median Latency':<15} | {'Throughput':<15}")
    print("-" * 50)
    
    # Sort by median time
    sorted_res = sorted(results.items(), key=lambda kv: kv[1]["median_ms"])
    winner = sorted_res[0][0]
    
    for name, stats in sorted_res:
        ms = stats["median_ms"]
        tokens_per_sec = (B * L) / (ms / 1000.0)
        marker = "✅ WINNER" if name == winner else ""
        print(f"{name:<15} | {ms:>10.3f} ms   | {tokens_per_sec:>10.1f} tok/s {marker}")

if __name__ == "__main__":
    print("===============================================================")
    print("   FusionML Transformer Encoder Benchmark (End-to-End)")
    print("===============================================================")
    
    # Small sequence (CPU dominant usually)
    benchmark_transformer_block(B=1, L=128, D=512)
    
    # Medium sequence
    benchmark_transformer_block(B=4, L=512, D=512)
    
    # Large sequence (GPU/ANE dominant)
    benchmark_transformer_block(B=8, L=1024, D=512)
    
    # Very Large
    benchmark_transformer_block(B=1, L=4096, D=512)
