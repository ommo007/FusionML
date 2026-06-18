#!/usr/bin/env python3
"""
Verify FusionML autograd gradients against MLX and PyTorch (MPS)
for correctness on Llama-3-8B decoder block components.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "python")))

import mlx.core as mx
import torch
from fusionml.tensor import Tensor, softmax, sigmoid

def main():
    print("===============================================================")
    # Set seed for reproducibility
    np.random.seed(42)
    
    B = 1
    L = 128
    D = 256  # smaller size for quick verify
    scale = 0.02
    
    # Generate static NumPy inputs
    x_np = np.random.randn(B, L, D).astype(np.float32)
    w_q_np = np.random.randn(D, D).astype(np.float32) * scale
    w_k_np = np.random.randn(D, D).astype(np.float32) * scale
    w_v_np = np.random.randn(D, D).astype(np.float32) * scale
    w_o_np = np.random.randn(D, D).astype(np.float32) * scale
    w_gate_np = np.random.randn(D, 2*D).astype(np.float32) * scale
    w_up_np = np.random.randn(D, 2*D).astype(np.float32) * scale
    w_down_np = np.random.randn(2*D, D).astype(np.float32) * scale

    # 1. PyTorch MPS
    x_pt = torch.from_numpy(x_np).to("mps")
    w_q_pt = torch.from_numpy(w_q_np).to("mps").requires_grad_(True)
    w_k_pt = torch.from_numpy(w_k_np).to("mps").requires_grad_(True)
    w_v_pt = torch.from_numpy(w_v_np).to("mps").requires_grad_(True)
    w_o_pt = torch.from_numpy(w_o_np).to("mps").requires_grad_(True)
    w_gate_pt = torch.from_numpy(w_gate_np).to("mps").requires_grad_(True)
    w_up_pt = torch.from_numpy(w_up_np).to("mps").requires_grad_(True)
    w_down_pt = torch.from_numpy(w_down_np).to("mps").requires_grad_(True)

    # PyTorch Forward
    x_flat_pt = x_pt.reshape(L, D)
    q_pt = x_flat_pt @ w_q_pt
    k_pt = x_flat_pt @ w_k_pt
    v_pt = x_flat_pt @ w_v_pt
    scores_pt = (q_pt @ k_pt.T) * (1.0 / np.sqrt(D))
    attn_pt = torch.softmax(scores_pt, dim=-1)
    out_pt = attn_pt @ v_pt
    attn_out_pt = out_pt @ w_o_pt
    h1_pt = x_flat_pt + attn_out_pt
    gate_pt = h1_pt @ w_gate_pt
    silu_gate_pt = gate_pt * torch.sigmoid(gate_pt)
    up_pt = h1_pt @ w_up_pt
    mlp_out_pt = (silu_gate_pt * up_pt) @ w_down_pt
    res_pt = h1_pt + mlp_out_pt
    loss_pt = torch.mean(res_pt)
    
    # PyTorch Backward
    loss_pt.backward()
    torch.mps.synchronize()

    # 2. MLX
    x_mlx = mx.array(x_np)
    w_q_mlx = mx.array(w_q_np)
    w_k_mlx = mx.array(w_k_np)
    w_v_mlx = mx.array(w_v_np)
    w_o_mlx = mx.array(w_o_np)
    w_gate_mlx = mx.array(w_gate_np)
    w_up_mlx = mx.array(w_up_np)
    w_down_mlx = mx.array(w_down_np)

    def loss_fn_mlx(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down):
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

    grad_fn_mlx = mx.value_and_grad(loss_fn_mlx, argnums=[1, 2, 3, 4, 5, 6, 7])
    loss_mlx, grads_mlx = grad_fn_mlx(x_mlx, w_q_mlx, w_k_mlx, w_v_mlx, w_o_mlx, w_gate_mlx, w_up_mlx, w_down_mlx)
    mx.eval(loss_mlx, grads_mlx)

    # 3. FusionML
    x_fs = Tensor(x_np).to_gpu()
    w_q_fs = Tensor(w_q_np, requires_grad=True).to_gpu()
    w_k_fs = Tensor(w_k_np, requires_grad=True).to_gpu()
    w_v_fs = Tensor(w_v_np, requires_grad=True).to_gpu()
    w_o_fs = Tensor(w_o_np, requires_grad=True).to_gpu()
    w_gate_fs = Tensor(w_gate_np, requires_grad=True).to_gpu()
    w_up_fs = Tensor(w_up_np, requires_grad=True).to_gpu()
    w_down_fs = Tensor(w_down_np, requires_grad=True).to_gpu()

    # FusionML Forward
    x_flat_fs = x_fs.reshape(L, D)
    q_fs = x_flat_fs @ w_q_fs
    k_fs = x_flat_fs @ w_k_fs
    v_fs = x_flat_fs @ w_v_fs
    scores_fs = (q_fs @ k_fs.T) * (1.0 / np.sqrt(D))
    attn_fs = softmax(scores_fs, axis=-1)
    out_fs = attn_fs @ v_fs
    attn_out_fs = out_fs @ w_o_fs
    h1_fs = x_flat_fs + attn_out_fs
    gate_fs = h1_fs @ w_gate_fs
    silu_gate_fs = gate_fs * sigmoid(gate_fs)
    up_fs = h1_fs @ w_up_fs
    mlp_out_fs = (silu_gate_fs * up_fs) @ w_down_fs
    res_fs = h1_fs + mlp_out_fs
    loss_fs = res_fs.mean()

    # FusionML Backward
    loss_fs.backward()
    loss_fs.eval()

    # Get PyTorch grads as numpy
    grads_pt = [
        w_q_pt.grad.cpu().numpy(),
        w_k_pt.grad.cpu().numpy(),
        w_v_pt.grad.cpu().numpy(),
        w_o_pt.grad.cpu().numpy(),
        w_gate_pt.grad.cpu().numpy(),
        w_up_pt.grad.cpu().numpy(),
        w_down_pt.grad.cpu().numpy(),
    ]

    # Get MLX grads as numpy
    grads_mlx_np = [np.array(g) for g in grads_mlx]

    # Get FusionML grads as numpy
    grads_fs = [
        w_q_fs.grad.numpy,
        w_k_fs.grad.numpy,
        w_v_fs.grad.numpy,
        w_o_fs.grad.numpy,
        w_gate_fs.grad.numpy,
        w_up_fs.grad.numpy,
        w_down_fs.grad.numpy,
    ]

    names = ["w_q", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down"]
    
    print("\n--- Correctness Verification: FusionML vs PyTorch (MPS) and MLX ---")
    all_pass = True
    for name, pt_g, mlx_g, fs_g in zip(names, grads_pt, grads_mlx_np, grads_fs):
        err_pt = np.max(np.abs(fs_g - pt_g))
        err_mlx = np.max(np.abs(fs_g - mlx_g))
        print(f"{name:<8} | Max Abs Error vs PyTorch: {err_pt:.2e} | vs MLX: {err_mlx:.2e}")
        if err_pt > 1e-4 or err_mlx > 1e-4:
            all_pass = False
            
    if all_pass:
        print("\n✅ Verification SUCCESSFUL! Gradients match PyTorch and MLX within 1e-4 tolerance.")
    else:
        print("\n❌ Verification FAILED! Gradient discrepancy detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()
