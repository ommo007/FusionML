import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from fusionml.tensor import Tensor, softmax, sigmoid

B = 1
L = 1024
D = 4096
scale = 0.02

print("Creating inputs...")
x_llama_np = np.random.randn(B, L, 4096).astype(np.float32)
w_q_np = np.random.randn(4096, 4096).astype(np.float32) * scale
w_k_np = np.random.randn(4096, 4096).astype(np.float32) * scale
w_v_np = np.random.randn(4096, 4096).astype(np.float32) * scale
w_o_np = np.random.randn(4096, 4096).astype(np.float32) * scale
w_gate_np = np.random.randn(4096, 14336).astype(np.float32) * scale
w_up_np = np.random.randn(4096, 14336).astype(np.float32) * scale
w_down_np = np.random.randn(14336, 4096).astype(np.float32) * scale

print("Wrapping in Tensors...")
x = Tensor(x_llama_np).to_gpu()
w_q = Tensor(w_q_np, requires_grad=True).to_gpu()
w_k = Tensor(w_k_np, requires_grad=True).to_gpu()
w_v = Tensor(w_v_np, requires_grad=True).to_gpu()
w_o = Tensor(w_o_np, requires_grad=True).to_gpu()
w_gate = Tensor(w_gate_np, requires_grad=True).to_gpu()
w_up = Tensor(w_up_np, requires_grad=True).to_gpu()
w_down = Tensor(w_down_np, requires_grad=True).to_gpu()

print("Running forward pass...")
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
loss = res.mean()

print("Forward pass completed. Loss:", loss.numpy)

print("Running backward pass...")
loss.backward()
print("Backward pass completed successfully!")
