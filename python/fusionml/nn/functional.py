"""
Functional operations - stateless functions (GPU-compatible)
"""

import numpy as np
from ..tensor import Tensor, relu as tensor_relu, sigmoid as tensor_sigmoid, softmax as tensor_softmax, log_softmax as tensor_log_softmax

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def relu(x: Tensor) -> Tensor:
    """ReLU activation"""
    return tensor_relu(x)


def gelu(x: Tensor) -> Tensor:
    """GELU activation"""
    if x._on_gpu and HAS_MLX:
        result = Tensor.__new__(Tensor)
        x_mlx = x._mlx
        result._mlx = 0.5 * x_mlx * (1 + mx.tanh(np.sqrt(2/np.pi) * (x_mlx + 0.044715 * x_mlx ** 3)))
        result._np = None
        result._on_gpu = True
        result.requires_grad = x.requires_grad
        result._ctx = None
        result.grad = None
        return result
    else:
        data = x.numpy
        result_data = 0.5 * data * (1 + np.tanh(np.sqrt(2/np.pi) * (data + 0.044715 * data**3)))
        return Tensor(result_data.astype(np.float32), requires_grad=x.requires_grad)


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation"""  
    return tensor_sigmoid(x)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax"""
    return tensor_softmax(x, axis=dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Log softmax"""
    return tensor_log_softmax(x, axis=dim)


def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    """
    Cross entropy loss
    input: [batch, classes] logits
    target: [batch] class indices
    """
    batch_size = input.shape[0]
    
    if input._on_gpu and HAS_MLX:
        logits = input._mlx
        
        # Softmax
        max_logits = mx.max(logits, axis=1, keepdims=True)
        exp_logits = mx.exp(logits - max_logits)
        probs = exp_logits / mx.sum(exp_logits, axis=1, keepdims=True)
        
        # Get target indices
        target_np = target.numpy.astype(int).flatten()
        
        # Cross entropy - need to index into probs
        # MLX doesn't have advanced indexing, so convert for this part
        mx.eval(probs)
        probs_np = np.array(probs)
        log_probs = np.log(probs_np[np.arange(batch_size), target_np] + 1e-7)
        loss = -np.mean(log_probs)
        
        result = Tensor(np.array([loss], dtype=np.float32), requires_grad=input.requires_grad)
        if input.requires_grad:
            result._ctx = ('cross_entropy', input, target, Tensor(probs_np))
        return result
    else:
        logits = input.numpy
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        target_idx = target.numpy.astype(int).flatten()
        log_probs = np.log(probs[np.arange(batch_size), target_idx] + 1e-7)
        loss = -np.mean(log_probs)
        
        result = Tensor(np.array([loss], dtype=np.float32), requires_grad=input.requires_grad)
        if input.requires_grad:
            result._ctx = ('cross_entropy', input, target, Tensor(probs))
        return result


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss"""
    if input._on_gpu and HAS_MLX:
        diff = input._mlx - (target._mlx if target._on_gpu else mx.array(target.numpy))
        loss = mx.mean(diff ** 2)
        mx.eval(loss)
        result = Tensor(np.array([float(loss)], dtype=np.float32), requires_grad=input.requires_grad)
    else:
        diff = input.numpy - target.numpy
        loss = np.mean(diff ** 2)
        result = Tensor(np.array([loss], dtype=np.float32), requires_grad=input.requires_grad)
    
    if input.requires_grad:
        result._ctx = ('mse', input, target)
    return result


def l1_loss(input: Tensor, target: Tensor) -> Tensor:
    """L1 loss"""
    if input._on_gpu and HAS_MLX:
        diff = input._mlx - (target._mlx if target._on_gpu else mx.array(target.numpy))
        loss = mx.mean(mx.abs(diff))
        mx.eval(loss)
        return Tensor(np.array([float(loss)], dtype=np.float32))
    else:
        diff = input.numpy - target.numpy
        loss = np.mean(np.abs(diff))
        return Tensor(np.array([loss], dtype=np.float32))


def linear(input: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    """Linear transformation"""
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out
