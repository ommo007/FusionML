"""
Functional operations - stateless functions
"""

import numpy as np
from ..tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """ReLU activation"""
    result = Tensor(np.maximum(x._data, 0), requires_grad=x.requires_grad)
    if x.requires_grad:
        result._ctx = ('relu', x)
    return result


def gelu(x: Tensor) -> Tensor:
    """GELU activation"""
    data = x._data
    result = 0.5 * data * (1 + np.tanh(np.sqrt(2/np.pi) * (data + 0.044715 * data**3)))
    return Tensor(result.astype(np.float32), requires_grad=x.requires_grad)


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation"""  
    return Tensor(1 / (1 + np.exp(-x._data)), requires_grad=x.requires_grad)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax"""
    exp_x = np.exp(x._data - np.max(x._data, axis=dim, keepdims=True))
    result = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    out = Tensor(result.astype(np.float32), requires_grad=x.requires_grad)
    if x.requires_grad:
        out._ctx = ('softmax', x)
    return out


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Log softmax"""
    max_x = np.max(x._data, axis=dim, keepdims=True)
    exp_x = np.exp(x._data - max_x)
    log_sum_exp = np.log(np.sum(exp_x, axis=dim, keepdims=True))
    result = x._data - max_x - log_sum_exp
    return Tensor(result.astype(np.float32), requires_grad=x.requires_grad)


def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    """
    Cross entropy loss
    input: [batch, classes] logits
    target: [batch] class indices
    """
    batch_size = input.shape[0]
    num_classes = input.shape[1]
    
    # Softmax
    logits = input._data
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Cross entropy
    target_idx = target._data.astype(int).flatten()
    log_probs = np.log(probs[np.arange(batch_size), target_idx] + 1e-7)
    loss = -np.mean(log_probs)
    
    result = Tensor(np.array([loss], dtype=np.float32), requires_grad=input.requires_grad)
    if input.requires_grad:
        # Store for backward
        result._ctx = ('cross_entropy', input, target, probs)
    return result


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss"""
    diff = input._data - target._data
    loss = np.mean(diff ** 2)
    
    result = Tensor(np.array([loss], dtype=np.float32), requires_grad=input.requires_grad)
    if input.requires_grad:
        result._ctx = ('mse', input, target)
    return result


def l1_loss(input: Tensor, target: Tensor) -> Tensor:
    """L1 loss"""
    diff = input._data - target._data
    loss = np.mean(np.abs(diff))
    return Tensor(np.array([loss], dtype=np.float32))


def linear(input: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    """Linear transformation"""
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out
