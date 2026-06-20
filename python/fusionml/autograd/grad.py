"""
Autograd - Automatic differentiation with computation graph (GPU-compatible)

GPU-native: backward ops stay on MLX where possible to avoid GPU→CPU→GPU roundtrips.
"""

from typing import Optional
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _get_data(tensor):
    """Get underlying data (MLX array if on GPU, numpy otherwise). No forced sync."""
    if tensor._on_gpu and tensor._mlx is not None:
        return tensor._mlx
    return tensor._np if tensor._np is not None else np.array([])


def _get_numpy(tensor) -> np.ndarray:
    """Get numpy array from tensor (handles both GPU and CPU) — forces sync."""
    if tensor._on_gpu:
        if tensor._mlx is not None:
            mx.eval(tensor._mlx)
            return np.array(tensor._mlx)
        return np.array([])
    return tensor._np if tensor._np is not None else np.array([])


def _is_gpu(tensor) -> bool:
    """Check if tensor is on GPU."""
    return hasattr(tensor, '_on_gpu') and tensor._on_gpu and HAS_MLX


def _create_zeros_like(tensor):
    """Create zeros tensor with same shape"""
    from ..tensor import Tensor
    if _is_gpu(tensor):
        t = Tensor.__new__(Tensor)
        t._mlx = mx.zeros(tensor.shape)
        t._np = None
        t._on_gpu = True
        t.requires_grad = False
        t._ctx = None
        t.grad = None
        return t
    else:
        return Tensor(np.zeros(tensor.shape, dtype=np.float32))


def _make_grad_tensor(data, on_gpu):
    """Create a gradient Tensor from raw data (mlx array or numpy array)."""
    from ..tensor import Tensor
    t = Tensor.__new__(Tensor)
    if on_gpu and HAS_MLX:
        if isinstance(data, mx.array):
            t._mlx = data
        else:
            t._mlx = mx.array(data)
        t._np = None
        t._on_gpu = True
    else:
        if isinstance(data, np.ndarray):
            t._np = data
        else:
            t._np = np.array(data)
        t._mlx = None
        t._on_gpu = False
    t.requires_grad = False
    t._ctx = None
    t.grad = None
    return t


def _add_grad(tensor, grad_data):
    """Add gradient to tensor. grad_data can be mlx array or numpy array.
    Stays on GPU when possible."""
    from ..tensor import Tensor

    if _is_gpu(tensor):
        # GPU path: keep everything on MLX
        if HAS_MLX and isinstance(grad_data, mx.array):
            grad_mlx = grad_data
        else:
            grad_mlx = mx.array(grad_data)

        if tensor.grad is None:
            tensor.grad = _make_grad_tensor(grad_mlx, on_gpu=True)
        else:
            tensor.grad._mlx = tensor.grad._mlx + grad_mlx
    else:
        # CPU path
        if isinstance(grad_data, np.ndarray):
            grad_np = grad_data
        elif HAS_MLX and isinstance(grad_data, mx.array):
            mx.eval(grad_data)
            grad_np = np.array(grad_data)
        else:
            grad_np = np.array(grad_data)

        if tensor.grad is None:
            tensor.grad = _make_grad_tensor(grad_np, on_gpu=False)
        else:
            tensor.grad._np = tensor.grad._np + grad_np


def _ensure_mlx(tensor_or_data):
    """Convert to MLX array if needed."""
    if isinstance(tensor_or_data, np.ndarray):
        return mx.array(tensor_or_data)
    if hasattr(tensor_or_data, '_mlx') and tensor_or_data._mlx is not None:
        return tensor_or_data._mlx
    if hasattr(tensor_or_data, '_np'):
        return mx.array(tensor_or_data._np)
    return mx.array(tensor_or_data)


def _ensure_np(tensor_or_data):
    """Convert to numpy array if needed."""
    if isinstance(tensor_or_data, np.ndarray):
        return tensor_or_data
    if hasattr(tensor_or_data, '_np') and tensor_or_data._np is not None:
        return tensor_or_data._np
    if hasattr(tensor_or_data, '_mlx') and tensor_or_data._mlx is not None:
        mx.eval(tensor_or_data._mlx)
        return np.array(tensor_or_data._mlx)
    return np.array(tensor_or_data)


def _reduce_grad(grad, target_shape):
    """Reduce gradient to match target shape (handle broadcasting).
    Works with both mlx and numpy arrays."""
    if isinstance(grad, np.ndarray):
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for i, (dt, dg) in enumerate(zip(target_shape, grad.shape)):
            if dt == 1 and dg > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    else:
        # MLX array
        while len(grad.shape) > len(target_shape):
            grad = mx.sum(grad, axis=0)
        for i, (dt, dg) in enumerate(zip(target_shape, grad.shape)):
            if dt == 1 and dg > 1:
                grad = mx.sum(grad, axis=i, keepdims=True)
        return grad


def backward(tensor, grad: Optional['Tensor'] = None):
    """
    Compute gradients via reverse-mode autodiff.
    
    GPU-native: when tensors are on GPU, backward ops use MLX directly
    to avoid expensive GPU→CPU→GPU roundtrips.
    """
    from ..tensor import Tensor
    
    if grad is None:
        if _is_gpu(tensor):
            grad = _make_grad_tensor(mx.ones(tensor.shape), on_gpu=True)
        else:
            grad = Tensor(np.ones(tensor.shape, dtype=np.float32))
    
    # Build topological order
    topo = []
    visited = set()
    
    def build_topo(t):
        if id(t) not in visited and t._ctx is not None:
            visited.add(id(t))
            op, *inputs = t._ctx
            for inp in inputs:
                if isinstance(inp, Tensor):
                    build_topo(inp)
            topo.append(t)
    
    build_topo(tensor)
    
    # Backward pass
    tensor.grad = grad
    
    for t in reversed(topo):
        if t._ctx is None:
            continue
            
        op, *args = t._ctx
        
        # Determine if we should use GPU path
        use_gpu = _is_gpu(t) and t.grad is not None and _is_gpu(t.grad)
        
        if use_gpu:
            g = t.grad._mlx
        else:
            g = _get_numpy(t.grad) if t.grad else np.ones(t.shape, dtype=np.float32)
        
        if op == 'add':
            a, b = args
            if isinstance(a, Tensor) and a.requires_grad:
                grad_a = _reduce_grad(g, a.shape)
                _add_grad(a, grad_a)
            if isinstance(b, Tensor) and b.requires_grad:
                grad_b = _reduce_grad(g, b.shape)
                _add_grad(b, grad_b)
                
        elif op == 'sub':
            a, b = args
            if isinstance(a, Tensor) and a.requires_grad:
                _add_grad(a, g)
            if isinstance(b, Tensor) and b.requires_grad:
                _add_grad(b, -g)
                
        elif op == 'mul':
            a, b = args
            if isinstance(a, Tensor) and a.requires_grad:
                if use_gpu:
                    b_mlx = b._mlx if _is_gpu(b) else (mx.array(b.numpy) if isinstance(b, Tensor) else mx.array(b))
                    grad_a = _reduce_grad(g * b_mlx, a.shape)
                else:
                    b_np = _ensure_np(b) if isinstance(b, Tensor) else np.array(b)
                    g_np = g if isinstance(g, np.ndarray) else _get_numpy(t.grad)
                    grad_a = _reduce_grad(g_np * b_np, a.shape)
                _add_grad(a, grad_a)
            if isinstance(b, Tensor) and b.requires_grad:
                if use_gpu:
                    a_mlx = a._mlx if _is_gpu(a) else (mx.array(a.numpy) if isinstance(a, Tensor) else mx.array(a))
                    grad_b = _reduce_grad(g * a_mlx, b.shape)
                else:
                    a_np = _ensure_np(a) if isinstance(a, Tensor) else np.array(a)
                    g_np = g if isinstance(g, np.ndarray) else _get_numpy(t.grad)
                    grad_b = _reduce_grad(g_np * a_np, b.shape)
                _add_grad(b, grad_b)
                
        elif op == 'matmul':
            a, b = args
            if use_gpu and _is_gpu(a) and _is_gpu(b):
                # GPU-native matmul backward — no CPU sync!
                if a.requires_grad:
                    _add_grad(a, g @ mx.transpose(b._mlx))
                if b.requires_grad:
                    _add_grad(b, mx.transpose(a._mlx) @ g)
            else:
                # CPU fallback
                a_np = _ensure_np(a)
                b_np = _ensure_np(b)
                g_np = g if isinstance(g, np.ndarray) else _get_numpy(t.grad)
                if a.requires_grad:
                    _add_grad(a, np.matmul(g_np, b_np.T))
                if b.requires_grad:
                    _add_grad(b, np.matmul(a_np.T, g_np))
                
        elif op == 'sum':
            a, dim, keepdim = args
            if a.requires_grad:
                if use_gpu:
                    _add_grad(a, mx.broadcast_to(g, a.shape))
                else:
                    _add_grad(a, np.broadcast_to(g, a.shape))
                
        elif op == 'mean':
            a, dim, keepdim = args
            if a.requires_grad:
                size = 1
                for d in a.shape:
                    size *= d
                if use_gpu:
                    _add_grad(a, mx.broadcast_to(g, a.shape) / size)
                else:
                    _add_grad(a, np.broadcast_to(g, a.shape) / size)
                
        elif op == 'relu':
            a, = args
            if a.requires_grad:
                if use_gpu and _is_gpu(a):
                    _add_grad(a, g * (a._mlx > 0).astype(mx.float32))
                else:
                    a_np = _ensure_np(a)
                    g_np = g if isinstance(g, np.ndarray) else _get_numpy(t.grad)
                    _add_grad(a, g_np * (a_np > 0))
                    
        elif op == 'sigmoid':
            a, = args
            if a.requires_grad:
                if use_gpu and _is_gpu(t):
                    sig = t._mlx  # sigmoid output
                    _add_grad(a, g * sig * (1.0 - sig))
                else:
                    t_np = _ensure_np(t)
                    g_np = g if isinstance(g, np.ndarray) else _get_numpy(t.grad)
                    _add_grad(a, g_np * t_np * (1.0 - t_np))

        elif op == 'silu':
            a, = args
            if a.requires_grad:
                if use_gpu and _is_gpu(a):
                    sig = mx.sigmoid(a._mlx)
                    # d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                    #                        = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                    dsilu = sig * (1 + a._mlx * (1 - sig))
                    _add_grad(a, g * dsilu)
                else:
                    a_np = _ensure_np(a)
                    g_np = g if isinstance(g, np.ndarray) else _get_numpy(t.grad)
                    sig = 1 / (1 + np.exp(-a_np))
                    dsilu = sig * (1 + a_np * (1 - sig))
                    _add_grad(a, g_np * dsilu)

        elif op == 'gelu':
            a, = args
            if a.requires_grad:
                # GELU'(x) via chain rule on tanh approximation
                if use_gpu and _is_gpu(a):
                    x = a._mlx
                    c = mx.sqrt(mx.array(2.0 / np.pi))
                    inner = c * (x + 0.044715 * x ** 3)
                    tanh_inner = mx.tanh(inner)
                    # d/dx = 0.5 * (1 + tanh) + 0.5 * x * (1 - tanh^2) * c * (1 + 3*0.044715*x^2)
                    dgelu = 0.5 * (1 + tanh_inner) + 0.5 * x * (1 - tanh_inner ** 2) * c * (1 + 3 * 0.044715 * x ** 2)
                    _add_grad(a, g * dgelu)
                else:
                    x_np = _ensure_np(a)
                    g_np = g if isinstance(g, np.ndarray) else _get_numpy(t.grad)
                    c = np.sqrt(2.0 / np.pi)
                    inner = c * (x_np + 0.044715 * x_np ** 3)
                    tanh_inner = np.tanh(inner)
                    dgelu = 0.5 * (1 + tanh_inner) + 0.5 * x_np * (1 - tanh_inner ** 2) * c * (1 + 3 * 0.044715 * x_np ** 2)
                    _add_grad(a, g_np * dgelu)

        elif op == 'layer_norm':
            x_in, gamma, beta, eps = args
            if x_in.requires_grad or gamma.requires_grad or beta.requires_grad:
                if use_gpu and _is_gpu(x_in):
                    x_mlx = x_in._mlx
                    g_mlx = gamma._mlx if _is_gpu(gamma) else mx.array(gamma._np)
                    mean = mx.mean(x_mlx, axis=-1, keepdims=True)
                    var = mx.var(x_mlx, axis=-1, keepdims=True)
                    std = mx.sqrt(var + eps)
                    x_hat = (x_mlx - mean) / std
                    D = x_mlx.shape[-1]

                    if beta.requires_grad:
                        # dL/dbeta = sum over batch dims
                        dbeta = mx.sum(g, axis=tuple(range(len(g.shape) - 1)))
                        _add_grad(beta, dbeta)

                    if gamma.requires_grad:
                        dgamma = mx.sum(g * x_hat, axis=tuple(range(len(g.shape) - 1)))
                        _add_grad(gamma, dgamma)

                    if x_in.requires_grad:
                        dx_hat = g * g_mlx
                        dvar = mx.sum(dx_hat * (x_mlx - mean) * (-0.5) * (var + eps) ** (-1.5), axis=-1, keepdims=True)
                        dmean = mx.sum(dx_hat * (-1.0 / std), axis=-1, keepdims=True)
                        dx = dx_hat / std + dvar * 2 * (x_mlx - mean) / D + dmean / D
                        _add_grad(x_in, dx)
                else:
                    x_np = _ensure_np(x_in)
                    g_np_val = _ensure_np(gamma)
                    g_np = g if isinstance(g, np.ndarray) else _get_numpy(t.grad)
                    mean = np.mean(x_np, axis=-1, keepdims=True)
                    var = np.var(x_np, axis=-1, keepdims=True)
                    std = np.sqrt(var + eps)
                    x_hat = (x_np - mean) / std
                    D = x_np.shape[-1]

                    if beta.requires_grad:
                        dbeta = np.sum(g_np, axis=tuple(range(len(g_np.shape) - 1)))
                        _add_grad(beta, dbeta)

                    if gamma.requires_grad:
                        dgamma = np.sum(g_np * x_hat, axis=tuple(range(len(g_np.shape) - 1)))
                        _add_grad(gamma, dgamma)

                    if x_in.requires_grad:
                        dx_hat = g_np * g_np_val
                        dvar = np.sum(dx_hat * (x_np - mean) * (-0.5) * (var + eps) ** (-1.5), axis=-1, keepdims=True)
                        dmean = np.sum(dx_hat * (-1.0 / std), axis=-1, keepdims=True)
                        dx = dx_hat / std + dvar * 2 * (x_np - mean) / D + dmean / D
                        _add_grad(x_in, dx)

        elif op == 'cross_entropy':
            input_tensor, target, probs = args
            if input_tensor.requires_grad:
                batch_size = input_tensor.shape[0]
                probs_np = _ensure_np(probs)
                target_np = _ensure_np(target).astype(int).flatten()
                
                grad_ce = probs_np.copy()
                grad_ce[np.arange(batch_size), target_np] -= 1
                grad_ce /= batch_size
                
                _add_grad(input_tensor, grad_ce)
                
        elif op == 'softmax':
            a, axis = args
            if a.requires_grad:
                if use_gpu and _is_gpu(t):
                    y = t._mlx
                    sum_gy = mx.sum(g * y, axis=axis, keepdims=True)
                    _add_grad(a, y * (g - sum_gy))
                else:
                    y_np = _ensure_np(t)
                    g_np = g if isinstance(g, np.ndarray) else _get_numpy(t.grad)
                    sum_gy = np.sum(g_np * y_np, axis=axis, keepdims=True)
                    _add_grad(a, y_np * (g_np - sum_gy))
                
        elif op == 'transpose':
            a, = args
            if a.requires_grad:
                if use_gpu:
                    _add_grad(a, mx.transpose(g))
                else:
                    _add_grad(a, g.T)
                
        elif op == 'reshape':
            a, orig_shape = args
            if a.requires_grad:
                if use_gpu:
                    _add_grad(a, g.reshape(orig_shape))
                else:
                    _add_grad(a, g.reshape(orig_shape))


def no_grad():
    """Context manager for disabling gradient computation"""
    class NoGrad:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
    return NoGrad()
