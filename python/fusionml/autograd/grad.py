"""
Autograd - Automatic differentiation with computation graph (GPU-compatible)
"""

from typing import Optional
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _get_numpy(tensor) -> np.ndarray:
    """Get numpy array from tensor (handles both GPU and CPU)"""
    if tensor._on_gpu:
        if tensor._mlx is not None:
            mx.eval(tensor._mlx)
            return np.array(tensor._mlx)
        return np.array([])
    return tensor._np if tensor._np is not None else np.array([])


def _create_zeros_like(tensor):
    """Create zeros tensor with same shape"""
    from ..tensor import Tensor
    if tensor._on_gpu and HAS_MLX:
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


def _add_grad(tensor, grad_np: np.ndarray):
    """Add gradient to tensor"""
    from ..tensor import Tensor
    if tensor._on_gpu and HAS_MLX:
        if tensor.grad is None:
            tensor.grad = _create_zeros_like(tensor)
        # Convert grad to MLX and add
        current = np.array(tensor.grad._mlx) if tensor.grad._mlx is not None else np.zeros(tensor.shape, dtype=np.float32)
        tensor.grad._mlx = mx.array(current + grad_np)
    else:
        if tensor.grad is None:
            tensor.grad = Tensor(np.zeros(tensor.shape, dtype=np.float32))
        tensor.grad._np = tensor.grad._np + grad_np


def backward(tensor, grad: Optional['Tensor'] = None):
    """
    Compute gradients via reverse-mode autodiff
    """
    from ..tensor import Tensor
    
    if grad is None:
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
        g = _get_numpy(t.grad) if t.grad else np.ones(t.shape, dtype=np.float32)
        
        if op == 'add':
            a, b = args
            if isinstance(a, Tensor) and a.requires_grad:
                grad_a = g
                while grad_a.ndim > len(a.shape):
                    grad_a = grad_a.sum(axis=0)
                for i, (da, dg) in enumerate(zip(a.shape, grad_a.shape)):
                    if da == 1 and dg > 1:
                        grad_a = grad_a.sum(axis=i, keepdims=True)
                _add_grad(a, grad_a)
            if isinstance(b, Tensor) and b.requires_grad:
                grad_b = g
                while grad_b.ndim > len(b.shape):
                    grad_b = grad_b.sum(axis=0)
                for i, (db, dg) in enumerate(zip(b.shape, grad_b.shape)):
                    if db == 1 and dg > 1:
                        grad_b = grad_b.sum(axis=i, keepdims=True)
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
                b_np = _get_numpy(b) if isinstance(b, Tensor) else np.array(b)
                _add_grad(a, g * b_np)
            if isinstance(b, Tensor) and b.requires_grad:
                a_np = _get_numpy(a) if isinstance(a, Tensor) else np.array(a)
                _add_grad(b, g * a_np)
                
        elif op == 'matmul':
            a, b = args
            a_np = _get_numpy(a)
            b_np = _get_numpy(b)
            if a.requires_grad:
                _add_grad(a, np.matmul(g, b_np.T))
            if b.requires_grad:
                _add_grad(b, np.matmul(a_np.T, g))
                
        elif op == 'sum':
            a, dim, keepdim = args
            if a.requires_grad:
                _add_grad(a, np.broadcast_to(g, a.shape))
                
        elif op == 'mean':
            a, dim, keepdim = args
            if a.requires_grad:
                size = np.prod(a.shape)
                _add_grad(a, np.broadcast_to(g, a.shape) / size)
                
        elif op == 'relu':
            a, = args
            if a.requires_grad:
                a_np = _get_numpy(a)
                _add_grad(a, g * (a_np > 0))
                
        elif op == 'cross_entropy':
            input_tensor, target, probs = args
            if input_tensor.requires_grad:
                batch_size = input_tensor.shape[0]
                probs_np = _get_numpy(probs)
                target_np = _get_numpy(target).astype(int).flatten()
                
                grad_ce = probs_np.copy()
                grad_ce[np.arange(batch_size), target_np] -= 1
                grad_ce /= batch_size
                
                _add_grad(input_tensor, grad_ce)
                
        elif op == 'softmax':
            # Softmax backward is complex, skip for now
            pass


def no_grad():
    """Context manager for disabling gradient computation"""
    class NoGrad:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
    return NoGrad()
