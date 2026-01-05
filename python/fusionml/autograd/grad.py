"""
Autograd - Automatic differentiation with computation graph
"""

from typing import Optional
import numpy as np


def backward(tensor, grad: Optional['Tensor'] = None):
    """
    Compute gradients via reverse-mode autodiff
    """
    from ..tensor import Tensor
    
    if grad is None:
        grad = Tensor(np.ones_like(tensor._data))
    
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
        g = t.grad._data if t.grad else np.ones_like(t._data)
        
        if op == 'add':
            a, b = args
            if isinstance(a, Tensor) and a.requires_grad:
                if a.grad is None:
                    a.grad = Tensor(np.zeros_like(a._data))
                # Handle broadcasting - sum along extra dims
                grad_a = g
                while grad_a.ndim > a.ndim:
                    grad_a = grad_a.sum(axis=0)
                for i, (da, dg) in enumerate(zip(a.shape, grad_a.shape)):
                    if da == 1 and dg > 1:
                        grad_a = grad_a.sum(axis=i, keepdims=True)
                a.grad._data += grad_a
            if isinstance(b, Tensor) and b.requires_grad:
                if b.grad is None:
                    b.grad = Tensor(np.zeros_like(b._data))
                grad_b = g
                while grad_b.ndim > b.ndim:
                    grad_b = grad_b.sum(axis=0)
                for i, (db, dg) in enumerate(zip(b.shape, grad_b.shape)):
                    if db == 1 and dg > 1:
                        grad_b = grad_b.sum(axis=i, keepdims=True)
                b.grad._data += grad_b
                
        elif op == 'sub':
            a, b = args
            if isinstance(a, Tensor) and a.requires_grad:
                if a.grad is None:
                    a.grad = Tensor(np.zeros_like(a._data))
                a.grad._data += g
            if isinstance(b, Tensor) and b.requires_grad:
                if b.grad is None:
                    b.grad = Tensor(np.zeros_like(b._data))
                b.grad._data -= g
                
        elif op == 'mul':
            a, b = args
            if isinstance(a, Tensor) and a.requires_grad:
                b_data = b._data if isinstance(b, Tensor) else b
                if a.grad is None:
                    a.grad = Tensor(np.zeros_like(a._data))
                a.grad._data += g * b_data
            if isinstance(b, Tensor) and b.requires_grad:
                a_data = a._data if isinstance(a, Tensor) else a
                if b.grad is None:
                    b.grad = Tensor(np.zeros_like(b._data))
                b.grad._data += g * a_data
                
        elif op == 'matmul':
            a, b = args
            # dL/dA = grad @ B^T
            # dL/dB = A^T @ grad
            if a.requires_grad:
                if a.grad is None:
                    a.grad = Tensor(np.zeros_like(a._data))
                a.grad._data += np.matmul(g, b._data.T)
            if b.requires_grad:
                if b.grad is None:
                    b.grad = Tensor(np.zeros_like(b._data))
                b.grad._data += np.matmul(a._data.T, g)
                
        elif op == 'sum':
            a, dim, keepdim = args
            if a.requires_grad:
                if a.grad is None:
                    a.grad = Tensor(np.zeros_like(a._data))
                a.grad._data += np.broadcast_to(g, a.shape)
                
        elif op == 'mean':
            a, dim, keepdim = args
            if a.requires_grad:
                if a.grad is None:
                    a.grad = Tensor(np.zeros_like(a._data))
                a.grad._data += np.broadcast_to(g, a.shape) / a.numel()
                
        elif op == 'relu':
            a, = args
            if a.requires_grad:
                if a.grad is None:
                    a.grad = Tensor(np.zeros_like(a._data))
                a.grad._data += g * (a._data > 0)
                
        elif op == 'softmax':
            # Softmax backward is complex, approximate
            pass


def no_grad():
    """Context manager for disabling gradient computation"""
    class NoGrad:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
    return NoGrad()
