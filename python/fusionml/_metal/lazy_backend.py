"""
MLX Lazy Evaluation Backend
Uses MLX's lazy evaluation for maximum performance on large operations
"""

import numpy as np
from typing import Optional, List

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class LazyTensor:
    """
    Tensor that stays on MLX GPU with lazy evaluation
    Only converts back to numpy when explicitly requested
    """
    
    def __init__(self, data, is_mlx: bool = False):
        if is_mlx:
            self._mlx = data
            self._np = None
        elif isinstance(data, np.ndarray):
            self._np = data
            self._mlx = None
        else:
            self._np = np.array(data, dtype=np.float32)
            self._mlx = None
    
    @property
    def mlx(self):
        """Get MLX array (lazy creation)"""
        if self._mlx is None and self._np is not None:
            self._mlx = mx.array(self._np)
        return self._mlx
    
    @property
    def numpy(self):
        """Get numpy array (forces evaluation)"""
        if self._np is None and self._mlx is not None:
            mx.eval(self._mlx)
            self._np = np.array(self._mlx)
        return self._np
    
    def __matmul__(self, other: 'LazyTensor') -> 'LazyTensor':
        """Matrix multiply - stays on GPU, no eval"""
        result_mlx = self.mlx @ other.mlx
        return LazyTensor(result_mlx, is_mlx=True)
    
    def __add__(self, other: 'LazyTensor') -> 'LazyTensor':
        result_mlx = self.mlx + other.mlx
        return LazyTensor(result_mlx, is_mlx=True)
    
    def __mul__(self, other) -> 'LazyTensor':
        if isinstance(other, LazyTensor):
            result_mlx = self.mlx * other.mlx
        else:
            result_mlx = self.mlx * other
        return LazyTensor(result_mlx, is_mlx=True)
    
    def __sub__(self, other: 'LazyTensor') -> 'LazyTensor':
        result_mlx = self.mlx - other.mlx
        return LazyTensor(result_mlx, is_mlx=True)
    
    def sum(self, axis=None):
        result_mlx = mx.sum(self.mlx, axis=axis)
        return LazyTensor(result_mlx, is_mlx=True)
    
    def mean(self, axis=None):
        result_mlx = mx.mean(self.mlx, axis=axis)
        return LazyTensor(result_mlx, is_mlx=True)
    
    def exp(self) -> 'LazyTensor':
        return LazyTensor(mx.exp(self.mlx), is_mlx=True)
    
    def log(self) -> 'LazyTensor':
        return LazyTensor(mx.log(self.mlx), is_mlx=True)
    
    def relu(self) -> 'LazyTensor':
        return LazyTensor(mx.maximum(self.mlx, 0), is_mlx=True)
    
    def softmax(self, axis=-1) -> 'LazyTensor':
        return LazyTensor(mx.softmax(self.mlx, axis=axis), is_mlx=True)
    
    def T(self) -> 'LazyTensor':
        return LazyTensor(self.mlx.T, is_mlx=True)
    
    def eval(self):
        """Force evaluation"""
        if self._mlx is not None:
            mx.eval(self._mlx)
        return self
    
    @property
    def shape(self):
        if self._mlx is not None:
            return tuple(self._mlx.shape)
        return self._np.shape


def batch_eval(*tensors: LazyTensor):
    """Evaluate multiple tensors in a single GPU sync"""
    mlx_arrays = [t._mlx for t in tensors if t._mlx is not None]
    if mlx_arrays:
        mx.eval(*mlx_arrays)


def lazy_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Single matmul using MLX - for when you need numpy result immediately
    """
    if not HAS_MLX:
        return np.matmul(a, b)
    
    a_mlx = mx.array(a)
    b_mlx = mx.array(b)
    c_mlx = a_mlx @ b_mlx
    mx.eval(c_mlx)
    return np.array(c_mlx)


def device_info():
    return {
        "backend": "mlx-lazy",
        "has_mlx": HAS_MLX,
        "strategy": "Lazy evaluation - batch operations, single GPU sync"
    }


__all__ = ['LazyTensor', 'batch_eval', 'lazy_matmul', 'device_info', 'HAS_MLX']
