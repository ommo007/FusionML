"""
Tensor - Core tensor class with GPU+CPU intelligent routing
"""

import numpy as np
from typing import Union, List, Optional, Tuple
from ._metal import HAS_METAL

# Import backends once at module level
try:
    from ._metal.hybrid_backend import hybrid_matmul, HAS_MLX
    from ._metal.parallel_tiled import parallel_tiled_matmul, find_optimal_ratio
    from ._metal.optimized_ops import (
        parallel_strassen_matmul,
        parallel_add,
        parallel_mul,
        parallel_exp,
        parallel_softmax,
        parallel_sum
    )
    HAS_PARALLEL = True
    HAS_OPTIMIZED = True
except ImportError:
    HAS_MLX = False
    HAS_PARALLEL = False
    HAS_OPTIMIZED = False
    hybrid_matmul = None
    parallel_tiled_matmul = None
    find_optimal_ratio = None
    parallel_strassen_matmul = None
    parallel_add = None
    parallel_mul = None
    parallel_exp = None
    parallel_softmax = None
    parallel_sum = None

ArrayLike = Union[np.ndarray, List, float, int]

class Tensor:
    """
    FusionML Tensor with automatic GPU+CPU routing
    """
    
    def __init__(self, data: ArrayLike, requires_grad: bool = False):
        if isinstance(data, Tensor):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        else:
            self._data = np.array(data, dtype=np.float32)
        
        self.requires_grad = requires_grad
        self.grad: Optional['Tensor'] = None
        self._grad_fn = None
        self._ctx = None  # For backward
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    @property  
    def ndim(self) -> int:
        return self._data.ndim
    
    @property
    def dtype(self):
        return self._data.dtype
    
    def __len__(self) -> int:
        return len(self._data)
    
    def numel(self) -> int:
        """Number of elements"""
        return self._data.size
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return self._data.copy()
    
    def item(self) -> float:
        """Get scalar value"""
        return float(self._data.flat[0])
    
    def __repr__(self) -> str:
        return f"Tensor({self._data}, requires_grad={self.requires_grad})"
    
    # ===== Factory Methods =====
    
    @staticmethod
    def zeros(*shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return Tensor(np.zeros(shape, dtype=np.float32))
    
    @staticmethod
    def ones(*shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return Tensor(np.ones(shape, dtype=np.float32))
    
    @staticmethod
    def rand(*shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return Tensor(np.random.rand(*shape).astype(np.float32))
    
    @staticmethod
    def randn(*shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return Tensor(np.random.randn(*shape).astype(np.float32))
    
    @staticmethod
    def eye(n: int) -> 'Tensor':
        return Tensor(np.eye(n, dtype=np.float32))
    
    @staticmethod  
    def from_numpy(arr: np.ndarray) -> 'Tensor':
        return Tensor(arr)

    # ===== Operations =====
    
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        other_data = other._data if isinstance(other, Tensor) else other
        
        # Use parallel add for large tensors
        if HAS_OPTIMIZED and parallel_add is not None and isinstance(other, Tensor) and self._data.size > 500_000:
            result_data = parallel_add(self._data, other_data)
        else:
            result_data = self._data + other_data
        
        result = Tensor(result_data)
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._ctx = ('add', self, other)
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        other_data = other._data if isinstance(other, Tensor) else other
        result = Tensor(self._data - other_data)
        
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._ctx = ('sub', self, other)
        return result
    
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        other_data = other._data if isinstance(other, Tensor) else other
        
        # Use parallel mul for large tensors
        if HAS_OPTIMIZED and parallel_mul is not None and isinstance(other, Tensor) and self._data.size > 500_000:
            result_data = parallel_mul(self._data, other_data)
        else:
            result_data = self._data * other_data
        
        result = Tensor(result_data)
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._ctx = ('mul', self, other)
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data / other_data)
    
    def __neg__(self) -> 'Tensor':
        return Tensor(-self._data, requires_grad=self.requires_grad)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication with intelligent routing"""
        return matmul(self, other)
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        result = Tensor(np.sum(self._data, axis=dim, keepdims=keepdim))
        if self.requires_grad:
            result.requires_grad = True
            result._ctx = ('sum', self, dim, keepdim)
        return result
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        result = Tensor(np.mean(self._data, axis=dim, keepdims=keepdim))
        if self.requires_grad:
            result.requires_grad = True
            result._ctx = ('mean', self, dim, keepdim)
        return result
    
    def reshape(self, *shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return Tensor(self._data.reshape(shape), requires_grad=self.requires_grad)
    
    def T(self) -> 'Tensor':
        """Transpose"""
        return Tensor(self._data.T, requires_grad=self.requires_grad)
    
    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Tensor(np.transpose(self._data, axes))
    
    def detach(self) -> 'Tensor':
        """Return tensor without gradient tracking"""
        return Tensor(self._data)
    
    def zero_grad(self):
        """Zero the gradient"""
        self.grad = None
    
    def backward(self, grad: Optional['Tensor'] = None):
        """Backward pass"""
        from .autograd import backward
        backward(self, grad)


# ===== Core Operations with GPU+CPU Routing =====

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication with OPTIMAL strategy selection
    
    Strategy:
    - Small (<1024): NumPy (Accelerate BLAS) - minimal overhead
    - Large (>=1024): MLX GPU - maximum throughput
    
    For best performance with multiple operations, use LazyTensor
    which enables batched GPU evaluation.
    """
    M, K = a.shape[0], a.shape[1] if a.ndim > 1 else 1
    N = b.shape[1] if b.ndim > 1 else 1
    min_dim = min(M, K, N)
    
    if min_dim < 1024:
        # Small: NumPy (Accelerate BLAS) - no GPU overhead
        result = np.matmul(a._data, b._data)
    elif HAS_MLX:
        # Large: Use MLX GPU directly - it's highly optimized
        import mlx.core as mx
        a_mlx = mx.array(a._data)
        b_mlx = mx.array(b._data)
        c_mlx = a_mlx @ b_mlx
        mx.eval(c_mlx)
        result = np.array(c_mlx)
    else:
        # Fallback to NumPy
        result = np.matmul(a._data, b._data)
    
    out = Tensor(result)
    if a.requires_grad or b.requires_grad:
        out.requires_grad = True
        out._ctx = ('matmul', a, b)
    return out


# ===== Convenience functions =====

def zeros(*shape) -> Tensor:
    return Tensor.zeros(*shape)

def ones(*shape) -> Tensor:
    return Tensor.ones(*shape)

def rand(*shape) -> Tensor:
    return Tensor.rand(*shape)

def randn(*shape) -> Tensor:
    return Tensor.randn(*shape)

def eye(n: int) -> Tensor:
    return Tensor.eye(n)
