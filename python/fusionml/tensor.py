"""
FusionML Tensor - MLX-Native Implementation
Maximum performance by keeping data on GPU
"""

import numpy as np
from typing import Union, List, Optional, Tuple

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

ArrayLike = Union[np.ndarray, List, float, int, 'mx.array'] if HAS_MLX else Union[np.ndarray, List, float, int]


class Tensor:
    """
    FusionML Tensor with MLX-native storage for maximum GPU performance
    
    Data is stored on GPU (MLX) by default for large tensors.
    Small tensors use numpy for minimal overhead.
    
    Key features:
    - Lazy evaluation: operations don't execute until .eval() or .numpy
    - Automatic backend selection based on size
    - Full autograd support
    """
    
    # Threshold for using GPU (elements)
    # Matches matmul threshold of 1500x1500 for consistency
    GPU_THRESHOLD = 1500 * 1500  # ~2.25M elements
    
    def __init__(self, data: ArrayLike, requires_grad: bool = False, _mlx_data=None):
        self.requires_grad = requires_grad
        self._ctx = None  # For autograd
        self.grad = None
        
        if _mlx_data is not None:
            # Direct MLX array (internal use)
            self._mlx = _mlx_data
            self._np = None
            self._on_gpu = True
        elif HAS_MLX and isinstance(data, mx.array):
            self._mlx = data
            self._np = None
            self._on_gpu = True
        elif isinstance(data, np.ndarray):
            if HAS_MLX and data.size >= self.GPU_THRESHOLD:
                self._mlx = mx.array(data.astype(np.float32))
                self._np = None
                self._on_gpu = True
            else:
                self._np = data.astype(np.float32)
                self._mlx = None
                self._on_gpu = False
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=np.float32)
            if HAS_MLX and arr.size >= self.GPU_THRESHOLD:
                self._mlx = mx.array(arr)
                self._np = None
                self._on_gpu = True
            else:
                self._np = arr
                self._mlx = None
                self._on_gpu = False
        elif isinstance(data, (int, float)):
            self._np = np.array(data, dtype=np.float32)
            self._mlx = None
            self._on_gpu = False
        elif isinstance(data, Tensor):
            if data._on_gpu:
                self._mlx = data._mlx
                self._np = None
                self._on_gpu = True
            else:
                self._np = data._np.copy()
                self._mlx = None
                self._on_gpu = False
        else:
            self._np = np.array(data, dtype=np.float32)
            self._mlx = None
            self._on_gpu = False
    
    @property
    def data(self):
        """Get data as MLX array if on GPU, numpy otherwise"""
        if self._on_gpu:
            return self._mlx
        return self._np
    
    @property
    def numpy(self) -> np.ndarray:
        """Get data as numpy array (forces GPU sync if needed)"""
        if self._np is not None:
            return self._np
        if self._mlx is not None:
            mx.eval(self._mlx)
            self._np = np.array(self._mlx)
            return self._np
        return np.array([])
    
    def eval(self) -> 'Tensor':
        """Force GPU evaluation"""
        if self._on_gpu and self._mlx is not None:
            mx.eval(self._mlx)
        return self
    
    @property
    def shape(self) -> Tuple:
        if self._on_gpu:
            return tuple(self._mlx.shape)
        return self._np.shape
    
    @property
    def ndim(self) -> int:
        if self._on_gpu:
            return len(self._mlx.shape)
        return self._np.ndim
    
    @property
    def size(self) -> int:
        if self._on_gpu:
            s = 1
            for d in self._mlx.shape:
                s *= d
            return s
        return self._np.size
    
    @property 
    def dtype(self):
        if self._on_gpu:
            return self._mlx.dtype
        return self._np.dtype
    
    def to_gpu(self) -> 'Tensor':
        """Move tensor to GPU"""
        if self._on_gpu or not HAS_MLX:
            return self
        t = Tensor.__new__(Tensor)
        t._mlx = mx.array(self._np)
        t._np = None
        t._on_gpu = True
        t.requires_grad = self.requires_grad
        t._ctx = None
        t.grad = None
        return t
    
    def to_cpu(self) -> 'Tensor':
        """Move tensor to CPU"""
        if not self._on_gpu:
            return self
        t = Tensor.__new__(Tensor)
        mx.eval(self._mlx)
        t._np = np.array(self._mlx)
        t._mlx = None
        t._on_gpu = False
        t.requires_grad = self.requires_grad
        t._ctx = None
        t.grad = None
        return t
    
    def __repr__(self):
        if self._on_gpu:
            return f"Tensor(shape={self.shape}, device='gpu')"
        return f"Tensor({self._np})"
    
    # ===== Operations =====
    
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, Tensor):
            if self._on_gpu or other._on_gpu:
                # GPU path
                a = self._mlx if self._on_gpu else mx.array(self._np)
                b = other._mlx if other._on_gpu else mx.array(other._np)
                result = Tensor.__new__(Tensor)
                result._mlx = a + b
                result._np = None
                result._on_gpu = True
            else:
                result = Tensor(self._np + other._np)
        else:
            if self._on_gpu:
                result = Tensor.__new__(Tensor)
                result._mlx = self._mlx + other
                result._np = None
                result._on_gpu = True
            else:
                result = Tensor(self._np + other)
        
        result.requires_grad = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)
        result._ctx = ('add', self, other) if result.requires_grad else None
        result.grad = None
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, Tensor):
            if self._on_gpu or other._on_gpu:
                a = self._mlx if self._on_gpu else mx.array(self._np)
                b = other._mlx if other._on_gpu else mx.array(other._np)
                result = Tensor.__new__(Tensor)
                result._mlx = a - b
                result._np = None
                result._on_gpu = True
            else:
                result = Tensor(self._np - other._np)
        else:
            if self._on_gpu:
                result = Tensor.__new__(Tensor)
                result._mlx = self._mlx - other
                result._np = None
                result._on_gpu = True
            else:
                result = Tensor(self._np - other)
        
        result.requires_grad = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)
        result._ctx = ('sub', self, other) if result.requires_grad else None
        result.grad = None
        return result
    
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, Tensor):
            if self._on_gpu or other._on_gpu:
                a = self._mlx if self._on_gpu else mx.array(self._np)
                b = other._mlx if other._on_gpu else mx.array(other._np)
                result = Tensor.__new__(Tensor)
                result._mlx = a * b
                result._np = None
                result._on_gpu = True
            else:
                result = Tensor(self._np * other._np)
        else:
            if self._on_gpu:
                result = Tensor.__new__(Tensor)
                result._mlx = self._mlx * other
                result._np = None
                result._on_gpu = True
            else:
                result = Tensor(self._np * other)
        
        result.requires_grad = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)
        result._ctx = ('mul', self, other) if result.requires_grad else None
        result.grad = None
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, Tensor):
            if self._on_gpu or other._on_gpu:
                a = self._mlx if self._on_gpu else mx.array(self._np)
                b = other._mlx if other._on_gpu else mx.array(other._np)
                result = Tensor.__new__(Tensor)
                result._mlx = a / b
                result._np = None
                result._on_gpu = True
            else:
                result = Tensor(self._np / other._np)
        else:
            if self._on_gpu:
                result = Tensor.__new__(Tensor)
                result._mlx = self._mlx / other
                result._np = None
                result._on_gpu = True
            else:
                result = Tensor(self._np / other)
        
        result.requires_grad = False
        result._ctx = None
        result.grad = None
        return result
    
    def __neg__(self) -> 'Tensor':
        if self._on_gpu:
            result = Tensor.__new__(Tensor)
            result._mlx = -self._mlx
            result._np = None
            result._on_gpu = True
        else:
            result = Tensor(-self._np)
        result.requires_grad = self.requires_grad
        result._ctx = None
        result.grad = None
        return result
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication - always on GPU for best performance"""
        return matmul(self, other)
    
    def __getitem__(self, key):
        if self._on_gpu:
            result = Tensor.__new__(Tensor)
            result._mlx = self._mlx[key]
            result._np = None
            result._on_gpu = True
        else:
            result = Tensor(self._np[key])
        result.requires_grad = self.requires_grad
        result._ctx = None
        result.grad = None
        return result
    
    # ===== Reduction Operations =====
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        if self._on_gpu:
            result = Tensor.__new__(Tensor)
            result._mlx = mx.sum(self._mlx, axis=axis, keepdims=keepdims)
            result._np = None
            result._on_gpu = True
        else:
            result = Tensor(np.sum(self._np, axis=axis, keepdims=keepdims))
        result.requires_grad = self.requires_grad
        result._ctx = ('sum', self, axis, keepdims) if result.requires_grad else None
        result.grad = None
        return result
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        if self._on_gpu:
            result = Tensor.__new__(Tensor)
            result._mlx = mx.mean(self._mlx, axis=axis, keepdims=keepdims)
            result._np = None
            result._on_gpu = True
        else:
            result = Tensor(np.mean(self._np, axis=axis, keepdims=keepdims))
        result.requires_grad = self.requires_grad
        result._ctx = ('mean', self, axis, keepdims) if result.requires_grad else None
        result.grad = None
        return result
    
    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        if self._on_gpu:
            result = Tensor.__new__(Tensor)
            result._mlx = mx.max(self._mlx, axis=axis, keepdims=keepdims)
            result._np = None
            result._on_gpu = True
        else:
            result = Tensor(np.max(self._np, axis=axis, keepdims=keepdims))
        result.requires_grad = False
        result._ctx = None
        result.grad = None
        return result
    
    @property
    def T(self) -> 'Tensor':
        if self._on_gpu:
            result = Tensor.__new__(Tensor)
            result._mlx = self._mlx.T
            result._np = None
            result._on_gpu = True
        else:
            result = Tensor(self._np.T)
        result.requires_grad = self.requires_grad
        result._ctx = None
        result.grad = None
        return result
    
    def reshape(self, *shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if self._on_gpu:
            result = Tensor.__new__(Tensor)
            result._mlx = self._mlx.reshape(shape)
            result._np = None
            result._on_gpu = True
        else:
            result = Tensor(self._np.reshape(shape))
        result.requires_grad = self.requires_grad
        result._ctx = None
        result.grad = None
        return result
    
    def backward(self, grad: Optional['Tensor'] = None):
        from .autograd import backward
        backward(self, grad)


# ===== Matrix Multiplication =====

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication with HYBRID strategy for maximum performance
    
    - Small matrices (<1500): NumPy (Accelerate BLAS) - no GPU overhead
    - Large matrices (>=1500): MLX GPU - maximum throughput
    
    This gives us the best of both worlds!
    """
    M = a.shape[0]
    K = a.shape[1] if len(a.shape) > 1 else 1
    N = b.shape[1] if len(b.shape) > 1 else 1
    min_dim = min(M, K, N)
    
    # Use NumPy for small/medium matrices (GPU overhead not worth it)
    if min_dim < 1500:
        # Get numpy arrays
        a_np = a.numpy if a._on_gpu else a._np
        b_np = b.numpy if b._on_gpu else b._np
        
        # NumPy matmul (uses Accelerate BLAS on macOS)
        result_np = np.matmul(a_np, b_np)
        
        result = Tensor.__new__(Tensor)
        result._np = result_np
        result._mlx = None
        result._on_gpu = False
    elif HAS_MLX:
        # Large matrices: Use MLX GPU
        a_mlx = a._mlx if a._on_gpu else mx.array(a._np)
        b_mlx = b._mlx if b._on_gpu else mx.array(b._np)
        
        result = Tensor.__new__(Tensor)
        result._mlx = a_mlx @ b_mlx
        result._np = None
        result._on_gpu = True
    else:
        # CPU fallback
        a_np = a._np if not a._on_gpu else a.numpy
        b_np = b._np if not b._on_gpu else b.numpy
        result = Tensor(np.matmul(a_np, b_np))
    
    result.requires_grad = a.requires_grad or b.requires_grad
    result._ctx = ('matmul', a, b) if result.requires_grad else None
    result.grad = None
    return result


# ===== Activation Functions =====

def relu(x: Tensor) -> Tensor:
    """ReLU activation"""
    if x._on_gpu:
        result = Tensor.__new__(Tensor)
        result._mlx = mx.maximum(x._mlx, 0)
        result._np = None
        result._on_gpu = True
    else:
        result = Tensor(np.maximum(x._np, 0))
    result.requires_grad = x.requires_grad
    result._ctx = ('relu', x) if result.requires_grad else None
    result.grad = None
    return result


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation"""
    if x._on_gpu:
        result = Tensor.__new__(Tensor)
        result._mlx = mx.sigmoid(x._mlx)
        result._np = None
        result._on_gpu = True
    else:
        result = Tensor(1 / (1 + np.exp(-x._np)))
    result.requires_grad = x.requires_grad
    result._ctx = ('sigmoid', x) if result.requires_grad else None
    result.grad = None
    return result


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax activation"""
    if x._on_gpu:
        result = Tensor.__new__(Tensor)
        result._mlx = mx.softmax(x._mlx, axis=axis)
        result._np = None
        result._on_gpu = True
    else:
        exp_x = np.exp(x._np - np.max(x._np, axis=axis, keepdims=True))
        result = Tensor(exp_x / np.sum(exp_x, axis=axis, keepdims=True))
    result.requires_grad = x.requires_grad
    result._ctx = ('softmax', x, axis) if result.requires_grad else None
    result.grad = None
    return result


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Log-Softmax for numerical stability"""
    if x._on_gpu:
        result = Tensor.__new__(Tensor)
        # log_softmax = x - log(sum(exp(x)))
        result._mlx = x._mlx - mx.logsumexp(x._mlx, axis=axis, keepdims=True)
        result._np = None
        result._on_gpu = True
    else:
        max_x = np.max(x._np, axis=axis, keepdims=True)
        log_sum_exp = max_x + np.log(np.sum(np.exp(x._np - max_x), axis=axis, keepdims=True))
        result = Tensor(x._np - log_sum_exp)
    result.requires_grad = x.requires_grad
    result._ctx = ('log_softmax', x, axis) if result.requires_grad else None
    result.grad = None
    return result


def exp(x: Tensor) -> Tensor:
    """Exponential"""
    if x._on_gpu:
        result = Tensor.__new__(Tensor)
        result._mlx = mx.exp(x._mlx)
        result._np = None
        result._on_gpu = True
    else:
        result = Tensor(np.exp(x._np))
    result.requires_grad = x.requires_grad
    result._ctx = ('exp', x) if result.requires_grad else None
    result.grad = None
    return result


def log(x: Tensor) -> Tensor:
    """Natural logarithm"""
    if x._on_gpu:
        result = Tensor.__new__(Tensor)
        result._mlx = mx.log(x._mlx)
        result._np = None
        result._on_gpu = True
    else:
        result = Tensor(np.log(x._np))
    result.requires_grad = x.requires_grad
    result._ctx = ('log', x) if result.requires_grad else None
    result.grad = None
    return result


# ===== Creation Functions =====

def zeros(*shape, requires_grad: bool = False) -> Tensor:
    """Create tensor of zeros"""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    size = 1
    for d in shape:
        size *= d
    if HAS_MLX and size >= Tensor.GPU_THRESHOLD:
        t = Tensor.__new__(Tensor)
        t._mlx = mx.zeros(shape)
        t._np = None
        t._on_gpu = True
    else:
        t = Tensor(np.zeros(shape, dtype=np.float32))
    t.requires_grad = requires_grad
    t._ctx = None
    t.grad = None
    return t


def ones(*shape, requires_grad: bool = False) -> Tensor:
    """Create tensor of ones"""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    size = 1
    for d in shape:
        size *= d
    if HAS_MLX and size >= Tensor.GPU_THRESHOLD:
        t = Tensor.__new__(Tensor)
        t._mlx = mx.ones(shape)
        t._np = None
        t._on_gpu = True
    else:
        t = Tensor(np.ones(shape, dtype=np.float32))
    t.requires_grad = requires_grad
    t._ctx = None
    t.grad = None
    return t


def rand(*shape, requires_grad: bool = False) -> Tensor:
    """Create tensor with random values [0, 1)"""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    size = 1
    for d in shape:
        size *= d
    if HAS_MLX and size >= Tensor.GPU_THRESHOLD:
        t = Tensor.__new__(Tensor)
        t._mlx = mx.random.uniform(shape=shape)
        t._np = None
        t._on_gpu = True
    else:
        t = Tensor(np.random.rand(*shape).astype(np.float32))
    t.requires_grad = requires_grad
    t._ctx = None
    t.grad = None
    return t


def randn(*shape, requires_grad: bool = False) -> Tensor:
    """Create tensor with random normal values"""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    size = 1
    for d in shape:
        size *= d
    if HAS_MLX and size >= Tensor.GPU_THRESHOLD:
        t = Tensor.__new__(Tensor)
        t._mlx = mx.random.normal(shape=shape)
        t._np = None
        t._on_gpu = True
    else:
        t = Tensor(np.random.randn(*shape).astype(np.float32))
    t.requires_grad = requires_grad
    t._ctx = None
    t.grad = None
    return t



# ===== Utility =====

def eye(n: int, requires_grad: bool = False) -> Tensor:
    """Create identity matrix"""
    if HAS_MLX and n * n >= Tensor.GPU_THRESHOLD:
        t = Tensor.__new__(Tensor)
        t._mlx = mx.eye(n)
        t._np = None
        t._on_gpu = True
    else:
        t = Tensor(np.eye(n, dtype=np.float32))
    t.requires_grad = requires_grad
    t._ctx = None
    t.grad = None
    return t


def batch_eval(*tensors: Tensor):
    """Evaluate multiple tensors in a single GPU sync"""
    if HAS_MLX:
        mlx_arrays = [t._mlx for t in tensors if t._on_gpu and t._mlx is not None]
        if mlx_arrays:
            mx.eval(*mlx_arrays)


def device_info():
    """Get device information"""
    if HAS_MLX:
        return {
            "backend": "MLX-native",
            "gpu": True,
            "gpu_threshold": Tensor.GPU_THRESHOLD,
        }
    return {"backend": "numpy", "gpu": False}
