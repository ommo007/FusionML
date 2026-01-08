"""
Optimized Algorithms for FusionML
- Strassen Matrix Multiplication: O(n^2.807) vs O(n^3)
- Parallel execution on GPU+CPU
"""

import numpy as np
from typing import Tuple, Optional
import concurrent.futures

# Try to import MLX for GPU
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# ============================================================================
# STRASSEN MATRIX MULTIPLICATION
# ============================================================================

def _pad_to_power_of_2(a: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Pad matrix to next power of 2 for Strassen"""
    m, n = a.shape
    size = max(m, n)
    # Find next power of 2
    new_size = 1
    while new_size < size:
        new_size *= 2
    
    if new_size == m == n:
        return a, m, n
    
    padded = np.zeros((new_size, new_size), dtype=a.dtype)
    padded[:m, :n] = a
    return padded, m, n


def _strassen_recursive(a: np.ndarray, b: np.ndarray, threshold: int = 64) -> np.ndarray:
    """
    Strassen's algorithm for matrix multiplication
    O(n^2.807) complexity vs O(n^3) for naive
    
    Uses divide-and-conquer with 7 multiplications instead of 8
    """
    n = a.shape[0]
    
    # Base case: use standard matmul for small matrices
    if n <= threshold:
        return np.matmul(a, b)
    
    # Split matrices into quadrants
    mid = n // 2
    
    a11, a12 = a[:mid, :mid], a[:mid, mid:]
    a21, a22 = a[mid:, :mid], a[mid:, mid:]
    
    b11, b12 = b[:mid, :mid], b[:mid, mid:]
    b21, b22 = b[mid:, :mid], b[mid:, mid:]
    
    # Strassen's 7 products (instead of 8)
    m1 = _strassen_recursive(a11 + a22, b11 + b22, threshold)
    m2 = _strassen_recursive(a21 + a22, b11, threshold)
    m3 = _strassen_recursive(a11, b12 - b22, threshold)
    m4 = _strassen_recursive(a22, b21 - b11, threshold)
    m5 = _strassen_recursive(a11 + a12, b22, threshold)
    m6 = _strassen_recursive(a21 - a11, b11 + b12, threshold)
    m7 = _strassen_recursive(a12 - a22, b21 + b22, threshold)
    
    # Compute result quadrants
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    
    # Combine quadrants
    c = np.zeros((n, n), dtype=a.dtype)
    c[:mid, :mid] = c11
    c[:mid, mid:] = c12
    c[mid:, :mid] = c21
    c[mid:, mid:] = c22
    
    return c


def strassen_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Strassen matrix multiplication with padding
    
    Faster than O(n^3) for large matrices
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    m, k1 = a.shape
    k2, n = b.shape
    assert k1 == k2, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    # Pad to square power of 2
    max_dim = max(m, k1, n)
    size = 1
    while size < max_dim:
        size *= 2
    
    a_padded = np.zeros((size, size), dtype=np.float32)
    b_padded = np.zeros((size, size), dtype=np.float32)
    a_padded[:m, :k1] = a
    b_padded[:k1, :n] = b
    
    c_padded = _strassen_recursive(a_padded, b_padded)
    
    return c_padded[:m, :n]


# ============================================================================
# PARALLEL STRASSEN (GPU+CPU)
# ============================================================================

def parallel_strassen_matmul(
    a: np.ndarray, 
    b: np.ndarray,
    gpu_ratio: float = 0.7
) -> np.ndarray:
    """
    Parallel Strassen using GPU+CPU
    
    Strategy: Split rows and compute in parallel
    - GPU: Top gpu_ratio rows using MLX
    - CPU: Bottom (1-gpu_ratio) rows using Strassen
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    
    M, K = a.shape
    K2, N = b.shape
    
    if not HAS_MLX or M < 512:
        # Fallback to pure Strassen
        return strassen_matmul(a, b)
    
    # Split point
    split = int(M * gpu_ratio)
    
    a_gpu = a[:split, :]
    a_cpu = a[split:, :]
    
    def gpu_work():
        """GPU computes using MLX (standard matmul - already optimized)"""
        if split > 0:
            a_mlx = mx.array(a_gpu)
            b_mlx = mx.array(b)
            c = a_mlx @ b_mlx
            mx.eval(c)
            return np.array(c)
        return None
    
    def cpu_work():
        """CPU computes using standard matmul (Accelerate BLAS is already fast)"""
        if M - split > 0:
            return np.matmul(a_cpu, b)
        return None
    
    # Execute in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        gpu_future = executor.submit(gpu_work)
        cpu_future = executor.submit(cpu_work)
        
        gpu_result = gpu_future.result()
        cpu_result = cpu_future.result()
    
    # Combine
    result = np.zeros((M, N), dtype=np.float32)
    if gpu_result is not None:
        result[:split, :] = gpu_result
    if cpu_result is not None:
        result[split:, :] = cpu_result
    
    return result


# ============================================================================
# FAST ELEMENTWISE OPERATIONS (GPU+CPU PARALLEL)
# ============================================================================

def parallel_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Parallel addition using GPU+CPU"""
    if not HAS_MLX or a.size < 1_000_000:
        return a + b
    
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    
    # Split along first axis
    split = a.shape[0] // 2
    
    def gpu_work():
        a_mlx = mx.array(a[:split])
        b_mlx = mx.array(b[:split])
        c = a_mlx + b_mlx
        mx.eval(c)
        return np.array(c)
    
    def cpu_work():
        return a[split:] + b[split:]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(gpu_work)
        f2 = executor.submit(cpu_work)
        r1, r2 = f1.result(), f2.result()
    
    return np.concatenate([r1, r2], axis=0)


def parallel_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Parallel element-wise multiplication using GPU+CPU"""
    if not HAS_MLX or a.size < 1_000_000:
        return a * b
    
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    
    split = a.shape[0] // 2
    
    def gpu_work():
        a_mlx = mx.array(a[:split])
        b_mlx = mx.array(b[:split])
        c = a_mlx * b_mlx
        mx.eval(c)
        return np.array(c)
    
    def cpu_work():
        return a[split:] * b[split:]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(gpu_work)
        f2 = executor.submit(cpu_work)
        r1, r2 = f1.result(), f2.result()
    
    return np.concatenate([r1, r2], axis=0)


def parallel_exp(a: np.ndarray) -> np.ndarray:
    """Parallel exponential using GPU+CPU"""
    if not HAS_MLX or a.size < 500_000:
        return np.exp(a)
    
    a = np.ascontiguousarray(a, dtype=np.float32)
    split = a.shape[0] // 2
    
    def gpu_work():
        a_mlx = mx.array(a[:split])
        c = mx.exp(a_mlx)
        mx.eval(c)
        return np.array(c)
    
    def cpu_work():
        return np.exp(a[split:])
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(gpu_work)
        f2 = executor.submit(cpu_work)
        r1, r2 = f1.result(), f2.result()
    
    return np.concatenate([r1, r2], axis=0)


def parallel_softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Parallel softmax using GPU+CPU"""
    if not HAS_MLX or a.size < 500_000:
        exp_a = np.exp(a - np.max(a, axis=axis, keepdims=True))
        return exp_a / np.sum(exp_a, axis=axis, keepdims=True)
    
    a = np.ascontiguousarray(a, dtype=np.float32)
    
    if a.ndim == 1:
        # 1D softmax - use GPU
        a_mlx = mx.array(a)
        c = mx.softmax(a_mlx, axis=axis)
        mx.eval(c)
        return np.array(c)
    
    # Split batch dimension
    split = a.shape[0] // 2
    
    def gpu_work():
        a_mlx = mx.array(a[:split])
        c = mx.softmax(a_mlx, axis=axis)
        mx.eval(c)
        return np.array(c)
    
    def cpu_work():
        x = a[split:]
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(gpu_work)
        f2 = executor.submit(cpu_work)
        r1, r2 = f1.result(), f2.result()
    
    return np.concatenate([r1, r2], axis=0)


# ============================================================================
# OPTIMIZED SUM/MEAN (PARALLEL REDUCTION)
# ============================================================================

def parallel_sum(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Parallel sum reduction"""
    if not HAS_MLX or a.size < 1_000_000:
        return np.sum(a, axis=axis)
    
    if axis is None:
        # Full reduction - split and sum both parts
        split = a.size // 2
        flat = a.flatten()
        
        def gpu_work():
            a_mlx = mx.array(flat[:split])
            return float(mx.sum(a_mlx))
        
        def cpu_work():
            return float(np.sum(flat[split:]))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(gpu_work)
            f2 = executor.submit(cpu_work)
            return f1.result() + f2.result()
    
    return np.sum(a, axis=axis)


# ============================================================================
# DEVICE INFO
# ============================================================================

def device_info() -> dict:
    return {
        "backend": "optimized-parallel",
        "algorithms": ["strassen", "parallel-matmul", "parallel-elementwise"],
        "has_mlx": HAS_MLX,
        "strassen_threshold": 64,
    }


__all__ = [
    'strassen_matmul',
    'parallel_strassen_matmul', 
    'parallel_add',
    'parallel_mul',
    'parallel_exp',
    'parallel_softmax',
    'parallel_sum',
    'device_info'
]
