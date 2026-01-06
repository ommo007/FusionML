"""
Parallel Tiled MatMul - The Core FusionML Innovation
Splits large matrices and runs GPU + CPU in parallel for maximum throughput
"""

import numpy as np
from typing import Tuple
import concurrent.futures
import threading

# Try to import MLX for GPU
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _mlx_matmul_tile(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute matmul on GPU using MLX"""
    a_mlx = mx.array(a.astype(np.float32))
    b_mlx = mx.array(b.astype(np.float32))
    c_mlx = a_mlx @ b_mlx
    mx.eval(c_mlx)
    return np.array(c_mlx)


def _cpu_matmul_tile(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute matmul on CPU using NumPy (Accelerate BLAS)"""
    return np.matmul(a.astype(np.float32), b.astype(np.float32))


def parallel_tiled_matmul(
    a: np.ndarray, 
    b: np.ndarray,
    gpu_ratio: float = 0.7,  # Portion to run on GPU
) -> np.ndarray:
    """
    Parallel matrix multiplication - THE FUSIONML INNOVATION
    
    Runs GPU and CPU computations simultaneously:
    - gpu_ratio of rows computed on GPU (MLX)
    - (1 - gpu_ratio) of rows computed on CPU (NumPy/Accelerate)
    
    This beats pure GPU because:
    1. MLX releases GIL during GPU compute
    2. CPU can work on remaining rows simultaneously
    3. Combined throughput > single device throughput
    
    For C = A @ B where A is (M, K) and B is (K, N):
    - GPU computes: C[0:split, :] = A[0:split, :] @ B
    - CPU computes: C[split:M, :] = A[split:M, :] @ B
    - Both run in parallel!
    
    Args:
        a: Left matrix (M, K)
        b: Right matrix (K, N)
        gpu_ratio: Fraction of rows to process on GPU (0.0-1.0)
    
    Returns:
        Result matrix (M, N)
    """
    if not HAS_MLX:
        return np.matmul(a, b)
    
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    # Calculate split point
    split = int(M * gpu_ratio)
    
    # Split A into GPU and CPU portions
    a_gpu = a[:split, :]
    a_cpu = a[split:, :]
    
    # Define work functions (capture arrays by closure)
    def gpu_work():
        if split > 0:
            a_mlx = mx.array(a_gpu)
            b_mlx = mx.array(b)
            c_mlx = a_mlx @ b_mlx
            mx.eval(c_mlx)
            return np.array(c_mlx)
        return None
    
    def cpu_work():
        if M - split > 0:
            return np.matmul(a_cpu, b)
        return None
    
    # Run in parallel - MLX releases GIL so true parallelism is possible
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        gpu_future = executor.submit(gpu_work)
        cpu_future = executor.submit(cpu_work)
        
        gpu_result = gpu_future.result()
        cpu_result = cpu_future.result()
    
    # Combine results
    result = np.zeros((M, N), dtype=np.float32)
    if gpu_result is not None:
        result[:split, :] = gpu_result
    if cpu_result is not None:
        result[split:, :] = cpu_result
    
    return result


def find_optimal_ratio(size: int) -> float:
    """
    Find optimal GPU/CPU ratio for a given matrix size
    Based on empirical testing on Apple M1
    """
    if size < 1500:
        return 0.0  # CPU only for small matrices
    elif size < 2048:
        return 0.65  # 65% GPU, 35% CPU
    elif size < 4096:
        return 0.70  # 70% GPU, 30% CPU
    else:
        return 0.75  # 75% GPU, 25% CPU for very large


def smart_parallel_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Intelligent parallel matmul that automatically selects best strategy
    """
    M, K = a.shape
    K2, N = b.shape
    
    min_dim = min(M, K, N)
    
    # For small matrices, direct NumPy is fastest
    if min_dim < 1024:
        return np.matmul(a, b)
    
    # For larger matrices, use parallel tiled approach
    ratio = find_optimal_ratio(min_dim)
    
    if ratio == 0.0:
        return np.matmul(a, b)
    
    return parallel_tiled_matmul(a, b, gpu_ratio=ratio)


def device_info() -> dict:
    """Get device information"""
    return {
        "backend": "parallel-tiled",
        "strategy": "GPU+CPU parallel (FusionML innovation)",
        "has_mlx": HAS_MLX,
        "default_gpu_ratio": 0.7
    }


# Export
__all__ = ['parallel_tiled_matmul', 'smart_parallel_matmul', 'find_optimal_ratio', 'device_info']
