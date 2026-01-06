"""
Hybrid Backend - Uses best available option for each operation size
- Small: NumPy (CPU, no overhead)
- Large: MLX if available (GPU optimized), else NumPy
"""

import numpy as np
from typing import Optional

# Check for MLX
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Check for Accelerate via numpy (always true on macOS)
HAS_ACCELERATE = True  # NumPy on macOS uses Accelerate


def hybrid_matmul(a: np.ndarray, b: np.ndarray, threshold: int = 2_000_000) -> np.ndarray:
    """
    Intelligent matrix multiplication with optimal backend selection
    
    Args:
        a: First matrix (M x K)
        b: Second matrix (K x N)  
        threshold: Size threshold for GPU (M * K * N)
    
    Returns:
        Result matrix (M x N)
    """
    M, K = a.shape
    K2, N = b.shape
    size = M * K * N
    
    # For matrix sizes >= 1024x1024, MLX GPU is faster
    # For smaller matrices, CPU (NumPy) is faster due to GPU overhead
    if size < threshold or not HAS_MLX:
        # Small or no MLX: use NumPy (Accelerate BLAS on macOS)
        return np.matmul(a, b)
    else:
        # Large + MLX available: use GPU
        return _mlx_matmul(a, b)


def _mlx_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiplication using MLX GPU"""
    # Convert to MLX arrays
    a_mlx = mx.array(a.astype(np.float32))
    b_mlx = mx.array(b.astype(np.float32))
    
    # Compute on GPU
    c_mlx = a_mlx @ b_mlx
    mx.eval(c_mlx)  # Force computation
    
    # Convert back to numpy
    return np.array(c_mlx)


def device_info() -> dict:
    """Get device information"""
    info = {
        "backend": "hybrid",
        "small_ops": "NumPy (Accelerate BLAS)",
        "large_ops": "MLX GPU" if HAS_MLX else "NumPy (Accelerate BLAS)",
        "has_mlx": HAS_MLX,
    }
    if HAS_MLX:
        # Get MLX device info
        info["mlx_version"] = mx.__version__ if hasattr(mx, '__version__') else "unknown"
    return info


# Export
__all__ = ['hybrid_matmul', 'device_info', 'HAS_MLX', 'HAS_ACCELERATE']
