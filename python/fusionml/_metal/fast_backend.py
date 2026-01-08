"""
Fast Metal Backend using CTypes
Direct memory access without PyObjC overhead
"""

import numpy as np
from typing import Optional
import ctypes
import ctypes.util

# Load Metal framework directly via ctypes
_metal_lib = None
_device = None
_command_queue = None

def _load_metal():
    """Load Metal framework using ctypes"""
    global _metal_lib
    if _metal_lib is None:
        try:
            # Load Metal framework
            metal_path = ctypes.util.find_library('Metal')
            if metal_path:
                _metal_lib = ctypes.CDLL(metal_path)
            else:
                # Try direct path on macOS
                _metal_lib = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
        except OSError:
            _metal_lib = None
    return _metal_lib

# Try loading Accelerate for BLAS
_accelerate = None
_cblas_sgemm = None

def _load_accelerate():
    """Load Accelerate framework for fast BLAS operations"""
    global _accelerate, _cblas_sgemm
    if _accelerate is None:
        try:
            acc_path = ctypes.util.find_library('Accelerate')
            if acc_path:
                _accelerate = ctypes.CDLL(acc_path)
            else:
                _accelerate = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
            
            # Get cblas_sgemm function
            _cblas_sgemm = _accelerate.cblas_sgemm
            _cblas_sgemm.restype = None
            _cblas_sgemm.argtypes = [
                ctypes.c_int,   # Order (CblasRowMajor = 101)
                ctypes.c_int,   # TransA (CblasNoTrans = 111)
                ctypes.c_int,   # TransB
                ctypes.c_int,   # M
                ctypes.c_int,   # N
                ctypes.c_int,   # K
                ctypes.c_float, # alpha
                ctypes.POINTER(ctypes.c_float),  # A
                ctypes.c_int,   # lda
                ctypes.POINTER(ctypes.c_float),  # B
                ctypes.c_int,   # ldb
                ctypes.c_float, # beta
                ctypes.POINTER(ctypes.c_float),  # C
                ctypes.c_int,   # ldc
            ]
        except OSError:
            _accelerate = None
    return _cblas_sgemm

# CBLAS constants
CblasRowMajor = 101
CblasNoTrans = 111

def accelerate_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Fast matrix multiplication using Accelerate BLAS (cblas_sgemm)
    This bypasses numpy and calls Apple's optimized BLAS directly
    """
    cblas_sgemm = _load_accelerate()
    
    if cblas_sgemm is None:
        # Fallback to numpy
        return np.matmul(a, b)
    
    # Ensure float32 and C-contiguous
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    # Create output array
    c = np.zeros((M, N), dtype=np.float32)
    
    # Get pointers
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call cblas_sgemm: C = alpha * A * B + beta * C
    cblas_sgemm(
        CblasRowMajor,  # Row major order
        CblasNoTrans,   # No transpose A
        CblasNoTrans,   # No transpose B
        M, N, K,        # Dimensions
        1.0,            # alpha
        a_ptr, K,       # A and leading dimension
        b_ptr, N,       # B and leading dimension
        0.0,            # beta
        c_ptr, N        # C and leading dimension
    )
    
    return c


# Check if Accelerate is available
HAS_ACCELERATE = _load_accelerate() is not None

def device_info() -> dict:
    """Get device information"""
    return {
        "name": "Apple Silicon (Accelerate BLAS)",
        "has_accelerate": HAS_ACCELERATE,
        "backend": "ctypes-accelerate"
    }
