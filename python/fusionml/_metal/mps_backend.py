"""
Metal Performance Shaders (MPS) Backend
Uses MPSMatrixMultiplication for GPU-accelerated matmul
"""

import numpy as np
from typing import Optional
import ctypes
import os

# Try to import objc for Objective-C runtime access
try:
    import objc
    from Foundation import NSObject
    import Metal
    import MetalPerformanceShaders as MPS
    HAS_MPS = True
except ImportError:
    HAS_MPS = False

_device = None
_command_queue = None


def get_device():
    """Get Metal device singleton"""
    global _device, _command_queue
    if _device is None and HAS_MPS:
        _device = Metal.MTLCreateSystemDefaultDevice()
        _command_queue = _device.newCommandQueue()
    return _device


def mps_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication using Metal Performance Shaders
    This is Apple's highly optimized GPU GEMM implementation
    """
    if not HAS_MPS:
        return np.matmul(a, b)
    
    device = get_device()
    if device is None:
        return np.matmul(a, b)
    
    # Ensure float32 and C-contiguous
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    # Create output array
    c = np.zeros((M, N), dtype=np.float32)
    
    # Create Metal buffers
    buffer_a = device.newBufferWithBytes_length_options_(
        a.tobytes(), a.nbytes, Metal.MTLResourceStorageModeShared
    )
    buffer_b = device.newBufferWithBytes_length_options_(
        b.tobytes(), b.nbytes, Metal.MTLResourceStorageModeShared
    )
    buffer_c = device.newBufferWithLength_options_(
        c.nbytes, Metal.MTLResourceStorageModeShared
    )
    
    # Create MPS matrix descriptors
    desc_a = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        M, K, K * 4, MPS.MPSDataTypeFloat32
    )
    desc_b = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        K, N, N * 4, MPS.MPSDataTypeFloat32
    )
    desc_c = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        M, N, N * 4, MPS.MPSDataTypeFloat32
    )
    
    # Create MPS matrices
    mat_a = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(buffer_a, desc_a)
    mat_b = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(buffer_b, desc_b)
    mat_c = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(buffer_c, desc_c)
    
    # Create matrix multiplication kernel
    matmul_kernel = MPS.MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
        device,
        False,  # transposeLeft
        False,  # transposeRight
        M, N, K,
        1.0,    # alpha
        0.0     # beta
    )
    
    # Execute
    command_buffer = _command_queue.commandBuffer()
    matmul_kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
        command_buffer, mat_a, mat_b, mat_c
    )
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    
    # Read result - use memoryview for faster access
    contents = buffer_c.contents()
    length = buffer_c.length()
    
    # Fast byte extraction using slice assignment
    result_bytes = bytearray(length)
    for i, b in enumerate(contents[:length]):
        result_bytes[i] = b[0] if isinstance(b, bytes) else b
    
    result = np.frombuffer(bytes(result_bytes), dtype=np.float32).reshape((M, N)).copy()
    
    return result


def device_info() -> dict:
    """Get device information"""
    if HAS_MPS:
        dev = get_device()
        if dev:
            return {
                "name": dev.name(),
                "backend": "MPS (Metal Performance Shaders)",
                "has_unified_memory": dev.hasUnifiedMemory()
            }
    return {"name": "CPU", "backend": "numpy"}
