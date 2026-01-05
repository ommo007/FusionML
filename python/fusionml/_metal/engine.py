"""
Metal Backend - GPU acceleration via PyObjC
"""

import numpy as np
from typing import Optional, Tuple
import ctypes

# Try to import Metal via PyObjC
try:
    import Metal
    import Accelerate
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    print("Warning: PyObjC Metal not available, using CPU fallback")

# Metal device singleton
_device = None
_command_queue = None

def get_device():
    """Get Metal device"""
    global _device, _command_queue
    if _device is None:
        if HAS_METAL:
            _device = Metal.MTLCreateSystemDefaultDevice()
            _command_queue = _device.newCommandQueue()
        else:
            _device = "cpu"
    return _device

def device_info() -> dict:
    """Get device information"""
    dev = get_device()
    if HAS_METAL:
        return {
            "name": dev.name(),
            "has_unified_memory": dev.hasUnifiedMemory(),
            "max_buffer_length": dev.maxBufferLength()
        }
    return {"name": "CPU (Metal not available)", "has_unified_memory": True}

class MetalBuffer:
    """GPU buffer backed by Metal"""
    
    def __init__(self, data: np.ndarray):
        self.shape = data.shape
        self.dtype = data.dtype
        self.size = data.nbytes
        
        if HAS_METAL:
            device = get_device()
            # Create Metal buffer from numpy data
            self._buffer = device.newBufferWithBytes_length_options_(
                data.ctypes.data_as(ctypes.c_void_p),
                self.size,
                Metal.MTLResourceStorageModeShared
            )
        else:
            # CPU fallback - just store numpy array
            self._data = data.copy()
    
    def to_numpy(self) -> np.ndarray:
        """Convert buffer to numpy array"""
        if HAS_METAL:
            # Read from Metal buffer
            ptr = self._buffer.contents()
            return np.ctypeslib.as_array(
                ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                shape=self.shape
            ).copy()
        return self._data.copy()

# Preload Metal shaders
_matmul_kernel = None

def _get_matmul_kernel():
    """Get compiled matmul kernel"""
    global _matmul_kernel
    if _matmul_kernel is None and HAS_METAL:
        device = get_device()
        source = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void matmul(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],
            device float* C [[buffer(2)]],
            constant uint& M [[buffer(3)]],
            constant uint& N [[buffer(4)]],
            constant uint& K [[buffer(5)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            if (gid.x >= N || gid.y >= M) return;
            
            float sum = 0.0f;
            for (uint k = 0; k < K; k++) {
                sum += A[gid.y * K + k] * B[k * N + gid.x];
            }
            C[gid.y * N + gid.x] = sum;
        }
        """
        options = Metal.MTLCompileOptions.new()
        library = device.newLibraryWithSource_options_error_(source, options, None)[0]
        _matmul_kernel = library.newFunctionWithName_("matmul")
    return _matmul_kernel
