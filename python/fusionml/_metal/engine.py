"""
Metal Backend - GPU acceleration via PyObjC
"""

import numpy as np
from typing import Optional, Tuple
import ctypes

# Try to import Metal via PyObjC
try:
    import Metal
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


def metal_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Perform matrix multiplication on GPU using Metal
    
    Args:
        a: numpy array of shape (M, K)
        b: numpy array of shape (K, N)
    
    Returns:
        numpy array of shape (M, N)
    """
    if not HAS_METAL:
        return np.matmul(a, b)
    
    device = get_device()
    kernel = _get_matmul_kernel()
    
    if kernel is None:
        return np.matmul(a, b)
    
    # Ensure float32 and contiguous
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    # Create output array
    c = np.zeros((M, N), dtype=np.float32)
    
    # Create Metal buffers using tobytes()
    buffer_a = device.newBufferWithBytes_length_options_(
        a.tobytes(),
        a.nbytes,
        Metal.MTLResourceStorageModeShared
    )
    buffer_b = device.newBufferWithBytes_length_options_(
        b.tobytes(),
        b.nbytes,
        Metal.MTLResourceStorageModeShared
    )
    buffer_c = device.newBufferWithLength_options_(
        c.nbytes,
        Metal.MTLResourceStorageModeShared
    )
    
    # Create dimension buffers
    m_val = np.array([M], dtype=np.uint32)
    n_val = np.array([N], dtype=np.uint32)
    k_val = np.array([K], dtype=np.uint32)
    
    buffer_m = device.newBufferWithBytes_length_options_(
        m_val.tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    buffer_n = device.newBufferWithBytes_length_options_(
        n_val.tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    buffer_k = device.newBufferWithBytes_length_options_(
        k_val.tobytes(), 4, Metal.MTLResourceStorageModeShared
    )
    
    # Create pipeline state
    pipeline_state = device.newComputePipelineStateWithFunction_error_(kernel, None)[0]
    
    # Create command buffer and encoder
    command_buffer = _command_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    
    encoder.setComputePipelineState_(pipeline_state)
    encoder.setBuffer_offset_atIndex_(buffer_a, 0, 0)
    encoder.setBuffer_offset_atIndex_(buffer_b, 0, 1)
    encoder.setBuffer_offset_atIndex_(buffer_c, 0, 2)
    encoder.setBuffer_offset_atIndex_(buffer_m, 0, 3)
    encoder.setBuffer_offset_atIndex_(buffer_n, 0, 4)
    encoder.setBuffer_offset_atIndex_(buffer_k, 0, 5)
    
    # Dispatch threads
    threads_per_group = Metal.MTLSize(16, 16, 1)
    num_groups = Metal.MTLSize(
        (N + 15) // 16,
        (M + 15) // 16,
        1
    )
    
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)
    encoder.endEncoding()
    
    # Execute and wait
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    
    # Read result from GPU buffer
    # PyObjC returns objc.varlist of bytes, join them
    contents = buffer_c.contents()
    length = buffer_c.length()
    raw_bytes = b''.join(contents[:length])
    result = np.frombuffer(raw_bytes, dtype=np.float32).reshape((M, N)).copy()
    
    return result
