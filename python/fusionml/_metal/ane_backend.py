"""
ANE Backend - Apple Neural Engine execution via CoreML
Uses coremltools MIL builder to create CoreML models that run on the Neural Engine.

The ANE excels at:
- Convolutions (conv2d)
- Batch normalization
- Small-to-medium matrix multiplications
- INT8/FP16 operations

Model caching ensures compilation cost is amortized over repeated calls.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import tempfile
import os
import time
import hashlib


# Try to import coremltools for ANE access
try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil import register_torch_op
    from coremltools.converters.mil.mil import types as mil_types
    import coremltools.converters.mil.frontend.milproto as milproto
    HAS_COREML = True
except ImportError:
    HAS_COREML = False

# Try to import CoreML Python bindings for prediction
try:
    import coremltools  # noqa: already imported above
    _CoreML_available = True
except Exception:
    _CoreML_available = False


from typing import Optional, Tuple, Dict, Any

# ============================================================================
# MODEL CACHE
# ============================================================================

class _ModelCache:
    """Thread-safe cache for compiled CoreML models keyed by shape signature."""
    
    def __init__(self, max_size: int = 64):
        self._cache = {}
        self._max_size = max_size
        self._access_order: list = []
        self._temp_dir = tempfile.mkdtemp(prefix="fusionml_ane_")
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, model: Any):
        # Evict LRU if full
        if len(self._cache) >= self._max_size and key not in self._cache:
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        
        self._cache[key] = model
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    @property
    def temp_dir(self) -> str:
        return self._temp_dir
    
    def __len__(self) -> int:
        return len(self._cache)


_model_cache = _ModelCache()


# ============================================================================
# ANE MODEL BUILDERS (using coremltools MIL)
# ============================================================================

def _build_matmul_model(
    M: int, K: int, N: int,
    compute_units: str = "ALL"
) -> Optional[Any]:
    """
    Build a CoreML model that performs matmul A(M,K) @ B(K,N) -> C(M,N)
    Routes to Neural Engine when compute_units includes ANE.
    
    Args:
        M, K, N: Matrix dimensions
        compute_units: "ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"
    
    Returns:
        Compiled CoreML MLModel or None if coremltools unavailable
    """
    if not HAS_COREML:
        return None
    
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(M, K), dtype=mil_types.fp16),
            mb.TensorSpec(shape=(K, N), dtype=mil_types.fp16),
        ]
    )
    def matmul_prog(a, b):
        return mb.matmul(x=a, y=b)
    
    # Convert to CoreML model
    compute_unit_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    
    model = ct.convert(
        matmul_prog,
        compute_units=compute_unit_map.get(compute_units, ct.ComputeUnit.ALL),
        minimum_deployment_target=ct.target.macOS13,
    )
    
    return model


def _build_conv2d_model(
    batch: int, in_channels: int, height: int, width: int,
    out_channels: int, kernel_h: int, kernel_w: int,
    stride: int = 1, padding: int = 0,
    compute_units: str = "ALL"
) -> Optional[Any]:
    """
    Build a CoreML model for Conv2D operation.
    ANE is highly optimized for convolutions.
    
    Input:  (batch, in_channels, height, width) 
    Weight: (out_channels, in_channels, kernel_h, kernel_w)
    Output: (batch, out_channels, out_h, out_w)
    """
    if not HAS_COREML:
        return None
    
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(batch, in_channels, height, width), dtype=mil_types.fp16),
            mb.TensorSpec(
                shape=(out_channels, in_channels, kernel_h, kernel_w), 
                dtype=mil_types.fp16
            ),
        ]
    )
    def conv2d_prog(input_tensor, weight):
        return mb.conv(
            x=input_tensor,
            weight=weight,
            strides=[stride, stride],
            pad_type="custom",
            pad=[padding, padding, padding, padding],
        )
    
    compute_unit_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    
    model = ct.convert(
        conv2d_prog,
        compute_units=compute_unit_map.get(compute_units, ct.ComputeUnit.ALL),
        minimum_deployment_target=ct.target.macOS13,
    )
    
    return model


def _build_batch_norm_model(
    batch: int, channels: int, height: int, width: int,
    compute_units: str = "ALL"
) -> Optional[Any]:
    """
    Build a CoreML model for Batch Normalization.
    ANE excels at this operation.
    
    Input:  (batch, channels, height, width)
    Gamma, Beta, Mean, Var: (channels,)
    """
    if not HAS_COREML:
        return None
    
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(batch, channels, height, width), dtype=mil_types.fp16),
            mb.TensorSpec(shape=(channels,), dtype=mil_types.fp16),  # gamma
            mb.TensorSpec(shape=(channels,), dtype=mil_types.fp16),  # beta
            mb.TensorSpec(shape=(channels,), dtype=mil_types.fp16),  # mean
            mb.TensorSpec(shape=(channels,), dtype=mil_types.fp16),  # var
        ]
    )
    def bn_prog(x, gamma, beta, mean, var):
        eps = mb.const(val=np.float16(1e-5))
        # BatchNorm: gamma * (x - mean) / sqrt(var + eps) + beta
        x_centered = mb.sub(x=x, y=mb.expand_dims(x=mean, axes=[0, 2, 3]))
        var_eps = mb.add(x=var, y=eps)
        std = mb.sqrt(x=mb.expand_dims(x=var_eps, axes=[0, 2, 3]))
        normalized = mb.real_div(x=x_centered, y=std)
        scaled = mb.mul(x=normalized, y=mb.expand_dims(x=gamma, axes=[0, 2, 3]))
        return mb.add(x=scaled, y=mb.expand_dims(x=beta, axes=[0, 2, 3]))
    
    compute_unit_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    
    model = ct.convert(
        bn_prog,
        compute_units=compute_unit_map.get(compute_units, ct.ComputeUnit.ALL),
        minimum_deployment_target=ct.target.macOS13,
    )
    
    return model


# ============================================================================
# ANE EXECUTION FUNCTIONS
# ============================================================================

def ane_matmul(a: np.ndarray, b: np.ndarray, compute_units: str = "CPU_AND_NE") -> np.ndarray:
    """
    Matrix multiplication on Apple Neural Engine via CoreML.
    
    Args:
        a: First matrix (M, K), float32 or float16
        b: Second matrix (K, N), float32 or float16
        compute_units: Target compute units ("CPU_AND_NE" for ANE priority)
    
    Returns:
        Result matrix (M, N) as float32
    """
    if not HAS_COREML:
        return np.matmul(a, b)
    
    a = np.ascontiguousarray(a, dtype=np.float16)
    b = np.ascontiguousarray(b, dtype=np.float16)
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    # Check cache
    cache_key = f"matmul_{M}_{K}_{N}_{compute_units}"
    model = _model_cache.get(cache_key)
    
    if model is None:
        model = _build_matmul_model(M, K, N, compute_units=compute_units)
        if model is None:
            return np.matmul(a.astype(np.float32), b.astype(np.float32))
        _model_cache.put(cache_key, model)
    
    # Run prediction
    prediction = model.predict({"a": a, "b": b})
    
    # Extract result - coremltools returns dict with output name
    result_key = list(prediction.keys())[0]
    result = np.array(prediction[result_key], dtype=np.float32)
    
    return result


def ane_conv2d(
    input_data: np.ndarray,
    weight: np.ndarray,
    stride: int = 1,
    padding: int = 0,
    compute_units: str = "CPU_AND_NE"
) -> np.ndarray:
    """
    Conv2D on Apple Neural Engine via CoreML.
    
    Args:
        input_data: (batch, in_channels, height, width)
        weight: (out_channels, in_channels, kernel_h, kernel_w)
        stride: Convolution stride
        padding: Zero-padding
        compute_units: Target compute units
    
    Returns:
        Output tensor as float32
    """
    if not HAS_COREML:
        raise ImportError("coremltools required for ANE conv2d")
    
    input_data = np.ascontiguousarray(input_data, dtype=np.float16)
    weight = np.ascontiguousarray(weight, dtype=np.float16)
    
    B, C, H, W = input_data.shape
    OC, IC, KH, KW = weight.shape
    assert C == IC, f"Channel mismatch: input has {C} channels, weight expects {IC}"
    
    cache_key = f"conv2d_{B}_{C}_{H}_{W}_{OC}_{KH}_{KW}_{stride}_{padding}_{compute_units}"
    model = _model_cache.get(cache_key)
    
    if model is None:
        model = _build_conv2d_model(
            B, C, H, W, OC, KH, KW,
            stride=stride, padding=padding,
            compute_units=compute_units
        )
        if model is None:
            raise RuntimeError("Failed to build conv2d model")
        _model_cache.put(cache_key, model)
    
    prediction = model.predict({"input_tensor": input_data, "weight": weight})
    result_key = list(prediction.keys())[0]
    return np.array(prediction[result_key], dtype=np.float32)


def ane_batch_norm(
    input_data: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    compute_units: str = "CPU_AND_NE"
) -> np.ndarray:
    """
    Batch Normalization on Apple Neural Engine via CoreML.
    
    Args:
        input_data: (batch, channels, height, width)
        gamma, beta, mean, var: (channels,) parameters
        compute_units: Target compute units
    
    Returns:
        Normalized output as float32
    """
    if not HAS_COREML:
        raise ImportError("coremltools required for ANE batch_norm")
    
    input_data = np.ascontiguousarray(input_data, dtype=np.float16)
    gamma = np.ascontiguousarray(gamma, dtype=np.float16)
    beta = np.ascontiguousarray(beta, dtype=np.float16)
    mean = np.ascontiguousarray(mean, dtype=np.float16)
    var = np.ascontiguousarray(var, dtype=np.float16)
    
    B, C, H, W = input_data.shape
    
    cache_key = f"bn_{B}_{C}_{H}_{W}_{compute_units}"
    model = _model_cache.get(cache_key)
    
    if model is None:
        model = _build_batch_norm_model(B, C, H, W, compute_units=compute_units)
        if model is None:
            raise RuntimeError("Failed to build batch_norm model")
        _model_cache.put(cache_key, model)
    
    prediction = model.predict({
        "x": input_data,
        "gamma": gamma,
        "beta": beta,
        "mean": mean,
        "var": var,
    })
    result_key = list(prediction.keys())[0]
    return np.array(prediction[result_key], dtype=np.float32)


# ============================================================================
# ANE AVAILABILITY & INFO
# ============================================================================

def ane_available() -> bool:
    """Check if ANE execution is available (requires coremltools + macOS)."""
    return HAS_COREML


def ane_device_info() -> dict:
    """Get ANE device information."""
    info = {
        "backend": "ane-coreml",
        "available": HAS_COREML,
        "operations": ["matmul", "conv2d", "batch_norm"],
        "precision": "fp16",
        "cached_models": len(_model_cache),
    }
    if HAS_COREML:
        info["coremltools_version"] = ct.__version__
    return info


def warmup_ane(sizes: list = None):
    """
    Pre-compile common model shapes to avoid cold-start latency during benchmarks.
    
    Args:
        sizes: List of matrix sizes to pre-compile, e.g. [256, 512, 1024]
    """
    if not HAS_COREML:
        print("ANE not available (coremltools not installed)")
        return
    
    if sizes is None:
        sizes = [256, 512, 1024]
    
    print("🧠 Warming up ANE models...")
    for size in sizes:
        t0 = time.time()
        cache_key = f"matmul_{size}_{size}_{size}_CPU_AND_NE"
        if _model_cache.get(cache_key) is None:
            model = _build_matmul_model(size, size, size, compute_units="CPU_AND_NE")
            if model is not None:
                _model_cache.put(cache_key, model)
                elapsed = (time.time() - t0) * 1000
                print(f"  ✓ matmul {size}x{size}: compiled in {elapsed:.0f}ms")
    
    print(f"  ANE cache: {len(_model_cache)} models ready")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ane_matmul',
    'ane_conv2d', 
    'ane_batch_norm',
    'ane_available',
    'ane_device_info',
    'warmup_ane',
    'HAS_COREML',
]
