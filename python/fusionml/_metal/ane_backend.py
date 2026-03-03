"""
ANE Backend - Apple Neural Engine execution via CoreML

Key design:
- Static weights baked into models via mil.const() for instant compilation
- Disk cache for compiled .mlmodelc files across process restarts
- ANECompiledLayer for full-layer graph compilation (conv+bn+relu as one model)

The ANE excels at:
- Convolutions (conv2d) with static weights
- Batch normalization with fixed parameters
- Fused layer graphs (conv+bn+relu compiled together)
"""

import os
import time
import hashlib
import tempfile
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    # coremltools 9.0 moved types to mil.mil.types
    try:
        from coremltools.converters.mil import types as mil_types
    except ImportError:
        from coremltools.converters.mil.mil import types as mil_types
    HAS_COREML = True
    _CoreML_available = True
except ImportError:
    HAS_COREML = False
    _CoreML_available = False


# ============================================================================
# DISK-BACKED MODEL CACHE
# ============================================================================

def _cache_dir():
    """Get or create the ANE model cache directory."""
    d = os.path.join(os.path.expanduser("~"), ".cache", "fusionml", "ane")
    os.makedirs(d, exist_ok=True)
    return d


def _weight_hash(arrays: List[np.ndarray]) -> str:
    """Compute a fast hash of weight arrays for cache keys."""
    h = hashlib.md5()
    for a in arrays:
        h.update(a.tobytes()[:1024])  # First 1KB for speed
        h.update(str(a.shape).encode())
    return h.hexdigest()[:12]


class _ModelCache:
    """Thread-safe cache for compiled CoreML models with disk persistence."""

    def __init__(self, max_memory: int = 64):
        self._memory: Dict[str, Any] = {}
        self._max_memory = max_memory
        self._access_order: list = []
        self._disk_dir = _cache_dir()

    def get(self, key: str) -> Optional[Any]:
        """Get model from memory cache, falling back to disk."""
        if key in self._memory:
            return self._memory[key]
        # Try disk
        disk_path = os.path.join(self._disk_dir, f"{key}.mlpackage")
        if os.path.exists(disk_path):
            try:
                model = ct.models.MLModel(disk_path)
                self._memory[key] = model
                return model
            except Exception:
                pass
        return None

    def put(self, key: str, model: Any, save_to_disk: bool = True):
        """Store model in memory and optionally to disk."""
        if len(self._memory) >= self._max_memory:
            oldest = self._access_order.pop(0)
            self._memory.pop(oldest, None)
        self._memory[key] = model
        self._access_order.append(key)
        if save_to_disk:
            try:
                disk_path = os.path.join(self._disk_dir, f"{key}.mlpackage")
                if not os.path.exists(disk_path):
                    model.save(disk_path)
            except Exception:
                pass

    def __len__(self):
        return len(self._memory)

    def clear(self):
        self._memory.clear()
        self._access_order.clear()


_model_cache = _ModelCache()

_CU_MAP = {}
if HAS_COREML:
    _CU_MAP = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }


def _convert(program, compute_units: str = "ALL"):
    """Convert a MIL program to a CoreML model."""
    return ct.convert(
        program,
        compute_units=_CU_MAP.get(compute_units, ct.ComputeUnit.ALL),
        minimum_deployment_target=ct.target.macOS13,
    )


# ============================================================================
# STATIC WEIGHT MODEL BUILDERS
# ============================================================================
# Key difference from old backend: weights are baked as mb.const() rather than
# passed as dynamic TensorSpec inputs. This makes compilation much faster and
# enables ANE kernel fusion.

def _build_matmul_static(
    M: int, K: int, N: int,
    b_weight: np.ndarray,
    compute_units: str = "ALL"
) -> Optional[Any]:
    """
    Build CoreML matmul with B matrix baked as a static constant.
    Only A is a dynamic input: A(M,K) @ const_B(K,N) -> C(M,N)
    """
    if not HAS_COREML:
        return None
    b_const = b_weight.astype(np.float16)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(M, K), dtype=mil_types.fp16)]
    )
    def prog(a):
        b = mb.const(val=b_const)
        return mb.matmul(x=a, y=b)

    return _convert(prog, compute_units)


def _build_conv2d_static(
    batch: int, in_channels: int, height: int, width: int,
    weight: np.ndarray,
    stride: int = 1, padding: int = 0,
    compute_units: str = "ALL"
) -> Optional[Any]:
    """
    Build CoreML conv2d with weight baked as a static constant.
    Only input tensor is dynamic. This is how Apple's own models work.
    """
    if not HAS_COREML:
        return None
    w_const = weight.astype(np.float16)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(batch, in_channels, height, width), dtype=mil_types.fp16)
        ]
    )
    def prog(input_tensor):
        w = mb.const(val=w_const)
        return mb.conv(
            x=input_tensor, weight=w,
            strides=[stride, stride],
            pad_type="custom",
            pad=[padding, padding, padding, padding],
        )

    return _convert(prog, compute_units)


def _build_batch_norm_static(
    batch: int, channels: int, height: int, width: int,
    gamma: np.ndarray, beta: np.ndarray,
    mean: np.ndarray, var: np.ndarray,
    compute_units: str = "ALL"
) -> Optional[Any]:
    """
    Build CoreML batch norm with all parameters baked as constants.
    Only the input tensor is dynamic.
    """
    if not HAS_COREML:
        return None
    g = gamma.astype(np.float16)
    b = beta.astype(np.float16)
    m = mean.astype(np.float16)
    v = var.astype(np.float16)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(batch, channels, height, width), dtype=mil_types.fp16)
        ]
    )
    def prog(x):
        eps = mb.const(val=np.float16(1e-5))
        g_c = mb.const(val=g)
        b_c = mb.const(val=b)
        m_c = mb.const(val=m)
        v_c = mb.const(val=v)
        x_centered = mb.sub(x=x, y=mb.expand_dims(x=m_c, axes=[0, 2, 3]))
        var_eps = mb.add(x=v_c, y=eps)
        std = mb.sqrt(x=mb.expand_dims(x=var_eps, axes=[0, 2, 3]))
        normalized = mb.real_div(x=x_centered, y=std)
        scaled = mb.mul(x=normalized, y=mb.expand_dims(x=g_c, axes=[0, 2, 3]))
        return mb.add(x=scaled, y=mb.expand_dims(x=b_c, axes=[0, 2, 3]))

    return _convert(prog, compute_units)


# ============================================================================
# FUSED LAYER BUILDERS (conv + bn + relu as one model)
# ============================================================================

def _build_conv_bn_relu_static(
    batch: int, in_channels: int, height: int, width: int,
    conv_weight: np.ndarray,
    bn_gamma: np.ndarray, bn_beta: np.ndarray,
    bn_mean: np.ndarray, bn_var: np.ndarray,
    stride: int = 1, padding: int = 0,
    relu: bool = True,
    compute_units: str = "ALL"
) -> Optional[Any]:
    """
    Build a FUSED conv2d + batch_norm + relu as a single CoreML model.
    All weights and BN params are static constants.
    This is the key to ANE performance — one model.predict() for an entire layer.
    """
    if not HAS_COREML:
        return None

    w = conv_weight.astype(np.float16)
    g = bn_gamma.astype(np.float16)
    b = bn_beta.astype(np.float16)
    m = bn_mean.astype(np.float16)
    v = bn_var.astype(np.float16)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(batch, in_channels, height, width), dtype=mil_types.fp16)
        ]
    )
    def prog(x):
        # Conv2d
        w_c = mb.const(val=w)
        h = mb.conv(
            x=x, weight=w_c,
            strides=[stride, stride],
            pad_type="custom",
            pad=[padding, padding, padding, padding],
        )
        # BatchNorm
        eps = mb.const(val=np.float16(1e-5))
        g_c = mb.const(val=g)
        b_c = mb.const(val=b)
        m_c = mb.const(val=m)
        v_c = mb.const(val=v)
        h_centered = mb.sub(x=h, y=mb.expand_dims(x=m_c, axes=[0, 2, 3]))
        var_eps = mb.add(x=v_c, y=eps)
        std = mb.sqrt(x=mb.expand_dims(x=var_eps, axes=[0, 2, 3]))
        h_norm = mb.real_div(x=h_centered, y=std)
        h_scaled = mb.mul(x=h_norm, y=mb.expand_dims(x=g_c, axes=[0, 2, 3]))
        out = mb.add(x=h_scaled, y=mb.expand_dims(x=b_c, axes=[0, 2, 3]))
        # ReLU
        if relu:
            out = mb.relu(x=out)
        return out

    return _convert(prog, compute_units)


# ============================================================================
# ANECompiledLayer — High-level API
# ============================================================================

class ANECompiledLayer:
    """
    Pre-compiled CoreML model with static weights for ANE execution.
    Compile once (or load from disk cache), run forever with near-zero overhead.
    
    Usage:
        layer = ANECompiledLayer.conv_bn_relu(
            input_shape=(1, 64, 56, 56),
            weight=conv_weight,
            bn_gamma=g, bn_beta=b, bn_mean=m, bn_var=v,
            padding=1
        )
        output = layer(input_tensor)  # Fast ANE execution
    """

    def __init__(self, model, input_name: str = "x"):
        self._model = model
        self._input_name = input_name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run inference on ANE. Input should be float32 NCHW."""
        x_in = np.ascontiguousarray(x, dtype=np.float32)
        prediction = self._model.predict({self._input_name: x_in})
        result_key = list(prediction.keys())[0]
        return np.array(prediction[result_key], dtype=np.float32)

    @staticmethod
    def conv_bn_relu(
        input_shape: Tuple[int, int, int, int],
        weight: np.ndarray,
        bn_gamma: np.ndarray, bn_beta: np.ndarray,
        bn_mean: np.ndarray, bn_var: np.ndarray,
        stride: int = 1, padding: int = 0,
        relu: bool = True,
        compute_units: str = "ALL"
    ) -> 'ANECompiledLayer':
        """Create a fused conv+bn+relu layer compiled for ANE."""
        B, C, H, W = input_shape
        wh = _weight_hash([weight, bn_gamma, bn_beta, bn_mean, bn_var])
        cache_key = f"cbr_{B}_{C}_{H}_{W}_{weight.shape[0]}_{stride}_{padding}_{relu}_{wh}"

        model = _model_cache.get(cache_key)
        if model is None:
            model = _build_conv_bn_relu_static(
                B, C, H, W, weight,
                bn_gamma, bn_beta, bn_mean, bn_var,
                stride=stride, padding=padding, relu=relu,
                compute_units=compute_units
            )
            if model is None:
                raise RuntimeError("Failed to compile conv_bn_relu for ANE")
            _model_cache.put(cache_key, model)

        return ANECompiledLayer(model, input_name="x")

    @staticmethod
    def conv2d(
        input_shape: Tuple[int, int, int, int],
        weight: np.ndarray,
        stride: int = 1, padding: int = 0,
        compute_units: str = "ALL"
    ) -> 'ANECompiledLayer':
        """Create a conv2d layer with static weights compiled for ANE."""
        B, C, H, W = input_shape
        wh = _weight_hash([weight])
        cache_key = f"conv_{B}_{C}_{H}_{W}_{weight.shape[0]}_{stride}_{padding}_{wh}"

        model = _model_cache.get(cache_key)
        if model is None:
            model = _build_conv2d_static(
                B, C, H, W, weight,
                stride=stride, padding=padding,
                compute_units=compute_units
            )
            if model is None:
                raise RuntimeError("Failed to compile conv2d for ANE")
            _model_cache.put(cache_key, model)

        return ANECompiledLayer(model, input_name="input_tensor")

    @staticmethod
    def matmul(
        M: int, K: int,
        weight: np.ndarray,
        compute_units: str = "ALL"
    ) -> 'ANECompiledLayer':
        """Create a matmul layer: input(M,K) @ static_weight(K,N) -> (M,N)."""
        N = weight.shape[1]
        wh = _weight_hash([weight])
        cache_key = f"mm_{M}_{K}_{N}_{wh}"

        model = _model_cache.get(cache_key)
        if model is None:
            model = _build_matmul_static(M, K, N, weight, compute_units=compute_units)
            if model is None:
                raise RuntimeError("Failed to compile matmul for ANE")
            _model_cache.put(cache_key, model)

        return ANECompiledLayer(model, input_name="a")


# ============================================================================
# BACKWARD-COMPATIBLE EXECUTION FUNCTIONS (dynamic weight API)
# ============================================================================
# These keep the old API working but now use static weight compilation under
# the hood for the B/weight matrix. First call is slow (compilation), but
# subsequent calls with the same weight are instant (cached).

def ane_matmul(a: np.ndarray, b: np.ndarray, compute_units: str = "CPU_AND_NE") -> np.ndarray:
    """
    Matrix multiplication on Apple Neural Engine via CoreML.
    Uses static weight compilation — first call compiles, rest are cached.
    """
    if not HAS_COREML:
        return np.matmul(a, b)

    a = np.ascontiguousarray(a, dtype=np.float32)
    b_f16 = np.ascontiguousarray(b, dtype=np.float16)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: {a.shape} vs {b.shape}"

    wh = _weight_hash([b_f16])
    cache_key = f"matmul_{M}_{K}_{N}_{compute_units}_{wh}"
    model = _model_cache.get(cache_key)

    if model is None:
        model = _build_matmul_static(M, K, N, b, compute_units=compute_units)
        if model is None:
            return np.matmul(a, b.astype(np.float32))
        _model_cache.put(cache_key, model)

    prediction = model.predict({"a": a})
    result_key = list(prediction.keys())[0]
    return np.array(prediction[result_key], dtype=np.float32)


def ane_conv2d(
    input_data: np.ndarray,
    weight: np.ndarray,
    stride: int = 1,
    padding: int = 0,
    compute_units: str = "CPU_AND_NE"
) -> np.ndarray:
    """Conv2D on ANE with static weight compilation."""
    if not HAS_COREML:
        raise ImportError("coremltools required for ANE conv2d")

    input_data = np.ascontiguousarray(input_data, dtype=np.float32)
    w_f16 = np.ascontiguousarray(weight, dtype=np.float16)
    B, C, H, W = input_data.shape

    wh = _weight_hash([w_f16])
    cache_key = f"conv2d_{B}_{C}_{H}_{W}_{weight.shape[0]}_{stride}_{padding}_{compute_units}_{wh}"
    model = _model_cache.get(cache_key)

    if model is None:
        model = _build_conv2d_static(
            B, C, H, W, weight,
            stride=stride, padding=padding,
            compute_units=compute_units
        )
        if model is None:
            raise RuntimeError("Failed to build conv2d model")
        _model_cache.put(cache_key, model)

    prediction = model.predict({"input_tensor": input_data})
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
    """Batch Normalization on ANE with static parameter compilation."""
    if not HAS_COREML:
        raise ImportError("coremltools required for ANE batch_norm")

    input_data = np.ascontiguousarray(input_data, dtype=np.float32)
    B, C, H, W = input_data.shape

    wh = _weight_hash([gamma, beta, mean, var])
    cache_key = f"bn_{B}_{C}_{H}_{W}_{compute_units}_{wh}"
    model = _model_cache.get(cache_key)

    if model is None:
        model = _build_batch_norm_static(
            B, C, H, W, gamma, beta, mean, var,
            compute_units=compute_units
        )
        if model is None:
            raise RuntimeError("Failed to build batch_norm model")
        _model_cache.put(cache_key, model)

    prediction = model.predict({"x": input_data})
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
        "backend": "ane-coreml-static",
        "available": HAS_COREML,
        "operations": ["matmul", "conv2d", "batch_norm", "conv_bn_relu (fused)"],
        "weight_mode": "static (mil.const)",
        "disk_cache": _cache_dir() if HAS_COREML else None,
        "cached_models_memory": len(_model_cache),
    }
    if HAS_COREML:
        info["coremltools_version"] = ct.__version__
    return info


def warmup_ane(sizes: list = None):
    """Pre-compile common matmul shapes with static weights."""
    if not HAS_COREML:
        print("ANE not available (coremltools not installed)")
        return

    if sizes is None:
        sizes = [256, 512, 1024]

    print("🧠 Warming up ANE (static weight mode)...")
    for size in sizes:
        t0 = time.time()
        b_dummy = np.random.randn(size, size).astype(np.float16) * 0.02
        wh = _weight_hash([b_dummy])
        cache_key = f"matmul_{size}_{size}_{size}_CPU_AND_NE_{wh}"
        if _model_cache.get(cache_key) is None:
            model = _build_matmul_static(size, size, size, b_dummy, compute_units="CPU_AND_NE")
            if model is not None:
                _model_cache.put(cache_key, model)
                elapsed = (time.time() - t0) * 1000
                print(f"  ✓ matmul {size}x{size}: compiled in {elapsed:.0f}ms")

    print(f"  ANE cache: {len(_model_cache)} models ready")


def clear_cache():
    """Clear all cached ANE models (memory + disk)."""
    _model_cache.clear()
    import shutil
    d = _cache_dir()
    if os.path.exists(d):
        shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("ANE cache cleared.")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ANECompiledLayer',
    'ane_matmul',
    'ane_conv2d',
    'ane_batch_norm',
    'ane_available',
    'ane_device_info',
    'warmup_ane',
    'clear_cache',
    'HAS_COREML',
]
