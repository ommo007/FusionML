"""
FusionML Optimized Engine — MAXIMUM PERFORMANCE
=================================================
Zero-overhead adaptive routing to beat PyTorch MPS and MLX
on every matrix size.

Strategy:
  1. CALIBRATE: Profile CPU, GPU, ANE on a range of sizes
  2. BUILD LOOKUP TABLE: For each size range, store the fastest backend
  3. ROUTE: matmul() = O(1) lookup + direct call to winner. No threads.

Key insight: Thread overhead (0.1-5ms) destroys any gains from parallelism
for operations under ~50ms. Instead, route the ENTIRE operation to the
single fastest backend with ZERO overhead.
"""

import numpy as np
import time
import json
import os
from typing import Dict, List, Optional, Tuple
import concurrent.futures
import threading

# Backend imports
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    from .ane_backend import ane_matmul, ane_conv2d, ane_batch_norm, HAS_COREML, warmup_ane
except ImportError:
    HAS_COREML = False


# ============================================================================
# PERFORMANCE DATABASE
# ============================================================================

class PerfDB:
    """Stores profiled latencies per backend per operation size."""
    
    def __init__(self):
        self._data: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
    
    def record(self, op: str, size: int, backend: str, time_ms: float):
        key = f"{op}_{size}"
        with self._lock:
            if key not in self._data:
                self._data[key] = {}
            old = self._data[key].get(backend, time_ms)
            self._data[key][backend] = 0.7 * time_ms + 0.3 * old
    
    def best_backend(self, op: str, size: int) -> Optional[str]:
        key = f"{op}_{size}"
        with self._lock:
            data = self._data.get(key, {})
        if not data:
            return None
        return min(data, key=data.get)
    
    def get_times(self, op: str, size: int) -> Dict[str, float]:
        key = f"{op}_{size}"
        with self._lock:
            return dict(self._data.get(key, {}))
    
    def save(self, path: str):
        with self._lock:
            with open(path, 'w') as f:
                json.dump(self._data, f, indent=2)
    
    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                with self._lock:
                    self._data = json.load(f)


# ============================================================================
# OPTIMIZED ENGINE — ZERO OVERHEAD ROUTING
# ============================================================================

class FusionEngine:
    """
    The FusionML engine — zero-overhead adaptive routing.
    
    After calibration, matmul() is a single dict lookup + direct backend call.
    No threads, no futures, no splitting. Just the fastest possible path.
    
    Usage:
        engine = FusionEngine()
        engine.calibrate()
        result = engine.matmul(a, b)  # Automatically picks fastest backend
    """
    
    def __init__(self):
        self.perfdb = PerfDB()
        self._calibrated = False
        
        # Routing table: ordered list of (threshold, backend)
        # "Use this backend for sizes <= threshold"
        self._route_table: List[Tuple[int, str]] = [
            (768, "cpu"),       # Default: CPU wins small
            (99999, "gpu"),     # Default: GPU wins medium+large
        ]
        

    


    def calibrate(self, sizes: List[int] = None, iterations: int = 10, verbose: bool = True):
        """Profile all backends, build optimal routing table."""
        if sizes is None:
            sizes = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
        
        if verbose:
            print("⚡ FusionML Engine Calibration")
            print("=" * 65)
        
        winners_by_size = {}
        
        for size in sizes:
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            times = {}
            
            # CPU
            _ = np.matmul(a, b)
            t_list = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = np.matmul(a, b)
                t_list.append((time.perf_counter() - t0) * 1000)
            times["cpu"] = np.median(t_list)
            self.perfdb.record("matmul", size, "cpu", times["cpu"])
            
            # GPU (MLX) — measure ACTUAL _gpu_matmul path
            if HAS_MLX:
                _ = self._gpu_matmul(a, b)  # warmup
                t_list = []
                for _ in range(iterations):
                    t0 = time.perf_counter()
                    _ = self._gpu_matmul(a, b)
                    t_list.append((time.perf_counter() - t0) * 1000)
                times["gpu"] = np.median(t_list)
                self.perfdb.record("matmul", size, "gpu", times["gpu"])
            
            # ANE (CoreML)
            if HAS_COREML:
                _ = ane_matmul(a, b)
                t_list = []
                for _ in range(iterations):
                    t0 = time.perf_counter()
                    _ = ane_matmul(a, b)
                    t_list.append((time.perf_counter() - t0) * 1000)
                times["ane"] = np.median(t_list)
                self.perfdb.record("matmul", size, "ane", times["ane"])


            
            best = min(times, key=times.get)
            winners_by_size[size] = best
            
            if verbose:
                parts = [f"{k}={v:.3f}ms" for k, v in sorted(times.items())]
                marker = {k: "★" if k == best else " " for k in times}
                print(f"  {size:>5}×{size:<5} " + 
                      " | ".join(f"{k}={v:.3f}ms{marker[k]}" for k, v in sorted(times.items())) +
                      f"  → {best.upper()}")
        
        # Build routing table from profiled data
        self._build_route_table(sizes, winners_by_size)
        self._calibrated = True
        
        if verbose:
            print(f"\n  Routing table:")
            for threshold, backend in self._route_table:
                print(f"    size ≤ {threshold:>5}: {backend.upper()}")
            print("=" * 65)

    def _build_route_table(self, sizes: List[int], winners: Dict[int, str]):
        """
        Build a compact routing table from per-size winners.
        Merges consecutive sizes with the same winner into ranges.
        """
        table = []
        prev_backend = None
        
        for size in sorted(sizes):
            backend = winners.get(size, "cpu")
            if backend != prev_backend:
                if prev_backend is not None:
                    table.append((size - 1, prev_backend))
                prev_backend = backend
        
        # Final entry covers everything above
        if prev_backend:
            table.append((999999, prev_backend))
        
        if table:
            self._route_table = table
    
    def _get_backend(self, size: int) -> str:
        """O(n) lookup in routing table. n is tiny (3-5 entries)."""
        for threshold, backend in self._route_table:
            if size <= threshold:
                return backend
        return self._route_table[-1][1] if self._route_table else "cpu"
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        ZERO-OVERHEAD optimal matmul.
        
        After calibration, this is just:
          1. Compute min dimension (1 comparison)
          2. Lookup routing table (3-5 comparisons)
          3. Direct function call to winning backend
        """
        min_dim = min(a.shape[0], a.shape[1] if a.ndim > 1 else 1,
                      b.shape[1] if b.ndim > 1 else 1)
        
        backend = self._get_backend(min_dim)
        
        if backend == "gpu" and HAS_MLX:
            return self._gpu_matmul(a, b)
        elif backend == "ane" and HAS_COREML:
            return ane_matmul(a, b)

        else:
            return np.matmul(a, b)
    
    def _gpu_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Direct GPU matmul via MLX — minimal overhead."""
        a_mx = mx.array(a)
        b_mx = mx.array(b)
        c = a_mx @ b_mx
        mx.eval(c)
        return np.array(c)


    
    # ── MLX-NATIVE METHODS (for chained operations) ──────────────────
    
    def matmul_mlx(self, a_mx, b_mx):
        """
        GPU matmul keeping data in MLX format.
        For chained operations where np↔mx conversion is unnecessary.
        Returns an mx.array (lazy — call mx.eval when you need the result).
        """
        return a_mx @ b_mx
    
    def forward_2layer(self, x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """
        Optimized 2-layer forward pass: matmul → relu → matmul.
        
        Uses MLX-native pipeline: data→GPU only at start, GPU→CPU only at end.
        Intermediate results stay on GPU, eliminating 2x conversion overhead.
        """
        min_dim = min(x.shape[0], w1.shape[0])
        
        if min_dim >= 1024 and HAS_MLX:
            # Keep everything in MLX — one np→mx at start, one mx→np at end
            x_mx = mx.array(x)
            w1_mx = mx.array(w1)
            w2_mx = mx.array(w2)
            h = x_mx @ w1_mx
            h = mx.maximum(h, mx.array(0.0))
            o = h @ w2_mx
            mx.eval(o)
            return np.array(o)
        else:
            # Small: CPU is faster, no conversion needed
            h = np.matmul(x, w1)
            h = np.maximum(h, 0)
            return np.matmul(h, w2)

    def transformer_encoder_block(self, x: np.ndarray, 
                                w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray, 
                                w_o: np.ndarray, w_ff1: np.ndarray, w_ff2: np.ndarray,
                                heads: int = 8) -> np.ndarray:
        """
        Optimized Transformer Encoder Block (Native MLX Pipeline).
        
        Operation:
          1. Multi-Head Attention (Q,K,V proj -> Scaled Dot Product -> Output proj)
          2. Add + Norm (simplified to Add for benchmark)
          3. Feed Forward (Linear -> GELU -> Linear)
          4. Add + Norm (simplified to Add for benchmark)
          
        Keep entire graph on GPU to eliminate 6+ conversions.
        """
        B, L, D = x.shape
        min_dim = min(B * L, D)
        
        # Use MLX path for larger workloads
        if min_dim >= 512 and HAS_MLX:
            x_mx = mx.array(x)
            q_mx, k_mx, v_mx = mx.array(w_q), mx.array(w_k), mx.array(w_v)
            o_mx = mx.array(w_o)
            ff1_mx, ff2_mx = mx.array(w_ff1), mx.array(w_ff2)
            
            # 1. Attention
            # Projections
            Q = x_mx @ q_mx
            K = x_mx @ k_mx
            V = x_mx @ v_mx
            
            # Reshape for heads (B, L, H, D//H) -> (B, H, L, D//H)
            d_head = D // heads
            Q = Q.reshape(B, L, heads, d_head).transpose(0, 2, 1, 3)
            K = K.reshape(B, L, heads, d_head).transpose(0, 2, 1, 3)
            V = V.reshape(B, L, heads, d_head).transpose(0, 2, 1, 3)
            
            # Scaled Dot Product
            scale = 1.0 / np.sqrt(d_head)
            scores = (Q @ K.transpose(0, 1, 3, 2)) * scale
            attn = mx.softmax(scores, axis=-1)
            out = attn @ V
            
            # Combine heads (B, H, L, D//H) -> (B, L, D)
            out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
            
            # Output projection + Residual
            out = (out @ o_mx) + x_mx
            
            # 2. Feed Forward (MLP)
            # Linear -> GELU -> Linear + Residual
            ff = out @ ff1_mx
            ff = mx.maximum(ff, mx.array(0.0)) # ReLU for simplicity/parity
            ff = ff @ ff2_mx
            output = ff + out
            
            mx.eval(output)
            return np.array(output)
            
        else:
            # CPU Fallback (simplified)
            # 1. Attention
            Q = x @ w_q
            K = x @ w_k
            V = x @ w_v
            
            d_head = D // heads
            Q = Q.reshape(B, L, heads, d_head).transpose(0, 2, 1, 3)
            K = K.reshape(B, L, heads, d_head).transpose(0, 2, 1, 3)
            V = V.reshape(B, L, heads, d_head).transpose(0, 2, 1, 3)
            
            scale = 1.0 / np.sqrt(d_head)
            scores = (Q @ K.transpose(0, 1, 3, 2)) * scale
            
            # Stable softmax
            scores_max = scores.max(axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
            
            out = attn @ V
            out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
            out = (out @ w_o) + x
            
            # 2. MLP
            ff = out @ w_ff1
            ff = np.maximum(ff, 0)
            ff = ff @ w_ff2
            output = ff + out
            
            return output


# ============================================================================
# GLOBAL ENGINE
# ============================================================================

_engine: Optional[FusionEngine] = None

def get_engine() -> FusionEngine:
    global _engine
    if _engine is None:
        _engine = FusionEngine()
    return _engine


def fast_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Top-level optimized matmul."""
    return get_engine().matmul(a, b)


__all__ = ['FusionEngine', 'get_engine', 'fast_matmul', 'PerfDB']

