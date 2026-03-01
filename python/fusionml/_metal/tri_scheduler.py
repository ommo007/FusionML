"""
Tri-Compute Scheduler - THE CORE FUSIONML INNOVATION
Adaptive parallel execution across GPU (MLX) + CPU (Accelerate) + ANE (CoreML)

Key insight: Apple Silicon has 3 compute units sharing unified memory.
By profiling each unit and splitting work optimally, we achieve higher
total throughput than any single unit alone.

Strategy:
1. Profile: Measure each backend's latency for an operation
2. Calibrate: Compute optimal split ratios (gpu_ratio, cpu_ratio, ane_ratio)
3. Execute: Split data, run all 3 backends in parallel, combine results
4. Adapt: Track history to improve ratios over time
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import concurrent.futures
import time
import json
import os

# Backend availability
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .ane_backend import ane_matmul, ane_available, HAS_COREML


# ============================================================================
# PERFORMANCE PROFILER
# ============================================================================

class BackendProfiler:
    """Profiles individual backend performance for specific operations."""
    
    def __init__(self):
        self.history: Dict[str, Dict[str, List[float]]] = {}
    
    def profile_matmul(self, size: int, iterations: int = 5) -> Dict[str, float]:
        """
        Profile matmul performance on all available backends.
        
        Returns dict of {backend_name: avg_time_ms}
        """
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        results = {}
        
        # CPU (NumPy / Accelerate BLAS)
        # Warmup
        _ = np.matmul(a, b)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = np.matmul(a, b)
            times.append((time.perf_counter() - t0) * 1000)
        results["cpu"] = np.median(times)
        
        # GPU (MLX)
        if HAS_MLX:
            a_mlx = mx.array(a)
            b_mlx = mx.array(b)
            # Warmup
            c = a_mlx @ b_mlx
            mx.eval(c)
            times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                c = a_mlx @ b_mlx
                mx.eval(c)
                times.append((time.perf_counter() - t0) * 1000)
            results["gpu"] = np.median(times)
        
        # ANE (CoreML)
        if HAS_COREML:
            # Warmup (includes compilation on first call)
            _ = ane_matmul(a, b)
            times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = ane_matmul(a, b)
                times.append((time.perf_counter() - t0) * 1000)
            results["ane"] = np.median(times)
        
        # Store in history
        key = f"matmul_{size}"
        if key not in self.history:
            self.history[key] = {}
        for backend, time_ms in results.items():
            if backend not in self.history[key]:
                self.history[key][backend] = []
            self.history[key][backend].append(time_ms)
            # Keep last 20 entries
            self.history[key][backend] = self.history[key][backend][-20:]
        
        return results


# ============================================================================
# RATIO CALCULATOR
# ============================================================================

def compute_optimal_ratios(
    profile: Dict[str, float],
    min_ratio: float = 0.05
) -> Dict[str, float]:
    """
    Compute optimal work-split ratios based on profiled backend speeds.
    
    Faster backends get proportionally more work.
    throughput_i = 1 / latency_i
    ratio_i = throughput_i / sum(throughputs)
    
    Args:
        profile: {backend: time_ms} from profiling
        min_ratio: Minimum ratio for any backend (avoids zero-work scenarios)
    
    Returns:
        {backend: ratio} where ratios sum to 1.0
    """
    if not profile:
        return {"cpu": 1.0}
    
    # Compute throughputs (inverse of latency)
    throughputs = {}
    for backend, time_ms in profile.items():
        if time_ms > 0:
            throughputs[backend] = 1.0 / time_ms
    
    total_throughput = sum(throughputs.values())
    
    if total_throughput == 0:
        # Equal split
        n = len(profile)
        return {k: 1.0 / n for k in profile}
    
    # Raw ratios based on throughput
    ratios = {k: v / total_throughput for k, v in throughputs.items()}
    
    # Enforce minimum ratio
    for k in ratios:
        if ratios[k] < min_ratio:
            ratios[k] = min_ratio
    
    # Renormalize
    total = sum(ratios.values())
    ratios = {k: v / total for k, v in ratios.items()}
    
    return ratios


# ============================================================================
# TRI-COMPUTE SCHEDULER
# ============================================================================

class TriComputeScheduler:
    """
    Adaptive scheduler for CPU + GPU + ANE parallel execution.
    
    Usage:
        scheduler = TriComputeScheduler()
        scheduler.calibrate(sizes=[256, 512, 1024, 2048])
        result = scheduler.tri_matmul(a, b)
    """
    
    def __init__(self, auto_calibrate: bool = True,
                 enable_gpu: bool = True, enable_cpu: bool = True,
                 enable_ane: bool = True, random_routing: bool = False):
        self.profiler = BackendProfiler()
        self.calibrated_ratios: Dict[str, Dict[str, float]] = {}
        self.auto_calibrate = auto_calibrate
        self._calibration_count = 0
        
        # Ablation control flags
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu
        self.enable_ane = enable_ane
        self.random_routing = random_routing
        
        # Size thresholds for routing decisions
        self.CPU_ONLY_THRESHOLD = 512      # Below this: CPU only
        self.DUAL_THRESHOLD = 1024         # Below this: GPU+CPU only
        # Above DUAL_THRESHOLD: tri-compute (GPU+CPU+ANE)
        
        # Persistent thread pool — eliminates thread creation overhead
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    
    def __del__(self):
        """Clean up thread pool."""
        if hasattr(self, '_pool'):
            self._pool.shutdown(wait=False)

    def calibrate(self, sizes: List[int] = None, iterations: int = 5, verbose: bool = True):
        """
        Contention-aware calibration: profiles both parallel AND sequential
        execution to determine whether splitting actually helps.
        
        On unified-memory SoCs, backends share memory bandwidth.
        Running them simultaneously may be SLOWER than a single backend.
        We measure both and pick the genuinely faster approach.
        """
        if sizes is None:
            sizes = [256, 512, 1024, 2048, 4096]
        
        if verbose:
            print("⚡ Tri-Compute Calibration (contention-aware)")
            print("=" * 60)
        
        for size in sizes:
            if verbose:
                print(f"\n  Profiling matmul {size}x{size}...")
            
            # Phase 1: Profile each backend independently
            profile = self.profiler.profile_matmul(size, iterations=iterations)
            ratios = compute_optimal_ratios(profile)
            
            # Phase 2: Profile PARALLEL execution with these ratios
            parallel_time = self._profile_parallel(size, ratios, iterations=iterations)
            
            # Phase 3: Find best single backend
            best_single_backend = min(profile, key=profile.get)
            best_single_time = profile[best_single_backend]
            
            # Decision: use parallel only if it genuinely beats best single
            if parallel_time is not None and parallel_time < best_single_time * 0.95:
                # Parallel wins (with 5% margin to avoid noise)
                key = f"matmul_{size}"
                self.calibrated_ratios[key] = ratios
                if verbose:
                    print(f"    ✅ Parallel wins: {parallel_time:.3f}ms vs {best_single_backend}={best_single_time:.3f}ms")
                    parts = [f"{k}={v:.1%}" for k, v in ratios.items()]
                    print(f"    Ratios: {', '.join(parts)}")
            else:
                # Single backend wins — store 100% for that backend
                key = f"matmul_{size}"
                self.calibrated_ratios[key] = {best_single_backend: 1.0}
                if verbose:
                    par_str = f"{parallel_time:.3f}ms" if parallel_time else "N/A"
                    print(f"    ⚡ Single-backend wins: {best_single_backend}={best_single_time:.3f}ms vs parallel={par_str}")
        
        self._calibration_count += 1
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"✓ Calibration complete ({len(sizes)} sizes)")
    
    def _profile_parallel(self, size: int, ratios: Dict[str, float], 
                          iterations: int = 5) -> Optional[float]:
        """Profile actual parallel execution with the given ratios."""
        if len(ratios) < 2:
            return None
        
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # Build a temporary scheduler with these ratios
        key = f"matmul_{size}"
        old_ratios = self.calibrated_ratios.get(key)
        self.calibrated_ratios[key] = ratios
        
        # Warmup
        try:
            self.tri_matmul(a, b)
        except Exception:
            self.calibrated_ratios.pop(key, None)
            return None
        
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.tri_matmul(a, b)
            times.append((time.perf_counter() - t0) * 1000)
        
        # Restore old ratios
        if old_ratios is not None:
            self.calibrated_ratios[key] = old_ratios
        else:
            self.calibrated_ratios.pop(key, None)
        
        return float(np.median(times))
    
    def get_ratios(self, size: int) -> Dict[str, float]:
        """
        Get the optimal ratios for a given matrix size.
        Interpolates between calibrated sizes.
        """
        key = f"matmul_{size}"
        
        # Exact match
        if key in self.calibrated_ratios:
            return self.calibrated_ratios[key]
        
        # Find nearest calibrated size
        calibrated_sizes = []
        for k in self.calibrated_ratios:
            if k.startswith("matmul_"):
                try:
                    calibrated_sizes.append(int(k.split("_")[1]))
                except ValueError:
                    pass
        
        if not calibrated_sizes:
            # Not calibrated — use defaults
            if size < self.CPU_ONLY_THRESHOLD:
                return {"cpu": 1.0}
            elif size < self.DUAL_THRESHOLD:
                return {"gpu": 0.7, "cpu": 0.3}
            else:
                return {"gpu": 0.6, "cpu": 0.25, "ane": 0.15}
        
        # Use closest calibrated size
        closest = min(calibrated_sizes, key=lambda s: abs(s - size))
        return self.calibrated_ratios[f"matmul_{closest}"]
    
    def tri_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication using all available compute units in parallel.
        
        Splits rows of A across GPU, CPU, and ANE based on calibrated ratios.
        All compute concurrently, results are combined.
        
        Optimized for zero overhead:
          - Persistent thread pool (no thread creation cost)
          - View-based slicing (no data copies)
          - Pre-allocated output buffer
        """
        M, K = a.shape
        K2, N = b.shape
        
        min_dim = min(M, K, N)
        
        # Small: CPU only (no parallelism overhead)
        if min_dim < self.CPU_ONLY_THRESHOLD:
            return np.matmul(a, b)
        
        # Get ratios
        ratios = self.get_ratios(min_dim)
        
        # Determine which backends to use
        backends = list(ratios.keys())
        
        # Filter to available AND enabled backends
        available_backends = []
        if self.enable_cpu:
            available_backends.append("cpu")
        if HAS_MLX and self.enable_gpu and "gpu" in backends:
            available_backends.append("gpu")
        if HAS_COREML and self.enable_ane and "ane" in backends:
            available_backends.append("ane")
        
        # Fallback: must have at least one backend
        if not available_backends:
            available_backends = ["cpu"]
        
        # Random routing (ablation baseline)
        if self.random_routing:
            import random
            active_ratios = {k: random.random() for k in available_backends}
            total = sum(active_ratios.values())
            active_ratios = {k: v / total for k, v in active_ratios.items()}
        else:
            # Recalculate ratios for available backends only
            active_ratios = {k: ratios.get(k, 0) for k in available_backends}
            total = sum(active_ratios.values())
            if total > 0:
                active_ratios = {k: v / total for k, v in active_ratios.items()}
            else:
                active_ratios = {available_backends[0]: 1.0}
        
        # If only one backend, run directly (zero overhead)
        if len(active_ratios) == 1:
            backend = list(active_ratios.keys())[0]
            if backend == "gpu":
                return self._gpu_matmul(a, b)
            elif backend == "ane":
                return ane_matmul(a, b)
            else:
                return np.matmul(a, b)
        
        # ============================================================
        # PARALLEL EXECUTION — optimized hot path
        # ============================================================
        
        # Compute split points
        splits = self._compute_splits(M, active_ratios)
        
        # Pre-allocate result
        result = np.empty((M, N), dtype=np.float32)
        
        # Submit all work to persistent pool using numpy views (zero-copy)
        futures = {}
        for backend, (start, end) in splits.items():
            if end > start:
                # numpy slicing returns a VIEW, not a copy
                future = self._pool.submit(
                    self._execute_into, backend, a[start:end], b, result, start, end
                )
                futures[future] = backend
        
        # Wait for all to complete
        concurrent.futures.wait(futures.keys())
        
        # Check for exceptions
        for future in futures:
            if future.exception() is not None:
                raise future.exception()
        
        return result
    
    def _compute_splits(
        self, M: int, ratios: Dict[str, float]
    ) -> Dict[str, Tuple[int, int]]:
        """Compute row split points from ratios."""
        splits = {}
        current = 0
        backends = list(ratios.keys())
        
        for i, backend in enumerate(backends):
            if i == len(backends) - 1:
                # Last backend gets remaining rows
                splits[backend] = (current, M)
            else:
                rows = max(1, int(M * ratios[backend]))
                splits[backend] = (current, min(current + rows, M))
                current = min(current + rows, M)
        
        return splits
    
    def _execute_backend(
        self, backend: str, a: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Execute matmul on a specific backend."""
        if backend == "gpu":
            return self._gpu_matmul(a, b)
        elif backend == "ane":
            return ane_matmul(a, b)
        else:  # cpu
            return np.matmul(a, b)
    
    def _execute_into(
        self, backend: str, a: np.ndarray, b: np.ndarray,
        out: np.ndarray, row_start: int, row_end: int
    ):
        """Execute matmul and write result directly into output buffer."""
        if backend == "gpu":
            out[row_start:row_end, :] = self._gpu_matmul(a, b)
        elif backend == "ane":
            out[row_start:row_end, :] = ane_matmul(a, b)
        else:  # cpu
            # NumPy matmul supports `out` parameter for zero-copy write
            np.matmul(a, b, out=out[row_start:row_end, :])
    
    def _gpu_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU matmul via MLX."""
        if not HAS_MLX:
            return np.matmul(a, b)
        a_mlx = mx.array(a)
        b_mlx = mx.array(b)
        c = a_mlx @ b_mlx
        mx.eval(c)
        return np.array(c)
    
    def save_calibration(self, path: str):
        """Save calibration results to JSON."""
        data = {
            "calibrated_ratios": self.calibrated_ratios,
            "profiler_history": self.profiler.history,
            "calibration_count": self._calibration_count,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_calibration(self, path: str):
        """Load calibration results from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.calibrated_ratios = data.get("calibrated_ratios", {})
        self._calibration_count = data.get("calibration_count", 0)
    
    def print_status(self):
        """Print scheduler status."""
        print(f"\n📊 Tri-Compute Scheduler Status")
        print(f"   Backends: CPU" + 
              (", GPU (MLX)" if HAS_MLX else "") +
              (", ANE (CoreML)" if HAS_COREML else ""))
        print(f"   Calibrations: {self._calibration_count}")
        print(f"   Calibrated sizes: {len(self.calibrated_ratios)}")
        
        if self.calibrated_ratios:
            print(f"\n   Ratios:")
            for key, ratios in sorted(self.calibrated_ratios.items()):
                parts = [f"{k}={v:.1%}" for k, v in ratios.items()]
                print(f"     {key}: {', '.join(parts)}")


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

# Global scheduler instance
_scheduler = None

def get_scheduler() -> TriComputeScheduler:
    """Get or create the global tri-compute scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TriComputeScheduler()
    return _scheduler


def tri_matmul(a: np.ndarray, b: np.ndarray, auto_calibrate: bool = True) -> np.ndarray:
    """
    Tri-compute matrix multiplication.
    
    First call auto-calibrates if no prior calibration exists.
    
    Args:
        a: Left matrix (M, K)
        b: Right matrix (K, N)
        auto_calibrate: Auto-calibrate on first call
    
    Returns:
        Result matrix (M, N)
    """
    scheduler = get_scheduler()
    
    # Auto-calibrate on first call with large matrices
    if auto_calibrate and scheduler._calibration_count == 0:
        size = min(a.shape[0], a.shape[1] if a.ndim > 1 else 1)
        if size >= scheduler.CPU_ONLY_THRESHOLD:
            scheduler.calibrate(sizes=[512, 1024, 2048], verbose=False)
    
    return scheduler.tri_matmul(a, b)


def calibrate(sizes: List[int] = None, verbose: bool = True):
    """Run calibration on the global scheduler."""
    scheduler = get_scheduler()
    scheduler.calibrate(sizes=sizes, verbose=verbose)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'TriComputeScheduler',
    'BackendProfiler',
    'compute_optimal_ratios',
    'tri_matmul',
    'calibrate',
    'get_scheduler',
]
