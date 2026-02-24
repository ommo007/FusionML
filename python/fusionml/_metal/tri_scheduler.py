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
    
    def __init__(self, auto_calibrate: bool = True):
        self.profiler = BackendProfiler()
        self.calibrated_ratios: Dict[str, Dict[str, float]] = {}
        self.auto_calibrate = auto_calibrate
        self._calibration_count = 0
        
        # Size thresholds for routing decisions
        self.CPU_ONLY_THRESHOLD = 512      # Below this: CPU only
        self.DUAL_THRESHOLD = 1024         # Below this: GPU+CPU only
        # Above DUAL_THRESHOLD: tri-compute (GPU+CPU+ANE)
    
    def calibrate(self, sizes: List[int] = None, iterations: int = 5, verbose: bool = True):
        """
        Run calibration suite: profile all backends at various sizes
        and compute optimal ratios.
        
        Args:
            sizes: Matrix sizes to calibrate
            iterations: Number of trials per size
            verbose: Print results
        """
        if sizes is None:
            sizes = [256, 512, 1024, 2048, 4096]
        
        if verbose:
            print("⚡ Tri-Compute Calibration")
            print("=" * 60)
        
        for size in sizes:
            if verbose:
                print(f"\n  Profiling matmul {size}x{size}...")
            
            profile = self.profiler.profile_matmul(size, iterations=iterations)
            ratios = compute_optimal_ratios(profile)
            
            key = f"matmul_{size}"
            self.calibrated_ratios[key] = ratios
            
            if verbose:
                print(f"    Latencies: ", end="")
                parts = [f"{k}={v:.3f}ms" for k, v in profile.items()]
                print(", ".join(parts))
                print(f"    Ratios:    ", end="")
                parts = [f"{k}={v:.1%}" for k, v in ratios.items()]
                print(", ".join(parts))
        
        self._calibration_count += 1
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"✓ Calibration complete ({len(sizes)} sizes)")
    
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
        All 3 compute concurrently, results are combined.
        
        Args:
            a: Left matrix (M, K)
            b: Right matrix (K, N)
        
        Returns:
            Result matrix (M, N)
        """
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Shape mismatch: {a.shape} vs {b.shape}"
        
        min_dim = min(M, K, N)
        
        # Small: CPU only (no parallelism overhead)
        if min_dim < self.CPU_ONLY_THRESHOLD:
            return np.matmul(a, b)
        
        # Get ratios
        ratios = self.get_ratios(min_dim)
        
        # Determine which backends to use
        backends = list(ratios.keys())
        
        # Filter to available backends
        available_backends = ["cpu"]  # CPU always available
        if HAS_MLX and "gpu" in backends:
            available_backends.append("gpu")
        if HAS_COREML and "ane" in backends:
            available_backends.append("ane")
        
        # Recalculate ratios for available backends only
        active_ratios = {k: ratios.get(k, 0) for k in available_backends}
        total = sum(active_ratios.values())
        if total > 0:
            active_ratios = {k: v / total for k, v in active_ratios.items()}
        else:
            active_ratios = {"cpu": 1.0}
        
        # If only one backend, run directly
        if len(active_ratios) == 1:
            backend = list(active_ratios.keys())[0]
            if backend == "gpu":
                return self._gpu_matmul(a, b)
            elif backend == "ane":
                return ane_matmul(a, b)
            else:
                return np.matmul(a, b)
        
        # Compute split points
        splits = self._compute_splits(M, active_ratios)
        
        # Define work functions
        work_items = []
        for backend, (start, end) in splits.items():
            if end > start:  # Non-empty slice
                a_slice = a[start:end, :]
                work_items.append((backend, start, end, a_slice))
        
        # Execute in parallel
        result = np.zeros((M, N), dtype=np.float32)
        
        if len(work_items) == 1:
            # Single backend
            backend, start, end, a_slice = work_items[0]
            result[start:end, :] = self._execute_backend(backend, a_slice, b)
        else:
            # Parallel execution across backends
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(work_items)) as executor:
                futures = {}
                for backend, start, end, a_slice in work_items:
                    future = executor.submit(self._execute_backend, backend, a_slice, b)
                    futures[future] = (start, end)
                
                for future in concurrent.futures.as_completed(futures):
                    start, end = futures[future]
                    result[start:end, :] = future.result()
        
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
