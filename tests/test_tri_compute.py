#!/usr/bin/env python3
"""
Tests for ANE Backend and Tri-Compute Scheduler
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


# ============================================================================
# ANE Backend Tests
# ============================================================================

class TestANEBackend:
    """Test CoreML-based ANE execution."""
    
    def test_ane_available(self):
        from fusionml._metal.ane_backend import ane_available
        # Should be True on macOS with coremltools
        assert isinstance(ane_available(), bool)
    
    def test_ane_device_info(self):
        from fusionml._metal.ane_backend import ane_device_info
        info = ane_device_info()
        assert "backend" in info
        assert info["backend"] == "ane-coreml"
        assert "available" in info
        assert "operations" in info
    
    def test_ane_matmul_small(self):
        from fusionml._metal.ane_backend import ane_matmul, ane_available
        if not ane_available():
            pytest.skip("ANE not available")
        
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)
        
        result = ane_matmul(a, b)
        expected = np.matmul(a, b)
        
        assert result.shape == expected.shape
        # fp16 tolerance
        rel_err = np.max(np.abs(result - expected)) / (np.max(np.abs(expected)) + 1e-8)
        assert rel_err < 0.02, f"Relative error too high: {rel_err}"
    
    def test_ane_matmul_medium(self):
        from fusionml._metal.ane_backend import ane_matmul, ane_available
        if not ane_available():
            pytest.skip("ANE not available")
        
        a = np.random.randn(256, 256).astype(np.float32)
        b = np.random.randn(256, 256).astype(np.float32)
        
        result = ane_matmul(a, b)
        expected = np.matmul(a, b)
        
        assert result.shape == expected.shape
        rel_err = np.max(np.abs(result - expected)) / (np.max(np.abs(expected)) + 1e-8)
        assert rel_err < 0.02, f"Relative error too high: {rel_err}"
    
    def test_ane_matmul_nonsquare(self):
        from fusionml._metal.ane_backend import ane_matmul, ane_available
        if not ane_available():
            pytest.skip("ANE not available")
        
        a = np.random.randn(128, 64).astype(np.float32)
        b = np.random.randn(64, 256).astype(np.float32)
        
        result = ane_matmul(a, b)
        expected = np.matmul(a, b)
        
        assert result.shape == (128, 256)
        rel_err = np.max(np.abs(result - expected)) / (np.max(np.abs(expected)) + 1e-8)
        assert rel_err < 0.02
    
    def test_ane_model_caching(self):
        """Verify that repeated calls with same shape use cached model."""
        from fusionml._metal.ane_backend import ane_matmul, ane_available, _model_cache
        if not ane_available():
            pytest.skip("ANE not available")
        
        a = np.random.randn(64, 64).astype(np.float32)
        b = np.random.randn(64, 64).astype(np.float32)
        
        # First call compiles model
        _ = ane_matmul(a, b)
        cache_size_after_first = len(_model_cache)
        
        # Second call should use cache (same shape)
        _ = ane_matmul(a, b)
        assert len(_model_cache) == cache_size_after_first


# ============================================================================
# Tri-Compute Scheduler Tests
# ============================================================================

class TestTriScheduler:
    """Test the tri-compute scheduler."""
    
    def test_compute_optimal_ratios(self):
        from fusionml._metal.tri_scheduler import compute_optimal_ratios
        
        profile = {"cpu": 10.0, "gpu": 5.0, "ane": 20.0}
        ratios = compute_optimal_ratios(profile)
        
        # Should sum to ~1.0
        assert abs(sum(ratios.values()) - 1.0) < 0.01
        
        # GPU should get most work (fastest = lowest time)
        assert ratios["gpu"] > ratios["cpu"]
        assert ratios["gpu"] > ratios["ane"]
    
    def test_compute_optimal_ratios_single(self):
        from fusionml._metal.tri_scheduler import compute_optimal_ratios
        
        profile = {"cpu": 10.0}
        ratios = compute_optimal_ratios(profile)
        assert ratios["cpu"] == 1.0
    
    def test_scheduler_creation(self):
        from fusionml._metal.tri_scheduler import TriComputeScheduler
        
        scheduler = TriComputeScheduler()
        assert scheduler.CPU_ONLY_THRESHOLD == 512
        assert scheduler.DUAL_THRESHOLD == 1024
    
    def test_scheduler_small_matmul(self):
        """Small matrices should go CPU-only (no parallelism overhead)."""
        from fusionml._metal.tri_scheduler import TriComputeScheduler
        
        scheduler = TriComputeScheduler()
        a = np.random.randn(128, 128).astype(np.float32)
        b = np.random.randn(128, 128).astype(np.float32)
        
        result = scheduler.tri_matmul(a, b)
        expected = np.matmul(a, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_scheduler_large_matmul_correct(self):
        """Large matrices: verify correctness of tri-compute result."""
        from fusionml._metal.tri_scheduler import TriComputeScheduler
        
        scheduler = TriComputeScheduler()
        scheduler.calibrate(sizes=[1024], iterations=2, verbose=False)
        
        a = np.random.randn(1024, 1024).astype(np.float32)
        b = np.random.randn(1024, 1024).astype(np.float32)
        
        result = scheduler.tri_matmul(a, b)
        expected = np.matmul(a, b)
        
        rel_err = np.max(np.abs(result - expected)) / (np.max(np.abs(expected)) + 1e-8)
        assert rel_err < 0.02, f"Relative error too high: {rel_err}"
    
    def test_scheduler_calibration(self):
        """Calibration should produce valid ratios."""
        from fusionml._metal.tri_scheduler import TriComputeScheduler
        
        scheduler = TriComputeScheduler()
        scheduler.calibrate(sizes=[256], iterations=2, verbose=False)
        
        ratios = scheduler.get_ratios(256)
        assert abs(sum(ratios.values()) - 1.0) < 0.01
        assert all(v >= 0 for v in ratios.values())
    
    def test_scheduler_get_ratios_default(self):
        """Default ratios when not calibrated."""
        from fusionml._metal.tri_scheduler import TriComputeScheduler
        
        scheduler = TriComputeScheduler()
        
        # Small
        ratios = scheduler.get_ratios(256)
        assert "cpu" in ratios
        
        # Large 
        ratios = scheduler.get_ratios(4096)
        assert len(ratios) >= 2
    
    def test_module_level_tri_matmul(self):
        """Test the convenience function."""
        from fusionml._metal.tri_scheduler import tri_matmul
        
        a = np.random.randn(64, 64).astype(np.float32)
        b = np.random.randn(64, 64).astype(np.float32)
        
        result = tri_matmul(a, b, auto_calibrate=False)
        expected = np.matmul(a, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test end-to-end integration."""
    
    def test_tensor_matmul_uses_tri_scheduler(self):
        """Verify Tensor matmul routes to tri-compute for large matrices."""
        from fusionml.tensor import Tensor, matmul
        
        # Large enough to trigger tri-compute
        a = Tensor(np.random.randn(2048, 2048).astype(np.float32))
        b = Tensor(np.random.randn(2048, 2048).astype(np.float32))
        
        result = matmul(a, b)
        expected = np.matmul(a.numpy, b.numpy)
        
        rel_err = np.max(np.abs(result.numpy - expected)) / (np.max(np.abs(expected)) + 1e-8)
        assert rel_err < 0.02
    
    def test_tensor_matmul_small_is_cpu(self):
        """Small matrices should still use CPU path."""
        from fusionml.tensor import Tensor, matmul
        
        a = Tensor(np.random.randn(64, 64).astype(np.float32))
        b = Tensor(np.random.randn(64, 64).astype(np.float32))
        
        result = matmul(a, b)
        expected = np.matmul(a.numpy, b.numpy)
        
        np.testing.assert_allclose(result.numpy, expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
