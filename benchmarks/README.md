# FusionML Benchmarks

This branch contains benchmark code and results for FusionML performance evaluation.

## 🎯 Goal

We encourage the community to benchmark FusionML on their hardware and submit results!

## 📊 Current Results

| Device | MatMul 1024² | MatMul 2048² | Training (MLP) |
|--------|--------------|--------------|----------------|
| Apple M1 | 0.96ms | 3.2ms | 64ms/epoch |
| Apple M1 Pro | TBD | TBD | TBD |
| Apple M1 Max | TBD | TBD | TBD |
| Apple M2 | TBD | TBD | TBD |
| Apple M3 | TBD | TBD | TBD |

## 🚀 Run Benchmarks

### Python

```bash
cd benchmarks/python
pip install -r requirements.txt
python benchmark_matmul.py
python benchmark_training.py
python benchmark_vs_pytorch.py  # Comparison
```

### Swift

```bash
cd FusionML
swift run BenchmarkExample
```

## 📝 Submit Your Results

1. Fork this repo
2. Run benchmarks on your device
3. Add results to `results/<your-device>.json`
4. Create a PR with:
   - Device specs (chip, RAM, macOS version)
   - Benchmark output
   - Any observations

### Result Format

```json
{
  "device": "Apple M1 Pro",
  "ram_gb": 16,
  "macos_version": "14.2",
  "date": "2026-01-05",
  "benchmarks": {
    "matmul_1024": {"fusionml_ms": 0.8, "pytorch_ms": 1.2},
    "matmul_2048": {"fusionml_ms": 2.9, "pytorch_ms": 4.1},
    "training_mlp": {"fusionml_ms_per_epoch": 58}
  }
}
```

## 🏆 Hall of Fame

Contributors who submitted verified benchmarks:
- Your name here!

## 📈 Why Benchmark?

FusionML's core innovation is **GPU+CPU parallel execution**. Benchmarks help us:
- Validate performance across different Apple Silicon chips
- Identify optimization opportunities
- Compare against PyTorch, JAX, MLX
