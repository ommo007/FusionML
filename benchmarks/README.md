# FusionML Benchmarks

Community-driven benchmarks for FusionML across Apple Silicon devices.

## 🚀 Quick Start (Easiest Way)

Just run the shell script - it handles everything!

```bash
cd benchmarks
./run_benchmarks.sh
```

This will:
1. ✅ Create a virtual environment
2. ✅ Install all dependencies
3. ✅ Run benchmarks
4. ✅ Save results with your system specs
5. ✅ Generate comparison plots

## 📊 Results

Results are automatically saved in the `results/` folder:

```
results/
├── Apple_M1_8GB_8cores_20260108.json    # Your benchmark data
├── Apple_M2_Pro_16GB_12cores_20260108.json
├── benchmark_comparison.png              # Comparison chart
├── benchmark_comparison.svg              # High-quality SVG
└── SUMMARY.md                            # Markdown summary table
```

### Current Results

| Device | MatMul 1024² | MatMul 2048² | Training |
|--------|--------------|--------------|----------|
| M1 (8GB) | 1.43ms | 10.1ms | 0.67ms |
| M1 Pro | Needs data! | | |
| M2 | Needs data! | | |
| M3 | Needs data! | | |

**Help us fill this table!** Submit your benchmarks.

## 🔧 Manual Setup

If you prefer to run manually:

```bash
cd benchmarks/python

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ../../python

# Run benchmarks
python run_benchmark.py

# Generate plots
python plot_results.py

# Run comparison with MLX
pip install mlx
python benchmark_vs_mlx.py
```

## 📝 Submit Your Results

1. Fork this repository
2. Run `./run_benchmarks.sh`
3. Find your result file in `results/`
4. Create a PR with your JSON file

### Result File Format

```json
{
  "system_info": {
    "cpu_brand": "Apple M2 Pro",
    "ram_gb": 16,
    "cpu_cores": 12,
    "gpu_name": "Apple M2 Pro"
  },
  "matmul": {
    "256": {"mean_ms": 0.03, "min_ms": 0.02, "max_ms": 0.04},
    "1024": {"mean_ms": 1.2, "min_ms": 1.1, "max_ms": 1.3}
  },
  "training": {
    "mean_ms": 0.8,
    "throughput_samples_per_sec": 40000
  }
}
```

## 📈 Benchmark Scripts

| Script | Description |
|--------|-------------|
| `run_benchmark.py` | Full benchmark suite (recommended) |
| `benchmark_matmul.py` | Matrix multiplication only |
| `benchmark_training.py` | Training performance only |
| `benchmark_vs_mlx.py` | Compare against MLX |
| `benchmark_vs_pytorch.py` | Compare against PyTorch |
| `plot_results.py` | Generate comparison plots |

## 🏆 Hall of Fame

Contributors who submitted verified benchmarks:

- *Your name here!*

---

*FusionML - Faster than MLX on Apple Silicon* 🚀
