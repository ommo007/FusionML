# FusionML Benchmarks

Community-driven benchmarks for FusionML across Apple Silicon devices.

## 🚀 Quick Start

Just run the shell script - it handles everything!

```bash
cd benchmarks
./run_benchmarks.sh
```

This will:
1. ✅ Create a virtual environment
2. ✅ Install all dependencies (numpy, mlx, torch)
3. ✅ Run all 4 benchmarks
4. ✅ Save results in a device-specific folder

## 📊 Results Structure

Results are organized by device:

```
results/
├── M1_8GB_8cores_926Gi/
│   ├── matmul.json
│   ├── training.json
│   ├── vs_mlx.json
│   └── vs_pytorch.json
├── M2_Pro_16GB_12cores_500Gi/
│   ├── matmul.json
│   └── ...
└── M4_24GB_10cores_1Ti/
    └── ...
```

Each device folder contains 4 JSON files:
- **matmul.json** - Matrix multiplication (256 to 4096)
- **training.json** - MLP training benchmark
- **vs_mlx.json** - FusionML vs MLX comparison
- **vs_pytorch.json** - FusionML vs PyTorch comparison

## 📝 Submit Your Results

1. Fork this repository
2. Run `./run_benchmarks.sh`
3. Create a PR with your device folder from `results/`

## 🔧 Manual Setup

```bash
cd benchmarks/python

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ../../python

# Run all benchmarks
python run_benchmark.py
```

## 🏆 Hall of Fame

Contributors who submitted verified benchmarks:

- *Your name here!*

---

*FusionML - Faster than MLX on Apple Silicon* 🚀
