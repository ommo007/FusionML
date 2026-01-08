#!/bin/bash
# ============================================================
# FusionML Benchmark Runner
# Easy setup and execution script - Runs ALL benchmarks
# ============================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/python"
VENV_DIR="$PYTHON_DIR/.venv"

echo "============================================================"
echo "🔥 FusionML Benchmark Runner"
echo "============================================================"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "   Install from: https://python.org"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment if needed
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$PYTHON_DIR/requirements.txt"

# Install FusionML from parent directory
echo "📥 Installing FusionML..."
pip install --quiet -e "$SCRIPT_DIR/../python"

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

echo ""
echo "============================================================"
echo "🚀 Running ALL Benchmarks..."
echo "============================================================"

cd "$PYTHON_DIR"

# 1. Run main benchmark (MatMul + Training)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 1/4: Main Benchmark (MatMul + Training)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_benchmark.py

# 2. Run MatMul benchmark
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 2/4: Matrix Multiplication Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python benchmark_matmul.py

# 3. Run Training benchmark
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 3/4: Training Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python benchmark_training.py

# 4. Run MLX comparison (if MLX is available)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 4/4: FusionML vs MLX Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python -c "import mlx" 2>/dev/null; then
    python benchmark_vs_mlx.py
else
    echo "   ⚠️  MLX not installed, skipping comparison"
    echo "   Install with: pip install mlx"
fi

# Generate plots
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 Generating Comparison Plots..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python plot_results.py

echo ""
echo "============================================================"
echo "✅ ALL BENCHMARKS COMPLETE!"
echo "============================================================"
echo ""
echo "📁 Results saved in: $SCRIPT_DIR/results/"
echo ""
echo "   📊 benchmark_comparison.png  - Comparison chart"
echo "   📝 SUMMARY.md                - Summary table"
echo "   📄 *.json                    - Raw benchmark data"
echo ""
echo "To submit results: Create a PR with your JSON file!"
echo "============================================================"
