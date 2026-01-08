#!/bin/bash
# ============================================================
# FusionML Benchmark Runner
# Easy setup and execution script
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
pip install --quiet numpy matplotlib

# Install FusionML from parent directory
echo "📥 Installing FusionML..."
pip install --quiet -e "$SCRIPT_DIR/../python"

# Check for MLX (optional)
if pip show mlx &> /dev/null; then
    echo "✓ MLX found"
else
    echo "ℹ️  MLX not installed (optional for comparison)"
    echo "   Install with: pip install mlx"
fi

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

echo ""
echo "============================================================"
echo "🚀 Running Benchmarks..."
echo "============================================================"

cd "$PYTHON_DIR"

# Run main benchmark
python run_benchmark.py

echo ""
echo "============================================================"
echo "✅ Benchmark Complete!"
echo "============================================================"
echo ""
echo "📁 Results saved in: $SCRIPT_DIR/results/"
echo "📊 View plots: $SCRIPT_DIR/results/benchmark_comparison.png"
echo "📝 Summary: $SCRIPT_DIR/results/SUMMARY.md"
echo ""
echo "To run again: ./run_benchmarks.sh"
echo "To compare with MLX: python benchmark_vs_mlx.py"
echo ""
