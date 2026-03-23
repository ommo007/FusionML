#!/bin/bash
# ============================================================
# FusionML Benchmark Runner
# Easy setup and execution - Runs ALL benchmarks
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

# Run comprehensive generic benchmark (creates device folder with all results)
echo "------------------------------------------------------------"
echo ">> Running Core Generic Benchmarks (MLP/MatMul)..."
python run_benchmark.py

# Run NeurIPS Scaling Benchmarks (ResNet / BERT throughput)
echo ""
echo "------------------------------------------------------------"
echo ">> Running NeurIPS Device Scaling Benchmarks (ResNet-50 & BERT-base)..."
python throughput_benchmark.py

# Run NeurIPS Ablation Benchmarks (Makespan Ratios)
echo ""
echo "------------------------------------------------------------"
echo ">> Running NeurIPS Scheduler Ablation Benchmarks..."
python ablation_benchmark.py

echo ""
echo "============================================================"
echo "To submit results: Create a PR with your device folder and benchmark JSONs!"
echo "============================================================"
