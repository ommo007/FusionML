#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===============================================================${NC}"
echo -e "${GREEN}   FusionML Reproducibility Benchmark Suite (NeurIPS 2026)${NC}"
echo -e "${GREEN}===============================================================${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 could not be found${NC}"
    exit 1
fi

# Detect Hardware
echo "Checking System..."
sysctl -n machdep.cpu.brand_string || echo "Not on macOS or sysctl failed"

# Setup Virtual Environment (if not already active)
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_bench
    source venv_bench/bin/activate
else
    echo "Using active virtual environment: $VIRTUAL_ENV"
fi

# Install Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r benchmarks/python/requirements.txt
# Install FusionML in editable mode
pip install -e python/

# Create Results Directory
mkdir -p benchmarks/results/reproducibility
REPORT_FILE="benchmarks/results/reproducibility/report_$(date +%Y%m%d_%H%M%S).txt"

echo -e "\n${GREEN}Starting Benchmarks... Results will be saved to $REPORT_FILE${NC}"

{
    echo "FusionML Reproducibility Report"
    echo "Date: $(date)"
    echo "---------------------------------------------------------------"
    echo "SYSTEM SPECIFICATIONS:"
    echo "---------------------------------------------------------------"
    system_profiler SPHardwareDataType | grep -E "Model Name|Chip|Total Number of Cores|Memory"
    sw_vers
    echo "---------------------------------------------------------------"
    
    echo -e "\n1. Running Comparative Matmul Benchmark (End-to-End)..."
    python3 benchmarks/python/head_to_head.py --quiet --output benchmarks/results/reproducibility
    
    echo -e "\n2. Running Transformer Encoder Block Benchmark..."
    python3 benchmarks/python/benchmark_transformer.py

} | tee "$REPORT_FILE"

echo -e "\n${GREEN}Success! Report generated at $REPORT_FILE${NC}"
echo "Ready for NeurIPS submission!"
