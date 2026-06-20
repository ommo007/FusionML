#!/bin/bash
# ==============================================================================
# 🔥 FusionML Unified Benchmark Runner
# ==============================================================================
# Run and manage all performance comparisons, scaling studies, ablation sweeps,
# and Swift-native parallel execution benchmarks from a single entry point.
# ==============================================================================

set -e

# ANSI Color Codes for Premium UI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print Premium Header
[ -t 1 ] && [ -n "$TERM" ] && clear || true
echo -e "${CYAN}${BOLD}"
echo " ╔═══════════════════════════════════════════════════════════════╗"
echo " ║                                                               ║"
echo " ║   ⚡ F u s i o n M L   U n i f i e d   B e n c h m a r k s    ║"
echo " ║                                                               ║"
echo " ╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Detect active directory structure
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$SCRIPT_DIR/python"
VENV_DIR="$PYTHON_DIR/.venv"

# Usage Helper
show_help() {
    echo -e "${BOLD}Usage:${NC}"
    echo "  ./run_benchmark.sh [options]"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${GREEN}--all${NC}               Run all Python benchmarks and Swift benchmarks (Default)"
    echo -e "  ${GREEN}--cross-framework${NC}   Run head-to-head decoder layers vs MLX & PyTorch (Inference/Training)"
    echo -e "  ${GREEN}--head-to-head${NC}      Run raw matrix multiplication routing vs CPU/GPU/ANE"
    echo -e "  ${GREEN}--ablation${NC}          Run scheduler ablation sweeps (MakeSpan routing accuracy)"
    echo -e "  ${GREEN}--throughput${NC}        Run NeurIPS device scaling throughput (ResNet-50 & BERT-base)"
    echo -e "  ${GREEN}--swift${NC}             Run Swift-native zero-copy parallel scheduler benchmarks"
    echo -e "  ${GREEN}--collate${NC}           Collate result JSONs and auto-format LaTeX publication tables"
    echo -e "  ${GREEN}--help${NC}              Show this help menu"
    echo ""
}

# Flag initialization
RUN_ALL=true
RUN_CROSS=false
RUN_H2H=false
RUN_ABLATION=false
RUN_THROUGHPUT=false
RUN_SWIFT=false
RUN_COLLATE=false

# Parse flags
if [ $# -gt 0 ]; then
    RUN_ALL=false
    while [ $# -gt 0 ]; do
        case "$1" in
            --all)
                RUN_ALL=true
                ;;
            --cross-framework)
                RUN_CROSS=true
                ;;
            --head-to-head)
                RUN_H2H=true
                ;;
            --ablation)
                RUN_ABLATION=true
                ;;
            --throughput)
                RUN_THROUGHPUT=true
                ;;
            --swift)
                RUN_SWIFT=true
                ;;
            --collate)
                RUN_COLLATE=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                show_help
                exit 1
                ;;
        esac
        shift
    done
fi

# Detect hardware metadata
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
RAM_GB=$((RAM_BYTES / 1073741824))
OS_VER=$(sw_vers -productVersion 2>/dev/null || echo "Unknown")

echo -e "${BOLD}${CYAN}Hardware Profile Detected:${NC}"
echo "  • Chip:          $CHIP"
echo "  • Memory:        ${RAM_GB} GB Unified RAM"
echo "  • Operating Sys: macOS $OS_VER"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Setup Python Virtual Environment
# ──────────────────────────────────────────────────────────────────────────────
setup_environment() {
    echo -e "${BOLD}${YELLOW}⚙️  Setting up Python Virtual Environment...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Error: Python 3 is required but not installed.${NC}"
        exit 1
    fi
    
    # Create virtual environment if missing
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "  • Creating venv at: ${CYAN}benchmarks/python/.venv${NC}"
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Install/upgrade pip and requirements
    echo -e "  • Installing dependencies from ${CYAN}requirements.txt${NC}..."
    pip install --quiet --upgrade pip
    pip install --quiet -r "$PYTHON_DIR/requirements.txt"
    
    # Install FusionML local package in editable mode
    echo -e "  • Registering local ${CYAN}fusionml${NC} package..."
    pip install --quiet -e "$ROOT_DIR/python"
    
    echo -e "${GREEN}✓ Environment setup complete!${NC}\n"
}

# ──────────────────────────────────────────────────────────────────────────────
# Executions
# ──────────────────────────────────────────────────────────────────────────────

run_cross_framework() {
    echo -e "${BOLD}${MAGENTA}▶ Running Cross-Framework Model Comparison (Llama-3 & GPT-2 Blocks)...${NC}"
    cd "$PYTHON_DIR"
    python model_comparison.py --framework all
}

run_head_to_head() {
    echo -e "${BOLD}${MAGENTA}▶ Running Matrix Multiplication Head-to-Head (CPU/GPU/ANE/FusionML)...${NC}"
    cd "$PYTHON_DIR"
    python head_to_head.py
}

run_ablation() {
    echo -e "${BOLD}${MAGENTA}▶ Running Scheduler Ablation Study (Makespan Calibration accuracy)...${NC}"
    cd "$PYTHON_DIR"
    python ablation_benchmark.py
}

run_throughput() {
    echo -e "${BOLD}${MAGENTA}▶ Running Throughput Scaling Benchmarks (ResNet-50 & BERT-base)...${NC}"
    cd "$PYTHON_DIR"
    python throughput_benchmark.py
}

run_swift() {
    echo -e "${BOLD}${MAGENTA}▶ Running Swift-Native Zero-Copy Parallel Benchmarks...${NC}"
    cd "$ROOT_DIR"
    # Build package in release configuration
    swift build -c release
    # Run Benchmark executable
    swift run -c release BenchmarkExample
}

run_collate() {
    echo -e "${BOLD}${MAGENTA}▶ Collating JSON results & generating LaTeX tables...${NC}"
    cd "$PYTHON_DIR"
    if [ -f "collate_results.py" ]; then
        python collate_results.py
    else
        echo -e "${RED}❌ Error: collate_results.py script not found.${NC}"
    fi
}

# Main Logic execution path
setup_environment

if [ "$RUN_ALL" = true ]; then
    run_cross_framework
    run_head_to_head
    run_ablation
    run_throughput
    run_swift
    run_collate
else
    [ "$RUN_CROSS" = true ] && run_cross_framework
    [ "$RUN_H2H" = true ] && run_head_to_head
    [ "$RUN_ABLATION" = true ] && run_ablation
    [ "$RUN_THROUGHPUT" = true ] && run_throughput
    [ "$RUN_SWIFT" = true ] && run_swift
    [ "$RUN_COLLATE" = true ] && run_collate
fi

echo -e "\n${BOLD}${GREEN}🎉 All requested benchmarks completed successfully!${NC}"
echo -e "JSON results saved under: ${CYAN}benchmarks/results/${NC}"
