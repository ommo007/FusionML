#!/usr/bin/env python3
"""
Plot benchmark results from all machines
Generates comparison charts
"""

import json
from pathlib import Path
import sys

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_all_results():
    """Load all benchmark results from results folder"""
    results = []
    
    for filepath in RESULTS_DIR.glob("*.json"):
        if filepath.name.startswith("plot_"):
            continue  # Skip plot data files
        try:
            with open(filepath) as f:
                data = json.load(f)
                data["_filename"] = filepath.stem
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return results


def get_device_label(result):
    """Get a short label for the device"""
    info = result.get("system_info", {})
    
    # Try to extract chip name
    cpu = info.get("cpu_brand", info.get("processor", "Unknown"))
    
    # Simplify Apple Silicon names
    if "Apple" in cpu:
        if "M1" in cpu:
            chip = "M1"
            if "Pro" in cpu:
                chip = "M1 Pro"
            elif "Max" in cpu:
                chip = "M1 Max"
            elif "Ultra" in cpu:
                chip = "M1 Ultra"
        elif "M2" in cpu:
            chip = "M2"
            if "Pro" in cpu:
                chip = "M2 Pro"
            elif "Max" in cpu:
                chip = "M2 Max"
            elif "Ultra" in cpu:
                chip = "M2 Ultra"
        elif "M3" in cpu:
            chip = "M3"
            if "Pro" in cpu:
                chip = "M3 Pro"
            elif "Max" in cpu:
                chip = "M3 Max"
        else:
            chip = cpu.split()[-1] if cpu else "Unknown"
    else:
        chip = cpu[:20] if cpu else "Unknown"
    
    ram = info.get("ram_gb", "?")
    return f"{chip} ({ram}GB)"


def generate_plots():
    """Generate comparison plots from all results"""
    if not HAS_MATPLOTLIB:
        print("Cannot generate plots without matplotlib")
        return
    
    results = load_all_results()
    
    if not results:
        print("No benchmark results found in", RESULTS_DIR)
        return
    
    print(f"Found {len(results)} benchmark result(s)")
    
    # Prepare data
    devices = []
    matmul_data = {256: [], 512: [], 1024: [], 2048: [], 4096: []}
    training_data = []
    
    for result in results:
        device = get_device_label(result)
        devices.append(device)
        
        # MatMul data
        matmul = result.get("matmul", {})
        for size in [256, 512, 1024, 2048, 4096]:
            if str(size) in matmul:
                matmul_data[size].append(matmul[str(size)].get("mean_ms", 0))
            else:
                matmul_data[size].append(0)
        
        # Training data
        training = result.get("training", {})
        training_data.append(training.get("mean_ms", 0))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FusionML Benchmark Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set2.colors
    
    # Plot 1: MatMul 1024x1024 comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(devices, matmul_data[1024], color=colors[:len(devices)])
    ax1.set_title('Matrix Multiplication (1024×1024)')
    ax1.set_ylabel('Time (ms) - Lower is Better')
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, matmul_data[1024]):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: MatMul 2048x2048 comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(devices, matmul_data[2048], color=colors[:len(devices)])
    ax2.set_title('Matrix Multiplication (2048×2048)')
    ax2.set_ylabel('Time (ms) - Lower is Better')
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, matmul_data[2048]):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Training comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(devices, training_data, color=colors[:len(devices)])
    ax3.set_title('Training (MLP 784→256→10, batch=32)')
    ax3.set_ylabel('Time per batch (ms) - Lower is Better')
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, training_data):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: All MatMul sizes comparison (line chart)
    ax4 = axes[1, 1]
    sizes = [256, 512, 1024, 2048, 4096]
    for i, device in enumerate(devices):
        times = [matmul_data[s][i] for s in sizes]
        ax4.plot(sizes, times, marker='o', label=device, color=colors[i % len(colors)])
    ax4.set_title('MatMul Scaling by Size')
    ax4.set_xlabel('Matrix Size')
    ax4.set_ylabel('Time (ms)')
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = RESULTS_DIR / "benchmark_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    # Also save as SVG for better quality
    svg_path = RESULTS_DIR / "benchmark_comparison.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved SVG to: {svg_path}")
    
    plt.close()
    
    # Generate summary table
    generate_summary_table(results)


def generate_summary_table(results):
    """Generate a markdown summary table"""
    
    lines = [
        "# Benchmark Results Summary\n",
        f"*Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "",
        "## Matrix Multiplication (ms)\n",
        "| Device | 256×256 | 512×512 | 1024×1024 | 2048×2048 | 4096×4096 |",
        "|--------|---------|---------|-----------|-----------|-----------|"
    ]
    
    for result in results:
        device = get_device_label(result)
        matmul = result.get("matmul", {})
        
        row = f"| {device} |"
        for size in [256, 512, 1024, 2048, 4096]:
            val = matmul.get(str(size), {}).get("mean_ms", "-")
            if isinstance(val, (int, float)):
                row += f" {val:.2f} |"
            else:
                row += f" {val} |"
        lines.append(row)
    
    lines.extend([
        "",
        "## Training Performance\n",
        "| Device | ms/batch | samples/sec |",
        "|--------|----------|-------------|"
    ])
    
    for result in results:
        device = get_device_label(result)
        training = result.get("training", {})
        ms = training.get("mean_ms", "-")
        throughput = training.get("throughput_samples_per_sec", "-")
        
        if isinstance(ms, (int, float)) and isinstance(throughput, (int, float)):
            lines.append(f"| {device} | {ms:.2f} | {throughput:.0f} |")
        else:
            lines.append(f"| {device} | {ms} | {throughput} |")
    
    # Write summary
    summary_path = RESULTS_DIR / "SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    generate_plots()
