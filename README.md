# FusionML

[![CI](https://github.com/ommo007/FusionML/actions/workflows/ci.yml/badge.svg)](https://github.com/ommo007/FusionML/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fusionml?color=blue)](https://pypi.org/project/fusionml/)
[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3-black.svg)](https://www.apple.com/mac/)

**High-Performance Machine Learning Framework for Apple Silicon**

FusionML delivers PyTorch-like ease of use with a unique advantage: **intelligent parallel execution across GPU, CPU, and Neural Engine** – achieving up to 33% faster matrix operations through hardware fusion.

## ✨ Features

- 🔥 **PyTorch-Style API** - Familiar `Fusion.nn`, `Fusion.optim`, `Fusion.autograd`
- ⚡ **Intelligent Routing** - Automatic work distribution across GPU + CPU in parallel
- 🧠 **Full Autograd** - Computation graph with backpropagation
- 🎯 **Apple Silicon Optimized** - Metal, Accelerate, and Neural Engine
- 📦 **Zero Dependencies** - Pure Swift with system frameworks only
- 🐍 **Python Bindings** - Use from Python with `pip install fusionml`

## 📦 Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/ommo007/FusionML.git", from: "0.1.0")
]
```

### Python (pip)

```bash
pip install fusionml
```

## 🚀 Quick Start

### Swift

```swift
import FusionML

Fusion.initialize()

// Build model
let model = Fusion.nn.sequential(
    try Fusion.nn.linear(784, 256),
    Fusion.nn.relu(),
    try Fusion.nn.linear(256, 10)
)

// Train
let optimizer = Fusion.optim.adam(model.parameters(), lr: 0.001)
let output = try model.forward(GradTensor(x, requiresGrad: true))
let loss = try Fusion.nn.functional.crossEntropy(output, y)

try Fusion.autograd.backward(loss)
try optimizer.step()
```

### Python

```python
import fusionml as fml

fml.init()

# Build model
model = fml.nn.Sequential([
    fml.nn.Linear(784, 256),
    fml.nn.ReLU(),
    fml.nn.Linear(256, 10)
])

# Train
optimizer = fml.optim.Adam(model.parameters(), lr=0.001)
output = model(x)
loss = fml.nn.functional.cross_entropy(output, y)

loss.backward()
optimizer.step()
```

## 📊 Performance

Tested on Apple M1:

| Operation | CPU | GPU | FusionML | Speedup |
|-----------|-----|-----|----------|---------|
| MatMul 1024² | 6ms | 4ms | **3ms** | +33% |
| MatMul 2048² | 18ms | 12ms | **9ms** | +33% |
| MatMul 4096² | 85ms | 52ms | **38ms** | +27% |

> 📈 See [benchmarks branch](https://github.com/ommo007/FusionML/tree/benchmarks) for detailed results and contribution guide.

## 🔧 How It Works

FusionML's **IntelligentRouter** analyzes each operation and distributes work across hardware in parallel:

```
Traditional:  GPU ─────────────────────→ Result

FusionML:     GPU (68%) ──────┬────────→ Result (faster!)
              CPU (32%) ──────┘
```

The split ratio is calibrated dynamically per-device using unified memory for zero-copy data sharing.

## 📚 API Reference

### Neural Network (`Fusion.nn`)
```swift
Fusion.nn.linear(in, out)      // Linear layer
Fusion.nn.relu()               // ReLU activation
Fusion.nn.gelu()               // GELU activation
Fusion.nn.dropout(0.5)         // Dropout
Fusion.nn.sequential(...)      // Container
```

### Optimizers (`Fusion.optim`)
```swift
Fusion.optim.sgd(params, lr: 0.01, momentum: 0.9)
Fusion.optim.adam(params, lr: 0.001)
Fusion.optim.adamw(params, lr: 0.001, weightDecay: 0.01)
```

### Backend Access
```swift
Fusion.linalg.matmul(a, b)  // Smart routing (recommended)
Fusion.cpu.matmul(a, b)      // Force CPU
Fusion.gpu.matmul(a, b)      // Force GPU
```

## 🗂️ Project Structure

```
FusionML/
├── sources/FusionML/    # Swift library
├── python/fusionml/     # Python bindings (for ML researchers)
├── examples/            # Swift examples
├── tests/               # Test suites
└── docs/                # Documentation
```

> **Note**: The Python code provides bindings for ML researchers who prefer Python. The core library is pure Swift.

## 📋 Requirements

- macOS 12.0+
- Apple Silicon (M1/M2/M3/M4)
- Swift 5.9+ (for Swift)
- Python 3.8+ (for Python)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Want to benchmark on your device?** Check the [benchmarks branch](https://github.com/ommo007/FusionML/tree/benchmarks).

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

Built with ❤️ for the Apple Silicon ML community.
