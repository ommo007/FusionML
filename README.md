# FusionML

**High-Performance Machine Learning Framework for Apple Silicon**

FusionML delivers PyTorch-like ease of use with a unique advantage: **intelligent parallel execution across GPU, CPU, and Neural Engine** – achieving up to 31% faster matrix operations through hardware fusion.

## Features

- 🔥 **PyTorch-Style API** - Familiar `Fusion.nn`, `Fusion.optim`, `Fusion.autograd`
- ⚡ **Intelligent Routing** - Automatic work distribution across GPU + CPU
- 🧠 **Full Autograd** - Computation graph with backpropagation
- 🎯 **Apple Silicon Optimized** - Metal, Accelerate, and Neural Engine
- 📦 **Zero Dependencies** - Pure Swift with system frameworks only

## Installation

### Swift Package Manager (Local)

```swift
dependencies: [
    .package(path: "../FusionML")
]
```

## Quick Start

```swift
import FusionML

// Initialize
Fusion.initialize()

// Create tensors
let x = try Fusion.rand([32, 784])
let y = try Fusion.randint(0, 10, [32])

// Build model
let model = Fusion.nn.sequential(
    Fusion.nn.linear(784, 256),
    Fusion.nn.relu(),
    Fusion.nn.linear(256, 10)
)

// Optimizer
let optimizer = Fusion.optim.adam(model.parameters(), lr: 0.001)

// Training loop
for epoch in 0..<10 {
    optimizer.zeroGrad()
    
    let output = try model.forward(GradTensor(x, requiresGrad: false))
    let loss = try Fusion.nn.functional.crossEntropy(output, y)
    
    try Fusion.autograd.backward(loss)
    try optimizer.step()
    
    print("Epoch \(epoch): Loss = \(loss.data.toArray()[0])")
}
```

## API Reference

### Neural Network (`Fusion.nn`)
```swift
Fusion.nn.linear(inFeatures, outFeatures)
Fusion.nn.relu()
Fusion.nn.gelu()
Fusion.nn.dropout(0.5)
Fusion.nn.layerNorm([hiddenSize])
Fusion.nn.sequential(layer1, layer2, ...)
```

### Functional (`Fusion.nn.functional`)
```swift
Fusion.nn.functional.relu(tensor)
Fusion.nn.functional.softmax(tensor)
Fusion.nn.functional.crossEntropy(predictions, targets)
Fusion.nn.functional.mse(predictions, targets)
```

### Optimizers (`Fusion.optim`)
```swift
Fusion.optim.sgd(parameters, lr: 0.01, momentum: 0.9)
Fusion.optim.adam(parameters, lr: 0.001)
Fusion.optim.adamw(parameters, lr: 0.001, weightDecay: 0.01)
```

### Hardware Backends
```swift
Fusion.linalg.matmul(a, b)  // Intelligent routing (fastest!)
Fusion.cpu.matmul(a, b)      // Force CPU
Fusion.gpu.matmul(a, b)      // Force GPU
```

## Performance

| Operation | CPU | GPU | Fusion (smart) | Speedup / Latency Reduction |
|-----------|-----|-----|----------------|-----------------------------|
| MatMul 1024² | 1.86 ms | 4.52 ms | 1.42 ms | +23.7% (vs CPU) |
| MatMul 2048² | 20.15 ms | 12.41 ms | 8.14 ms | +34.4% (vs GPU) |
| MatMul 4096² | 163.33 ms | 93.36 ms | 52.12 ms | +44.2% (vs GPU via ANE) |

## How It Works

FusionML's **IntelligentRouter** analyzes each operation and distributes work:

```
Traditional:  GPU ────────────────→ Result

FusionML:     GPU (68%) ────┬─────→ Result (faster!)
              CPU (32%) ────┘
```

The split ratio is calibrated per-device for optimal throughput.

## Requirements

- macOS 12.0+
- Apple Silicon (M series)
- Swift 5.9+

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
