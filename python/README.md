# FusionML - Python

High-Performance ML Framework for Apple Silicon with GPU+CPU parallel execution.

## Installation

```bash
pip install fusionml

# With Metal GPU support
pip install fusionml[metal]
```

## Quick Start

```python
import fusionml as fml

# Initialize
fml.init()

# Create tensors
x = fml.rand(32, 784)
y = fml.Tensor([0, 1, 2, 3])  # Labels

# Build model
model = fml.nn.Sequential([
    fml.nn.Linear(784, 256),
    fml.nn.ReLU(),
    fml.nn.Linear(256, 10)
])

# Optimizer
optimizer = fml.optim.Adam(model.parameters(), lr=0.001)

# Training step
output = model(x)
loss = fml.nn.functional.cross_entropy(output, y)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
```

## Features

- 🔥 PyTorch-like API
- ⚡ GPU+CPU parallel execution
- 🧠 Full autograd support
- 🎯 Apple Silicon optimized

## API

### Tensors
```python
fml.rand(2, 3)      # Random uniform
fml.randn(2, 3)     # Random normal
fml.zeros(2, 3)     # Zeros
fml.ones(2, 3)      # Ones
fml.eye(3)          # Identity
```

### Layers
```python
fml.nn.Linear(in, out)
fml.nn.ReLU()
fml.nn.GELU()
fml.nn.Dropout(0.5)
fml.nn.Sequential([...])
```

### Optimizers
```python
fml.optim.SGD(params, lr=0.01, momentum=0.9)
fml.optim.Adam(params, lr=0.001)
```

### Functional
```python
fml.nn.functional.relu(x)
fml.nn.functional.softmax(x)
fml.nn.functional.cross_entropy(pred, target)
fml.nn.functional.mse_loss(pred, target)
```

## License

MIT
