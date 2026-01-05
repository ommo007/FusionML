# Getting Started with FusionML

## Installation

### Python
```bash
pip install fusionml
```

### Swift
Add to `Package.swift`:
```swift
dependencies: [
    .package(url: "https://github.com/yourname/FusionML.git", from: "0.1.0")
]
```

## Quick Start

### Python
```python
import fusionml as fml

fml.init()

# Create tensors
x = fml.rand(32, 784)

# Build model
model = fml.nn.Sequential([
    fml.nn.Linear(784, 256),
    fml.nn.ReLU(),
    fml.nn.Linear(256, 10)
])

# Train
optimizer = fml.optim.Adam(model.parameters(), lr=0.001)
output = model(x)
loss = fml.nn.functional.cross_entropy(output, target)
loss.backward()
optimizer.step()
```

### Swift
```swift
import FusionML

Fusion.initialize()

let model = Fusion.nn.sequential(
    try Fusion.nn.linear(784, 256),
    Fusion.nn.relu(),
    try Fusion.nn.linear(256, 10)
)

let optimizer = Fusion.optim.adam(model.parameters(), lr: 0.001)
```

## Next Steps

- See [API.md](API.md) for full API reference
- See [architecture.md](architecture.md) for design details
