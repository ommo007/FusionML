# FusionML API Reference

## Python API

### Tensor
```python
import fusionml as fml

x = fml.rand(2, 3)          # Random [0, 1)
x = fml.randn(2, 3)         # Random normal
x = fml.zeros(2, 3)         # Zeros
x = fml.ones(2, 3)          # Ones
x = fml.Tensor([1, 2, 3])   # From list
```

### Neural Network
```python
# Layers
fml.nn.Linear(in, out)
fml.nn.ReLU()
fml.nn.GELU()
fml.nn.Sigmoid()
fml.nn.Dropout(0.5)
fml.nn.Sequential([...])

# Functional
fml.nn.functional.relu(x)
fml.nn.functional.softmax(x)
fml.nn.functional.cross_entropy(pred, target)
fml.nn.functional.mse_loss(pred, target)
```

### Optimizers
```python
fml.optim.SGD(params, lr=0.01, momentum=0.9)
fml.optim.Adam(params, lr=0.001, betas=(0.9, 0.999))
```

---

## Swift API

### Tensor
```swift
let x = try Fusion.rand([2, 3])
let y = try Fusion.zeros([2, 3])
```

### Neural Network
```swift
let model = Fusion.nn.sequential(
    try Fusion.nn.linear(784, 256),
    Fusion.nn.relu(),
    try Fusion.nn.linear(256, 10)
)
```

### Optimizers
```swift
let opt = Fusion.optim.adam(model.parameters(), lr: 0.001)
```

### Backend Access
```swift
Fusion.linalg.matmul(a, b)  // Intelligent routing
Fusion.cpu.matmul(a, b)      // Force CPU
Fusion.gpu.matmul(a, b)      // Force GPU
```
