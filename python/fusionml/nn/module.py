"""
Neural Network Module - Base class and layers (GPU-compatible)
"""

from typing import List, Iterator, Tuple
import numpy as np
from ..tensor import Tensor, relu as tensor_relu, sigmoid as tensor_sigmoid

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class Module:
    """Base class for all neural network modules"""
    
    def __init__(self):
        self._modules: dict = {}
        self._parameters: dict = {}
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameters(self) -> Iterator[Tensor]:
        """Get all parameters"""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Get named parameters"""
        for name, param in self._parameters.items():
            yield name, param
        for mod_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                yield f"{mod_name}.{param_name}", param
    
    def train(self):
        """Set training mode"""
        self.training = True
        for module in self._modules.values():
            module.train()
    
    def eval(self):
        """Set evaluation mode"""
        self.training = False
        for module in self._modules.values():
            module.eval()
    
    def zero_grad(self):
        """Zero all gradients"""
        for param in self.parameters():
            param.grad = None


class Linear(Module):
    """Linear layer: y = x @ W + b"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier initialization
        k = np.sqrt(1 / in_features)
        self.weight = Tensor(
            np.random.uniform(-k, k, (in_features, out_features)).astype(np.float32),
            requires_grad=True
        )
        self._parameters['weight'] = self.weight
        
        if bias:
            self.bias = Tensor(
                np.zeros((1, out_features), dtype=np.float32),
                requires_grad=True
            )
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    """ReLU activation"""
    
    def forward(self, x: Tensor) -> Tensor:
        return tensor_relu(x)


class GELU(Module):
    """GELU activation"""
    
    def forward(self, x: Tensor) -> Tensor:
        if x._on_gpu and HAS_MLX:
            # Use MLX GELU
            result = Tensor.__new__(Tensor)
            # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            x_mlx = x._mlx
            result._mlx = 0.5 * x_mlx * (1 + mx.tanh(np.sqrt(2/np.pi) * (x_mlx + 0.044715 * x_mlx ** 3)))
            result._np = None
            result._on_gpu = True
            result.requires_grad = x.requires_grad
            result._ctx = None
            result.grad = None
            return result
        else:
            data = x.numpy
            result_data = 0.5 * data * (1 + np.tanh(np.sqrt(2/np.pi) * (data + 0.044715 * data**3)))
            return Tensor(result_data.astype(np.float32), requires_grad=x.requires_grad)


class Sigmoid(Module):
    """Sigmoid activation"""
    
    def forward(self, x: Tensor) -> Tensor:
        return tensor_sigmoid(x)


class Tanh(Module):
    """Tanh activation"""
    
    def forward(self, x: Tensor) -> Tensor:
        if x._on_gpu and HAS_MLX:
            result = Tensor.__new__(Tensor)
            result._mlx = mx.tanh(x._mlx)
            result._np = None
            result._on_gpu = True
            result.requires_grad = x.requires_grad
            result._ctx = None
            result.grad = None
            return result
        else:
            return Tensor(np.tanh(x.numpy), requires_grad=x.requires_grad)


class Dropout(Module):
    """Dropout layer"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        
        if x._on_gpu and HAS_MLX:
            mask = mx.random.uniform(shape=x.shape) > self.p
            result = Tensor.__new__(Tensor)
            result._mlx = x._mlx * mask.astype(mx.float32) / (1 - self.p)
            result._np = None
            result._on_gpu = True
            result.requires_grad = x.requires_grad
            result._ctx = None
            result.grad = None
            return result
        else:
            mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
            return Tensor(x.numpy * mask / (1 - self.p))


class Sequential(Module):
    """Sequential container"""
    
    def __init__(self, layers: List[Module]):
        super().__init__()
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x
