"""
Neural Network Module - Base class and layers
"""

from typing import List, Iterator, Tuple
import numpy as np
from ..tensor import Tensor


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
            param.zero_grad()


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
            # Broadcast bias
            out = out + self.bias
        return out


class ReLU(Module):
    """ReLU activation"""
    
    def forward(self, x: Tensor) -> Tensor:
        result = Tensor(np.maximum(x._data, 0), requires_grad=x.requires_grad)
        if x.requires_grad:
            result._ctx = ('relu', x)
        return result


class GELU(Module):
    """GELU activation"""
    
    def forward(self, x: Tensor) -> Tensor:
        # GELU approximation
        data = x._data
        result = 0.5 * data * (1 + np.tanh(np.sqrt(2/np.pi) * (data + 0.044715 * data**3)))
        return Tensor(result.astype(np.float32), requires_grad=x.requires_grad)


class Sigmoid(Module):
    """Sigmoid activation"""
    
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(1 / (1 + np.exp(-x._data)), requires_grad=x.requires_grad)


class Tanh(Module):
    """Tanh activation"""
    
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.tanh(x._data), requires_grad=x.requires_grad)


class Dropout(Module):
    """Dropout layer"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        return Tensor(x._data * mask / (1 - self.p))


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
