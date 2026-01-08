"""
Optimizers - SGD, Adam (GPU-compatible)
"""

from typing import Iterator
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from ..tensor import Tensor


class Optimizer:
    """Base optimizer class"""
    
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01):
        self.parameters = list(parameters)
        self.lr = lr
    
    def zero_grad(self):
        """Zero all gradients"""
        for param in self.parameters:
            param.grad = None
    
    def step(self):
        """Update parameters"""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum"""
    
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01, 
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize velocities lazily
        self.velocities = None
    
    def _init_state(self):
        if self.velocities is None:
            self.velocities = []
            for p in self.parameters:
                if p._on_gpu:
                    self.velocities.append(mx.zeros(p.shape))
                else:
                    self.velocities.append(np.zeros(p.shape, dtype=np.float32))
    
    def step(self):
        self._init_state()
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            if param._on_gpu:
                grad = param.grad._mlx if param.grad._on_gpu else mx.array(param.grad.numpy)
                
                # Weight decay
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param._mlx
                
                # Momentum
                if self.momentum != 0:
                    self.velocities[i] = self.momentum * self.velocities[i] + grad
                    grad = self.velocities[i]
                
                # Update in-place
                param._mlx = param._mlx - self.lr * grad
            else:
                grad = param.grad.numpy
                
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param._np
                
                if self.momentum != 0:
                    self.velocities[i] = self.momentum * self.velocities[i] + grad
                    grad = self.velocities[i]
                
                param._np = param._np - self.lr * grad


class Adam(Optimizer):
    """Adam optimizer (GPU-compatible)"""
    
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = None
        self.v = None
    
    def _init_state(self):
        if self.m is None:
            self.m = []
            self.v = []
            for p in self.parameters:
                if p._on_gpu:
                    self.m.append(mx.zeros(p.shape))
                    self.v.append(mx.zeros(p.shape))
                else:
                    self.m.append(np.zeros(p.shape, dtype=np.float32))
                    self.v.append(np.zeros(p.shape, dtype=np.float32))
    
    def step(self):
        self._init_state()
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            if param._on_gpu and HAS_MLX:
                grad = param.grad._mlx if param.grad._on_gpu else mx.array(param.grad.numpy)
                
                # Weight decay (AdamW style)
                if self.weight_decay != 0:
                    param._mlx = param._mlx - self.lr * self.weight_decay * param._mlx
                
                # Update moments
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # Update
                param._mlx = param._mlx - self.lr * m_hat / (mx.sqrt(v_hat) + self.eps)
            else:
                grad = param.grad.numpy
                
                if self.weight_decay != 0:
                    param._np = param._np - self.lr * self.weight_decay * param._np
                
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                param._np = param._np - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
