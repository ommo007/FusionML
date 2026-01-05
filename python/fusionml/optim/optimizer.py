"""
Optimizers - SGD, Adam
"""

from typing import Iterator
import numpy as np
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
        self.velocities = [np.zeros_like(p._data) for p in self.parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad._data
            
            # Weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param._data
            
            # Momentum
            if self.momentum != 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                grad = self.velocities[i]
            
            # Update
            param._data -= self.lr * grad


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # Moment estimates
        self.m = [np.zeros_like(p._data) for p in self.parameters]
        self.v = [np.zeros_like(p._data) for p in self.parameters]
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad._data
            
            # Weight decay (AdamW style)
            if self.weight_decay != 0:
                param._data -= self.lr * self.weight_decay * param._data
            
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update
            param._data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
