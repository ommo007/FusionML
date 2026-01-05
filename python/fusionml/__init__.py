"""
FusionML - High-Performance ML Framework for Apple Silicon
GPU + CPU parallel execution for optimal performance
"""

__version__ = "0.1.0"

from .tensor import Tensor, zeros, ones, rand, randn, eye
from . import nn
from . import optim
from . import autograd
from . import functional as F
from ._metal import device_info

def init():
    """Initialize FusionML backend"""
    info = device_info()
    print(f"🔥 FusionML {__version__} initialized")
    print(f"   Device: {info['name']}")
    print(f"   GPU: ✓ | CPU: ✓")
