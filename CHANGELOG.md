# Changelog

All notable changes to FusionML will be documented in this file.

## [0.1.0] - 2026-01-05

### Added
- **Python Package** (`fusionml`)
  - Tensor class with numpy interop
  - Neural network layers: Linear, ReLU, GELU, Sequential
  - Optimizers: SGD, Adam
  - Autograd with backward pass
  - Functional API: relu, softmax, cross_entropy

- **Swift Package** (`FusionML`)
  - Core Tensor with Metal GPU backend
  - GradTensor for automatic differentiation
  - Neural network Module protocol
  - PyTorch-style `Fusion.nn`, `Fusion.optim` API
  - Intelligent GPU+CPU parallel execution

### Core Innovation
- **Intelligent Router**: Automatic work distribution between GPU and CPU
- **24-31% speedup** on matrix operations via parallel execution

## [Unreleased]

### Planned
- Conv2d, Conv3d layers
- PyObjC Metal integration for Python
- MNIST training example
- Model save/load
