# FusionML Architecture

## Core Innovation: Intelligent Parallel Execution

FusionML's key innovation is **GPU+CPU parallel execution** for matrix operations.

```
Traditional:  GPU ────────────────→ Result

FusionML:     GPU (68%) ────┬─────→ Result (faster!)
              CPU (32%) ────┘
```

## System Architecture

```
┌─────────────────────────────────────────┐
│            User API                     │
│   Fusion.nn | Fusion.optim | Tensor     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Intelligent Router              │
│   Analyzes op type, size, history       │
└──────┬────────────┬────────────┬────────┘
       │            │            │
┌──────▼──────┐ ┌───▼───┐ ┌──────▼──────┐
│  GPU Engine │ │  CPU  │ │ ANE Engine  │
│   (Metal)   │ │(Accel)│ │  (CoreML)   │
└─────────────┘ └───────┘ └─────────────┘
```

## Components

### Tensor
- Core data structure
- Unified memory buffer (Metal shared)
- Shape, dtype, operations

### Autograd
- GradTensor wraps Tensor
- Computation graph via GradNode
- Backward pass for gradients

### IntelligentRouter
- Routes ops to optimal backend
- Calibrates GPU/CPU split ratio
- Size-based decisions

### Memory Manager
- Buffer pooling and reuse
- 99% reuse rate achieved
