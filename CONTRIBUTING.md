# Contributing to FusionML

Thank you for your interest in contributing to FusionML!

## Getting Started

1. Copy the repository locally
2. Create a branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `swift test`
5. Commit: `git commit -m "Add my feature"`

## Development Setup

```bash
cd FusionML
swift build
swift test
swift run APIDemo
```

## Code Style

- Use Swift standard naming conventions
- Add documentation comments for public APIs
- Keep functions focused and small
- Write tests for new functionality

## Areas for Contribution

- **New Layers**: Conv2d, RNN, LSTM, Attention
- **Optimizers**: AdaGrad, RMSprop, LAMB
- **Loss Functions**: Focal loss, Dice loss
- **Performance**: Metal shader optimization
- **Documentation**: Tutorials, examples
- **Tests**: Unit tests, integration tests

## Reporting Issues

When reporting issues, please include:
- macOS version
- Apple Silicon chip (M1/M2/M3)
- Swift version
- Minimal reproduction code

## Questions?

Reach out to the maintainers directly.
