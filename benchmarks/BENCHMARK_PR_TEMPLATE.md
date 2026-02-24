# PR Template for Benchmark Submissions

## Device Information
- **Chip**: Apple M_
- **RAM**: _ GB
- **macOS Version**: 
- **Python Version** (if applicable): 
- **Swift Version** (if applicable): 

## Benchmark Results

### Matrix Multiplication

| Size | FusionML (ms) | PyTorch (ms) | Speedup |
|------|---------------|--------------|---------|
| 256  |               |              |         |
| 512  |               |              |         |
| 1024 |               |              |         |
| 2048 |               |              |         |
| 4096 |               |              |         |

### Backend Comparison (1024x1024)

| Backend | Time (ms) |
|---------|-----------|
| CPU     |           |
| GPU     |           |
| Smart   |           |

### Training (MLP 784→256→10, batch=32)

- **Time per batch**: _ ms
- **Throughput**: _ samples/sec

## Checklist
- [ ] Results are reproducible
- [ ] JSON file added to `results/` folder
- [ ] Device info is accurate
- [ ] Multiple runs averaged (min 5)

## Observations
<!-- Any interesting findings or notes -->
