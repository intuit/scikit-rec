# Installation

## Prerequisites

- **Python 3.9** or higher

## Install from PyPI

```bash
pip install scikit-rec
```

The package is also available under the alias `skrec`:

```bash
pip install skrec
```

## Optional Dependencies

### Torch Support

For PyTorch-based models (NCF, Two-Tower, DCN, NeuralFactorization, SASRec, HRNN, DeepFM):

```bash
pip install scikit-rec[torch]
```

### AWS Support

For S3 data loading:

```bash
pip install scikit-rec[aws]
```

### Development Dependencies

If you're contributing to the library:

```bash
# Clone the repository
git clone https://github.com/intuit/scikit-rec.git
cd scikit-rec

# Install in editable mode with dev extras
pip install -e ".[dev]"
```

## Verify Installation

```python
import skrec
print(skrec.__version__)

# Test with example datasets
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_users,
    sample_binary_reward_items,
)

print("Installation successful!")
print(f"Example interactions: {sample_binary_reward_interactions.fetch_data().shape}")
```

## Environment Support

The library works in multiple environments:

- Local development (Jupyter notebooks, Python scripts)
- Cloud notebooks (SageMaker, Colab, etc.)
- Batch processing (Spark, Airflow)
- Real-time inference (API endpoints)

## macOS notes

On macOS (especially Apple Silicon), if you train a tabular estimator (e.g. MF/ALS, which is numpy-heavy) and a torch estimator (NCF, Two-Tower, DCN, NeuralFactorization) in the **same Python process**, torch may segfault during training (process exits with status 139).

This is a known interaction between numpy's bundled OpenBLAS+libomp and PyTorch's use of Apple Accelerate: two OpenMP runtimes loaded into one process can leave threading state that crashes subsequent BLAS calls. It is **not specific to scikit-rec** — any pipeline mixing numpy and torch on macOS is exposed.

### Fix

Set these environment variables **before** Python imports numpy or torch (in your shell, your launcher, or at the top of your script before any other imports):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

The cost is single-threaded BLAS for the tabular training path, which is usually negligible at typical recommender-system data sizes. Linux installations are not affected and do not need these settings.

## Troubleshooting

### ImportError: No module named 'skrec'

**Solution**: Ensure you have installed the package:

```bash
pip install scikit-rec
```

### Permission Denied

**Solution**: Use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install scikit-rec
```

## Next Steps

- **[Quick Start Tutorial](quick-start.md)** - Build your first recommender in 5 minutes
- **[Dataset Preparation Guide](datasets.md)** - Learn about data requirements
- **[Architecture Overview](../user-guide/architecture.md)** - Understand the library structure
