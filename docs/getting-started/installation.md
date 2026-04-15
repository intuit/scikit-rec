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

For deep learning models (DeepFM, NCF, Two-Tower, SASRec, HRNN):

```bash
pip install scikit-rec[torch]
```

### AWS Support

For S3 data loading:

```bash
pip install scikit-rec[aws]
```

### Explainability

For SHAP-based feature explanations:

```bash
pip install scikit-rec[explain]
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
