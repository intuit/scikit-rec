# Config-Driven Orchestration

The library provides tools to build complete recommender pipelines from configuration dictionaries, enabling config-driven development for MLOps and Kubeflow pipelines.

## Overview

**Purpose**: Build entire pipelines (Estimator → Scorer → Recommender) from a single configuration, eliminating manual instantiation.

**Key Function**: `create_recommender_pipeline()` in `recommender.orchestrator.factory`

## Basic Usage

### Simple Configuration

```python
from skrec.orchestrator.factory import create_recommender_pipeline

# Define complete pipeline in config
config = {
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 5
        }
    },
    "scorer_type": "universal",
    "recommender_type": "propensity"
}

# Create pipeline from config
recommender = create_recommender_pipeline(config)

# Use as normal
recommender.train(
    interactions_ds=interactions_dataset,
    users_ds=users_dataset,
    items_ds=items_dataset
)

recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5
)
```

## Configuration Structure

### RecommenderConfig

The configuration dictionary follows the `RecommenderConfig` structure:

```python
config = {
    # Estimator configuration
    "estimator_config": {
        "ml_task": str,        # "classification" or "regression"
        "<estimator_name>": {  # e.g., "xgboost", "lightgbm", "logistic_regression"
            # Estimator-specific parameters
        },
        "hpo": {               # Optional: estimator-level HPO
            # HPO configuration
        }
    },
    
    # Scorer type
    "scorer_type": str,  # "universal", "independent", "multiclass", "multioutput"
    
    # Recommender type
    "recommender_type": str  # "propensity", "bandits", etc.
}
```

### Available Estimators

#### Classification

```python
# XGBoost
"estimator_config": {
    "ml_task": "classification",
    "xgboost": {
        "objective": "binary:logistic",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "use_label_encoder": False
    }
}

# LightGBM
"estimator_config": {
    "ml_task": "classification",
    "lightgbm": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31
    }
}

# Logistic Regression
"estimator_config": {
    "ml_task": "classification",
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 100
    }
}
```

#### Regression

```python
"estimator_config": {
    "ml_task": "regression",
    "xgboost": {
        "objective": "reg:squarederror",
        "n_estimators": 100,
        "learning_rate": 0.1
    }
}
```

### Available Scorers

```python
# Universal Scorer (global model with item features)
"scorer_type": "universal"

# Independent Scorer (per-item models)
"scorer_type": "independent"

# Multiclass Scorer (items as classes)
"scorer_type": "multiclass"

# Multioutput Scorer (multiple outcomes)
"scorer_type": "multioutput"
```

### Available Recommenders

```python
# Propensity Recommender (most common)
"recommender_type": "propensity"

# Contextual Bandits
"recommender_type": "bandits"
# Note: Requires additional bandit-specific config

# Uplift Models
"recommender_type": "uplift_t_learner"
"recommender_type": "uplift_s_learner"
```

## Complete Examples

### Example 1: XGBoost + Universal Scorer

```python
config = {
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {
            "objective": "binary:logistic",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False
        }
    },
    "scorer_type": "universal",
    "recommender_type": "propensity"
}

recommender = create_recommender_pipeline(config)
```

### Example 2: LightGBM + Independent Scorer

```python
config = {
    "estimator_config": {
        "ml_task": "classification",
        "lightgbm": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20
        }
    },
    "scorer_type": "independent",
    "recommender_type": "propensity"
}

recommender = create_recommender_pipeline(config)
```

### Example 3: With Estimator-Level HPO

```python
config = {
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {
            "objective": "binary:logistic",
            "use_label_encoder": False,
            "n_jobs": 1
        },
        "hpo": {
            "hpo_method": "GRID_SEARCH_CV",
            "param_space": {
                "xgboost.n_estimators": [50, 100, 200],
                "xgboost.learning_rate": [0.01, 0.05, 0.1],
                "xgboost.max_depth": [3, 5, 7]
            },
            "optimizer_params": {
                "cv": 3,
                "scoring": "roc_auc"
            }
        }
    },
    "scorer_type": "universal",
    "recommender_type": "propensity"
}

recommender = create_recommender_pipeline(config)
```

## Loading Config from Files

### YAML Configuration

Create `config.yaml`:

```yaml
estimator_config:
  ml_task: classification
  xgboost:
    objective: binary:logistic
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 5
    use_label_encoder: false

scorer_type: universal
recommender_type: propensity
```

Load and use:

```python
import yaml
from skrec.orchestrator.factory import create_recommender_pipeline

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create pipeline
recommender = create_recommender_pipeline(config)
```

### JSON Configuration

Create `config.json`:

```json
{
  "estimator_config": {
    "ml_task": "classification",
    "xgboost": {
      "objective": "binary:logistic",
      "n_estimators": 100,
      "learning_rate": 0.1,
      "max_depth": 5,
      "use_label_encoder": false
    }
  },
  "scorer_type": "universal",
  "recommender_type": "propensity"
}
```

Load and use:

```python
import json
from skrec.orchestrator.factory import create_recommender_pipeline

with open('config.json', 'r') as f:
    config = json.load(f)

recommender = create_recommender_pipeline(config)
```

## Use Cases

### 1. Kubeflow Pipelines

```python
from kfp import dsl

@dsl.component
def train_recommender(config_path: str, data_path: str):
    import yaml
    from skrec.orchestrator.factory import create_recommender_pipeline
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create and train
    recommender = create_recommender_pipeline(config)
    recommender.train(...)
    
    # Save model
    ...
```

### 2. A/B Testing Different Configs

```python
# Config A: XGBoost
config_a = {
    "estimator_config": {"ml_task": "classification", "xgboost": {...}},
    "scorer_type": "universal",
    "recommender_type": "propensity"
}

# Config B: LightGBM
config_b = {
    "estimator_config": {"ml_task": "classification", "lightgbm": {...}},
    "scorer_type": "universal",
    "recommender_type": "propensity"
}

# Train both
recommender_a = create_recommender_pipeline(config_a)
recommender_b = create_recommender_pipeline(config_b)

# Compare
ndcg_a = recommender_a.evaluate(...)
ndcg_b = recommender_b.evaluate(...)
```

### 3. Environment-Specific Configs

```python
import os

# Different configs for dev/staging/prod
env = os.getenv("ENVIRONMENT", "dev")

configs = {
    "dev": {
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 50}  # Faster for dev
        },
        "scorer_type": "universal",
        "recommender_type": "propensity"
    },
    "prod": {
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 200}  # Better for prod
        },
        "scorer_type": "universal",
        "recommender_type": "propensity"
    }
}

recommender = create_recommender_pipeline(configs[env])
```

## Advanced Features

### LightGBM Training Parameters

LightGBM estimators support additional training parameters:

```python
config = {
    "estimator_config": {
        "ml_task": "classification",
        "lightgbm": {
            "n_estimators": 100,
            "learning_rate": 0.05
        },
        "training_params": {  # Additional training parameters
            "early_stopping_rounds": 10,
            "verbose": False
        }
    },
    "scorer_type": "universal",
    "recommender_type": "propensity"
}
```

### XGBoost In-Place Prediction

XGBoost estimators can use in-place prediction for faster inference:

```python
config = {
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {
            "n_estimators": 100,
            "use_inplace_predict": True  # Faster inference
        }
    },
    "scorer_type": "universal",
    "recommender_type": "propensity"
}
```

## Best Practices

### 1. Version Your Configs

```python
config = {
    "version": "v1.2.0",  # Track config versions
    "estimator_config": {...},
    "scorer_type": "universal",
    "recommender_type": "propensity"
}
```

### 2. Use Config Validation

```python
def validate_config(config):
    assert "estimator_config" in config
    assert "scorer_type" in config
    assert "recommender_type" in config
    assert config["estimator_config"]["ml_task"] in ["classification", "regression"]
    return True

validate_config(config)
recommender = create_recommender_pipeline(config)
```

### 3. Separate Sensitive Data

```python
# config.yaml - committed to git
estimator_config:
  ml_task: classification
  xgboost:
    n_estimators: 100

# secrets.yaml - NOT committed
aws:
  access_key: ${AWS_ACCESS_KEY}
  secret_key: ${AWS_SECRET_KEY}
```

### 4. Document Your Configs

```yaml
# Production recommender config - v2.1.0
# Last updated: 2025-01-15
# Owner: Data Science Team
# Performance: NDCG@5 = 0.78

estimator_config:
  ml_task: classification
  xgboost:
    n_estimators: 200  # Optimized via HPO
    learning_rate: 0.08  # Optimized via HPO
    max_depth: 6  # Optimized via HPO

scorer_type: universal
recommender_type: propensity
```

## Troubleshooting

### Issue: "Unknown estimator type"

**Solution**: Check estimator name matches available types (xgboost, lightgbm, logistic_regression, etc.)

### Issue: "Invalid ml_task"

**Solution**: Ensure `ml_task` is "classification" or "regression"

### Issue: Config not loading from YAML

**Solution**: 
- Check YAML syntax
- Ensure proper indentation
- Use `yaml.safe_load()` not `yaml.load()`

## Next Steps

- **[HPO Guide](hpo.md)** - Add hyperparameter optimization to configs
- **[Training Guide](../user-guide/training.md)** - Train config-driven pipelines
- **[Production Guide](production.md)** - Deploy config-driven systems

