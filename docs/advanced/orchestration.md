# Config-Driven Orchestration

Build complete recommender pipelines from a single configuration dictionary using `create_recommender_pipeline()`. This eliminates manual instantiation and provides a single entry point for MLOps pipelines, agent-driven systems, and A/B testing.

## Basic Usage

```python
from skrec.orchestrator import create_recommender_pipeline

config = {
    "recommender_type": "ranking",
    "scorer_type": "universal",
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
        },
    },
}

recommender = create_recommender_pipeline(config)

recommender.train(
    interactions_ds=interactions_dataset,
    items_ds=items_dataset,
    users_ds=users_dataset,
)

recommendations = recommender.recommend(interactions=query_df, top_k=5)
```

## Configuration Reference

### Top-Level Config (`RecommenderConfig`)

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `recommender_type` | str | **Yes** | `"ranking"`, `"bandits"`, `"sequential"`, `"hierarchical_sequential"`, `"uplift"`, `"gcsl"` |
| `scorer_type` | str | **Yes** | `"universal"`, `"independent"`, `"multiclass"`, `"multioutput"`, `"sequential"`, `"hierarchical"` |
| `estimator_config` | dict | Yes | Estimator configuration (see below) |
| `recommender_params` | dict | No | Per-recommender parameters (see below) |

### Estimator Config (`EstimatorConfig`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `estimator_type` | str | `"tabular"` | `"tabular"`, `"embedding"`, or `"sequential"` |
| `ml_task` | str | `"classification"` | `"classification"` or `"regression"` (tabular only) |
| `xgboost` | dict | `{}` | XGBoost hyperparameters (tabular only) |
| `hpo` | dict | — | HPO configuration (tabular only) |
| `weights` | dict | — | Sample/feature weighting (tabular only) |
| `embedding` | dict | — | Embedding model config (embedding only) |
| `sequential` | dict | — | Sequential model config (sequential only) |

### Embedding Config

```python
"embedding": {
    "model_type": str,  # Required. See table below.
    "params": dict,     # Constructor kwargs passed to the estimator.
}
```

| `model_type` | Class | Requires PyTorch |
|-------------|-------|-----------------|
| `"matrix_factorization"` | `MatrixFactorizationEstimator` | No (NumPy) |
| `"ncf"` | `NCFEstimator` | Yes |
| `"two_tower"` | `ContextualizedTwoTowerEstimator` | Yes |
| `"deep_cross_network"` | `DeepCrossNetworkEstimator` | Yes |
| `"neural_factorization"` | `NeuralFactorizationEstimator` | Yes |

### Sequential Config

```python
"sequential": {
    "model_type": str,  # Required. See table below.
    "params": dict,     # Constructor kwargs passed to the estimator.
}
```

| `model_type` | Class | Description |
|-------------|-------|-------------|
| `"sasrec_classifier"` | `SASRecClassifierEstimator` | Self-attentive sequential (binary) |
| `"sasrec_regressor"` | `SASRecRegressorEstimator` | Self-attentive sequential (continuous) |
| `"hrnn_classifier"` | `HRNNClassifierEstimator` | Hierarchical RNN (binary) |
| `"hrnn_regressor"` | `HRNNRegressorEstimator` | Hierarchical RNN (continuous) |

### Recommender Params

Per-recommender constructor parameters. Only keys relevant to the chosen `recommender_type` are used.

| Key | Used by | Type | Description |
|-----|---------|------|-------------|
| `max_len` | `sequential` | int | Maximum sequence length (default: 50) |
| `max_sessions` | `hierarchical_sequential` | int | Max past sessions (default: 10) |
| `max_session_len` | `hierarchical_sequential` | int | Max items per session (default: 20) |
| `session_timeout_minutes` | `hierarchical_sequential` | float | Session boundary timeout (default: 30.0) |
| `control_item_id` | `uplift` | str | **Required.** Control group item ID |
| `mode` | `uplift` | str | `"t_learner"`, `"s_learner"`, or `"x_learner"`. Auto-detects if omitted. |
| `inference_method` | `gcsl` | dict | Goal-injection method (see below) |
| `retriever` | `ranking`, `gcsl` | dict | Candidate retriever (see below) |

### Inference Method Config (GCSL)

```python
"inference_method": {
    "type": str,    # "mean_scalarization", "percentile_value", "predefined_value"
    "params": dict, # Constructor kwargs
}
```

### Retriever Config

```python
"retriever": {
    "type": str,    # "popularity", "content_based", "embedding"
    "params": dict, # Constructor kwargs (e.g. {"top_k": 200})
}
```

## Compatibility Rules

The factory validates these constraints at pipeline creation time and raises `ValueError` with a clear message if violated:

| Rule | Why |
|------|-----|
| `sequential` / `hierarchical_sequential` recommender requires `estimator_type: "sequential"` | Sequential models need sequence data |
| `sequential` recommender requires `scorer_type: "sequential"` | SASRec needs SequentialScorer |
| `hierarchical_sequential` recommender requires `scorer_type: "hierarchical"` | HRNN needs HierarchicalScorer |
| `sequential` / `hierarchical` scorer requires `estimator_type: "sequential"` | Scorer delegates to SequentialEstimator |
| `embedding` estimator only works with `scorer_type: "universal"` | IndependentScorer/MulticlassScorer/MultioutputScorer reject embedding estimators |
| `uplift` recommender requires `scorer_type: "independent"` or `"universal"` | UpliftRecommender needs T-Learner or S-Learner compatible scorer |

## Complete Examples

### 1. Tabular: XGBoost + Universal + Ranking

The simplest pipeline — a pointwise XGBoost ranker.

```yaml
recommender_type: ranking
scorer_type: universal
estimator_config:
  ml_task: classification
  xgboost:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
```

### 2. Embedding: Matrix Factorization + Universal + Ranking

Collaborative filtering via learned user/item embeddings.

```yaml
recommender_type: ranking
scorer_type: universal
estimator_config:
  estimator_type: embedding
  embedding:
    model_type: matrix_factorization
    params:
      n_factors: 64
      algorithm: als
      epochs: 30
```

### 3. Embedding: NCF + Universal + Ranking with Retriever

Neural collaborative filtering with a two-stage retrieval pipeline.

```yaml
recommender_type: ranking
scorer_type: universal
estimator_config:
  estimator_type: embedding
  embedding:
    model_type: ncf
    params:
      ncf_type: neumf
      gmf_embedding_dim: 32
      mlp_embedding_dim: 32
      epochs: 20
recommender_params:
  retriever:
    type: embedding
    params:
      top_k: 200
```

### 4. Sequential: SASRec

Self-attentive sequential recommendation from interaction history.

```yaml
recommender_type: sequential
scorer_type: sequential
estimator_config:
  estimator_type: sequential
  sequential:
    model_type: sasrec_classifier
    params:
      hidden_units: 64
      num_blocks: 2
      num_heads: 2
      max_len: 50
      epochs: 100
recommender_params:
  max_len: 50
```

### 5. Sequential: HRNN (Hierarchical)

Session-aware recommendation with hierarchical GRU.

```yaml
recommender_type: hierarchical_sequential
scorer_type: hierarchical
estimator_config:
  estimator_type: sequential
  sequential:
    model_type: hrnn_classifier
    params:
      hidden_units: 64
      max_sessions: 10
      max_session_len: 20
      epochs: 100
recommender_params:
  max_sessions: 10
  max_session_len: 20
  session_timeout_minutes: 30.0
```

### 6. Uplift: T-Learner

Causal treatment effect estimation with per-treatment models.

```yaml
recommender_type: uplift
scorer_type: independent
estimator_config:
  ml_task: classification
  xgboost:
    n_estimators: 200
recommender_params:
  control_item_id: "control_arm"
  mode: t_learner
```

### 7. GCSL: Goal-Conditioned Supervised Learning

Multi-objective recommendation with goal injection at inference time.

```yaml
recommender_type: gcsl
scorer_type: universal
estimator_config:
  ml_task: classification
  xgboost:
    n_estimators: 200
recommender_params:
  inference_method:
    type: percentile_value
    params:
      percentiles:
        OUTCOME_revenue: 80
        OUTCOME_clicks: 75
```

### 8. Contextual Bandits

Exploration/exploitation via epsilon-greedy or static action strategies.

```yaml
recommender_type: bandits
scorer_type: universal
estimator_config:
  ml_task: classification
  xgboost:
    n_estimators: 100
```

> **Note:** Bandit strategies are set after pipeline creation via `recommender.set_strategy()`.

### 9. Tabular with HPO

Hyperparameter optimization using grid search or randomized search.

```yaml
recommender_type: ranking
scorer_type: universal
estimator_config:
  ml_task: classification
  xgboost:
    objective: binary:logistic
    n_jobs: 1
  hpo:
    hpo_method: grid_search_cv
    param_space:
      n_estimators: [50, 100, 200]
      learning_rate: [0.01, 0.05, 0.1]
      max_depth: [3, 5, 7]
    optimizer_params:
      cv: 3
      scoring: roc_auc
```

### 10. Tabular with Sample Weighting

Feature-level and item-level weighting for imbalanced datasets.

```yaml
recommender_type: ranking
scorer_type: universal
estimator_config:
  ml_task: classification
  xgboost:
    n_estimators: 100
    colsample_bynode: 0.9
  weights:
    action_weight: 0.8
    item_sample_weights:
      itemA: 1.2
      itemB: 0.5
```

## Loading Config from Files

### YAML

```python
import yaml
from skrec.orchestrator import create_recommender_pipeline

with open("config.yaml") as f:
    config = yaml.safe_load(f)

recommender = create_recommender_pipeline(config)
```

### JSON

```python
import json
from skrec.orchestrator import create_recommender_pipeline

with open("config.json") as f:
    config = json.load(f)

recommender = create_recommender_pipeline(config)
```

## Use Cases

### Kubeflow Pipelines

```python
from kfp import dsl

@dsl.component
def train_recommender(config_path: str, data_path: str):
    import yaml
    from skrec.orchestrator import create_recommender_pipeline

    with open(config_path) as f:
        config = yaml.safe_load(f)

    recommender = create_recommender_pipeline(config)
    recommender.train(...)
```

### A/B Testing

```python
configs = {
    "variant_a": {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 100, "max_depth": 5},
        },
    },
    "variant_b": {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {
            "estimator_type": "embedding",
            "embedding": {"model_type": "ncf", "params": {"embedding_dim": 32}},
        },
    },
}

for name, config in configs.items():
    recommender = create_recommender_pipeline(config)
    recommender.train(...)
    metrics = recommender.evaluate(...)
    print(f"{name}: {metrics}")
```

### Environment-Specific Configs

```python
import os

env = os.getenv("ENVIRONMENT", "dev")

configs = {
    "dev": {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 50},
        },
    },
    "prod": {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 500, "max_depth": 8},
        },
    },
}

recommender = create_recommender_pipeline(configs[env])
```

## Error Handling

The factory raises clear errors for invalid configurations:

```python
# Missing required key
>>> create_recommender_pipeline({"scorer_type": "universal"})
ValueError: 'recommender_type' must be specified in the configuration.

# Typo in recommender_type
>>> create_recommender_pipeline({..., "recommender_type": "ranknig"})
NotImplementedError: Recommender type 'ranknig' not supported.
    Supported types: 'ranking', 'bandits', 'sequential',
    'hierarchical_sequential', 'uplift', 'gcsl'.

# Incompatible estimator + scorer
>>> create_recommender_pipeline({..., "estimator_type": "embedding", "scorer_type": "independent"})
ValueError: scorer_type 'independent' does not support embedding estimators.
    Use scorer_type='universal' with embedding estimators.
```

## Next Steps

- **[HPO Guide](hpo.md)** — Add hyperparameter optimization to configs
- **[Training Guide](../user-guide/training.md)** — Train config-driven pipelines
- **[Capability Matrix](../user-guide/capability-matrix.md)** — Full compatibility reference
- **[Production Guide](production.md)** — Deploy config-driven systems
