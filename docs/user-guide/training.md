# Training Guide

This guide covers how to train recommendation models using the library.

## Basic Training

All recommenders follow the same training interface:

```python
recommender.train(
    interactions_ds=interactions_dataset,
    users_ds=users_dataset,      # Optional for some scorers
    items_ds=items_dataset        # Optional for some scorers
)
```

## Complete Example

See the [Quick Start Tutorial](../getting-started/quick-start.md) for a complete walkthrough.

## Dataset Requirements

Dataset requirements vary by scorer type. See:
- **[Scorer Selection Guide](scorers.md)** - Dataset requirements for each scorer
- **[Dataset Preparation](../getting-started/datasets.md)** - How to prepare datasets

## Training Different Recommender Types

### RankingRecommender
```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.universal import UniversalScorer

estimator = XGBClassifierEstimator({"learning_rate": 0.1, "n_estimators": 100})
scorer = UniversalScorer(estimator)
recommender = RankingRecommender(scorer)

recommender.train(
    interactions_ds=interactions_dataset,
    users_ds=users_dataset,
    items_ds=items_dataset
)
```

**Learn more**: [RankingRecommender Guide](../recommender-types/ranking.md)

### SequentialRecommender (SASRec)

```python
from skrec.estimator.sequential.sasrec_estimator import SASRecClassifierEstimator
from skrec.recommender.sequential.sequential_recommender import SequentialRecommender
from skrec.scorer.sequential import SequentialScorer

estimator = SASRecClassifierEstimator(
    hidden_units=50, num_blocks=2, num_heads=1,
    dropout_rate=0.2, epochs=200, max_len=50,
    early_stopping_patience=5,   # Stop if val loss stalls for 5 epochs
    restore_best_weights=True,
)
scorer = SequentialScorer(estimator)
recommender = SequentialRecommender(scorer, max_len=50)

# Leave-last-two-out split for validation
interactions["rank"] = interactions.groupby("USER_ID").cumcount(ascending=False)
train_df = interactions.drop(columns=["rank"])
valid_df = interactions[interactions["rank"] == 1]

recommender.train(
    items_ds=items_dataset,
    interactions_ds=InteractionsDataset("train.csv"),
    valid_interactions_ds=InteractionsDataset("valid.csv"),  # enables early stopping
)
```

**Learn more**: [SASRec Guide](sasrec.md) | [SequentialRecommender Guide](../recommender-types/sequential.md)

### HierarchicalSequentialRecommender (HRNN)

```python
from skrec.estimator.sequential.hrnn_estimator import HRNNClassifierEstimator
from skrec.recommender.sequential.hierarchical_recommender import HierarchicalSequentialRecommender
from skrec.scorer.hierarchical import HierarchicalScorer

estimator = HRNNClassifierEstimator(
    hidden_units=50, num_layers=1,
    max_sessions=10, max_session_len=20,
    epochs=200, early_stopping_patience=5,
)
scorer = HierarchicalScorer(estimator)
recommender = HierarchicalSequentialRecommender(
    scorer, max_sessions=10, max_session_len=20, session_timeout_minutes=30
)

recommender.train(
    items_ds=items_dataset,
    interactions_ds=InteractionsDataset("train.csv"),
    valid_interactions_ds=InteractionsDataset("valid.csv"),
)
```

**Learn more**: [HRNN Guide](hrnn.md) | [HierarchicalSequentialRecommender Guide](../recommender-types/hierarchical.md)

### ContextualBanditsRecommender
```python
from skrec.recommender.bandits.contextual_bandits import ContextualBanditsRecommender
from skrec.recommender.bandits.datatypes import StrategyType

recommender = ContextualBanditsRecommender(
    scorer=scorer,
    strategy_type=StrategyType.EPSILON_GREEDY,
    strategy_params={"epsilon": 0.1}
)

recommender.train(
    interactions_ds=interactions_dataset,
    users_ds=users_dataset,
    items_ds=items_dataset
)
```

**Learn more**: [ContextualBanditsRecommender Guide](../recommender-types/bandits.md)

## Config-Driven Training (Orchestration)

For production pipelines, use config-driven orchestration:

```python
from skrec.orchestrator.factory import create_recommender_pipeline

config = {
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {
            "learning_rate": 0.1,
            "n_estimators": 100
        }
    },
    "scorer_type": "universal",
    "recommender_type": "propensity"
}

recommender = create_recommender_pipeline(config)
recommender.train(interactions_ds, users_ds, items_ds)
```

**Learn more**: [Orchestration Guide](../advanced/orchestration.md)

## Hyperparameter Optimization

### Estimator-Level HPO
```python
from skrec.estimator.classification.xgb_classifier import TunedXGBClassifierEstimator
from skrec.estimator.datatypes import HPOType

param_space = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]}
optimizer_params = {"cv": 5, "scoring": "roc_auc"}

estimator = TunedXGBClassifierEstimator(
    hpo_method=HPOType.GRID_SEARCH_CV,
    param_space=param_space,
    optimizer_params=optimizer_params,
)

scorer = UniversalScorer(estimator)
recommender = RankingRecommender(scorer)
recommender.train(...)
```

### Recommender-Level HPO
```python
from skrec.orchestrator.hpo import HyperparameterOptimizer

hpo_manager = HyperparameterOptimizer(
    base_config=base_pipeline_config,
    search_space=hpo_search_space,
    metric_definitions=["NDCG@5", "Precision@5"],
    training_interactions_ds=train_interactions_ds,
    validation_interactions_ds=val_interactions_ds,
    evaluator_type="simple"
)

results_df, study = hpo_manager.run_optimization(
    n_trials=30,
    objective_metric="NDCG@5",
    sampler="tpe",  # or "gp", "cmaes", "random", "qmc"
)
```

**Learn more**: [HPO Guide](../advanced/hpo.md)

## Best Practices

### 1. Data Splitting
```python
# Split by time for temporal validation
train_data = interactions[interactions['timestamp'] < cutoff]
test_data = interactions[interactions['timestamp'] >= cutoff]
```

### 2. Feature Engineering
- Normalize numerical features
- Handle categorical features with vocab or hash_buckets
- Remove highly correlated features

### 3. Model Monitoring
- Track training metrics (loss, accuracy)
- Validate on held-out set
- Monitor for overfitting
- For SASRec/HRNN: use `early_stopping_patience` + `valid_interactions_ds` to automatically stop at the best epoch and avoid wasted compute

### 4. Reproducibility
```python
# Set random seeds
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# XGBoost seed
estimator = XGBClassifierEstimator({"random_state": 42, ...})
```

## Common Issues

### Issue: KeyError on required columns

**Solution**: Ensure datasets have required columns (`USER_ID`, `ITEM_ID`, `OUTCOME`). See [Dataset Guide](../getting-started/datasets.md).

### Issue: Training is slow

**Solution**:
- Use LightGBM instead of XGBoost
- Reduce data size for experimentation
- Use fewer estimator iterations
- Enable GPU training if available

### Issue: Poor validation performance

**Solution**:
- Check for data leakage
- Improve feature engineering
- Try different estimators
- Tune hyperparameters

## Next Steps

- **[Inference Guide](inference.md)** - Make recommendations after training
- **[Evaluation Guide](evaluation.md)** - Evaluate model performance
- **[HPO Guide](../advanced/hpo.md)** - Optimize hyperparameters
- **[Production Guide](../advanced/production.md)** - Deploy to production

