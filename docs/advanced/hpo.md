# Hyperparameter Optimization

This library offers comprehensive hyperparameter optimization (HPO) capabilities at both estimator and recommender levels.

## Overview

**Two Main Approaches:**

1. **Estimator-level HPO**: Optimizes estimator parameters using ML metrics (e.g., log loss, accuracy)
2. **Recommender-level HPO**: Optimizes the entire pipeline using recommendation metrics (e.g., NDCG@10)

## Estimator-Level HPO

### Using Tuned Estimators

Wrap a base estimator with a "Tuned" variant to perform hyperparameter search:

```python
from skrec.estimator.classification.xgb_classifier import TunedXGBClassifierEstimator
from skrec.estimator.datatypes import HPOType

# Define search space
param_space = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}

# Define optimization parameters
optimizer_params = {
    "cv": 5,
    "scoring": "roc_auc",
}

# Create tuned estimator
estimator = TunedXGBClassifierEstimator(
    hpo_method=HPOType.GRID_SEARCH_CV,
    param_space=param_space,
    optimizer_params=optimizer_params,
)

# Use with scorer and recommender as usual
from skrec.scorer.universal import UniversalScorer
from skrec.recommender.ranking.ranking_recommender import RankingRecommender

scorer = UniversalScorer(estimator)
recommender = RankingRecommender(scorer)

# Training will perform HPO automatically
recommender.train(
    interactions_ds=interactions_dataset,
    users_ds=users_dataset,
    items_ds=items_dataset
)
```

### Available HPO Methods

- **`GRID_SEARCH_CV`**: Exhaustive search over parameter grid
- **`RANDOMIZED_SEARCH_CV`**: Random sampling from parameter distributions

### Custom Scoring Functions

Define custom metrics for optimization:

```python
from sklearn.metrics import make_scorer
import pandas as pd

def weighted_average_accuracy(actual_class, pred_class):
    pred_df = pd.DataFrame(pred_class)
    all_outputs = actual_class.columns.to_list()
    pred_df.columns = all_outputs
    
    all_accuracies = []
    for output in all_outputs:
        preds = pred_df[output]
        actuals = actual_class[output]
        accuracy = accuracy_score(actuals, preds)
        all_accuracies.append(accuracy)
    
    return sum(all_accuracies) / len(all_accuracies)

optimizer_params = {
    "cv": 5,
    "scoring": make_scorer(weighted_average_accuracy)
}

estimator = TunedXGBClassifierEstimator(
    hpo_method=HPOType.GRID_SEARCH_CV,
    param_space=param_space,
    optimizer_params=optimizer_params,
)
```

### Multioutput Estimators

Special handling for multioutput scenarios:

```python
from xgboost import XGBClassifier
from skrec.estimator.classification.multioutput_classifier import TunedMultiOutputClassifierEstimator
from skrec.scorer.multioutput import MultioutputScorer

param_space = {
    "n_estimators": [100, 200, 1000],
    "max_depth": [1, 2, 3],
}

optimizer_params = {"cv": 5, "scoring": "f1_score"}

estimator = TunedMultiOutputClassifierEstimator(
    base_estimator=XGBClassifier,
    hpo_method=HPOType.GRID_SEARCH_CV,
    param_space=param_space,
    optimizer_params=optimizer_params,
)

scorer = MultioutputScorer(estimator)
recommender = RankingRecommender(scorer)
```

## Recommender-Level HPO

For optimizing the entire pipeline using recommendation metrics, use the `HyperparameterOptimizer` powered by [Optuna](https://optuna.readthedocs.io/).

### Basic Setup

```python
from skrec.orchestrator.hpo import HyperparameterOptimizer

# Define base pipeline configuration
base_pipeline_config = {
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {
            "objective": "binary:logistic",
            "use_label_encoder": False,
            "n_jobs": 1
        }
    },
    "scorer_type": "independent",
    "recommender_type": "propensity"
}

# Define search space using plain dicts
hpo_search_space = {
    "estimator_config.xgboost.n_estimators": {"type": "int", "low": 50, "high": 300},
    "estimator_config.xgboost.learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
    "estimator_config.xgboost.max_depth": {"type": "int", "low": 2, "high": 7},
    "estimator_config.xgboost.subsample": {"type": "float", "low": 0.6, "high": 1.0},
}

# Define metrics to track
hpo_metric_definitions = ["NDCG@5", "Precision@5", "MAP@5"]
```

### Create HPO Manager

```python
hpo_manager = HyperparameterOptimizer(
    base_config=base_pipeline_config,
    search_space=hpo_search_space,
    metric_definitions=hpo_metric_definitions,
    training_interactions_ds=train_interactions_ds,
    validation_interactions_ds=val_interactions_ds,
    training_users_ds=train_users_ds,           # Optional
    training_items_ds=train_items_ds,           # Optional
    validation_users_ds=val_users_ds,           # Optional
    evaluator_type="simple",
    persistence_path="s3://my-bucket/hpo_results.parquet"  # Optional
)
```

### Run Optimization

```python
# Run with TPE (default) — Optuna's Tree-structured Parzen Estimator
results_df, study = hpo_manager.run_optimization(
    n_trials=30,
    objective_metric="NDCG@5",
)

# View results
print(results_df.sort_values(by="NDCG@5", ascending=False).head(10))

# Get best parameters from the study
print(f"\nBest parameters: {study.best_params}")
print(f"Best NDCG@5: {study.best_value:.4f}")
```

### Choosing a Sampler

Any Optuna sampler can be used by name or as an instance:

```python
# TPE (default) — best general-purpose sampler
results_df, study = hpo_manager.run_optimization(
    n_trials=50, objective_metric="NDCG@5", sampler="tpe"
)

# GP — Gaussian Process (available in optuna >= 4.0)
results_df, study = hpo_manager.run_optimization(
    n_trials=30, objective_metric="NDCG@5", sampler="gp"
)

# CMA-ES — good for continuous parameter spaces
results_df, study = hpo_manager.run_optimization(
    n_trials=50, objective_metric="NDCG@5", sampler="cmaes"
)

# QMC — Quasi-Monte Carlo for better space coverage than random
results_df, study = hpo_manager.run_optimization(
    n_trials=30, objective_metric="NDCG@5", sampler="qmc"
)

# Random — pure random search via optuna
results_df, study = hpo_manager.run_optimization(
    n_trials=100, objective_metric="NDCG@5", sampler="random"
)

# Custom sampler instance with full control over kwargs
import optuna
sampler = optuna.samplers.TPESampler(n_startup_trials=20, seed=42)
results_df, study = hpo_manager.run_optimization(
    n_trials=50, objective_metric="NDCG@5", sampler=sampler
)
```

Available sampler names: `"tpe"`, `"gp"`, `"cmaes"`, `"random"`, `"qmc"`, `"grid"`.

### Complete Example

```python
from skrec.orchestrator.hpo import HyperparameterOptimizer

# 1. Define base config
base_config = {
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {
            "objective": "binary:logistic",
            "use_label_encoder": False
        }
    },
    "scorer_type": "universal",
    "recommender_type": "propensity"
}

# 2. Define search space
search_space = {
    "estimator_config.xgboost.n_estimators": {"type": "int", "low": 50, "high": 300},
    "estimator_config.xgboost.learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
    "estimator_config.xgboost.max_depth": {"type": "int", "low": 3, "high": 10},
}

# 3. Create HPO manager
hpo = HyperparameterOptimizer(
    base_config=base_config,
    search_space=search_space,
    metric_definitions=["NDCG@5", "Precision@5"],
    training_interactions_ds=train_interactions_ds,
    validation_interactions_ds=val_interactions_ds,
    training_users_ds=train_users_ds,
    training_items_ds=train_items_ds,
    evaluator_type="simple"
)

# 4. Run optimization (TPE by default)
results_df, study = hpo.run_optimization(
    n_trials=50,
    objective_metric="NDCG@5"
)

# 5. Analyze results
print("Top 5 configurations:")
print(results_df.sort_values("NDCG@5", ascending=False).head())

print(f"\nBest config: {study.best_params}")
print(f"Best NDCG@5: {study.best_value:.4f}")
```

## Key Features

### Persistence

Save HPO trials to resume later:

```python
hpo = HyperparameterOptimizer(
    ...,
    persistence_path="s3://bucket/hpo_results.parquet"
)

# First run
results_df, _ = hpo.run_optimization(n_trials=20, objective_metric="NDCG@5")

# Later: Resume with more trials
# HPO manager will load previous results and warm-start the study
results_df, _ = hpo.run_optimization(n_trials=20, objective_metric="NDCG@5")
```

### Multiple Metrics

Track multiple metrics simultaneously:

```python
metric_definitions = [
    "NDCG@5",
    "NDCG@10",
    "Precision@5",
    "MAP@5",
    "ROC_AUC"
]

# Optimize for one, track others
results_df, study = hpo.run_optimization(
    n_trials=30,
    objective_metric="NDCG@5"  # Primary objective
)

# All metrics available in results_df
print(results_df[["NDCG@5", "NDCG@10", "Precision@5", "MAP@5", "ROC_AUC"]])
```

### Search Space Options

```python
search_space = {
    # Integer parameters
    "estimator_config.xgboost.n_estimators": {"type": "int", "low": 50, "high": 500},
    
    # Integer with step
    "estimator_config.xgboost.n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
    
    # Float with log scale (good for learning rates)
    "estimator_config.xgboost.learning_rate": {"type": "float", "low": 0.001, "high": 0.5, "log": True},
    
    # Float with uniform scale
    "estimator_config.xgboost.subsample": {"type": "float", "low": 0.5, "high": 1.0},
    
    # Categorical parameters
    "scorer_type": {"type": "categorical", "choices": ["universal", "independent"]},
}
```

## Best Practices

### 1. Start Small
```python
# Quick iteration: fewer trials, coarse grid
quick_search_space = {
    "estimator_config.xgboost.n_estimators": {"type": "int", "low": 50, "high": 200},
    "estimator_config.xgboost.learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
}

results, _ = hpo.run_optimization(n_trials=10, objective_metric="NDCG@5")
```

### 2. Refine Promising Regions
```python
# After finding promising region, refine
refined_search_space = {
    "estimator_config.xgboost.n_estimators": {"type": "int", "low": 150, "high": 250},
    "estimator_config.xgboost.learning_rate": {"type": "float", "low": 0.03, "high": 0.07, "log": True},
    "estimator_config.xgboost.max_depth": {"type": "int", "low": 4, "high": 8},
}
```

### 3. Use Log Scale for Learning Rates
```python
# Learning rate spans orders of magnitude — use log scale
"learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True}

# Subsample is bounded and linear — no log needed
"subsample": {"type": "float", "low": 0.6, "high": 1.0}
```

### 4. Choose the Right Sampler
```python
# TPE (default): best for most cases, handles mixed types well
results, _ = hpo.run_optimization(n_trials=50, objective_metric="NDCG@5", sampler="tpe")

# CMA-ES: better for purely continuous spaces with many parameters
results, _ = hpo.run_optimization(n_trials=50, objective_metric="NDCG@5", sampler="cmaes")

# GP: most sample-efficient for low-dimensional continuous spaces
results, _ = hpo.run_optimization(n_trials=30, objective_metric="NDCG@5", sampler="gp")
```

### 5. Monitor Convergence
```python
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history

# Optuna's built-in visualization
fig = plot_optimization_history(study)
fig.show()

# Or plot from results_df
plt.plot(results_df.index, results_df["NDCG@5"])
plt.xlabel("Trial")
plt.ylabel("NDCG@5")
plt.title("HPO Convergence")
plt.show()
```

## Comparison: Estimator vs Recommender HPO

| Aspect | Estimator-Level | Recommender-Level |
|--------|----------------|-------------------|
| **Optimization Metric** | ML metrics (accuracy, AUC) | Recommendation metrics (NDCG@k) |
| **Speed** | Faster | Slower (full pipeline) |
| **Scope** | Single estimator | Entire pipeline |
| **Flexibility** | Limited | High (can tune scorer, recommender) |
| **Best For** | Quick iteration | Final optimization |
| **Use Case** | Initial tuning | Production-ready models |

## Troubleshooting

### Issue: HPO is too slow

**Solutions**:
- Reduce `n_trials`
- Use smaller validation set
- Start with estimator-level HPO
- Use fewer cross-validation folds

### Issue: Not finding good parameters

**Solutions**:
- Expand search space
- Increase `n_trials`
- Try a different sampler (e.g., `"gp"` for small budgets)
- Check if data/features are good

### Issue: Results not reproducible

**Solutions**:
- Pass a seeded sampler: `optuna.samplers.TPESampler(seed=42)`
- Use persistence to save results
- Version your data

## Next Steps

- **[Training Guide](../user-guide/training.md)** - Train optimized models
- **[Orchestration](orchestration.md)** - Config-driven pipelines
- **[Evaluation](../user-guide/evaluation.md)** - Evaluate optimized models
- **[Production](production.md)** - Deploy to production
