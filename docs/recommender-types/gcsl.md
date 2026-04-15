# GcslRecommender

**GcslRecommender** implements **GCSL** (Goal-Conditioned Supervised Learning): one model that produces different recommendations depending on the *goals* you specify at inference time — no retraining required.

## Overview

**Purpose**: Multi-objective recommendation where you want to steer results toward specific outcome targets — e.g. maximize engagement AND revenue, or trade off popularity against novelty.

**Key Insight**: Standard recommenders drop outcome columns before training. `GcslRecommender` keeps them as **input features**, so the model learns:

```
P(positive | user, item, context, outcome_1, outcome_2, ...)
```

At inference, an *inference method* fills in desired goal values. Items whose feature profile is most consistent with those goals score highest. Change the goals, change the recommendations — same model weights.

## When to Use

✅ **Perfect For**:

- Balancing multiple metrics (engagement + revenue + diversity)
- Steering between popular and niche recommendations
- A/B testing different business objectives without retraining
- Any scenario where you want one model to serve multiple objectives

❌ **Not Ideal For**:

- Single-objective ranking → Use [RankingRecommender](ranking.md)
- Causal impact measurement → Use [UpliftRecommender](uplift.md)
- Exploration-exploitation → Use [ContextualBanditsRecommender](bandits.md)

## Basic Usage

### 1. Build the Pipeline

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.recommender.gcsl.gcsl_recommender import GcslRecommender
from skrec.recommender.gcsl.inference.predefined_value import PredefinedValue
from skrec.scorer.universal import UniversalScorer

# Layer 1: Choose an estimator
estimator = XGBClassifierEstimator({
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 5
})

# Layer 2: Choose a scorer
scorer = UniversalScorer(estimator)

# Layer 3: Create the recommender with an inference method
inference = PredefinedValue({"OUTCOME_engagement": 0.8, "OUTCOME_revenue": 0.5})
recommender = GcslRecommender(scorer, inference_method=inference)
```

### 2. Prepare Multi-Outcome Data

Your interactions dataset must include `OUTCOME_*` columns alongside the primary `OUTCOME` label. These columns become input features during training.

```python
import pandas as pd

# Interactions with multiple outcome dimensions
interactions_df = pd.DataFrame({
    "USER_ID":            ["u1", "u1", "u2", "u2"],
    "ITEM_ID":            ["A",  "B",  "A",  "C"],
    "OUTCOME":            [1,    0,    1,    1],      # primary label
    "OUTCOME_engagement": [0.9,  0.2,  0.7,  0.8],   # kept as feature
    "OUTCOME_revenue":    [5.0,  0.0,  3.0,  8.0],   # kept as feature
})
```

### 3. Train

```python
recommender.train(
    interactions_ds=interactions_dataset,
    items_ds=items_dataset
)
```

### 4. Recommend

```python
recommendations = recommender.recommend(
    interactions=inference_df,
    users=users_df,
    top_k=10
)
```

### 5. Swap Goals Without Retraining

```python
from skrec.recommender.gcsl.inference.percentile_value import PercentileValue

# Switch to a different goal — auto-fits using stored training data
recommender.set_inference_method(
    PercentileValue({"OUTCOME_engagement": 80, "OUTCOME_revenue": 50})
)

# Same model, different recommendations
new_recommendations = recommender.recommend(
    interactions=inference_df,
    users=users_df,
    top_k=10
)
```

## Inference Methods

The inference method controls how goal values are injected at scoring time. All methods follow the sklearn-style `fit()` / `transform()` lifecycle.

### PredefinedValue — Fixed Goals

Set exact goal values. Simplest method.

```python
from skrec.recommender.gcsl.inference.predefined_value import PredefinedValue

inference = PredefinedValue({
    "OUTCOME_engagement": 0.8,
    "OUTCOME_revenue": 5.0
})
```

**When to use**: You know the exact outcome values you want to target.

### PercentileValue — Percentile-Based Goals

Target a specific percentile of the training distribution. Goals are **always within the training range** — the safest option.

```python
from skrec.recommender.gcsl.inference.percentile_value import PercentileValue

inference = PercentileValue({
    "OUTCOME_engagement": 80,  # 80th percentile
    "OUTCOME_revenue": 75      # 75th percentile
})
```

**When to use**: You want goals relative to observed data. Percentile 50 = "average"; 80 = "top-20%"; 95 = "elite".

### MeanScalarization — Scaled Training Mean

Multiply the per-outcome training mean by a scalar. A scalar of 1.0 targets the average; scalars above 1.0 push toward above-average outcomes.

```python
from skrec.recommender.gcsl.inference.mean_scalarization import MeanScalarization

inference = MeanScalarization({
    "OUTCOME_engagement": 1.2,  # 20% above mean engagement
    "OUTCOME_revenue": 0.8      # 20% below mean revenue
})
```

**When to use**: You want to express goals as "X% above/below average".

### Out-of-Distribution Warnings

`PredefinedValue` and `MeanScalarization` emit a `UserWarning` when a goal falls outside the observed training range — the model has no signal for values it never saw. `PercentileValue` is bounded by construction and never triggers this warning.

```python
# This will warn: goal 99.0 is outside the training range [0.0, 5.0]
inference = PredefinedValue({"OUTCOME_revenue": 99.0})
```

## Writing a Custom Inference Method

Subclass `BaseInference` and implement `fit()` and `transform()`:

```python
from skrec.recommender.gcsl.inference.base_inference import BaseInference

class ClippedValue(BaseInference):
    """Goals clipped to [training_min, training_max]."""

    def __init__(self, goal_values):
        super().__init__()
        self.goal_values = goal_values

    def fit(self, interactions_df, outcome_cols):
        self._ranges = {
            col: (float(interactions_df[col].min()),
                  float(interactions_df[col].max()))
            for col in outcome_cols
        }
        self.outcome_cols_ = outcome_cols
        self._fitted = True
        return self

    def transform(self, interactions):
        self._check_fitted()
        interactions = interactions.copy()
        for col in self.outcome_cols_:
            lo, hi = self._ranges[col]
            interactions[col] = max(lo, min(hi, self.goal_values[col]))
        return interactions
```

`set_inference_method()` auto-calls `fit()` when the recommender is already trained, so custom methods work immediately:

```python
recommender.set_inference_method(
    ClippedValue({"OUTCOME_engagement": 999.0, "OUTCOME_revenue": -5.0})
)
recommendations = recommender.recommend(interactions=df, top_k=10)
```

## How It Works

### Training

`GcslRecommender` overrides `_process_outcome_columns()` to be a no-op. Where the base class drops `OUTCOME_*` columns, GCSL keeps them — so the model trains on:

```
X = [user_features, item_features, context_features, OUTCOME_engagement, OUTCOME_revenue]
y = OUTCOME (primary label)
```

### Inference

The inference method's `transform()` overwrites the outcome columns with goal values before scoring:

```
inference_df["OUTCOME_engagement"] = 0.8   # goal value
inference_df["OUTCOME_revenue"]    = 5.0   # goal value
→ scorer.score_items(inference_df)         # score all items given these goals
```

Items that historically co-occurred with the requested outcome profile score highest.

### Why This Works

The model learns a conditional distribution: `P(positive | user, item, goals)`. By conditioning on different goals at inference, you query: "which items are most consistent with achieving these outcomes for this user?" Change the goals, change the answer — same weights.

**Limitation**: the model learns *correlation*, not *causation*. Injecting `revenue=10` doesn't guarantee revenue=10 — it recommends items that historically appeared alongside high revenue.

## Scorer Compatibility

`GcslRecommender` works with **all non-sequential scorer types**:

| Scorer | Compatible | Notes |
|---|---|---|
| `UniversalScorer` | ✅ | Most common choice |
| `IndependentScorer` | ✅ | Per-item models |
| `MulticlassScorer` | ✅ | Items as classes |

**Not compatible**: `SequentialScorer` — use [SequentialRecommender](sequential.md) for sequence models.

## Evaluation

```python
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
import numpy as np

eval_data = {
    "logged_items": np.array([["item_A"], ["item_B"]]),
    "logged_rewards": np.array([[1.0], [0.5]])
}

ndcg = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.NDCG_AT_K,
    eval_top_k=5,
    score_items_kwargs={"interactions": inference_df, "users": users_df},
    eval_kwargs=eval_data
)
```

All standard evaluators and metrics are supported — see [Evaluation Guide](../user-guide/evaluation.md).

## Best Practices

### 1. Stay In-Distribution
Use `PercentileValue` when possible — goals are bounded by construction. If using `PredefinedValue`, keep goals within the observed training range.

### 2. Validate Feature Importance
After training, check that outcome columns have non-trivial feature importance. If they rank low, goal conditioning has no effect and a standard `RankingRecommender` would be simpler.

### 3. Use Real Per-Interaction Outcomes
Outcome columns that vary per interaction (dwell time, purchase amount, explicit rating) give the model more signal than static item properties. Static properties can be learned from item features alone.

### 4. Compare Goals Systematically
Use `set_inference_method()` to sweep goals and compare top-k lists or evaluation metrics without retraining. This is the core workflow GCSL enables.

## Common Issues

### Issue: Goals don't change recommendations

**Solution**: Check feature importance — if `OUTCOME_*` columns have low gain, the model ignores them. Increase `n_estimators` or add more varied training data.

### Issue: `NotFittedError` when calling `recommend()`

**Solution**: The inference method needs `fit()` before `transform()`. If you construct an inference method manually (outside `set_inference_method()`), call `fit()` explicitly:

```python
method = PredefinedValue({"OUTCOME_engagement": 0.8})
method.fit(training_df, ["OUTCOME_engagement"])
recommender.inference_method = method
```

Using `set_inference_method()` or passing the method to the constructor avoids this — `fit()` is called automatically.

### Issue: `recommend_online()` raises `NotImplementedError`

**Expected**: `recommend_online()` is not supported for GCSL because the single-row fast path bypasses goal injection. Use `recommend()` instead.

## Next Steps

- **[Example Notebook](../examples/index.md)** — End-to-end GCSL demo on MovieLens 1M
- **[RankingRecommender](ranking.md)** — For single-objective ranking
- **[Architecture Overview](../user-guide/architecture.md)** — How the 3-layer design enables composability
- **[Evaluation Guide](../user-guide/evaluation.md)** — Offline evaluation strategies
