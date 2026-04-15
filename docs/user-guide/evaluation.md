# Evaluation Guide

This guide covers how to evaluate recommendation models using various evaluation strategies and metrics.

## Quick Start

```python
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
import numpy as np

# Prepare ground truth
eval_data = {
    "logged_items": np.array([["item_A"], ["item_B"]]),
    "logged_rewards": np.array([[1.0], [0.5]])
}

# Evaluate
ndcg = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.NDCG_AT_K,
    eval_top_k=5,
    score_items_kwargs={"interactions": interactions_df, "users": users_df},
    eval_kwargs=eval_data
)

print(f"NDCG@5: {ndcg:.4f}")
```

## Evaluation Method Overview

The `evaluate()` method accepts:

- **`eval_type`**: A `RecommenderEvaluatorType` that determines the evaluation technique (SIMPLE, IPS, DR, etc.)
- **`metric_type`**: A `RecommenderMetricType` that specifies which metric to calculate (NDCG, Precision, ROC-AUC, etc.)
- **`score_items_kwargs`**: Parameters passed to the internal `score_items()` method, including:
  - `interactions`: DataFrame with interaction context features
  - `users`: DataFrame with user features
    - For **embedding-based models** (a `BaseEmbeddingEstimator` subclass with `UniversalScorer`), this can include pre-computed embeddings in the `USER_EMBEDDING_NAME` column for real-time inference evaluation
- **`eval_kwargs`**: Ground truth data including `logged_items` and `logged_rewards`, and optionally `logging_proba` for off-policy evaluation. Pass `None` or an empty dict `{}` to reuse cached modified rewards when still valid — see [Caching and Performance](#caching-and-performance).

**Learn more**: [Inference Guide](inference.md) for details on `score_items()` parameters

## Contextual bandits and `evaluate()` {: #contextual-bandits-and-evaluate }

[`ContextualBanditsRecommender`](../recommender-types/bandits.md#evaluation) applies a **bandit strategy** on top of scorer outputs. Offline `evaluate()` is **policy-aligned** wherever that strategy shapes rankings or target probabilities (same idea as `recommend()`):

- Configure the strategy with **`set_strategy()`** (or in the constructor) **before** calling `evaluate()` when the code path ranks via the policy. If you omit this, you may see `RuntimeError: Strategy not set. Call set_strategy() before recommend().` This applies in particular to **`STATIC_ACTION`** with **non-probabilistic** evaluators (e.g. Simple, ReplayMatch), where full rankings are built through `_recommend_from_scores`.
- Metrics describe **deployed policy behavior**, not necessarily “sort items by raw score.” For **base-model-only** ranking metrics, use [`RankingRecommender`](../recommender-types/ranking.md) (or another non-bandit recommender) with the same underlying scorer.

See the [bandits guide](../recommender-types/bandits.md#evaluation) for examples and off-policy (IPS / DR / SNIPS) notes.

## Available Evaluators

### 1. SimpleEvaluator (On-Policy)
Standard evaluation assuming recommendations match logging policy.

```python
result = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=metric_type,
    eval_top_k=5,
    score_items_kwargs={"interactions": interactions_df, "users": users_df},
    eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards}
)
```

**Use when**: Evaluating on data collected from the same recommender policy.

### 2. ReplayMatchEvaluator
Replay-based evaluation that only considers recommendations matching logged actions.

```python
result = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.REPLAY_MATCH,
    ...
)
```

**Use when**: You want conservative estimates by only evaluating on matched recommendations.

### 3. IPSEvaluator (Inverse Propensity Scoring)
Off-policy evaluation using propensity scores.

```python
eval_data = {
    "logged_items": logged_items,
    "logged_rewards": logged_rewards,
    "logging_proba": logging_probabilities  # Required!
}

result = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.IPS,
    eval_kwargs=eval_data,
    ...
)
```

**Use when**: Evaluating on data collected from a different policy. Requires logging probabilities.

### 4. DREvaluator (Doubly Robust)
Combines direct method and IPS for robust off-policy evaluation.

```python
result = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.DR,
    eval_kwargs=eval_data,
    ...
)
```

**Use when**: Want robust off-policy estimates with lower variance than IPS.

### 5. SNIPSEvaluator (Self-Normalized IPS)
Self-normalized variant of IPS for lower variance.

```python
result = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SNIPS,
    eval_kwargs=eval_data,
    ...
)
```

**Use when**: IPS estimates have high variance.

### 6. PolicyWeightedEvaluator
Policy-weighted evaluation for off-policy scenarios.

```python
result = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.POLICY_WEIGHTED,
    eval_kwargs=eval_data,
    ...
)
```

## Available Metrics

### Ranking Metrics

- **`PRECISION_AT_K`**: Precision@k - Fraction of relevant items in top-k
- **`NDCG_AT_K`**: Normalized Discounted Cumulative Gain@k
- **`MAP_AT_K`**: Mean Average Precision@k
- **`MRR_AT_K`**: Mean Reciprocal Rank@k

```python
precision = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.PRECISION_AT_K,
    eval_top_k=5,
    ...
)
```

### Classification Metrics

- **`ROC_AUC`**: ROC-AUC score
- **`PR_AUC`**: Precision-Recall AUC

```python
roc_auc = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.ROC_AUC,
    eval_top_k=5,  # Still required but not used
    ...
)
```

### Reward Metrics

- **`AVERAGE_REWARD_AT_K`**: Expected reward in top-k

```python
reward = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.AVERAGE_REWARD_AT_K,
    eval_top_k=5,
    ...
)
```

## Evaluation Parameters

### Key Parameters

- **`eval_type`**: Which evaluator to use (SIMPLE, IPS, DR, etc.)
- **`metric_type`**: Which metric to calculate (NDCG, Precision, etc.)
- **`eval_top_k`**: Top-k cutoff for ranking metrics
- **`temperature`**: Temperature for softmax conversion (default: 1.0)
- **`score_items_kwargs`**: Arguments for `score_items()` method
- **`eval_kwargs`**: Ground truth data (logged_items, logged_rewards, etc.)

### Ground Truth Data

```python
eval_kwargs = {
    "logged_items": np.array([["item_A"], ["item_B"]]),      # Actual items
    "logged_rewards": np.array([[1.0], [0.5]]),              # Actual rewards
    "logging_proba": np.array([[0.7], [0.3]])                 # Optional: for off-policy
}
```

## Complete Example

```python
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
import pandas as pd
import numpy as np

# Prepare test data
interactions_df = pd.DataFrame({"USER_ID": ["user_1", "user_2"]})
users_df = pd.DataFrame({
    "USER_ID": ["user_1", "user_2"],
    "age": [25, 35],
    "income": [50000, 75000]
})

# Ground truth
eval_data = {
    "logged_items": np.array([["item_3"], ["item_2"]]),
    "logged_rewards": np.array([[1.0], [1.0]])
}

# Evaluate multiple metrics efficiently:
# Pass score_items_kwargs and eval_kwargs on the first call to compute scores
# and modified rewards. Omit both on subsequent calls — only the metric
# calculation reruns, the expensive intermediate results are reused.
metrics = [
    (RecommenderMetricType.NDCG_AT_K, "NDCG@5"),
    (RecommenderMetricType.PRECISION_AT_K, "Precision@5"),
    (RecommenderMetricType.MAP_AT_K, "MAP@5"),
    (RecommenderMetricType.ROC_AUC, "ROC-AUC"),
]

score_kwargs = {"interactions": interactions_df, "users": users_df}
for i, (metric_type, name) in enumerate(metrics):
    score = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=metric_type,
        eval_top_k=5,
        score_items_kwargs=score_kwargs if i == 0 else None,  # score once
        eval_kwargs=eval_data if i == 0 else None,            # modified rewards once
    )
    print(f"{name}: {score:.4f}")
```

## Evaluating Embedding-Based Models

When evaluating recommenders that use **`BaseEmbeddingEstimator`** subclasses (e.g., `NeuralFactorizationEstimator`, `ContextualizedTwoTowerEstimator`) with **`UniversalScorer`**, you can evaluate in two modes:

### Batch Evaluation Mode

```python
# Evaluator uses internally learned user embeddings
result = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.NDCG_AT_K,
    eval_top_k=5,
    score_items_kwargs={
        "interactions": interactions_df,
        # users=None - model uses internal embeddings
    },
    eval_kwargs=eval_data
)
```

### Real-Time Inference Evaluation Mode

```python
from skrec.constants import USER_EMBEDDING_NAME

# Prepare users DataFrame with pre-computed embeddings
users_df = pd.DataFrame({
    "USER_ID": ["user_1", "user_2"],
    USER_EMBEDDING_NAME: [user_emb_1, user_emb_2],  # Pre-computed embeddings
    # Optionally include other user features
})

# Evaluate using pre-computed embeddings (simulates real-time inference)
result = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.NDCG_AT_K,
    eval_top_k=5,
    score_items_kwargs={
        "interactions": interactions_df,
        "users": users_df  # Contains pre-computed embeddings
    },
    eval_kwargs=eval_data
)
```

**Learn more**: [Embedding Estimators Guide](estimators.md#embedding-estimators) | [Inference Guide](inference.md#embedding-based-inference-real-time-with-pre-computed-embeddings)

## Caching and Performance

`evaluate()` internally caches intermediate results to avoid redundant computation. Understanding when each cache is invalidated lets you structure calls efficiently.

### What gets cached and when it's recomputed

| Cached value | Recomputed when |
|---|---|
| Recommendation scores | `score_items_kwargs` is provided |
| Modified rewards | **Non-empty** `eval_kwargs` is provided, or scores change, or `eval_type`/`eval_factory_kwargs` change |
| Metric result | Always recomputed (cheap pure function) |

Concretely:

- **Only `metric_type` or `eval_top_k` changed** → only the final metric calculation reruns. Scores and modified rewards are reused.
- **Non-empty `eval_kwargs` provided** → modified rewards are recomputed. (`None` and `{}` are treated the same for reuse; there is no identity check on the data when you do pass a non-empty mapping.)
- **`score_items_kwargs` provided** → scores and modified rewards are both recomputed.
- **`eval_type` or `eval_factory_kwargs` changed** → a new evaluator is created and modified rewards are recomputed.
- **`recommender.clear_evaluation_cache()`** → clears scores, ranks, modified rewards, and the evaluator handle. The next `evaluate()` call must supply `score_items_kwargs` again (and non-empty `eval_kwargs` when modified rewards are missing).

### Sweeping metrics or top-k values

The most common pattern where caching matters is computing multiple metrics or top-k cutoffs over the same data. Pass `score_items_kwargs` and `eval_kwargs` only on the first call; on subsequent calls pass `eval_kwargs=None` or `eval_kwargs={}` to reuse cached modified rewards:

```python
eval_data = {"logged_items": logged_items, "logged_rewards": logged_rewards}
score_kwargs = {"interactions": interactions_df, "users": users_df}

# First call: scores computed, modified rewards computed, metric computed
ndcg = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.NDCG_AT_K,
    eval_top_k=10,
    score_items_kwargs=score_kwargs,
    eval_kwargs=eval_data,
)

# Subsequent calls: only the metric calculation reruns
precision = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.PRECISION_AT_K,
    eval_top_k=10,
    # score_items_kwargs and eval_kwargs omitted — caches reused
)

map_score = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.MAP_AT_K,
    eval_top_k=5,
    # different top_k is fine — still reuses scores and modified rewards
)
```

### Switching evaluator type

Switching `eval_type` invalidates the modified rewards cache. You must provide `eval_kwargs` again for the new evaluator:

```python
# Compute with SIMPLE
simple_score = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.NDCG_AT_K,
    eval_top_k=10,
    score_items_kwargs=score_kwargs,
    eval_kwargs=eval_data,
)

# Switching to IPS — must provide eval_kwargs (and logging_proba)
ips_score = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.IPS,
    metric_type=RecommenderMetricType.NDCG_AT_K,
    eval_top_k=10,
    eval_kwargs={**eval_data, "logging_proba": logging_proba},
    # score_items_kwargs can be omitted — scores are reused
)
```

---

## Best Practices

### 1. Use Multiple Metrics
```python
# Different metrics capture different aspects
metrics_to_evaluate = [
    RecommenderMetricType.NDCG_AT_K,      # Ranking quality
    RecommenderMetricType.PRECISION_AT_K,  # Relevance
    RecommenderMetricType.ROC_AUC          # Classification performance
]
```

### 2. Choose Right Evaluator
- **On-policy data** (same recommender) → SIMPLE
- **Off-policy data** (different recommender) → IPS, DR, or SNIPS
- **Conservative estimate** → REPLAY_MATCH

### 3. Off-Policy Evaluation
```python
# Always include logging probabilities for off-policy
eval_data = {
    "logged_items": logged_items,
    "logged_rewards": logged_rewards,
    "logging_proba": logging_proba  # Critical!
}
```

### 4. Temporal Validation
```python
# Split by time for realistic evaluation
train_data = data[data['timestamp'] < cutoff]
test_data = data[data['timestamp'] >= cutoff]
```

## Common Issues

### Issue: "logged_items must be provided"

**Solution**: Include ground truth in `eval_kwargs`:
```python
eval_kwargs={"logged_items": ..., "logged_rewards": ...}
```

### Issue: Off-policy estimates seem biased

**Solution**: 
- Check logging probabilities are correct
- Use DR or SNIPS for lower variance
- Collect more data

### Issue: Switching `eval_type` raises "eval_kwargs is required"

Changing `eval_type` invalidates the modified rewards cache. You must supply `eval_kwargs` for the new evaluator to recompute them.

```python
# ✅ Correct — provide eval_kwargs when switching eval_type
ips_score = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.IPS,
    metric_type=metric_type,
    eval_top_k=5,
    eval_kwargs={**eval_data, "logging_proba": logging_proba},
)
```

## Next Steps

- **[Metrics Guide](../user-guide/architecture.md)** - Deep dive into metrics
- **[Evaluator Guide](../user-guide/architecture.md)** - Deep dive into evaluators
- **[Training Guide](training.md)** - Train models for evaluation
- **[Production Guide](../advanced/production.md)** - Deploy evaluated models

