# Sequential Recommendation (SASRec)

SASRec (Self-Attentive Sequential Recommendation) is a transformer-based model that captures long-range dependencies in a user's interaction history to predict the next item they are likely to engage with.

!!! tip "When to use SASRec vs NCF"
    Use SASRec when **order matters** — e-commerce browsing sessions, video watch history, music listening queues. Use [NCF](ncf.md) when interactions are unordered or when you need user/item feature integration.

## Overview

`SASRecClassifierEstimator` and `SASRecRegressorEstimator` implement the architecture from Kang & McAuley (ICDM 2018) with the following design choices:

- **Pre-norm transformer blocks** with multi-head self-attention and GELU feed-forward (4× expansion)
- **Right-aligned padding** — sequences are left-padded so the most recent item is always at the last position
- **Causal (autoregressive) masking** — each position attends only to itself and earlier positions
- **Weight-tied item embeddings** — input and output embeddings share the same weight matrix

### Variants

| Class | Loss | Use for |
|---|---|---|
| `SASRecClassifierEstimator` | Binary Cross-Entropy (BCE) | Implicit feedback (clicks, purchases, views) |
| `SASRecRegressorEstimator` | Mean Squared Error (MSE) | Explicit ratings (1–5 stars) |

## Quick Start

```python
from skrec.estimator.sequential.sasrec_estimator import SASRecClassifierEstimator
from skrec.scorer.sequential import SequentialScorer
from skrec.recommender.sequential.sequential_recommender import SequentialRecommender
from skrec.dataset.interactions_dataset import InteractionsDataset

# 1. Define the model
estimator = SASRecClassifierEstimator(
    hidden_units=50,               # Embedding and attention dimension
    num_blocks=2,                  # Number of transformer blocks
    num_heads=1,                   # Number of attention heads
    dropout_rate=0.2,
    num_negatives=1,               # Negatives sampled per positive during training
    max_len=50,                    # Maximum sequence length
    learning_rate=0.001,
    epochs=200,                    # Upper bound — early stopping may stop sooner
    batch_size=128,
    early_stopping_patience=5,     # Stop if val loss doesn't improve for 5 epochs
    restore_best_weights=True,     # Restore weights from best val-loss epoch
    random_state=42,
)

# 2. Wrap in scorer and recommender
scorer = SequentialScorer(estimator)
recommender = SequentialRecommender(scorer, max_len=50)

# 3. Train (optionally pass a validation split for early stopping)
recommender.train(
    items_ds=items_dataset,
    interactions_ds=interactions_dataset,         # Must contain TIMESTAMP column
    valid_interactions_ds=valid_interactions_ds,  # Optional: enables early stopping
)

# 4. Recommend
recs = recommender.recommend(
    interactions=recent_interactions_df,    # USER_ID, ITEM_ID, TIMESTAMP
    top_k=10,
)
```

## Input Data Requirements

The interactions dataset **must** contain a `TIMESTAMP` column. The `SequentialRecommender` sorts interactions by timestamp per user to form sequences before training.

```python
import pandas as pd

interactions_df = pd.DataFrame({
    "USER_ID":   ["u1", "u1", "u1", "u2", "u2"],
    "ITEM_ID":   ["i1", "i2", "i3", "i4", "i5"],
    "OUTCOME":   [1.0,  1.0,  1.0,  1.0,  1.0],
    "TIMESTAMP": [100,  200,  300,  150,  250],
})
```

Use the provided schema file for dataset validation:

```yaml
# skrec/dataset/required_schemas/interactions_schema_with_timestamp_training.yaml
columns:
  - name: USER_ID
    type: str
  - name: ITEM_ID
    type: str
  - name: OUTCOME
    type: float
  - name: TIMESTAMP
    type: int
```

## Architecture Parameters

```python
SASRecClassifierEstimator(
    hidden_units=50,                  # d_model: size of all embedding and attention layers
    num_blocks=2,                     # Number of stacked transformer blocks
    num_heads=1,                      # Attention heads (hidden_units must be divisible)
    dropout_rate=0.2,                 # Applied to attention weights and feed-forward layers
    num_negatives=1,                  # Negatives per positive per training step (≥1)
    max_len=50,                       # Sequence truncation length
    learning_rate=0.001,
    epochs=200,                       # Max epochs — early stopping may stop sooner
    batch_size=128,
    weight_decay=0.0,                 # L2 regularization (e.g. 1e-4 to prevent overfitting)
    early_stopping_patience=None,     # Epochs without val-loss improvement before stopping
    restore_best_weights=True,        # Restore best-epoch weights when early stopping fires
    random_state=42,
)
```

### Early Stopping

When `valid_interactions_ds` is passed to `SequentialRecommender.train()`, the estimator evaluates validation loss after each epoch. Set `early_stopping_patience` to automatically stop training once the val loss stops improving:

```python
estimator = SASRecClassifierEstimator(
    epochs=200,                   # Upper bound — acts as a safety cap
    early_stopping_patience=5,    # Stop if no improvement for 5 consecutive epochs
    restore_best_weights=True,    # Roll back to the best epoch's weights (default True)
)

recommender.train(
    items_ds=items_ds,
    interactions_ds=interactions_ds,
    valid_interactions_ds=valid_inter_ds,   # Single interaction per user (e.g. second-to-last)
)
```

**Preparing a validation split** (leave-last-two-out):

```python
# Sort by timestamp and rank from the end
interactions["rank"] = interactions.groupby("USER_ID").cumcount(ascending=False)

train_df = interactions.drop(columns=["rank"])           # ALL interactions (last item is final target)
valid_df = interactions[interactions["rank"] == 1]       # Second-to-last per user

interactions_ds = InteractionsDataset(data_location="train.csv")
valid_inter_ds  = InteractionsDataset(data_location="valid.csv")
```

!!! note "Raising ValueError"
    If `early_stopping_patience` is set but `valid_interactions_ds` is not passed to `train()`, a `ValueError` is raised at training time.

### Choosing `hidden_units` and `num_blocks`

| Catalogue size | Recommended config |
|---|---|
| < 10K items | `hidden_units=50, num_blocks=2` |
| 10K–100K items | `hidden_units=100, num_blocks=2` |
| > 100K items | `hidden_units=128, num_blocks=3, num_heads=2` |

### Choosing `max_len`

`max_len` controls how many past interactions the model can attend to. Longer sequences improve recall for power users but increase training time.

```python
max_len=50    # General default
max_len=200   # Power-user catalogues (e.g. MovieLens 1M)
max_len=20    # Short-session use cases (e.g. browsing sessions)
```

!!! note "max_len and SequentialRecommender"
    The `SequentialRecommender` owns sequence truncation and syncs `max_len` to the estimator at training time. Pass the same value to both to avoid a warning.

### Choosing `num_negatives`

```python
num_negatives=1    # Default — sufficient for most datasets
num_negatives=3    # More negatives, harder training signal
num_negatives=10   # Very large catalogues with many cold items
```

The negative loss is automatically normalized by `num_negatives` to maintain a 1:1 gradient ratio between positives and negatives at any value.

## Soft-Label Training (Explicit Ratings)

To use rating values as training signal with the classifier, set `OUTCOME` to a normalized score:

```python
# Normalize ratings to [0, 1]
interactions_df["OUTCOME"] = interactions_df["rating"] / 5.0

estimator = SASRecClassifierEstimator(
    hidden_units=50, num_blocks=2, num_heads=1,
    epochs=200
)
```

Alternatively, use `SASRecRegressorEstimator` with raw rating values:

```python
from skrec.estimator.sequential.sasrec_estimator import SASRecRegressorEstimator

estimator = SASRecRegressorEstimator(
    hidden_units=50, num_blocks=2, num_heads=1,
    epochs=200
)
```

## Performance on MovieLens 1M

Evaluated with leave-last-two-out split (test = last item, valid = second-to-last), sampled ranking (1 positive + 100 random negatives per user, 6,034 users):

| Setup | HR@10 | NDCG@10 | Config |
|---|---|---|---|
| Classifier, BCE, positives-only (rating ≥ 4), early stopping | **0.8842** | **0.6247** | `patience=5`, stopped epoch 114/200 |
| Classifier, soft-label BCE, all ratings | 0.8548 | 0.5728 | `epochs=200` |
| Regressor, MSE, all ratings | 0.8224 | 0.5574 | `epochs=200` |

Paper baseline (all interactions as implicit feedback, full-catalogue ranking, different split): HR@10 = 0.585

!!! note "Why our numbers exceed the paper's"
    Two reasons: (1) we train on **positive-only** interactions (rating ≥ 4), reducing noise; (2) we use **sampled ranking** (1 positive vs. 100 random negatives), which is easier than the paper's full-catalogue ranking (1 positive vs. ~3,706 items). The numbers are not directly comparable.

## Hyperparameter Tuning

### Learning Rate

```python
learning_rate=0.0001  # Conservative
learning_rate=0.001   # Default (recommended)
learning_rate=0.005   # Aggressive (watch for instability)
```

### Regularization

```python
dropout_rate=0.0   # No dropout (use for small datasets / overfit tests)
dropout_rate=0.2   # Recommended for production
dropout_rate=0.5   # Heavy regularization for very large models

weight_decay=0.0   # Default
weight_decay=1e-4  # Light L2 regularization
weight_decay=1e-3  # Strong L2 regularization
```

### Epochs and Early Stopping

`epochs` is always an upper bound. When `early_stopping_patience` is set, training may stop much earlier:

```python
epochs=50    # Quick experiments / unit tests
epochs=200   # Standard training (use with early_stopping_patience=5)
epochs=500   # When dataset is small and model needs more iterations

# Recommended production config
estimator = SASRecClassifierEstimator(
    epochs=200,
    early_stopping_patience=5,
    restore_best_weights=True,
)
```

## When to Use SASRec

✅ **Use SASRec when:**

- Interaction order matters (browsing sessions, watch history, purchase sequences)
- Users have medium-to-long histories (≥5 interactions)
- You want state-of-the-art sequential ranking without a GPU

❌ **Don't use SASRec when:**

- Interactions are unordered or timestamp is unavailable
- Users have very short histories (≤2 interactions) — use [NCF](ncf.md) or [Collaborative Filtering](collaborative-filtering.md)
- You need user/item side features — SASRec is ID-based only

## Comparison with Other Estimators

| Feature | SASRec | NCF | MatrixFactorization |
|---|---|---|---|
| **Captures item order** | ✅ Yes | ❌ No | ❌ No |
| **Transformer architecture** | ✅ Yes | ❌ No | ❌ No |
| **User/item side features** | ❌ No | ✅ Yes | ❌ No |
| **PyTorch required** | ✅ Yes | ✅ Yes | ❌ No (NumPy-only) |
| **Training speed** | 🐢 Moderate | 🐢 Moderate | 🚀 Fast |
| **Best for** | **Sequential patterns** | Collaborative filtering | CF without GPU |

## Troubleshooting

### Model not learning

1. Ensure `num_negatives ≥ 1` (default is 1; setting it to 0 disables negative sampling and degrades BCE to trivial loss)
2. Increase `epochs` — SASRec typically needs 100–300 epochs on small datasets
3. Reduce `dropout_rate` to 0.0 to confirm the model can overfit before adding regularization

### Poor ranking quality

1. Check that `TIMESTAMP` ordering is correct — wrong order produces near-random sequences
2. Verify the `SequentialRecommender.max_len` and `SASRecClassifierEstimator.max_len` match
3. Filter to positive interactions only (`OUTCOME ≥ threshold`) rather than using all interactions

### Memory / performance

1. Reduce `max_len` — attention is O(max_len²) in memory
2. Reduce `batch_size` if GPU OOM
3. Set `verbose=0` to suppress per-epoch logging in production

## References

- Kang, W. C., & McAuley, J. (2018). [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). In ICDM 2018.

## Next Steps

- **[SequentialRecommender](../recommender-types/sequential.md)** — Full guide to the recommender layer
- **[HRNN Guide](hrnn.md)** — Session-aware alternative for session-structured interaction data
- **[Training Guide](training.md)** — General training patterns
- **[Evaluation Guide](evaluation.md)** — How to measure ranking quality
