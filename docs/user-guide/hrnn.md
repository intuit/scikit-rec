# Hierarchical Session Recommendation (HRNN)

HRNN (Hierarchical Recurrent Neural Network) is a session-aware model that uses a two-level GRU hierarchy to capture both within-session dynamics and long-term user preferences across sessions.

!!! tip "When to use HRNN vs SASRec"
    Use HRNN when your data has **natural session boundaries** — e-commerce visits, app sessions, TV viewing sittings. HRNN explicitly models how user preferences evolve from one session to the next. Use [SASRec](sasrec.md) when interactions form a single continuous history without clear session structure.

## Overview

`HRNNClassifierEstimator` and `HRNNRegressorEstimator` implement the architecture from Quadrana et al. (RecSys 2017) with the following design:

- **Session GRU**: encodes the sequence of items within a single session
- **User GRU**: evolves user state across sessions; its output seeds the session GRU's initial hidden state for the next session
- **Right-aligned padding** in both dimensions — most recent session at the last index, most recent item at the last position within each session
- **No user embedding table** — user representation is derived dynamically from session history at inference time

### Variants

| Class | Loss | Use for |
|---|---|---|
| `HRNNClassifierEstimator` | Binary Cross-Entropy (BCE) | Implicit feedback or binary labels |
| `HRNNRegressorEstimator` | Mean Squared Error (MSE) | Explicit ratings or continuous rewards |

## Quick Start

```python
from skrec.estimator.sequential.hrnn_estimator import HRNNClassifierEstimator
from skrec.scorer.hierarchical import HierarchicalScorer
from skrec.recommender.sequential.hierarchical_recommender import HierarchicalSequentialRecommender
from skrec.dataset.interactions_dataset import InteractionsDataset

# 1. Define the model
estimator = HRNNClassifierEstimator(
    hidden_units=50,               # GRU hidden size and item embedding dimension
    num_layers=1,                  # GRU depth
    dropout_rate=0.2,
    num_negatives=1,               # Negatives sampled per positive during training
    max_sessions=10,               # Max past sessions retained per user
    max_session_len=20,            # Max items per session
    learning_rate=0.001,
    epochs=200,                    # Upper bound — early stopping may stop sooner
    batch_size=128,
    early_stopping_patience=5,     # Stop if val loss doesn't improve for 5 epochs
    restore_best_weights=True,     # Restore weights from best val-loss epoch
    random_state=42,
)

# 2. Wrap in scorer and recommender
scorer = HierarchicalScorer(estimator)
recommender = HierarchicalSequentialRecommender(
    scorer,
    max_sessions=10,
    max_session_len=20,
    session_timeout_minutes=30,   # Gap threshold for automatic session splitting
)

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

The interactions dataset **must** contain a `TIMESTAMP` column. Session boundaries are detected automatically from timestamp gaps unless you supply an explicit `SESSION_ID` column.

```python
import pandas as pd

interactions_df = pd.DataFrame({
    "USER_ID":   ["u1", "u1", "u1", "u1"],
    "ITEM_ID":   ["i1", "i2", "i3", "i4"],
    "OUTCOME":   [1.0,  1.0,  1.0,  1.0],
    "TIMESTAMP": [100,  200,  5000, 5100],   # gap between i2→i3 starts new session
})
```

Use the shared schema for dataset validation:

```python
InteractionsDataset(
    data_location="interactions.csv",
    client_schema_path="skrec/dataset/required_schemas/interactions_schema_with_timestamp_training.yaml",
)
```

### Explicit Session IDs

If your data already has session boundaries defined, pass a `SESSION_ID` column — it takes priority over timestamp-based detection:

```python
interactions_df["SESSION_ID"] = interactions_df["session"]
```

## Session Detection

`HierarchicalSequentialRecommender` detects session boundaries before training and inference (in priority order):

1. **Explicit** — `SESSION_ID` column present in the interactions DataFrame
2. **Implicit** — `session_timeout_minutes` set; a new session starts whenever the gap between consecutive interactions exceeds the timeout

```python
# Explicit session boundaries
recommender = HierarchicalSequentialRecommender(
    scorer, max_sessions=10, max_session_len=20
    # SESSION_ID column in data takes priority
)

# Implicit — 30-minute timeout (default)
recommender = HierarchicalSequentialRecommender(
    scorer, max_sessions=10, max_session_len=20,
    session_timeout_minutes=30,
)

# Implicit — 24-hour timeout (e.g. daily sessions)
recommender = HierarchicalSequentialRecommender(
    scorer, max_sessions=15, max_session_len=30,
    session_timeout_minutes=1440,
)
```

!!! note "Numeric vs. datetime timestamps"
    Numeric timestamps (int/float) are treated as **Unix epoch seconds** — the standard for datasets like MovieLens and Amazon Reviews. Datetime or string timestamps are parsed with `pd.to_datetime`. Do **not** pass Unix epoch seconds as datetime strings, as pandas would interpret the integers as nanoseconds.

## Architecture Parameters

```python
HRNNClassifierEstimator(
    hidden_units=50,                  # GRU hidden size and item embedding dimension
    num_layers=1,                     # GRU depth (1 is standard; >1 adds inter-layer dropout)
    dropout_rate=0.2,                 # Applied to item embeddings and GRU inter-layer connections
    num_negatives=1,                  # Negatives per positive per training step (≥1)
    max_sessions=10,                  # Sessions retained per user (synced from recommender at train time)
    max_session_len=20,               # Items retained per session (synced from recommender at train time)
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

When `valid_interactions_ds` is passed to `HierarchicalSequentialRecommender.train()`, the estimator evaluates validation loss after each epoch. Set `early_stopping_patience` to stop training automatically when the val loss plateaus:

```python
estimator = HRNNClassifierEstimator(
    epochs=200,
    early_stopping_patience=5,    # Stop if no improvement for 5 consecutive epochs
    restore_best_weights=True,    # Roll back to the best epoch's weights (default True)
)

recommender.train(
    items_ds=items_ds,
    interactions_ds=interactions_ds,
    valid_interactions_ds=valid_inter_ds,   # Single interaction per user (e.g. second-to-last positive)
)
```

**Preparing a validation split**:

```python
interactions["rank"] = interactions.groupby("USER_ID").cumcount(ascending=False)

train_df = interactions.drop(columns=["rank"])       # ALL interactions
valid_df = interactions[interactions["rank"] == 1]   # Second-to-last per user

interactions_ds = InteractionsDataset(data_location="train.csv")
valid_inter_ds  = InteractionsDataset(data_location="valid.csv")
```

!!! note "HRNN on ML-1M: early stopping may not fire"
    On datasets where the model improves monotonically throughout training (e.g. ML-1M with `epochs=100`), `early_stopping_patience` will never trigger. This is expected — it means the model kept improving and more epochs would help further. Consider increasing `epochs` in that case.

!!! note "max_sessions / max_session_len ownership"
    `HierarchicalSequentialRecommender` owns these parameters. It syncs them to the estimator at training time and logs a warning if they differ.

### Choosing `max_sessions` and `max_session_len`

| Use case | Suggested config |
|---|---|
| E-commerce (short, frequent sessions) | `max_sessions=10, max_session_len=15` |
| Video streaming (longer sessions) | `max_sessions=10, max_session_len=30` |
| Daily app sessions | `max_sessions=15, max_session_len=30` |
| ML-1M benchmark | `max_sessions=15, max_session_len=30` |

## Training Modes

HRNN supports four data configurations — only the data preparation cell changes; the estimator, scorer, and recommender setup is identical.

| Mode | OUTCOME values | Estimator | Notes |
|---|---|---|---|
| **Positives only** | All `1.0` | Classifier | Filter to engaged interactions before training |
| **Binary 0/1** | `1.0` for positive, `0.0` for negative | Classifier | E.g. rating ≥ 4 → 1.0, rating ≤ 2 → 0.0 |
| **Soft-label** | Normalized to `[0, 1]` | Classifier | E.g. `rating / 5.0` |
| **Continuous** | Raw float values | Regressor | Revenue, time-spent, raw ratings |

```python
# Mode 1: Positives only
interactions_df["OUTCOME"] = 1.0

# Mode 2: Binary 0/1
interactions_df["OUTCOME"] = interactions_df["rating"].apply(
    lambda r: 1.0 if r >= 4 else (0.0 if r <= 2 else None)
)
interactions_df = interactions_df.dropna(subset=["OUTCOME"])

# Mode 3: Soft-label BCE
interactions_df["OUTCOME"] = interactions_df["rating"] / 5.0

# Mode 4: Continuous (use HRNNRegressorEstimator)
from skrec.estimator.sequential.hrnn_estimator import HRNNRegressorEstimator
estimator = HRNNRegressorEstimator(hidden_units=50, ...)
```

## Performance on MovieLens 1M

Evaluated with leave-last-two-out split on positives (rating ≥ 4), sampled ranking (1 positive + 100 random negatives per user, 6,038 users):

| Setup | HR@10 | NDCG@10 | Config |
|---|---|---|---|
| Classifier, BCE, positives-only (rating ≥ 4), with val loss logging | **0.6812** | **0.3969** | `epochs=100, patience=5` — val loss improved all 100 epochs |

Config: `hidden_units=50, num_layers=1, max_sessions=15, max_session_len=30, session_timeout_minutes=1440, epochs=100, early_stopping_patience=5`

Val loss curve: 1.0033 → 0.7271 across 100 epochs (monotonically decreasing — patience never triggered).

!!! note "SASRec vs HRNN on ML-1M"
    HRNN achieves HR@10=0.68 vs SASRec's 0.88 on ML-1M. This is expected: ML-1M users have long flat histories (100+ interactions) that don't naturally segment into short sessions. HRNN's two-level GRU is optimised for short, frequent sessions — it closes the gap on e-commerce and app session datasets.

## Hyperparameter Tuning

### Session boundaries

```python
session_timeout_minutes=30     # Standard web session (Google Analytics default)
session_timeout_minutes=1440   # Daily sessions (24 hours)
session_timeout_minutes=None   # Requires explicit SESSION_ID in data
```

### Regularization

```python
dropout_rate=0.0   # No dropout (use for small datasets / overfit tests)
dropout_rate=0.2   # Recommended default
dropout_rate=0.5   # Heavy regularization for large models

weight_decay=0.0   # Default
weight_decay=1e-4  # Light L2 regularization
```

### Epochs and Early Stopping

`epochs` is an upper bound. On datasets where the model keeps improving, early stopping won't fire — increase `epochs` to let it run longer:

```python
epochs=50     # Quick experiments
epochs=100    # Standard training
epochs=200    # With early stopping as a safety cap
epochs=500    # Small datasets or slow-converging configs

learning_rate=0.001   # Default (recommended)
learning_rate=0.0005  # More conservative

# Recommended production config
estimator = HRNNClassifierEstimator(
    epochs=200,
    early_stopping_patience=5,
    restore_best_weights=True,
)
```

## When to Use HRNN

✅ **Use HRNN when:**

- Interactions naturally group into sessions (shopping visits, app opens, TV sittings)
- You want to model how user preferences shift between sessions
- You have timestamp data and sessions can be inferred from gaps

❌ **Don't use HRNN when:**

- Interactions have no natural session structure — use [SASRec](sasrec.md)
- No `TIMESTAMP` available — use [RankingRecommender](../recommender-types/ranking.md) with [NCF](ncf.md)

## Comparison with SASRec

| Feature | HRNN | SASRec |
|---|---|---|
| **Session-aware** | ✅ Yes — explicit two-level hierarchy | ❌ No — single flat sequence |
| **Architecture** | GRU + GRU | Transformer (self-attention) |
| **Requires sessions** | ✅ Yes (detected automatically) | ❌ No |
| **Long-range dependencies** | Via user GRU across sessions | Via attention over full sequence |
| **Training speed** | 🐢 Moderate | 🐢 Moderate |
| **PyTorch required** | ✅ Yes | ✅ Yes |

## Troubleshooting

### All users appear to have only one session

The most common cause is passing Unix epoch second timestamps to `pd.to_datetime`, which interprets integers as nanoseconds — making all inter-item gaps appear to be microseconds. HRNN automatically handles numeric timestamps correctly (direct second comparison), but verify your timestamp column type:

```python
print(interactions_df["TIMESTAMP"].dtype)   # should be int64 or float64
```

### Model not learning

1. Ensure `num_negatives ≥ 1` — setting it to 0 means unseen items receive no gradient, producing near-random scores
2. Verify `session_timeout_minutes` produces multiple sessions per user (check with `len(sessions_df["SESSION_SEQUENCES"].iloc[0])`)
3. Reduce `dropout_rate` to 0.0 and increase `epochs` to confirm the model can overfit before adding regularization

### Poor ranking quality

1. When using binary 0/1 labels, evaluate on last **positive** interaction per user (not last overall)
2. Increase `max_sessions` and `max_session_len` if users have rich histories being truncated
3. Try `num_negatives=3` for harder training signal on large catalogues

## References

- Quadrana, M., Cremonesi, P., & Jannach, D. (2017). [Personalizing Session-Based Recommendations with Hierarchical Recurrent Neural Networks](https://dl.acm.org/doi/10.1145/3109859.3109896). In RecSys 2017.

## Next Steps

- **[HierarchicalSequentialRecommender](../recommender-types/hierarchical.md)** — Full guide to the recommender layer
- **[SASRec Guide](sasrec.md)** — Compare with the transformer-based sequential model
- **[Training Guide](training.md)** — General training patterns
- **[Evaluation Guide](evaluation.md)** — How to measure ranking quality
