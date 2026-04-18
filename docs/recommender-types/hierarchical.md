# HierarchicalSequentialRecommender

`HierarchicalSequentialRecommender` wraps a `HierarchicalScorer` + HRNN estimator and adds session boundary detection. It converts raw interaction logs into per-user session sequences before training and inference.

## Overview

```
HierarchicalSequentialRecommender
    → HierarchicalScorer
        → HRNNClassifierEstimator  (or HRNNRegressorEstimator)
```

**Owns:**

- Detecting session boundaries (from `SESSION_ID` column or timestamp gaps)
- Sorting interactions by `TIMESTAMP` per user
- Grouping items into sessions and truncating to `max_sessions` / `max_session_len`
- Handling the presence/absence of `OUTCOME` (training vs. inference)

**Does not own:**

- Tensor construction (estimator)
- Item vocabulary and score matrix (scorer)

## Quick Start

```python
from skrec.estimator.sequential.hrnn_estimator import HRNNClassifierEstimator
from skrec.scorer.hierarchical import HierarchicalScorer
from skrec.recommender.sequential.hierarchical_recommender import HierarchicalSequentialRecommender
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset

estimator = HRNNClassifierEstimator(
    hidden_units=50, num_layers=1,
    dropout_rate=0.2, epochs=200,
    max_sessions=10, max_session_len=20,
    early_stopping_patience=5,     # Optional: stop when val loss plateaus
    restore_best_weights=True,
)
scorer = HierarchicalScorer(estimator)
recommender = HierarchicalSequentialRecommender(
    scorer,
    max_sessions=10,
    max_session_len=20,
    session_timeout_minutes=30,
)

# Train (with optional validation split for early stopping)
recommender.train(
    items_ds=ItemsDataset(data_location="items.csv"),
    interactions_ds=InteractionsDataset(data_location="interactions.csv"),
    valid_interactions_ds=InteractionsDataset(data_location="valid.csv"),  # Optional
)

# Recommend
recs = recommender.recommend(
    interactions=recent_interactions_df,   # USER_ID, ITEM_ID, TIMESTAMP
    top_k=10,
)
```

## Training Data Requirements

| Column | Type | Required |
|---|---|---|
| `USER_ID` | str / int | ✅ |
| `ITEM_ID` | str / int | ✅ |
| `TIMESTAMP` | int / float / datetime | ✅ |
| `OUTCOME` | float | ✅ at training, optional at inference |
| `SESSION_ID` | any | Optional — overrides timeout-based detection |

Provide the built-in schema to enable validation:

```python
InteractionsDataset(
    data_location="interactions.csv",
    client_schema_path="skrec/dataset/required_schemas/interactions_schema_with_timestamp_training.yaml",
)
```

## Session Building

`_build_session_sequences` is the core transformation step:

1. **Sort** all rows by `(USER_ID, TIMESTAMP)` ascending
2. **Detect session boundaries** — via `SESSION_ID` column (explicit) or timestamp gap > `session_timeout_minutes` (implicit)
3. **Group** items into `(user, session)` pairs → one list per session per user
4. **Aggregate** sessions into a list-of-lists per user
5. **Truncate** — keep the most recent `max_sessions` sessions; within each session keep the most recent `max_session_len` items
6. **Include `SESSION_OUTCOMES`** only when `OUTCOME` is present (training mode)

```python
# Training mode output (one row per user):
sessions_df columns: USER_ID, SESSION_SEQUENCES, SESSION_OUTCOMES
# SESSION_SEQUENCES: List[List[str]]  — oldest session first, oldest item first
# SESSION_OUTCOMES:  List[List[float]]

# Inference mode output:
sessions_df columns: USER_ID, SESSION_SEQUENCES
```

## Session Detection

Two strategies, tried in order:

### 1. Explicit SESSION_ID

If `SESSION_ID` is present in the interactions DataFrame, it is used directly as the session grouping key. No timestamp gap computation is performed.

```python
interactions_df["SESSION_ID"] = interactions_df["visit_id"]
```

### 2. Implicit timeout

When no `SESSION_ID` column is present, sessions are split on inactivity gaps:

```python
# 30-minute timeout (web sessions)
recommender = HierarchicalSequentialRecommender(scorer, ..., session_timeout_minutes=30)

# 24-hour timeout (daily app sessions)
recommender = HierarchicalSequentialRecommender(scorer, ..., session_timeout_minutes=1440)
```

!!! note "Numeric timestamps"
    Integer/float timestamps are treated as **Unix epoch seconds** and compared directly. Datetime or string timestamps are parsed with `pd.to_datetime`. Do not rely on `pd.to_datetime` for integer Unix timestamps — it interprets them as nanoseconds.

## `max_sessions` and `max_session_len` Parameters

Both parameters must match between `HierarchicalSequentialRecommender` and the estimator. The recommender always wins — it syncs its values to the estimator at training time and logs a warning if they differ:

```python
# Correct: same values on both
estimator = HRNNClassifierEstimator(..., max_sessions=10, max_session_len=20)
recommender = HierarchicalSequentialRecommender(scorer, max_sessions=10, max_session_len=20)

# Also works: recommender values override, with a warning
estimator = HRNNClassifierEstimator(..., max_sessions=5, max_session_len=10)  # overridden
recommender = HierarchicalSequentialRecommender(scorer, max_sessions=10, max_session_len=20)
```

## Validation Split and Early Stopping

Pass `valid_interactions_ds` to `train()` to enable per-epoch val-loss monitoring. The recommender builds session sequences from all interactions except the last per user, so the last target in each validation session sequence is the validation item.

```python
# Leave-last-two-out split
interactions["rank"] = interactions.groupby("USER_ID").cumcount(ascending=False)
train_df = interactions.drop(columns=["rank"])       # ALL interactions
valid_df = interactions[interactions["rank"] == 1]   # Second-to-last per user

interactions_ds = InteractionsDataset(data_location="train.csv")
valid_inter_ds  = InteractionsDataset(data_location="valid.csv")

recommender.train(
    items_ds=items_ds,
    interactions_ds=interactions_ds,
    valid_interactions_ds=valid_inter_ds,   # Enables early stopping in estimator
)
```

!!! note "Validation data format"
    `valid_interactions_ds` uses the same raw interactions format as `interactions_ds`. The recommender applies session boundary detection (`_build_session_sequences`) to the validation data internally using the same `session_timeout_minutes` as training.

## Inference

At inference time, pass raw interaction rows in the same format as training. `OUTCOME` is optional.

```python
# Score all items for each user (returns ndarray shape: n_users × n_items)
scores = recommender.score_items(interactions=recent_df)

# Recommend top-k items (returns ndarray shape: n_users × top_k)
recs = recommender.recommend(interactions=recent_df, top_k=10)
```

!!! note "History window"
    Only the most recent `max_sessions` sessions (each up to `max_session_len` items) are used per user. Older data is silently dropped.

## Probabilistic Sampling

```python
recs = recommender.recommend(
    interactions=recent_df,
    top_k=10,
    sampling_temperature=0.5,   # 0 = deterministic, >0 = probabilistic
)
```

## When to Use HierarchicalSequentialRecommender

✅ **Use HierarchicalSequentialRecommender when:**

- Interactions naturally group into sessions (shopping visits, app opens, viewing sessions)
- You want cross-session preference evolution modelled explicitly
- You have `TIMESTAMP` data (or explicit `SESSION_ID`)

❌ **Don't use it when:**

- No `TIMESTAMP` available — use [RankingRecommender](ranking.md)
- Interactions have no natural session structure — use [SequentialRecommender](sequential.md) with SASRec

## Comparison with SequentialRecommender

| | HierarchicalSequentialRecommender | SequentialRecommender |
|---|---|---|
| **Session-aware** | ✅ Yes — explicit 2-level structure | ❌ No — single flat sequence |
| **Architecture** | HRNN (GRU + GRU) | SASRec (Transformer) |
| **Requires TIMESTAMP** | ✅ Yes | ✅ Yes |
| **SESSION_ID support** | ✅ Yes | ❌ No |
| **Compatible estimators** | HRNN only | SASRec only |

## Next Steps

- **[HRNN Estimator Guide](../user-guide/hrnn.md)** — Architecture, hyperparameters, training modes, performance benchmarks
- **[SequentialRecommender](sequential.md)** — The SASRec-based alternative for non-session data
- **[Training Guide](../user-guide/training.md)** — General training patterns
- **[Evaluation Guide](../user-guide/evaluation.md)** — Measuring ranking quality (HR@K, NDCG@K)
