# SequentialRecommender

`SequentialRecommender` wraps a `SequentialScorer` + SASRec estimator and adds the sequence-building logic required for temporal interaction data: it sorts interactions by timestamp, groups by user, truncates long histories, and passes the resulting sequences downstream.

## Overview

```
SequentialRecommender
    → SequentialScorer
        → SASRecClassifierEstimator  (or SASRecRegressorEstimator)
```

**Owns:**

- Sorting interactions by `TIMESTAMP` per user
- Truncating sequences to `max_len + 1`
- Handling the presence/absence of `OUTCOME` (training vs. inference)

**Does not own:**

- Tensor construction (estimator)
- Item vocabulary and score matrix (scorer)

## Quick Start

```python
from skrec.estimator.sequential.sasrec_estimator import SASRecClassifierEstimator
from skrec.scorer.sequential import SequentialScorer
from skrec.recommender.sequential.sequential_recommender import SequentialRecommender
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset

estimator = SASRecClassifierEstimator(
    hidden_units=50, num_blocks=2, num_heads=1,
    dropout_rate=0.2, epochs=200, max_len=50,
    early_stopping_patience=5,    # Optional: stop when val loss plateaus
    restore_best_weights=True,
)
scorer = SequentialScorer(estimator)
recommender = SequentialRecommender(scorer, max_len=50)

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

The interactions dataset must have at minimum:

| Column | Type | Required |
|---|---|---|
| `USER_ID` | str / int | ✅ |
| `ITEM_ID` | str / int | ✅ |
| `TIMESTAMP` | int / float | ✅ |
| `OUTCOME` | float | ✅ at training, optional at inference |

Provide the built-in schema to enable validation:

```python
InteractionsDataset(
    data_location="interactions.csv",
    client_schema_path="skrec/dataset/required_schemas/interactions_schema_with_timestamp_training.yaml",
)
```

## Validation Split and Early Stopping

Pass `valid_interactions_ds` to `train()` to enable per-epoch val-loss monitoring. The recommender builds validation sequences using the same `_build_sequences` logic applied to all interactions except the last per user — so the last target in each validation sequence is the validation item (the second-to-last interaction in the full history).

```python
# Leave-last-two-out split
interactions["rank"] = interactions.groupby("USER_ID").cumcount(ascending=False)
train_df = interactions.drop(columns=["rank"])         # ALL interactions
valid_df = interactions[interactions["rank"] == 1]     # Second-to-last per user

interactions_ds = InteractionsDataset(data_location="train.csv")
valid_inter_ds  = InteractionsDataset(data_location="valid.csv")

recommender.train(
    items_ds=items_ds,
    interactions_ds=interactions_ds,
    valid_interactions_ds=valid_inter_ds,   # Enables early stopping in estimator
)
```

!!! note "Validation data format"
    `valid_interactions_ds` uses the same raw interactions format as `interactions_ds` — one row per interaction with `USER_ID`, `ITEM_ID`, `OUTCOME`, `TIMESTAMP`. The recommender converts it to sequences internally.

## Sequence Building

`_build_sequences` is the core transformation step:

1. **Sort** all rows by `(USER_ID, TIMESTAMP)` ascending
2. **Group** by `USER_ID` — one row per user with a list of item IDs
3. **Truncate** to the most recent `max_len + 1` items (one extra so the estimator can form `input = seq[:-1]`, `target = seq[1:]` with all `max_len` positions filled for long histories)
4. **Include `OUTCOME_SEQUENCE`** only when `OUTCOME` is present (training mode)

```python
# Training mode: OUTCOME present
sequences_df columns: USER_ID, ITEM_SEQUENCE, OUTCOME_SEQUENCE

# Inference mode: no OUTCOME
sequences_df columns: USER_ID, ITEM_SEQUENCE
```

## `max_len` Parameter

`max_len` must match between `SequentialRecommender` and `SASRecClassifierEstimator`. The recommender always wins at training time — it syncs its value to the estimator and logs a warning if they differ:

```python
# Correct: same max_len on both
estimator = SASRecClassifierEstimator(..., max_len=50)
recommender = SequentialRecommender(scorer, max_len=50)

# Also works: recommender value overrides silently with a warning
estimator = SASRecClassifierEstimator(..., max_len=99)  # overridden to 50
recommender = SequentialRecommender(scorer, max_len=50)
```

## Inference

At inference time, pass raw interaction rows — the same `TIMESTAMP`-ordered format as training. `OUTCOME` is optional.

```python
# Score all items for each user (returns ndarray shape: n_users × n_items)
scores = recommender.score_items(interactions=recent_df)

# Recommend top-k items (returns ndarray shape: n_users × top_k)
recs = recommender.recommend(interactions=recent_df, top_k=10)
```

!!! note "History window"
    Only the most recent `max_len` interactions are used per user at inference. Older interactions beyond `max_len` are silently dropped.

## Probabilistic Sampling

Like `RankingRecommender`, you can sample from the score distribution instead of taking the argmax:

```python
recs = recommender.recommend(
    interactions=recent_df,
    top_k=10,
    sampling_temperature=0.5,   # 0 = deterministic, >0 = probabilistic
)
```

## When to Use SequentialRecommender

✅ **Use SequentialRecommender when:**

- Interaction order matters (video watch history, browsing sessions, purchase journeys)
- You have `TIMESTAMP` data for all interactions
- Users have at least a few historical interactions

❌ **Don't use SequentialRecommender when:**

- No `TIMESTAMP` available — use [RankingRecommender](ranking.md) with [NCF](../user-guide/ncf.md)
- Interactions are independent and unordered

## Comparison with RankingRecommender

| | SequentialRecommender | RankingRecommender |
|---|---|---|
| **Order-aware** | ✅ Yes | ❌ No |
| **Requires TIMESTAMP** | ✅ Yes | ❌ No |
| **Compatible estimators** | SASRec only | All estimators |
| **User/item features** | ❌ No (ID-based) | ✅ Yes |
| **Inference input** | Raw interactions + TIMESTAMP | User features or interactions |

## Next Steps

- **[SASRec Estimator Guide](../user-guide/sasrec.md)** — Architecture, hyperparameters, performance benchmarks
- **[HierarchicalSequentialRecommender](hierarchical.md)** — Session-aware alternative using HRNN
- **[Training Guide](../user-guide/training.md)** — General training patterns
- **[Evaluation Guide](../user-guide/evaluation.md)** — Measuring ranking quality (HR@K, NDCG@K)
