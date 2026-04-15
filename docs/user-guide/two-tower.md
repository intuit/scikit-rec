# Contextualized Two-Tower Estimator

`ContextualizedTwoTowerEstimator` is a two-tower (dual-encoder) model where a **user tower** and an **item tower** independently produce fixed-size embeddings that are combined to produce a relevance score. It supports three modes for incorporating interaction context features (e.g. time-of-day, device, session length), controlled by the `context_mode` parameter.

## Overview

```
User ID + User Features + [Context]  →  User Tower  →  user_rep (final_embedding_dim)
Item ID + Item Features              →  Item Tower  →  item_rep (final_embedding_dim)
                                                               ↓
                                                    Score = f(user_rep, item_rep, context)
```

**Key properties:**

- Item embeddings are always context-free and precomputable offline
- User embedding precomputability depends on `context_mode` (see table below)
- All three modes support early stopping via `early_stopping_patience`

## Context Modes

`ContextMode` is a `str` enum — you can pass the string value directly:

```python
from skrec.estimator.embedding.contextualized_two_tower_estimator import (
    ContextualizedTwoTowerEstimator,
    ContextMode,
)

estimator = ContextualizedTwoTowerEstimator(context_mode="trilinear")
# or
estimator = ContextualizedTwoTowerEstimator(context_mode=ContextMode.TRILINEAR)
```

### Mode Comparison

| Mode | Where context goes | Score formula | ANN retrieval | User embeddings precomputable | Context sensitivity |
|---|---|---|---|---|---|
| `user_tower` | Concatenated into user tower input | `dot(user_tower(u, ctx), item_tower(i))` | ✅ Yes (compute user+ctx at request time) | ❌ No (context-dependent) | Medium |
| `trilinear` | Hadamard-modulates user tower output | `dot(user_rep * ctx_emb, item_rep)` | ✅ Yes (multiply cached user_rep by runtime ctx_emb) | ✅ Yes | High |
| `scoring_layer` | Concatenated into final scoring layer | `linear([user_rep, item_rep, ctx_rep])` | ❌ No (score is not a dot product) | ✅ Yes | Low (linear layer struggles with weak signals) |

### `user_tower` (default)

Context is concatenated into the user tower's input alongside the user ID embedding and profile features. The score is the dot product of user and item tower outputs.

```
user_tower_input = [user_id_emb, user_features, context_features]
score = dot(user_tower(input), item_tower([item_id_emb, item_features]))
```

**ANN**: Compute user+context embedding at request time, then search the precomputed item FAISS index.

**`get_user_embeddings()`**: Raises `NotImplementedError` when context features are present — user representations are context-dependent and cannot be cached per-user.

### `trilinear`

Context is projected to `final_embedding_dim` and applied via an elementwise Hadamard product with the context-free user tower output.

```
user_rep  = user_tower([user_id_emb, user_features])
ctx_emb   = context_projection(context_features)
score     = dot(user_rep * ctx_emb, item_rep)
```

**ANN**: Cache `user_rep` offline per user. At request time: multiply by `ctx_emb`, then search item index.

**Most context-sensitive** — context modulates every dimension of the user embedding before retrieval.

### `scoring_layer`

Context is projected to `final_embedding_dim` and concatenated with both tower outputs. A final linear layer maps the concatenation to a scalar score.

```
user_rep  = user_tower([user_id_emb, user_features])
ctx_rep   = context_projection(context_features)
score     = linear([user_rep, item_rep, ctx_rep])
```

**ANN**: Not supported — score is not decomposable as a dot product. Full model must run at serving time.

**Most expressive** in principle, but the linear scoring layer can fail to exploit weak context signals in practice.

---

## Quick Start

```python
from skrec.estimator.embedding.contextualized_two_tower_estimator import (
    ContextualizedTwoTowerEstimator, ContextMode,
)
from skrec.scorer.universal import UniversalScorer
from skrec.recommender.ranking.ranking_recommender import RankingRecommender

estimator = ContextualizedTwoTowerEstimator(
    user_embedding_dim=32,
    item_embedding_dim=32,
    final_embedding_dim=16,
    context_mode=ContextMode.TRILINEAR,   # or "trilinear"
    learning_rate=0.001,
    epochs=50,
    batch_size=256,
    early_stopping_patience=5,
    restore_best_weights=True,
)

scorer = UniversalScorer(estimator)
recommender = RankingRecommender(scorer)

recommender.train(
    interactions_ds=interactions_ds,   # Must include context feature columns
    users_ds=users_ds,
    items_ds=items_ds,
)
```

## Architecture Parameters

```python
ContextualizedTwoTowerEstimator(
    user_embedding_dim=64,              # Learned user ID embedding size
    item_embedding_dim=64,              # Learned item ID embedding size
    final_embedding_dim=32,             # Output dimension of both towers (must match)
    context_mode="user_tower",          # "user_tower" | "trilinear" | "scoring_layer"
    user_tower_hidden_dim1=None,        # First hidden layer of user tower (default: final_embedding_dim * 2)
    user_tower_hidden_dim2=None,        # Second hidden layer (optional)
    item_tower_hidden_dim1=None,        # First hidden layer of item tower (default: final_embedding_dim * 2)
    item_tower_hidden_dim2=None,
    learning_rate=0.001,
    epochs=10,
    batch_size=32,
    optimizer_name="adam",
    loss_fn_name="bce",
    early_stopping_patience=None,       # Epochs without val-loss improvement before stopping
    restore_best_weights=True,
    random_state=None,
)
```

Tower hidden layers default to `final_embedding_dim * 2` when not specified. To use a single linear projection (no hidden layer), pass `user_tower_hidden_dim1=None` explicitly.

## Context Features

Context features are interaction-level features — columns in `interactions_df` other than `USER_ID`, `ITEM_ID`, `OUTCOME`, and `TIMESTAMP`. Examples:

```python
# Time-of-day features derived from real timestamps
interactions_df["hour_norm"] = pd.to_datetime(interactions_df["TIMESTAMP"], unit="s").dt.hour / 23.0
interactions_df["is_weekend"] = (
    pd.to_datetime(interactions_df["TIMESTAMP"], unit="s").dt.dayofweek >= 5
).astype(float)

# Device type (requires prior one-hot encoding)
# interactions_df["is_mobile"] = ...
```

!!! note "Categorical context features"
    `ContextualizedTwoTowerEstimator` expects **numerical** interaction features. One-hot encode categorical context variables before training.

## Performance on MovieLens 1M (with Temporal Context)

Evaluated using real ML-1M timestamps to derive `hour_norm` (time of day) and `is_weekend` as context features. Leave-last-out split, sampled ranking (1 positive + 100 random negatives per user, 6,040 users). `early_stopping_patience=3`.

| Mode | HR@10 | Context sensitivity (5% vs 95% hour overlap) |
|---|---|---|
| `user_tower` | **0.2270** | 33% list overlap — moderate |
| `trilinear` | 0.2210 | **5% list overlap** — highest sensitivity |
| `scoring_layer` | 0.2053 | 100% list overlap — zero sensitivity |

**Note**: These numbers are lower than typical two-tower benchmarks because ML-1M has thin user/item features (only timestamps as context) and uses sampled evaluation with 101 candidates. With richer features the gap between modes widens.

### Context sensitivity interpretation

The "context sensitivity" column shows how much the top-10 recommendations change between morning (hour < 12) and evening (hour ≥ 18) context:

- **`trilinear`** (5% overlap): Nearly completely different lists — context strongly modulates which items surface. The Hadamard product acts as a per-dimension gate that can suppress or amplify any item dimension based on context.
- **`user_tower`** (33% overlap): Meaningful but less dramatic context-dependence — context shifts the user representation but the tower weights constrain how much any single feature can move it.
- **`scoring_layer`** (100% overlap): Identical lists — the final linear layer assigns near-zero weight to `hour_norm`/`is_weekend` because these two weak features don't reliably predict item preference across the full catalogue.

## Embedding Precomputation

```python
# Get user embeddings (user_rep for each user in the training set)
user_embeddings_df = estimator.get_user_embeddings()
# Returns DataFrame with columns: USER_ID, USER_EMBEDDING

# user_tower + context features → raises NotImplementedError
# trilinear, scoring_layer → returns (n_users, final_embedding_dim) embeddings
```

### ANN Serving Pattern for `trilinear`

```python
# Offline: cache user and item embeddings
user_embs = estimator.get_user_embeddings()   # (n_users, final_embedding_dim)
# Build FAISS index from item embeddings (from estimator.model.item_tower)

# Online: modulate cached user embedding by runtime context
ctx_emb = estimator.model.context_projection(runtime_context_tensor)
query_vec = user_embs[user_id] * ctx_emb.numpy()   # Hadamard modulation
# Search FAISS index with query_vec
```

## Early Stopping

`ContextualizedTwoTowerEstimator` inherits early stopping from `BasePyTorchEmbeddingEstimator`. Pass `valid_interactions_ds` to `RankingRecommender.train()`:

```python
recommender.train(
    interactions_ds=train_ds,
    users_ds=users_ds,
    items_ds=items_ds,
    valid_interactions_ds=valid_ds,   # Enables early stopping
)
```

## When to Use Which Mode

| Scenario | Recommended mode |
|---|---|
| Context features are strong predictors (e.g. session type, intent signals) | `trilinear` |
| You need ANN retrieval + precomputed user embeddings | `trilinear` |
| You need ANN retrieval but context is secondary | `user_tower` |
| Expressiveness matters more than ANN compatibility | `scoring_layer` |
| No context features at all | Any (context path is bypassed) |

## References

- Yi, X. et al. (2019). [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996). RecSys 2019. (Google Two-Tower)

## Next Steps

- **[Estimator Guide](estimators.md)** — Overview of all estimators
- **[Inference Guide](inference.md)** — Embedding-based real-time inference patterns
- **[Training Guide](training.md)** — General training patterns
