# Collaborative Filtering (Matrix Factorization)

Native matrix factorization for collaborative filtering using only NumPy—no PyTorch or external libraries like Surprise. Learns user and item latent factors from (user_id, item_id, outcome) interactions and supports multiple algorithms and outcome types.

## Overview

The `MatrixFactorizationEstimator` implements classical collaborative filtering:

- **Model:** Predicted reward = global mean + user latent vector · item latent vector (with optional sigmoid or clamp for binary/ordinal).
- **Training:** Either **ALS** (Alternating Least Squares) or **SGD** (Stochastic Gradient Descent).
- **Outcome types:** Continuous (real-valued), binary (0/1), or ordinal (e.g. 1–5 star ratings).

Use this when you want a lightweight, dependency-free CF option or interpretable latent factors; use **NCF** when you need neural architectures and richer feature integration.

## Quick Start

```python
from skrec.estimator.datatypes import MFAlgorithm, MFOutcomeType
from skrec.estimator.embedding.matrix_factorization_estimator import MatrixFactorizationEstimator
from skrec.scorer.universal import UniversalScorer
from skrec.recommender.ranking.ranking_recommender import RankingRecommender

# Native CF: NumPy-only matrix factorization
estimator = MatrixFactorizationEstimator(
    n_factors=32,
    algorithm=MFAlgorithm.ALS,   # or MFAlgorithm.SGD
    outcome_type=MFOutcomeType.CONTINUOUS,  # or BINARY, ORDINAL
    regularization=0.01,
    epochs=20,
    random_state=42,
)
scorer = UniversalScorer(estimator=estimator)
recommender = RankingRecommender(scorer=scorer)

recommender.train(
    interactions_ds=interactions_ds,
    users_ds=users_ds,   # optional
    items_ds=items_ds,   # optional
)

recommendations = recommender.recommend(
    interactions=interactions_df,
    users=None,
    top_k=5,
)
```

!!! important "Scorer compatibility"
    Embedding estimators, including `MatrixFactorizationEstimator`, work **only with UniversalScorer**. Using another scorer (e.g. IndependentScorer) will raise a `TypeError` at construction time.

## Algorithms

Choose the solver via the `MFAlgorithm` enum.

### ALS (Alternating Least Squares)

- **Enum:** `MFAlgorithm.ALS`
- Fix item factors, solve for each user’s latent vector in closed form (ridge regression over that user’s observed items); then fix user factors and solve for each item. Alternate for several epochs.
- **Pros:** No learning rate, stable, good for explicit feedback.
- **Cons:** Per-user/per-item linear systems; can be slower than SGD on very large datasets when implemented naively.

```python
from skrec.estimator.datatypes import MFAlgorithm

estimator = MatrixFactorizationEstimator(
    algorithm=MFAlgorithm.ALS,
    n_factors=32,
    regularization=0.01,
    epochs=20,
)
```

### SGD (Stochastic Gradient Descent)

- **Enum:** `MFAlgorithm.SGD`
- Loop over (user, item, reward) triplets (shuffled each epoch); update user and item vectors with gradient step plus L2 regularization. Uses **MSE** for continuous/ordinal and **BCE** (sigmoid) for binary.
- **Pros:** Simple, flexible (supports binary loss), one hyperparameter (`learning_rate`).
- **Cons:** Sensitive to learning rate and may need more epochs.

```python
from skrec.estimator.datatypes import MFAlgorithm

estimator = MatrixFactorizationEstimator(
    algorithm=MFAlgorithm.SGD,
    learning_rate=0.02,
    n_factors=32,
    regularization=0.01,
    epochs=50,
    random_state=42,
)
```

## Outcome Types

Control reward semantics with the `MFOutcomeType` enum and optional bounds for ordinal.

### CONTINUOUS

- Real-valued outcomes (e.g. watch time, revenue). **MSE** loss; predictions are raw scores (any real number).

```python
from skrec.estimator.datatypes import MFOutcomeType

estimator = MatrixFactorizationEstimator(
    outcome_type=MFOutcomeType.CONTINUOUS,
    algorithm=MFAlgorithm.ALS,
    epochs=20,
)
```

### BINARY

- 0/1 outcomes (e.g. click, like). With **SGD**: **BCE** (sigmoid of logit); predictions are probabilities in [0, 1]. With **ALS**: MSE on 0/1, then sigmoid applied at prediction time so outputs stay in [0, 1].

```python
estimator = MatrixFactorizationEstimator(
    outcome_type=MFOutcomeType.BINARY,
    algorithm=MFAlgorithm.SGD,  # recommended for proper BCE
    learning_rate=0.02,
    epochs=50,
)
```

### ORDINAL

- Ordered discrete levels (e.g. 1–5 star ratings). **MSE** on the numeric levels; predictions can be **clamped** to an optional scale with `ordinal_min` and `ordinal_max`.

```python
estimator = MatrixFactorizationEstimator(
    outcome_type=MFOutcomeType.ORDINAL,
    ordinal_min=1.0,
    ordinal_max=5.0,
    algorithm=MFAlgorithm.ALS,
    epochs=20,
)
```

## Main Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_factors` | Latent dimension (rank) of the factorization | 32 |
| `algorithm` | `MFAlgorithm.ALS` or `MFAlgorithm.SGD` | ALS |
| `outcome_type` | `MFOutcomeType.CONTINUOUS`, `BINARY`, or `ORDINAL` | CONTINUOUS |
| `ordinal_min` / `ordinal_max` | For ORDINAL: optional clamp bounds (e.g. 1 and 5) | None |
| `regularization` | L2 regularization for factor updates | 0.01 |
| `learning_rate` | Step size for SGD (ignored for ALS) | 0.01 |
| `epochs` | Number of ALS alternations or SGD epochs | 20 |
| `random_state` | Seed for reproducibility | None |
| `verbose` | Log progress every epoch if > 0 | 0 |

## Real-time Inference with User Embeddings

Like other embedding estimators, you can precompute user embeddings and pass them at score time:

```python
# After training
user_embeddings_df = estimator.get_user_embeddings()  # USER_ID, EMBEDDING columns

# At inference: pass users DataFrame with EMBEDDING column
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=user_embeddings_df,
    top_k=5,
)
```

## Comparison: Matrix Factorization vs NCF

| Feature | MatrixFactorizationEstimator | NCFEstimator |
|---------|-------------------------------|--------------|
| **Dependencies** | NumPy only | PyTorch |
| **Algorithms** | ALS, SGD | GMF, MLP, NeuMF |
| **Outcome types** | Continuous, binary, ordinal (with clamp) | Implicit (BCE) or explicit (MSE) |
| **User/item features** | IDs only (no feature columns) | User + item + context features |
| **Training speed** | Fast (no GPU needed) | Slower (neural nets) |
| **Best for** | Lightweight CF, interpretable factors, no-GPU environments | Richer features, maximum accuracy, GPU available |

## When to Use

✅ **Use MatrixFactorizationEstimator when:**

- You want collaborative filtering without PyTorch or Surprise.
- Your rewards are continuous, binary, or ordinal (e.g. 1–5 ratings).
- You prefer interpretable latent factors or a small dependency set.
- Training or deployment environment has no GPU.

❌ **Consider NCF or other estimators when:**

- You have user/item/context features to fuse (NCF supports them).
- You need maximum accuracy and have GPU and enough data (NCF).
- You need content-based or two-tower retrieval (other embedding estimators).

## References

- Alternating Least Squares for implicit feedback: Hu et al., "Collaborative Filtering for Implicit Feedback Datasets" (2008).
- Neural Collaborative Filtering (alternative in this library): He et al., "Neural Collaborative Filtering" (WWW 2017); see [NCF Guide](ncf.md).
