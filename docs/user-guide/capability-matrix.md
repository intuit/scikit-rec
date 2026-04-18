# Capability matrix

This page summarizes **which combinations of components are supported** in the current codebase: training paths, inference APIs, retrievers, and batch training. It is derived from the implementation (not aspirational). If behavior changes in a release, this matrix should be updated with it.

For the layered mental model, see [Architecture overview](architecture.md).

---

## Legend

| Symbol | Meaning |
|--------|---------|
| **Yes** | Supported for typical configurations. |
| **No** | Raises, documented as unsupported, or would violate the model contract. |
| **Partial** | Works with constraints (see notes). |
| **N/A** | Concept does not apply (e.g. no estimator plane). |

---

## Estimator training planes

Models fall into three **training / scoring planes**. Each plane uses different dataset shaping and entrypoints.

| Plane | Base type (typical) | Training entrypoint | Primary scoring path |
|-------|---------------------|---------------------|----------------------|
| **Tabular** | `BaseClassifier`, `BaseRegressor` (e.g. XGB, LightGBM, sklearn, DeepFM as classifier) | `fit(X, y)` on a joined feature matrix from the scorer | `predict` / `predict_proba` on joined rows |
| **Embedding** | `BaseEmbeddingEstimator` (e.g. MF, NCF, two-tower, DCN variants) | `fit_embedding_model(users, items, interactions, …)` | `predict_proba_with_embeddings(…)` |
| **Sequential** | `SequentialEstimator` (SASRec, HRNN) | `fit_embedding_model(…)` via `SequentialScorer` (not `BaseEmbeddingEstimator`) | Forward pass over sequences; not Cartesian tabular join |

**Dispatch:** `recommender.recommender.training_coordinator.coordinate_training` chooses batch tabular vs tabular vs embedding vs (via scorer) sequential training.

---

## Scorer × estimator plane

| Scorer | Tabular estimator | Embedding (`BaseEmbeddingEstimator`) | Sequential (`SequentialEstimator`) |
|--------|-------------------|--------------------------------------|-------------------------------------|
| **UniversalScorer** | **Yes** (factory yields `TabularUniversalScorer`) | **Yes** (`EmbeddingUniversalScorer`) | **No** (use `SequentialScorer`) |
| **IndependentScorer** | **Yes** (single or dict of estimators) | **No** (raises at init) | **No** |
| **MulticlassScorer** | **Yes** | **No** (raises at init) | **No** |
| **MultioutputScorer** | **Yes** | **No** (raises at init) | **No** |
| **SequentialScorer** | **No** | **No** | **Yes** |
| **HierarchicalScorer** | **No** | **No** | **Yes** (HRNN estimators) |
| **UpliftScorerAdapter** | **Yes** (internal tabular scorers) | **No** (not exposed for uplift) | **No** |

---

## `recommend()` vs `recommend_online()`

| Recommender | `recommend()` | `recommend_online()` | Notes |
|-------------|---------------|----------------------|-------|
| **RankingRecommender** | **Yes** | **Partial** | With an attached **retriever**, `recommend()` uses two-stage candidates. **`recommend_online()` does not use the retriever** — it scores the **full catalog** (warning logged). Thread-safety: see [retrieval](retrieval.md) when using retriever + shared instance. |
| **GcslRecommender** | **Yes** | **No** | Goal injection is skipped on the fast path; use `recommend()`. |
| **SequentialRecommender** | **Yes** | **No** | Use `recommend()`; sequence models use their own forward path. |
| **HierarchicalSequentialRecommender** | **Yes** | **No** | Same as sequential (inherits blocked `recommend_online`). |
| **ContextualBanditsRecommender** | **Yes** | **Partial** | Inherits base fast path: same **`score_fast` / `_score_fast_np`** constraints as the underlying scorer. **Strategy must be set** before `recommend()` / sampling paths; embedding scorers still **cannot** use `recommend_online()`. |
| **UpliftRecommender** | **Yes** | **No** | `UpliftScorerAdapter.score_fast` raises; use `recommend()`. |

### `recommend_online()` and scorers (Ranking + tabular path)

`BaseRecommender.recommend_online` builds a single-row feature row, then calls **`MultioutputScorer.score_fast`** (special return type) or **`_score_fast_np`** on other scorers.

| Scorer | `recommend_online` | Notes |
|--------|-------------------|-------|
| **Tabular UniversalScorer** | **Yes** | Single-row fast path. |
| **Embedding UniversalScorer** | **No** | `score_fast` raises; use `recommend()`. |
| **IndependentScorer** | **Yes** | When estimator is tabular; `score_fast` validates exactly one row. |
| **MulticlassScorer** | **Yes** | Same one-row contract. |
| **MultioutputScorer** | **Partial** | Returns a **DataFrame** of predicted labels per output column; **`top_k` is ignored** (documented on `BaseRecommender.recommend_online`). |
| **SequentialScorer** | **No** | `score_fast` raises. |
| **UpliftScorerAdapter** | **No** | Use `recommend()`. |

---

## Optional retrievers (`RankingRecommender` / `GcslRecommender`)

Retriever is **optional** on these classes. `SequentialRecommender`’s public constructor does not expose a retriever argument (inherits `RankingRecommender` with retriever default **None**).

| Retriever | Required `train()` datasets | Estimator constraint |
|-----------|----------------------------|----------------------|
| **EmbeddingRetriever** | Depends on model; embedding index uses fitted estimator | **`BaseEmbeddingEstimator`** on the scorer |
| **ContentBasedRetriever** | **`items_ds`** | None specific |
| **PopularityRetriever** | **`interactions_ds`** | None specific |

---

## Batch (partitioned) training

| Support | Details |
|---------|---------|
| **Estimator** | In the shipped code, **`BatchXGBClassifierEstimator`** implements `_batch_fit_model` (see `skrec/estimator/classification/xgb_classifier.py`). Other estimators use in-memory `fit` unless extended. |
| **Training** | `coordinate_training` uses the batch path when `estimator.support_batch_training()` is true: **`interactions_ds` required**; **`items_ds` required** for partitioned catalogue setup; validation datasets must satisfy the same rules as tabular validation (e.g. `valid_users_ds` when users + valid interactions). |
| **Embedding** | **No** batch-training branch for `BaseEmbeddingEstimator` in the same iterator style. |
| **Sequential** | **`support_batch_training()` is False** for `SequentialEstimator`. |

---

## Evaluation (`evaluate()`)

All recommenders built on **`BaseRecommender`** share the evaluation session API, but **metrics are only meaningful** when the recommender’s ranking / probability behavior matches what you intend to measure.

| Topic | Note |
|-------|------|
| **ContextualBanditsRecommender** | Offline `evaluate()` uses scorer scores but **ranking / probabilities follow the bandit policy** when the strategy applies. For “raw model” metrics, use **`RankingRecommender`** (see docstring on `ContextualBanditsRecommender`). |
| **STATIC_ACTION** | Probabilistic `recommend()` / temperature paths may **raise** (`NotImplementedError`); evaluation has a dedicated bundle path for static action. |

---

## Thread safety (selected)

| Component | Concurrence |
|-----------|-------------|
| **RankingRecommender + retriever** | **Not thread-safe** on one instance: per-user retrieval mutates `scorer` item subset. |
| **IndependentScorer** | Docstring: **not thread-safe** for parallel inference / subset / executor configuration. |

---

## Related docs

- [Architecture overview](architecture.md) — three layers and data flow  
- [Inference](inference.md) — batch vs online patterns  
- [Retrieval](retrieval.md) — two-stage retrieval and scaling  
- [Training](training.md) — validation splits and training options  
- [Recommender types comparison](../recommender-types/comparison.md) — choosing a recommender class  
