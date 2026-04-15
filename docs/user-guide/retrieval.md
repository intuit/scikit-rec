# Two-Stage Retrieval

As catalogs grow, scoring every item for every user becomes expensive.
Two-stage recommendation addresses this: a fast **retrieval** step narrows the catalog to a
small candidate set, then the **ranking** model re-scores only those candidates with full
feature richness.

## How It Fits the Architecture

Two-stage retrieval is a **first-class, built-in feature** of `RankingRecommender` —
not a plugin or external system. Three retrievers ship with the library out of the box.
The interface is a single constructor argument:

```python
RankingRecommender(scorer=..., retriever=EmbeddingRetriever(top_k=200))
#                              ↑ this one argument enables the full two-stage pipeline
```

At the same time, retrieval is **entirely optional**. Omitting it gives you exactly the
same API and semantics — `RankingRecommender` silently scores the full catalog instead:

```python
RankingRecommender(scorer=...)   # no retriever — full-catalog scoring, same interface
```

The **3-layer core (Recommender → Scorer → Estimator) is unchanged in both cases.**
Retrieval is a pre-stage that sits in front of the ranking layer without altering it:

```
# Without retriever (default)
[RankingRecommender → Scorer → Estimator]
score all N items per user → rank → top-k

# With retriever
[Retriever]  ← built-in, runs inside recommend()
    ↓ top-200 candidates
[RankingRecommender → Scorer → Estimator]
score 200 items per user → rank → top-k
```

This design means you can start without a retriever on a small catalog, then add one
later with a single line change — no restructuring of your pipeline required.

## Quick Start

```python
from skrec.retriever import EmbeddingRetriever
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.estimator.embedding.matrix_factorization_estimator import MatrixFactorizationEstimator
from skrec.scorer.universal import UniversalScorer

recommender = RankingRecommender(
    scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=64)),
    retriever=EmbeddingRetriever(top_k=200),
)
recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
recommendations = recommender.recommend(interactions=df, top_k=10)
```

The retriever is built automatically at the end of `train()`. Inference is fully
transparent — `recommend()` runs retrieval and ranking in one call.

## Choosing a Retriever

| Retriever | Best for | Estimator requirement |
|---|---|---|
| `EmbeddingRetriever` | Personalized retrieval using learned embeddings (brute-force, works well up to tens of thousands of items) | Embedding estimator (MF, NCF, Two-Tower) |
| `ContentBasedRetriever` | New/frequently-changing items; any estimator type | None (uses item features) |
| `PopularityRetriever` | Non-personalized baseline; no features required | None |

**Decision tree:**

1. Do you have an embedding estimator (MF / NCF / Two-Tower)? → **EmbeddingRetriever**
2. Do items have numeric features (price, rating, category encoding)? → **ContentBasedRetriever**
3. No embeddings, no features? → **PopularityRetriever** as baseline

You can also skip built-in retrievers entirely and pass candidates directly via
`item_subset` — useful when your retrieval system lives outside the library
(e.g. Elasticsearch, FAISS service).

---

## EmbeddingRetriever

Uses brute-force dot-product search over learned item embeddings.
Compatible with any `BaseEmbeddingEstimator`: `MatrixFactorizationEstimator`,
`NCFEstimator`, `ContextualizedTwoTowerEstimator`.

!!! warning "ContextualizedTwoTowerEstimator compatibility"
    `EmbeddingRetriever` requires precomputed user embeddings. This is not possible
    when using `ContextualizedTwoTowerEstimator` with `context_mode='user_tower'`
    (the default) **and** context interaction features — in that mode, user
    representations depend on request-time context and cannot be cached upfront.
    Use `context_mode='trilinear'` instead, which supports precomputed user embeddings,
    or use `predict_proba_with_embeddings()` at serving time without a retriever.

```python
from skrec.retriever import EmbeddingRetriever

recommender = RankingRecommender(
    scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=64)),
    retriever=EmbeddingRetriever(top_k=200),
)
```

**How it works:**

1. After training, `build_index()` extracts item embeddings from the estimator and stores
   them as an `(n_items, k)` matrix.
2. At inference, it extracts user embeddings and computes `item_matrix @ user_vec` for each
   user — brute-force cosine similarity via `np.argpartition` (O(n) per user).
3. Unknown users (cold-start) fall back to the globally most popular items.

!!! note "Space compatibility"
    User and item embeddings are in the **same learned latent space** — this is guaranteed
    by construction (both come from the same factorized model). The dot product is valid.

!!! note "Implementation scope"
    The built-in `EmbeddingRetriever` uses brute-force NumPy dot-product search — no
    approximate nearest-neighbor index. This is correct and fast enough for catalogs in
    the tens of thousands. For larger catalogs, swap in a [custom retriever](#custom-retriever)
    backed by FAISS or a vector database — the pluggable interface makes this about 30 lines.

---

## ContentBasedRetriever

Retrieves candidates by cosine similarity between a user profile vector and item
feature vectors. No embeddings required — works with XGBoost, LightGBM, and any
other estimator.

```python
from skrec.retriever import ContentBasedRetriever

recommender = RankingRecommender(
    scorer=UniversalScorer(estimator=XGBClassifierEstimator({...})),
    retriever=ContentBasedRetriever(
        top_k=200,
        feature_columns=["price", "avg_rating", "category_enc"],
    ),
)
```

**How it works:**

1. Item feature vectors are L2-normalized at index time.
2. Each user's profile is the (optionally outcome-weighted) mean of their interacted item
   vectors, also L2-normalized.
3. Cosine similarity `item_matrix @ user_profile` ranks candidates.
4. Cold-start users fall back to the globally most popular items.

### Post-training items

New items added **after** training are immediately retrievable as long as their features
appear in the `items` DataFrame:

```python
# Add new items after training
new_items = pd.DataFrame({
    ITEM_ID_NAME: ["new_item_1", "new_item_2"],
    "price": [25.0, 45.0],
    "avg_rating": [4.2, 3.8],
    "category_enc": [1, 3],
})
extended_items_df = pd.concat([original_items_df, new_items], ignore_index=True)

# Rebuild the retriever index — no model retraining needed.
# The ranking model is unchanged; only the retrieval pool expands.
recommender.retriever.build_index(
    interactions=train_interactions_df,
    items=extended_items_df,
)
# New items are now immediately available in retrieve() calls.
```

### Outcome weighting

Weight interacted items by their outcome value (e.g. star ratings):

```python
retriever = ContentBasedRetriever(
    top_k=200,
    weight_by_outcome=True,   # items rated 5 stars count more than 1-star items
)
```

**Requirements**: Only numeric columns are supported. Encode categoricals
(one-hot or ordinal) before passing the items DataFrame. `ITEM_ID_NAME` must
not appear in `feature_columns` — it is an identifier, not a feature, and
passing it explicitly raises a `ValueError`.

---

## PopularityRetriever

Returns the globally most interacted items for every user. No personalization, but
fast and requires no additional setup.

```python
from skrec.retriever import PopularityRetriever

recommender = RankingRecommender(
    scorer=UniversalScorer(estimator=XGBClassifierEstimator({...})),
    retriever=PopularityRetriever(top_k=200),
)
```

**Use as a baseline** to measure the lift from personalized retrievers. The ranker
still personalizes within the popular candidate pool.

---

## External Retrieval via `item_subset`

If your retrieval system lives outside the library (Elasticsearch, proprietary ANN
service, business rules), pass candidates directly via `item_subset`:

```python
# Retrieve candidates from your own system
candidates = elasticsearch.query(user_id, top_k=500)

# Pass them to the ranker
recommender.set_item_subset(candidates)
recommendations = recommender.recommend(interactions=df, top_k=10)
recommender.clear_item_subset()
```

This bypasses the built-in retriever entirely. You can combine approaches:
set a `retriever` for batch traffic and override with `set_item_subset()` for
specific requests.

---

## Custom Retriever

The built-in retrievers cover common cases, but the architecture is designed to be
extended. Subclassing `BaseCandidateRetriever` with two methods is all it takes to
plug in any retrieval system — an ANN index, a vector database, or a domain-specific
heuristic — and it integrates transparently with `RankingRecommender.train()` and
`recommend()`.

Here is a FAISS-backed retriever as an example (~30 lines):

```python
import numpy as np
import faiss
from skrec.constants import ITEM_EMBEDDING_NAME, ITEM_ID_NAME, USER_EMBEDDING_NAME, USER_ID_NAME
from skrec.retriever.base_retriever import BaseCandidateRetriever

class FAISSRetriever(BaseCandidateRetriever):
    def __init__(self, top_k: int = 200):
        super().__init__(top_k=top_k)
        self._index = None
        self._item_ids = None
        self._user_emb_by_id = None  # user_id -> embedding (float32 array)

    def build_index(self, estimator, interactions=None, items=None):
        # Build item ANN index.
        item_emb_df = estimator.get_item_embeddings()
        matrix = np.stack(item_emb_df[ITEM_EMBEDDING_NAME].values).astype("float32")
        faiss.normalize_L2(matrix)
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)
        self._item_ids = item_emb_df[ITEM_ID_NAME].values

        # Cache user embeddings. get_user_embeddings() takes no arguments —
        # it returns all user embeddings as a DataFrame.
        user_emb_df = estimator.get_user_embeddings()
        user_ids_arr = user_emb_df[USER_ID_NAME].values
        user_vecs_arr = np.stack(user_emb_df[USER_EMBEDDING_NAME].values).astype("float32")
        self._user_emb_by_id = dict(zip(user_ids_arr, user_vecs_arr))

    def retrieve(self, user_ids, top_k):
        # Look up each user's embedding. Unknown users are not handled here —
        # add a cold-start fallback (e.g. popularity) for production use.
        user_vecs = np.stack([self._user_emb_by_id[uid] for uid in user_ids])
        faiss.normalize_L2(user_vecs)
        _, indices = self._index.search(user_vecs, top_k)
        return {uid: self._item_ids[idx].tolist()
                for uid, idx in zip(user_ids, indices)}

# Use it like any built-in retriever
recommender = RankingRecommender(
    scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=64)),
    retriever=FAISSRetriever(top_k=200),
)
```

---

## Combining Retrievers

Built-in retrievers are single-stage, but you can combine approaches manually by
merging candidate lists before passing to `item_subset`:

```python
embedding_candidates = embedding_retriever.retrieve([user_id], top_k=150)[user_id]
popular_candidates   = popularity_retriever.retrieve([user_id], top_k=50)[user_id]

combined = list(dict.fromkeys(embedding_candidates + popular_candidates))  # dedup, preserve order
recommender.set_item_subset(combined)
recommendations = recommender.recommend(interactions=df, top_k=10)
recommender.clear_item_subset()
```

---

## Setting the Candidate Pool Size (`top_k`)

The retriever's `top_k` controls the candidate pool, not the final recommendation count.
A larger pool gives the ranker more signal but increases ranking cost.

A reasonable starting point: set retriever `top_k` to 10–20× the final recommendation
count (e.g. retrieve 200 candidates, recommend top 10). Tune based on your catalog size
and latency budget.

Final `top_k` passed to `recommend()` is independent and should be much smaller
(typically 5–50).

---

## Notebook Example

See `nb_examples/retrieval_two_stage.ipynb` for a complete end-to-end walkthrough
covering all three retrievers, post-training items, and external `item_subset` usage.

## Next Steps

- **[RankingRecommender](../recommender-types/ranking.md)** — full ranking API reference
- **[Collaborative Filtering](collaborative-filtering.md)** — EmbeddingRetriever requires an embedding estimator
- **[Architecture Overview](architecture.md)** — understand where retrieval fits
