# Example Notebooks

End-to-end worked examples on real datasets. All notebooks are in the `nb_examples/` directory at the root of the repository.

---

## Sequential Recommendations — SASRec on MovieLens-1M

Three notebooks showing SASRec on the MovieLens-1M dataset (leave-last-out evaluation, sampled 1 positive + 100 random negatives).

| Notebook | Loss | Labels | HR@10 | NDCG@10 |
|---|---|---|---|---|
| `nb_examples/sasrec_movielens1m_positives.ipynb` | BCE | Positives only (rating ≥ 4) | **0.895** | **0.633** |
| `nb_examples/sasrec_movielens1m_ratings.ipynb` | BCE | All ratings as soft labels | 0.855 | 0.573 |
| `nb_examples/sasrec_movielens1m_ratings_mse.ipynb` | MSE | All ratings (regression) | 0.822 | 0.557 |

**What they cover:**

- Loading and preprocessing MovieLens-1M with the `InteractionsDataset` API
- Configuring `SASRecClassifierEstimator` / `SASRecRegressorEstimator`
- Wiring up `SequentialRecommender` with early stopping and validation
- Sampled leave-last-out evaluation with HR@K and NDCG@K metrics

---

## Hierarchical Session Recommendation — HRNN on MovieLens-1M

`nb_examples/hrnn_movielens1m.ipynb`

Demonstrates `HierarchicalSequentialRecommender` (HRNN) on MovieLens-1M, segmenting interactions into sessions via a configurable `session_timeout_minutes` gap.

**What it covers:**

- Session construction and the two-level GRU architecture (session GRU + user GRU)
- Choosing `session_timeout_minutes` for different datasets
- Comparing within-session vs. cross-session patterns
- Early stopping with validation data

---

## Two-Stage Retrieval

`nb_examples/retrieval_two_stage.ipynb`

End-to-end walkthrough of the retrieval layer plugged into `RankingRecommender`.

**What it covers:**

- `EmbeddingRetriever` with `MatrixFactorizationEstimator` — personalized retrieval via learned embeddings
- `ContentBasedRetriever` — feature-based retrieval that immediately surfaces post-training items
- `PopularityRetriever` — non-personalized baseline
- External retrieval via `item_subset` for integration with Elasticsearch or custom ANN systems
- Choosing retriever `top_k` and measuring the recall/speed trade-off

---

## Contextualized Two-Tower — Context Mode Comparison

`nb_examples/contextualized_two_tower_context_modes.ipynb`

Side-by-side comparison of all three `ContextMode` variants on a synthetic dataset.

**What it covers:**

- `USER_TOWER` — context fused into the user tower; ANN-compatible at request time
- `TRILINEAR` — context modulates user embedding via Hadamard product; user embeddings precomputable offline
- `SCORING_LAYER` — most expressive; context, user, and item representations combined in a final linear layer

See the [Contextualized Two-Tower Guide](../user-guide/two-tower.md) for architecture details and ANN serving patterns.
