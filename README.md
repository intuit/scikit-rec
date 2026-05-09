# scikit-rec

A composable, scikit-style recommender systems library.

**scikit-rec** provides a 3-layer architecture that cleanly separates business logic, scoring strategy, and ML models. Any recommender works with any compatible scorer and estimator, giving you a mix-and-match toolkit for building recommendation systems.

```
Recommender (business logic)  -->  Scorer (item scoring)  -->  Estimator (ML model)
```

### Why scikit-rec?

**Composable by design.** Each layer is independently extensible. Swap XGBoost for a Two-Tower model without changing your recommender. Add a new bandit strategy without touching the scorer. The library spans XGBoost, LightGBM, and scikit-learn alongside deep learning models (NCF, Two-Tower, DeepFM, SASRec, HRNN), with GPU optional — a pure-NumPy matrix factorization (ALS/SGD) requires no PyTorch. The composable architecture also accommodates novel research: a Goal-Conditioned Supervised Learning (GCSL) recommender for multi-objective recommendation was implemented as a single `Recommender` subclass — no new scorer or estimator required. Contributions welcome: implement one abstract class and it works with everything else.

**Beyond ranking.** Contextual bandits (epsilon-greedy, static-action) and heterogeneous treatment effect estimation (T/S/X-Learner) are first-class paradigms, not afterthoughts. All share the same evaluation infrastructure, so you can directly compare a ranking policy against a bandit or uplift policy on the same logged data.

**Production-grade evaluation.** The most complete offline policy evaluation suite in any recommendation library: IPS, Doubly Robust, SNIPS, Direct Method, Policy-Weighted, and Replay Match, paired with ten ranking, classification, and regression metrics (Precision, Recall, MAP, MRR, NDCG, ROC-AUC, PR-AUC, Expected Reward, RMSE, MAE) — enabling counterfactual policy comparison from logged data with a single call. Multi-label classification and multi-target regression workloads (wide-format `MultioutputScorer`) get per-label diagnostics and macro-averaged metrics out of the same `evaluate()` API.

**Production readiness.** Config-driven pipeline factory with Optuna HPO, low-latency single-user inference (`recommend_online`), two-stage retrieval-then-ranking, and batch training.

**Agent-friendly.** Optionally pair with [scikit-rec-agent](https://github.com/intuit/scikit-rec-agent/) so an LLM agent can build, train, and tune models against this library's contracts.

**Learn by example.** Ten end-to-end Jupyter notebooks on MovieLens 1M cover ranking, bandits, uplift, sequential recommendations, multi-objective optimization, hyperparameter tuning, two-stage retrieval, and contextual two-tower models. Our SASRec achieves HR@10 = 0.8953 and NDCG@10 = 0.6331 on MovieLens-1M (leave-last-out, 1 positive + 100 negatives). Each notebook downloads data, trains, evaluates, and shows sample recommendations — ready to run.

## Installation

```bash
pip install scikit-rec
```

Optional extras:

```bash
pip install scikit-rec[torch]    # Deep learning models (DeepFM, NCF, SASRec, HRNN, Two-Tower)
pip install scikit-rec[aws]      # S3 data loading
```

## Quick Start

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.scorer.universal import UniversalScorer
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_users,
    sample_binary_reward_items,
)

# Build the pipeline: Estimator -> Scorer -> Recommender
estimator = XGBClassifierEstimator({"learning_rate": 0.1, "max_depth": 5})
scorer = UniversalScorer(estimator)
recommender = RankingRecommender(scorer)

# Train
recommender.train(
    interactions_ds=sample_binary_reward_interactions,
    users_ds=sample_binary_reward_users,
    items_ds=sample_binary_reward_items,
)

# Recommend
interactions_df = sample_binary_reward_interactions.fetch_data()
users_df = sample_binary_reward_users.fetch_data()
recommendations = recommender.recommend(interactions=interactions_df, users=users_df, top_k=5)
```

## Components

### Recommenders

| Recommender | Description |
|---|---|
| `RankingRecommender` | Rank items by predicted score |
| `ContextualBanditsRecommender` | Exploration-exploitation strategies (epsilon-greedy, static action) |
| `UpliftRecommender` | Uplift modeling (S-Learner, T-Learner, X-Learner) |
| `SequentialRecommender` | Sequence-aware recommendations |
| `HierarchicalSequentialRecommender` | Session-aware hierarchical sequences (HRNN) |
| `GcslRecommender` | Multi-objective goal-conditioned supervised learning |

### Scorers

| Scorer | Description |
|---|---|
| `UniversalScorer` | Single global model using item features (auto-dispatches tabular vs. embedding) |
| `IndependentScorer` | Separate model per item |
| `MulticlassScorer` | Items as competing classes |
| `MultioutputScorer` | Wide-format multi-label binary classification or multi-target regression (one `ITEM_<name>` column per target) |
| `SequentialScorer` | For sequential estimators (SASRec) |
| `HierarchicalScorer` | For HRNN estimators |

### Estimators

| Type | Models |
|---|---|
| **Tabular** | XGBoost, LightGBM, Logistic Regression, sklearn classifiers/regressors, DeepFM |
| **Embedding** | Matrix Factorization, NCF, Two-Tower, Deep Cross Network, Neural Factorization Machine |
| **Sequential** | SASRec, HRNN |

### Evaluators

| Evaluator | Description |
|---|---|
| `SimpleEvaluator` | Standard offline evaluation on held-out data |
| `IPSEvaluator` | Inverse Propensity Scoring for counterfactual evaluation |
| `DREvaluator` | Doubly Robust — combines direct estimation with IPS |
| `SNIPSEvaluator` | Self-Normalized IPS — reduces variance of IPS |
| `DirectMethodEvaluator` | Uses a reward model to estimate policy value |
| `PolicyWeightedEvaluator` | Weights logged rewards by policy/logging probability ratio |
| `ReplayMatchEvaluator` | Unbiased evaluation using only matching logged actions |

### Metrics

Precision@k, Recall@k, MAP, MRR, NDCG, ROC-AUC, PR-AUC, Expected Reward, RMSE, MAE (the last two for multi-target regression via `MultioutputScorer`).

### Retrievers

Two-stage retrieval: Popularity, Content-Based, Embedding-Based.

## Example Notebooks

Notebooks are grouped by dataset under [examples/](examples/):

### MovieLens-1M ([examples/movielens-1m/](examples/movielens-1m/))

| Notebook | What it demonstrates |
|---|---|
| [Ranking with XGBoost](examples/movielens-1m/ranking_xgboost_movielens1m.ipynb) | Feature-based ranking with demographics and genre features |
| [GCSL Multi-Objective](examples/movielens-1m/gcsl_multi_objective_movielens1m.ipynb) | Goal-conditioned recommendations — steer quality vs. novelty |
| [HPO with Optuna](examples/movielens-1m/hpo_xgboost_movielens1m.ipynb) | Hyperparameter tuning with TPE, GP, and CMA-ES samplers |
| [Two-Tower Models](examples/movielens-1m/contextualized_two_tower_context_modes.ipynb) | Three context modes: user_tower, trilinear, scoring_layer |
| [SASRec (Positives)](examples/movielens-1m/sasrec_movielens1m_positives.ipynb) | Self-attentive sequential recommendation on positive interactions |
| [SASRec (Ratings)](examples/movielens-1m/sasrec_movielens1m_ratings.ipynb) | SASRec with explicit ratings as soft labels |
| [SASRec (MSE)](examples/movielens-1m/sasrec_movielens1m_ratings_mse.ipynb) | SASRec regressor with MSE loss |
| [HRNN](examples/movielens-1m/hrnn_movielens1m.ipynb) | Hierarchical RNN for session-aware recommendations |

### Amazon Books ([examples/amazon-books/](examples/amazon-books/))

| Notebook | What it demonstrates |
|---|---|
| [LightGBM](examples/amazon-books/lightgbm_amazon_books.ipynb) | Fast tabular ranking on Amazon Books with categorical book metadata |
| [DeepFM](examples/amazon-books/deepfm_amazon_books.ipynb) | Sparse-feature interaction learning with FM + MLP + cross network |
| [NCF](examples/amazon-books/ncf_amazon_books.ipynb) | Neural Collaborative Filtering with NeuMF + embedding-based two-stage retrieval |
| [SASRec (Positives)](examples/amazon-books/sasrec_amazon_books_positives.ipynb) | Sequential recommendation — mirrors the original SASRec paper's Amazon protocol |

### Generic ([examples/generic/](examples/generic/)) — dataset-agnostic library mechanics

| Notebook | What it demonstrates |
|---|---|
| [Factory Pipeline](examples/generic/factory_pipeline_demo.ipynb) | Config-driven recommender construction on the shipped sample dataset |
| [Two-Stage Retrieval](examples/generic/retrieval_two_stage.ipynb) | Popularity, content-based, and embedding retrieval + ranking on synthetic data |
| [Uplift Modeling](examples/generic/uplift_modeling.ipynb) | S-Learner, T-Learner, X-Learner treatment effect estimation on synthetic data |

The MovieLens-1M notebooks download data automatically from GroupLens. The four Amazon Books
notebooks share a cached download from McAuley-Lab's *Amazon Reviews 2023* via HuggingFace
`datasets` (the first run pays the download cost; subsequent runs reuse the cache).
All notebooks include training, evaluation, and sample recommendations.

## Documentation

Full documentation is available at **[intuit.github.io/scikit-rec](https://intuit.github.io/scikit-rec)**.

- [Architecture overview](https://intuit.github.io/scikit-rec/user-guide/architecture/)
- [Capability matrix](https://intuit.github.io/scikit-rec/user-guide/capability-matrix/)
- [Quick start tutorial](https://intuit.github.io/scikit-rec/getting-started/quick-start/)
- [Dataset preparation](https://intuit.github.io/scikit-rec/getting-started/datasets/)
- [Evaluation guide](https://intuit.github.io/scikit-rec/user-guide/evaluation/)

## Development

```bash
git clone https://github.com/intuit/scikit-rec.git
cd scikit-rec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

## License

[Apache 2.0](LICENSE)
