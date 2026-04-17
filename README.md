# scikit-rec

A composable, scikit-style recommender systems library.

**scikit-rec** provides a 3-layer architecture that cleanly separates business logic, scoring strategy, and ML models. Any recommender works with any compatible scorer and estimator, giving you a mix-and-match toolkit for building recommendation systems.

```
Recommender (business logic)  -->  Scorer (item scoring)  -->  Estimator (ML model)
```

### Why scikit-rec?

**Composable by design.** Each layer is independently extensible. Swap XGBoost for a Two-Tower model without changing your recommender. Add a new bandit strategy without touching the scorer. The library spans XGBoost, LightGBM, and scikit-learn alongside deep learning models (NCF, Two-Tower, DeepFM, SASRec, HRNN), with GPU optional — a pure-NumPy matrix factorization (ALS/SGD) requires no PyTorch. The composable architecture also accommodates novel research: a Goal-Conditioned Supervised Learning (GCSL) recommender for multi-objective recommendation was implemented as a single `Recommender` subclass — no new scorer or estimator required. Contributions welcome: implement one abstract class and it works with everything else.

**Beyond ranking.** Contextual bandits (epsilon-greedy, static-action) and heterogeneous treatment effect estimation (T/S/X-Learner) are first-class paradigms, not afterthoughts. All share the same evaluation infrastructure, so you can directly compare a ranking policy against a bandit or uplift policy on the same logged data.

**Production-grade evaluation.** The most complete offline policy evaluation suite in any recommendation library: IPS, Doubly Robust, SNIPS, Direct Method, Policy-Weighted, and Replay Match, paired with eight ranking and classification metrics (Precision, Recall, MAP, MRR, NDCG, ROC-AUC, PR-AUC, Expected Reward) — enabling counterfactual policy comparison from logged data with a single call.

**Production readiness.** Config-driven pipeline factory with Optuna HPO, low-latency single-user inference (`recommend_online`), two-stage retrieval-then-ranking, and batch training.

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
| `MultioutputScorer` | Multiple outcomes per prediction |
| `SequentialScorer` | For sequential estimators (SASRec) |
| `HierarchicalScorer` | For HRNN estimators |

### Estimators

| Type | Models |
|---|---|
| **Tabular** | XGBoost, LightGBM, Logistic Regression, sklearn classifiers/regressors |
| **Embedding** | Matrix Factorization, NCF, Two-Tower, DCN, DeepFM |
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

Precision@k, Recall@k, MAP, MRR, NDCG, ROC-AUC, PR-AUC, Expected Reward.

### Retrievers

Two-stage retrieval: Popularity, Content-Based, Embedding-Based.

## Example Notebooks

| Notebook | What it demonstrates |
|---|---|
| [Ranking with XGBoost](examples/ranking_xgboost_movielens1m.ipynb) | Feature-based ranking with demographics and genre features |
| [Uplift Modeling](examples/uplift_modeling.ipynb) | S-Learner, T-Learner, X-Learner treatment effect estimation |
| [GCSL Multi-Objective](examples/gcsl_multi_objective_movielens1m.ipynb) | Goal-conditioned recommendations — steer quality vs. novelty |
| [HPO with Optuna](examples/hpo_xgboost_movielens1m.ipynb) | Hyperparameter tuning with TPE, GP, and CMA-ES samplers |
| [Two-Stage Retrieval](examples/retrieval_two_stage.ipynb) | Popularity, content-based, and embedding retrieval + ranking |
| [Two-Tower Models](examples/contextualized_two_tower_context_modes.ipynb) | Three context modes: user_tower, trilinear, scoring_layer |
| [SASRec (Positives)](examples/sasrec_movielens1m_positives.ipynb) | Self-attentive sequential recommendation on positive interactions |
| [SASRec (Ratings)](examples/sasrec_movielens1m_ratings.ipynb) | SASRec with explicit ratings as soft labels |
| [SASRec (MSE)](examples/sasrec_movielens1m_ratings_mse.ipynb) | SASRec regressor with MSE loss |
| [HRNN](examples/hrnn_movielens1m.ipynb) | Hierarchical RNN for session-aware recommendations |

All notebooks use MovieLens 1M (downloaded automatically) and include training, evaluation, and sample recommendations.

## Documentation

See the [docs/](docs/) directory for:

- [Architecture overview](docs/user-guide/architecture.md)
- [Capability matrix](docs/user-guide/capability-matrix.md)
- [Quick start tutorial](docs/getting-started/quick-start.md)
- [Dataset preparation](docs/getting-started/datasets.md)
- [Evaluation guide](docs/user-guide/evaluation.md)

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
