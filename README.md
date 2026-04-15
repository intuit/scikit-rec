# scikit-rec

A composable, scikit-style recommender systems library.

**scikit-rec** provides a 3-layer architecture that cleanly separates business logic, scoring strategy, and ML models. Any recommender works with any compatible scorer and estimator, giving you a mix-and-match toolkit for building recommendation systems.

```
Recommender (business logic)  -->  Scorer (item scoring)  -->  Estimator (ML model)
```

## Installation

```bash
pip install scikit-rec
```

Optional extras:

```bash
pip install scikit-rec[torch]    # Deep learning models (DeepFM, NCF, SASRec, HRNN, Two-Tower)
pip install scikit-rec[aws]      # S3 data loading
pip install scikit-rec[explain]  # SHAP explanations
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

Offline policy evaluation methods: Simple, IPS, Doubly Robust, SNIPS, Direct Method, Policy Weighted, Replay Match.

### Metrics

Precision@k, Recall@k, MAP, MRR, NDCG, ROC-AUC, PR-AUC, Expected Reward.

### Retrievers

Two-stage retrieval: Popularity, Content-Based, Embedding-Based.

## Documentation

See the [docs/](docs/) directory for:

- [Architecture overview](docs/user-guide/architecture.md)
- [Capability matrix](docs/user-guide/capability-matrix.md)
- [Quick start tutorial](docs/getting-started/quick-start.md)
- [Dataset preparation](docs/getting-started/datasets.md)
- [Evaluation guide](docs/user-guide/evaluation.md)
- [Example notebooks](examples/)

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
