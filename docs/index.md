# Recommender Library Documentation

Welcome to the **recommender** library documentation! This library provides a scikit-learn style framework for building, training, and evaluating production-ready recommendation systems.

## What is Recommender?

The **recommender** library is a comprehensive Python package that standardizes the development of recommendation models. It provides:

- 🏗️ **Modular 3-layer architecture** (Recommender → Scorer → Estimator)
- 📊 **Multiple evaluation strategies** (Simple, IPS, DR, SNIPS, and more)
- 🎯 **Rich metrics library** (NDCG, MAP, Precision, ROC-AUC, Expected Reward)
- 🚀 **Production-ready** with real-time and batch inference support
- 🔧 **Hyperparameter optimization** at both estimator and recommender levels
- 🎛️ **Config-driven orchestration** for Kubeflow pipelines and MLOps

## Quick Links

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting Started](getting-started/quick-start.md)**

    ---
    5-minute tutorial to build your first recommender

-   :material-book-open-variant: **[User Guide](user-guide/architecture.md)**

    ---
    Complete walkthrough of the library's components

-   :material-view-dashboard: **[Recommender Types](recommender-types/comparison.md)**

    ---
    Detailed guides for each recommender type

-   :material-cog: **[Advanced Topics](advanced/hpo.md)**

    ---
    HPO, production deployment, and hybrid strategies

</div>

## Key Features

### 3-Layer Architecture

The library uses a clean, modular architecture that separates concerns:

```
Recommender (Business Logic)
    ↓
Scorer (Item Scoring Strategy)
    ↓
Estimator (ML Model)
```

This design allows you to mix and match components to build custom recommendation systems.

**Learn more:** [Architecture Overview](user-guide/architecture.md) · [Capability matrix](user-guide/capability-matrix.md)

### Multiple Recommender Types

| Recommender | Best For |
|------------|----------|
| **RankingRecommender** | Standard ranking; also powers Two-Tower, NeuralFactorization, NCF embedding models |
| **SequentialRecommender** | Order-aware history (SASRec transformer) |
| **HierarchicalSequentialRecommender** | Session-structured history with cross-session memory (HRNN) |
| **ContextualBanditsRecommender** | Exploration & A/B testing |
| **UpliftRecommender** | Causal impact estimation (T/S/X-Learner) |
| **GcslRecommender** | Multi-objective optimization (GCSL) |

**Learn more:** [Recommender Types Comparison](recommender-types/comparison.md)

### Comprehensive Evaluation

Built-in support for state-of-the-art evaluation techniques:

- **SimpleEvaluator**: Standard on-policy evaluation
- **ReplayMatchEvaluator**: Replay-based evaluation
- **IPSEvaluator**: Inverse Propensity Scoring (off-policy)
- **DREvaluator**: Doubly Robust estimation
- **SNIPSEvaluator**: Self-Normalized IPS
- **PolicyWeightedEvaluator**: Policy-weighted evaluation

**Learn more:** [Evaluation Guide](user-guide/evaluation.md)

## Quick Example

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.universal import UniversalScorer
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_users,
    sample_binary_reward_items
)

# Build the pipeline: Estimator → Scorer → Recommender
estimator = XGBClassifierEstimator({"learning_rate": 0.1, "n_estimators": 100})
scorer = UniversalScorer(estimator)
recommender = RankingRecommender(scorer)

# Train
recommender.train(
    interactions_ds=sample_binary_reward_interactions,
    users_ds=sample_binary_reward_users,
    items_ds=sample_binary_reward_items
)

# Recommend
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5
)
```

## Version Information

!!! info "Version Status"
    - **Recommender 3.0**: Currently in testing mode
    - **Recommender 2.x**: Stable and production-ready

## Support

- **Issues**: Report bugs in [GitHub Issues](https://github.com/intuit/scikit-rec/issues)

## Next Steps

<div class="grid" markdown>

1. **[Install the library](getting-started/installation.md)**
2. **[Follow the quick-start tutorial](getting-started/quick-start.md)**
3. **[Understand the architecture](user-guide/architecture.md)**
4. **[Choose your recommender type](recommender-types/comparison.md)**

</div>

