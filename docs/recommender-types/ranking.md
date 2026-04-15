# RankingRecommender

The **RankingRecommender** is the most commonly used recommender in the library. It predicts the likelihood of positive rewards for each item and ranks them accordingly.

## Overview

**Purpose**: Standard ranking and recommendation tasks where you want to rank items by predicted engagement, click-through rate, conversion probability, or expected reward.

**Key Insight**: The recommender learns patterns from historical interactions to predict which items users are most likely to engage with, then ranks items deterministically by score.

## When to Use

✅ **Perfect For**:
- E-commerce product recommendations
- Content recommendation (articles, videos, music)
- Search result ranking
- Email campaign item selection
- Any standard ranking task with interaction data

❌ **Not Ideal For**:
- Need explicit exploration strategies → Use [ContextualBanditsRecommender](bandits.md)
- Measuring causal impact → Use [UpliftRecommender](uplift.md)

## Basic Usage

### 1. Build the Pipeline

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.universal import UniversalScorer

# Layer 1: Choose an estimator
estimator = XGBClassifierEstimator({
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 5
})

# Layer 2: Choose a scorer
scorer = UniversalScorer(estimator)

# Layer 3: Create the recommender
recommender = RankingRecommender(scorer)
```

### 2. Train the Model

```python
recommender.train(
    interactions_ds=interactions_dataset,
    users_ds=users_dataset,
    items_ds=items_dataset
)
```

### 3. Make Recommendations

```python
import pandas as pd

# Prepare inference data
interactions_df = pd.DataFrame({"USER_ID": ["user_1", "user_2"]})
users_df = pd.DataFrame({
    "USER_ID": ["user_1", "user_2"],
    "age": [25, 35],
    "location": ["CA", "TX"]
})

# Get top-5 recommendations
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5
)

print(recommendations)
# Output: array of shape (2, 5) with top 5 items for each user
```

## Advanced Features

### 1. Deterministic Ranking (Default)

By default, the recommender ranks items deterministically by score:

```python
# Always returns the same top-k items for the same input
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5
)
```

### Probabilistic Sampling (Exploration) {#probabilistic-sampling-exploration}

Use `sampling_temperature` to add randomness for exploration:

```python
# Sample from probability distribution based on scores
# Lower temperature → closer to deterministic
# Higher temperature → more random
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5,
    sampling_temperature=0.3,  # Controls randomness
    replace=False  # Sample without replacement
)
```

**How it works**:
1. Scores are converted to probabilities using softmax with temperature
2. Items are sampled based on these probabilities
3. Temperature controls the "softness" of the distribution

**Use cases**:
- Breaking near-ties randomly
- Gentle exploration without full bandit strategies
- Diversifying recommendations

### 3. Item Subsetting

Recommend from a specific subset of items:

```python
# Only recommend from a pre-filtered list
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5,
    item_subset=["item_A", "item_B", "item_C", "item_D"]
)
```

**Use cases**:
- Apply business rules (e.g., in-stock items only)
- Category-specific recommendations
- Staged filtering (rules first, then ranking)

### 4. Two-Stage Retrieval (Large Catalogs)

At scale, scoring every item for every user is too slow. Attach a retriever to narrow
the candidate set before ranking:

```python
from skrec.retriever import EmbeddingRetriever, ContentBasedRetriever, PopularityRetriever

# Personalized retrieval via learned embeddings (requires embedding estimator)
recommender = RankingRecommender(
    scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=64)),
    retriever=EmbeddingRetriever(top_k=200),
)

# Feature-based retrieval (works with any estimator, supports new post-training items)
recommender = RankingRecommender(
    scorer=UniversalScorer(estimator=XGBClassifierEstimator({...})),
    retriever=ContentBasedRetriever(top_k=200, feature_columns=["price", "rating"]),
)

# Popularity baseline
recommender = RankingRecommender(
    scorer=UniversalScorer(estimator=XGBClassifierEstimator({...})),
    retriever=PopularityRetriever(top_k=200),
)
```

The retriever is built automatically at the end of `train()`. `recommend()` runs
retrieval and ranking transparently in one call.

For external retrieval systems (Elasticsearch, custom ANN), pass candidates via
`item_subset` instead:

```python
candidates = elasticsearch.query(user_id, top_k=500)
recommendations = recommender.recommend(
    interactions=df,
    item_subset=candidates,
    top_k=10,
)
```

**Learn more**: [Two-Stage Retrieval Guide](../user-guide/retrieval.md)

### 5. Getting Item Scores

To see underlying scores:

```python
# Get scores for all items
scores = recommender.score_items(
    interactions=interactions_df,
    users=users_df
)
# Returns: DataFrame with rows=users, columns=items, values=scores
```

### 6. Set Item Subset Globally

For inference efficiency, set item subset once:

```python
# Set globally for all subsequent recommendations
recommender.set_item_subset(["item_A", "item_B", "item_C"])

# Now all recommend() calls use this subset
recommendations = recommender.recommend(interactions_df, users_df, top_k=3)
```

## Scorer Compatibility

RankingRecommender works with **all scorer types**:

### Universal Scorer (Most Common)
```python
from skrec.scorer.universal import UniversalScorer

scorer = UniversalScorer(estimator)
recommender = RankingRecommender(scorer)
```
- Builds a single global model
- Uses item features
- **Requires items dataset**

### Independent Scorer
```python
from skrec.scorer.independent import IndependentScorer

scorer = IndependentScorer(estimator)
recommender = RankingRecommender(scorer)
```
- Builds separate model per item
- Doesn't use item features
- **Does not require items dataset**

### Multiclass Scorer
```python
from skrec.scorer.multiclass import MulticlassScorer

scorer = MulticlassScorer(estimator)
recommender = RankingRecommender(scorer)
```
- Treats items as competing classes
- One row per user in interactions
- **Does not require items or users dataset**

### Multioutput Scorer
```python
from skrec.scorer.multioutput import MultioutputScorer

scorer = MultioutputScorer(estimator)
recommender = RankingRecommender(scorer)
```
- Multiple outcomes per user
- One row per user with `OUTCOME_*` columns
- **Does not require items or users dataset**

**Learn more**: [Scorer Selection Guide](../user-guide/scorers.md)

## Evaluation

```python
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
import numpy as np

# Prepare ground truth
eval_data = {
    "logged_items": np.array([["item_A"], ["item_B"]]),  # Actual items
    "logged_rewards": np.array([[1.0], [0.5]])            # Actual rewards
}

# Evaluate
ndcg = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.NDCG_AT_K,
    eval_top_k=5,
    score_items_kwargs={"interactions": interactions_df, "users": users_df},
    eval_kwargs=eval_data
)

print(f"NDCG@5: {ndcg:.4f}")
```

**Available Metrics**:
- `PRECISION_AT_K`: Precision@k
- `NDCG_AT_K`: Normalized Discounted Cumulative Gain
- `MAP_AT_K`: Mean Average Precision
- `MRR_AT_K`: Mean Reciprocal Rank
- `ROC_AUC`: ROC-AUC score
- `PR_AUC`: Precision-Recall AUC
- `AVERAGE_REWARD_AT_K`: Expected reward

**Available Evaluators**:
- `SIMPLE`: Standard on-policy evaluation
- `REPLAY_MATCH`: Replay-based evaluation
- `IPS`: Inverse Propensity Scoring (off-policy)
- `DR`: Doubly Robust (off-policy)
- `SNIPS`: Self-Normalized IPS
- `POLICY_WEIGHTED`: Policy-weighted evaluation

**Learn more**: [Evaluation Guide](../user-guide/evaluation.md)

## Complete Example

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.universal import UniversalScorer
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_users,
    sample_binary_reward_items
)
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
import pandas as pd
import numpy as np

# 1. Build pipeline
estimator = XGBClassifierEstimator({"learning_rate": 0.1, "n_estimators": 100})
scorer = UniversalScorer(estimator)
recommender = RankingRecommender(scorer)

# 2. Train
recommender.train(
    interactions_ds=sample_binary_reward_interactions,
    users_ds=sample_binary_reward_users,
    items_ds=sample_binary_reward_items
)

# 3. Recommend
inference_interactions = pd.DataFrame({"USER_ID": ["user_1", "user_2"]})
inference_users = pd.DataFrame({
    "USER_ID": ["user_1", "user_2"],
    "age": [25, 35],
    "income": [50000, 75000]
})

# Deterministic top-5
recommendations = recommender.recommend(
    interactions=inference_interactions,
    users=inference_users,
    top_k=5
)
print("Top-5 recommendations:", recommendations)

# With exploration
recommendations_explore = recommender.recommend(
    interactions=inference_interactions,
    users=inference_users,
    top_k=5,
    sampling_temperature=0.3,
    replace=False
)
print("Top-5 with exploration:", recommendations_explore)

# 4. Evaluate
eval_data = {
    "logged_items": np.array([["item_3"], ["item_2"]]),
    "logged_rewards": np.array([[1.0], [1.0]])
}

ndcg = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.NDCG_AT_K,
    eval_top_k=5,
    score_items_kwargs={"interactions": inference_interactions, "users": inference_users},
    eval_kwargs=eval_data
)
print(f"NDCG@5: {ndcg:.4f}")
```

## Best Practices

### 1. Feature Engineering
- Include relevant user and item features
- Normalize numerical features
- Use appropriate categorical encoding

### 2. Hyperparameter Tuning
- Use [Recommender-level HPO](../advanced/hpo.md) to optimize end-to-end
- Or use [Estimator-level HPO](../user-guide/estimators.md#tuned-estimators) for faster iteration

### 3. Evaluation Strategy
- Use multiple metrics (NDCG, Precision, Expected Reward)
- Test with different evaluators (Simple, IPS, DR)
- Validate on held-out temporal data

### 4. Production Deployment
- Use `recommend_online()` for real-time single-user inference (avoids joins)
- Cache scores when possible
- Set `item_subset` to reduce computation

### 5. Exploration
- Start with deterministic recommendations
- Add `sampling_temperature` for gentle exploration
- Consider [ContextualBanditsRecommender](bandits.md) for explicit exploration

## Common Issues

### Issue: Recommendations are always the same

**Solution**: This is expected for deterministic ranking. Use `sampling_temperature` for variety, or use [ContextualBanditsRecommender](bandits.md).

### Issue: Low scores for all items

**Solution**: Check feature scaling and model training. Use `score_items()` to debug scores.

### Issue: Items dataset required error

**Solution**: Universal scorer requires items dataset. Use Independent scorer if you don't have item features.

## Next Steps

- **[Two-Stage Retrieval Guide](../user-guide/retrieval.md)** - Scale to large catalogs
- **[ContextualBanditsRecommender](bandits.md)** - Add explicit exploration
- **[Evaluation Guide](../user-guide/evaluation.md)** - Deep dive into evaluation
- **[HPO Guide](../advanced/hpo.md)** - Optimize hyperparameters
- **[Production Guide](../advanced/production.md)** - Deploy to production

