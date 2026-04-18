# Inference Guide

This guide covers how to make recommendations after training your models.

## Basic Inference

All recommenders follow the same recommendation interface:

```python
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5
)
```

**Parameters**:

- **`interactions`** (optional): DataFrame containing interaction context features and `USER_ID`
- **`users`** (optional): DataFrame containing user features and `USER_ID`
  - For most recommenders, this provides user-level features
  - When using a **`BaseEmbeddingEstimator`** subclass with **`UniversalScorer`** for real-time inference, this DataFrame should contain `USER_ID` and `USER_EMBEDDING_NAME` (pre-computed embeddings), and optionally other user features
- **`top_k`**: Number of items to recommend per user

**Output**: NumPy array of shape `(n_users, top_k)` with recommended item IDs.

## Complete Example

See the [Quick Start Tutorial](../getting-started/quick-start.md#step-5-make-recommendations) for a complete walkthrough.

## Inference Patterns

### 1. Deterministic Ranking (Default)

```python
# Returns top-k items by score (deterministic)
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5
)
```

### 2. Probabilistic Sampling

```python
# Sample from score distribution (exploration)
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5,
    sampling_temperature=0.3,  # Lower = more deterministic
    replace=False               # Sample without replacement
)
```

**Learn more**: [RankingRecommender Guide](../recommender-types/ranking.md#probabilistic-sampling-exploration)

### 3. Item Subsetting

```python
# Recommend from specific item subset
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5,
    item_subset=["item_A", "item_B", "item_C"]
)
```

### 4. Getting Scores

```python
# Get underlying scores for all items
scores = recommender.score_items(
    interactions=interactions_df,
    users=users_df
)
# Returns: DataFrame with rows=users, columns=items, values=scores
```

## Real-Time Inference Optimization

### 1. Single-User Mode (`recommend_online` and `score_fast`)

For real-time APIs with one user at a time, use `recommend_online()` — it skips the
pandas join entirely, applies schema validation, and returns ranked item names directly:

```python
# Preferred: handles schema + feature merging + ranking in one call
recommendations = recommender.recommend_online(
    interactions=single_user_interactions_df,
    users=single_user_df,
    top_k=5,
)
```

For scoring only (without ranking), call `scorer.score_fast()` directly with a
pre-merged single-row DataFrame (no `USER_ID`):

```python
# Pre-merge interactions and user features yourself
features_df = pd.DataFrame({"feat1": [18], "feat2": [0]})  # no USER_ID
scores_df = recommender.scorer.score_fast(features_df)
# Returns: DataFrame with item names as columns
```

**Supported scorers**: `UniversalScorer`, `MulticlassScorer`, `MultioutputScorer`,
and `IndependentScorer`. Not supported for embedding-based estimators
(NCF, Two-Tower, DeepFM) — use `score_items()` for those.

### 2. Parallel Inference (Independent Scorer)

```python
from skrec.scorer.independent import IndependentScorer

scorer = IndependentScorer(estimator)
scorer.set_parallel_inference(parallel_inference_status=True, num_cores=4)

recommender = RankingRecommender(scorer)
# Inference now parallelized across items
```

## Batch vs Real-Time

### Batch Inference
```python
# Process many users at once
large_interactions_df = pd.DataFrame({"USER_ID": user_ids})  # 1000s of users
large_users_df = pd.DataFrame({"USER_ID": user_ids, ...})

# Single batch call
all_recommendations = recommender.recommend(
    interactions=large_interactions_df,
    users=large_users_df,
    top_k=5
)
```

### Real-Time Inference
```python
# Process one user at a time (API endpoint) — no join overhead
def get_recommendations_for_user(user_id, user_features):
    interactions_df = pd.DataFrame({"USER_ID": [user_id]})
    users_df = pd.DataFrame({"USER_ID": [user_id], **user_features})

    recommendations = recommender.recommend_online(
        interactions=interactions_df,
        users=users_df,
        top_k=5,
    )

    return recommendations.tolist()
```

## Inference by Recommender Type

### RankingRecommender
```python
# Standard deterministic or sampled recommendations
recommendations = recommender.recommend(interactions_df, users_df, top_k=5)
```

**Learn more**: [RankingRecommender Guide](../recommender-types/ranking.md)

### ContextualBanditsRecommender
```python
# Recommendations with exploration
recommendations = recommender.recommend(interactions_df, users_df, top_k=5)

# Check which were exploratory
flags = recommender.get_latest_strategy_flags()  # 0=exploit, 1=explore
```

**Learn more**: [ContextualBanditsRecommender Guide](../recommender-types/bandits.md)

### Embedding-Based Inference (Real-Time with Pre-Computed Embeddings)

When using **`BaseEmbeddingEstimator`** subclasses (e.g., `NeuralFactorizationEstimator`, `ContextualizedTwoTowerEstimator`) with **`UniversalScorer`**, you can leverage pre-computed user embeddings for efficient real-time inference:

```python
from skrec.constants import USER_EMBEDDING_NAME
import numpy as np

# Assume embeddings are pre-computed and stored externally (e.g., in Redis)
user_embeddings = get_user_embeddings_from_store(user_ids)  # Shape: (n_users, embedding_dim)

# Create users DataFrame with pre-computed embeddings
users_df = pd.DataFrame({
    "USER_ID": user_ids,
    USER_EMBEDDING_NAME: list(user_embeddings),  # List of numpy arrays
    # Optionally include other user features if model uses them
    "age": user_ages,
    "income": user_incomes
})

interactions_df = pd.DataFrame({
    "USER_ID": user_ids,
    # Include any interaction context features
    "time_of_day": ["morning", "evening"]
})

# Real-time inference with pre-computed embeddings
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,  # Contains pre-computed embeddings
    top_k=5
)
```

**Workflow**:

1. **Training**: Train the embedding estimator on historical data
2. **Extract Embeddings**: Use `estimator.get_user_embeddings()` to extract learned embeddings
3. **Store Embeddings**: Save embeddings to an external store (Redis, database, etc.)
4. **Real-Time Inference**: Fetch embeddings and pass them in the `users` DataFrame with `USER_EMBEDDING_NAME` column

**Benefits**:

- **Faster inference**: No need to re-compute user embeddings at inference time
- **Scalability**: User embeddings can be managed separately (cached, updated periodically)
- **Flexibility**: Works with externally managed embedding stores

**Learn more**: [Embedding Estimators Guide](estimators.md#embedding-estimators)

## Performance Tips

### 1. Pre-filter Items
```python
# Apply business rules before scoring
valid_items = get_in_stock_items()
recommender.set_item_subset(valid_items)
recommendations = recommender.recommend(...)
```

### 2. Batch Requests
```python
# Process multiple users together (more efficient)
# Good: 100 users in one call
recommendations = recommender.recommend(large_batch_df, ...)

# Bad: 100 separate calls
for user in users:
    recommendations = recommender.recommend(single_user_df, ...)
```

### 3. Use Lighter Models
```python
# For real-time: LogisticRegression or LightGBM
estimator = LightGBMClassifierEstimator({"n_estimators": 50})  # Faster

# For batch: XGBoost or DeepFM
estimator = XGBClassifierEstimator({"n_estimators": 200})  # Better quality
```

### 4. Cache Scores
```python
# Cache scores if user context doesn't change frequently
scores_cache = {}

def get_recommendations(user_id):
    if user_id not in scores_cache:
        scores_cache[user_id] = recommender.score_items(...)
    return scores_cache[user_id].nlargest(5)
```

## Common Issues

### Issue: Inference is slow

**Solution**:
- Use `recommend_online()` for single-user requests (no join overhead)
- Enable parallel inference for Independent scorer
- Use lighter models (LightGBM, LogisticRegression)
- Batch requests when possible

### Issue: Out of memory for large batches

**Solution**:
- Process in smaller chunks
- Reduce model complexity
- Use sparse features

### Issue: Recommendations don't match expectations

**Solution**:
- Check `score_items()` to debug scores
- Verify input data format and features
- Check if `item_subset` is set correctly

## Next Steps

- **[Evaluation Guide](evaluation.md)** - Evaluate recommendation quality
- **[Production Guide](../advanced/production.md)** - Deploy to production
- **[Recommender Types](../recommender-types/comparison.md)** - Explore different recommenders

