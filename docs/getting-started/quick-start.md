# Quick Start

This tutorial will walk you through building your first recommender in 5 minutes using the library's example datasets.

## Overview

We'll build a simple recommendation system that:

1. Trains on historical user-item interactions
2. Predicts which items users are likely to engage with
3. Generates top-5 recommendations for new users

## Step 1: Import Required Components

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.universal import UniversalScorer
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_users,
    sample_binary_reward_items
)
import pandas as pd
```

## Step 2: Load Example Datasets

The library provides sample datasets for experimentation:

```python
# Load datasets (these are Dataset objects)
interactions_ds = sample_binary_reward_interactions
users_ds = sample_binary_reward_users
items_ds = sample_binary_reward_items

# Preview the data
print("Interactions shape:", interactions_ds.fetch_data().shape)
print("Users shape:", users_ds.fetch_data().shape)
print("Items shape:", items_ds.fetch_data().shape)
```

### Dataset Structure

- **Interactions**: Each row represents a user-item interaction with an outcome (e.g., click, purchase)
- **Users**: Each row is a user with features (e.g., age, location)
- **Items**: Each row is an item with features (e.g., category, price)

**Learn more:** [Dataset Preparation Guide](datasets.md)

## Step 3: Build the Recommendation Pipeline

The library uses a 3-layer architecture:

```python
# Layer 1: Estimator (the ML model)
estimator = XGBClassifierEstimator({
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 5
})

# Layer 2: Scorer (how items are scored)
scorer = UniversalScorer(estimator)

# Layer 3: Recommender (business logic)
recommender = RankingRecommender(scorer)
```

**Understanding the layers:**

- **Estimator**: XGBoost classifier that predicts engagement probability
- **Scorer**: Universal scorer builds a single model using all item features
- **Recommender**: RankingRecommender ranks items by predicted engagement

**Learn more:** [Architecture Overview](../user-guide/architecture.md)

## Step 4: Train the Model

```python
recommender.train(
    interactions_ds=interactions_ds,
    users_ds=users_ds,
    items_ds=items_ds
)
print("Training complete!")
```

## Step 5: Make Recommendations

```python
# Create inference data (users for whom we want recommendations)
inference_interactions = pd.DataFrame({
    "USER_ID": ["user_1", "user_2", "user_3"]
})

inference_users = pd.DataFrame({
    "USER_ID": ["user_1", "user_2", "user_3"],
    "feat1": [2000, 1500, 3000],
    "feat2": [100, 50, 200]
})

# Generate top-5 recommendations for each user
recommendations = recommender.recommend(
    interactions=inference_interactions,
    users=inference_users,
    top_k=5
)

print("Recommendations shape:", recommendations.shape)
print("\nTop 5 items for each user:")
print(recommendations)
```

### Output Format

The output is a NumPy array where:
- **Rows**: Users (in the same order as input)
- **Columns**: Top-k recommended items (ranked by score)

```
array([['item_3', 'item_1', 'item_5', 'item_2', 'item_4'],
       ['item_2', 'item_3', 'item_1', 'item_4', 'item_5'],
       ['item_1', 'item_3', 'item_2', 'item_5', 'item_4']], dtype='<U6')
```

## Step 6: Get Item Scores (Optional)

To see the underlying scores:

```python
scores = recommender.score_items(
    interactions=inference_interactions,
    users=inference_users
)

print("Scores for each item:")
print(scores)
```

## Step 7: Evaluate the Model

```python
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
import numpy as np

# Prepare evaluation data
eval_interactions = pd.DataFrame({
    "USER_ID": ["user_1", "user_2"]
})

eval_users = pd.DataFrame({
    "USER_ID": ["user_1", "user_2"],
    "feat1": [2000, 1500],
    "feat2": [100, 50]
})

# Ground truth: actual items users engaged with and rewards
eval_data = {
    "logged_items": np.array([["item_3"], ["item_2"]]),  # Actual items
    "logged_rewards": np.array([[1.0], [1.0]])           # Actual rewards
}

# Evaluate using Precision@5
precision_at_5 = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.PRECISION_AT_K,
    eval_top_k=5,
    score_items_kwargs={
        "interactions": eval_interactions,
        "users": eval_users
    },
    eval_kwargs=eval_data
)

print(f"Precision@5: {precision_at_5:.4f}")
```

**Learn more:** [Evaluation Guide](../user-guide/evaluation.md)

## Complete Example

Here's the complete code in one block:

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

# Step 1: Load datasets
interactions_ds = sample_binary_reward_interactions
users_ds = sample_binary_reward_users
items_ds = sample_binary_reward_items

# Step 2: Build pipeline
estimator = XGBClassifierEstimator({"learning_rate": 0.1, "n_estimators": 100})
scorer = UniversalScorer(estimator)
recommender = RankingRecommender(scorer)

# Step 3: Train
recommender.train(
    interactions_ds=interactions_ds,
    users_ds=users_ds,
    items_ds=items_ds
)

# Step 4: Recommend
inference_interactions = pd.DataFrame({"USER_ID": ["user_1", "user_2"]})
inference_users = pd.DataFrame({
    "USER_ID": ["user_1", "user_2"],
    "feat1": [2000, 1500],
    "feat2": [100, 50]
})

recommendations = recommender.recommend(
    interactions=inference_interactions,
    users=inference_users,
    top_k=5
)

print("Recommendations:", recommendations)

# Step 5: Evaluate
eval_data = {
    "logged_items": np.array([["item_3"], ["item_2"]]),
    "logged_rewards": np.array([[1.0], [1.0]])
}

precision = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.SIMPLE,
    metric_type=RecommenderMetricType.PRECISION_AT_K,
    eval_top_k=5,
    score_items_kwargs={"interactions": inference_interactions, "users": inference_users},
    eval_kwargs=eval_data
)

print(f"Precision@5: {precision:.4f}")
```

## Next Steps

Now that you've built your first recommender, explore:

- **[Dataset Preparation](datasets.md)** - Learn how to prepare your own data
- **[Recommender Types](../recommender-types/comparison.md)** - Explore different recommender options
- **[Architecture Deep Dive](../user-guide/architecture.md)** - Understand the 3-layer design
- **[Scorer Selection](../user-guide/scorers.md)** - Choose the right scorer for your use case
- **[Hyperparameter Optimization](../advanced/hpo.md)** - Optimize your model's performance

