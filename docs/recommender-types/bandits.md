# ContextualBanditsRecommender

The **ContextualBanditsRecommender** implements contextual multi-armed bandit strategies that explicitly balance **exploration** (trying new items to learn) vs **exploitation** (recommending known good items).

## Overview

**Purpose**: Online learning scenarios where you need to continuously learn from user feedback while balancing exploration and exploitation.

**Key Insight**: Traditional recommenders always exploit (recommend best-known items). Bandits add controlled exploration to discover potentially better items.

## When to Use

✅ **Perfect For**:
- Cold-start scenarios (new items or users with limited data)
- Online learning systems with continuous feedback
- A/B testing and experimentation
- Need explicit control over exploration rate
- Learning item quality over time

❌ **Not Ideal For**:
- Batch offline recommendations (use [RankingRecommender](ranking.md))
- When you have enough data and don't need exploration

## Available Strategies

### 1. Epsilon-Greedy

**How it works**: With probability ε, select random items (explore); otherwise select top-scoring items (exploit).

**Parameters**:
- `epsilon` (float, 0-1): Probability of exploration

**Example**:
```python
from skrec.recommender.bandits.contextual_bandits import ContextualBanditsRecommender
from skrec.recommender.bandits.datatypes import StrategyType

recommender = ContextualBanditsRecommender(
    scorer=scorer,
    strategy_type=StrategyType.EPSILON_GREEDY,
    strategy_params={"epsilon": 0.1}  # 10% exploration, 90% exploitation
)
```

**Best for**: Simple, interpretable exploration strategy

### 2. Static Action

**How it works**: Always recommend a fixed set of items (no model scoring).

**Parameters**:
- `static_items` (list): List of item IDs to always recommend

**Example**:
```python
recommender = ContextualBanditsRecommender(
    scorer=scorer,
    strategy_type=StrategyType.STATIC_ACTION,
    strategy_params={"static_items": ["item_A", "item_B", "item_C"]}
)
```

**Best for**: Control groups in A/B tests, baseline comparisons

## Basic Usage

### 1. Build the Pipeline

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.recommender.bandits.contextual_bandits import ContextualBanditsRecommender
from skrec.recommender.bandits.datatypes import StrategyType
from skrec.scorer.universal import UniversalScorer

# Create base scorer (same as RankingRecommender)
estimator = XGBClassifierEstimator({"learning_rate": 0.1, "n_estimators": 100})
scorer = UniversalScorer(estimator)

# Create bandit recommender with epsilon-greedy strategy
recommender = ContextualBanditsRecommender(
    scorer=scorer,
    strategy_type=StrategyType.EPSILON_GREEDY,
    strategy_params={"epsilon": 0.2}  # 20% exploration
)
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

# Get recommendations (automatically applies bandit strategy)
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5
)

print(recommendations)
# Some items will be exploratory (random), others exploitative (top-scoring)
```

### 4. Track Exploration vs Exploitation

```python
# Get flags indicating which recommendations were exploratory
strategy_flags = recommender.get_latest_strategy_flags()

print(strategy_flags)
# Output: array([[0, 1, 0, 0, 0],  # User 1: 2nd item was exploratory
#                [0, 0, 0, 1, 0]]) # User 2: 4th item was exploratory
# 0 = exploitation (top-scoring item)
# 1 = exploration (random item)
```

**Use case**: Log these flags for analysis and model retraining with proper off-policy correction.

## Complete Example

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.recommender.bandits.contextual_bandits import ContextualBanditsRecommender
from skrec.recommender.bandits.datatypes import StrategyType
from skrec.scorer.universal import UniversalScorer
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_users,
    sample_binary_reward_items
)
import pandas as pd

# 1. Build pipeline
estimator = XGBClassifierEstimator({"learning_rate": 0.1, "n_estimators": 100})
scorer = UniversalScorer(estimator)
recommender = ContextualBanditsRecommender(
    scorer=scorer,
    strategy_type=StrategyType.EPSILON_GREEDY,
    strategy_params={"epsilon": 0.1}
)

# 2. Train
recommender.train(
    interactions_ds=sample_binary_reward_interactions,
    users_ds=sample_binary_reward_users,
    items_ds=sample_binary_reward_items
)

# 3. Recommend
interactions_df = pd.DataFrame({"USER_ID": ["user_1", "user_2", "user_3"]})
users_df = pd.DataFrame({
    "USER_ID": ["user_1", "user_2", "user_3"],
    "age": [25, 35, 45],
    "income": [50000, 75000, 100000]
})

recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5
)

print("Recommendations:", recommendations)

# 4. Check which were exploratory
flags = recommender.get_latest_strategy_flags()
print("Strategy flags (0=exploit, 1=explore):")
print(flags)

# 5. Log for later analysis
exploration_rate = flags.mean()
print(f"Actual exploration rate: {exploration_rate:.2%}")
```

## Tuning Epsilon

### Cold-Start Phase (High Exploration)
```python
# Start with higher epsilon to learn quickly
recommender_coldstart = ContextualBanditsRecommender(
    scorer=scorer,
    strategy_type=StrategyType.EPSILON_GREEDY,
    strategy_params={"epsilon": 0.3}  # 30% exploration
)
```

### Warm-Start Phase (Lower Exploration)
```python
# Reduce epsilon as you collect more data
recommender_warmstart = ContextualBanditsRecommender(
    scorer=scorer,
    strategy_type=StrategyType.EPSILON_GREEDY,
    strategy_params={"epsilon": 0.05}  # 5% exploration
)
```

### Epsilon Decay Schedule
```python
# Implement epsilon decay over time
initial_epsilon = 0.3
min_epsilon = 0.05
decay_rate = 0.995

for iteration in range(num_iterations):
    current_epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** iteration))
    
    # Recreate recommender with new epsilon
    recommender = ContextualBanditsRecommender(
        scorer=scorer,
        strategy_type=StrategyType.EPSILON_GREEDY,
        strategy_params={"epsilon": current_epsilon}
    )
    
    # Make recommendations and collect feedback
    recommendations = recommender.recommend(...)
```

## A/B Testing with Static Action

Use Static Action strategy for control groups:

```python
# Treatment group: Epsilon-greedy bandit
treatment_recommender = ContextualBanditsRecommender(
    scorer=scorer,
    strategy_type=StrategyType.EPSILON_GREEDY,
    strategy_params={"epsilon": 0.1}
)

# Control group: Static baseline items
control_recommender = ContextualBanditsRecommender(
    scorer=scorer,  # Not used for static action, but required
    strategy_type=StrategyType.STATIC_ACTION,
    strategy_params={"static_items": ["item_popular_1", "item_popular_2", "item_popular_3"]}
)

# Assign users to groups
if user_id % 2 == 0:
    recommendations = treatment_recommender.recommend(...)
else:
    recommendations = control_recommender.recommend(...)
```

## Evaluation

Bandit recommenders use the same **`evaluate()` API** as `RankingRecommender`, but the **numbers are policy-aligned**, not “base model only”:

- **Strategy is required** wherever rankings are produced through the bandit policy—the same as for `recommend()`. Call `set_strategy()` (or pass strategy in the constructor) **before** `evaluate()` when your path uses that ranking (notably **`StrategyType.STATIC_ACTION`** with **non-probabilistic** evaluators such as Simple or ReplayMatch). If you skip this, you will get `RuntimeError: Strategy not set. Call set_strategy() before recommend().`
- **What gets measured:** The scorer still produces per-item scores, but **full rankings / target distributions** are built using the **bandit strategy** (exploration, static ordering, blended probabilities, etc.). That matches how the system behaves online. It is **not** the same as ranking items by raw `argmax(score)` unless your strategy reduces to that.
- **Base-model-only metrics:** Use [`RankingRecommender`](ranking.md) (or another non-bandit recommender) with the same scorer if you want offline metrics **without** the exploration policy.

The example below uses an off-policy evaluator; adjust `eval_type` and `eval_kwargs` for your setup.

```python
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
import numpy as np

# For off-policy evaluation, include logging probabilities
eval_data = {
    "logged_items": np.array([["item_A"], ["item_B"]]),
    "logged_rewards": np.array([[1.0], [0.5]]),
    "logging_proba": np.array([[0.7], [0.3]])  # Probability of logged action
}

# Use IPS or DR evaluator for off-policy evaluation
ndcg = recommender.evaluate(
    eval_type=RecommenderEvaluatorType.IPS,  # Off-policy evaluator
    metric_type=RecommenderMetricType.NDCG_AT_K,
    eval_top_k=5,
    score_items_kwargs={"interactions": interactions_df, "users": users_df},
    eval_kwargs=eval_data
)

print(f"Off-policy NDCG@5: {ndcg:.4f}")
```

**Important**: For **off-policy** learning from logged data, prefer evaluators such as IPS, DR, or SNIPS and supply `logging_proba` where required. That path uses `BaseRecommender.evaluate` (probabilistic target policy), which still reflects **strategy-blended** probabilities when the strategy supports them—not raw softmax of scores alone.

**Learn more**: [Evaluation Guide](../user-guide/evaluation.md) (including [bandit semantics](../user-guide/evaluation.md#contextual-bandits-and-evaluate))

## Best Practices

### 1. Epsilon Selection
- **Cold-start**: Start with ε=0.2-0.3
- **Warm-start**: Use ε=0.05-0.1
- **Mature system**: Use ε=0.01-0.05 or switch to RankingRecommender

### 2. Logging
```python
# Always log exploration flags for analysis
flags = recommender.get_latest_strategy_flags()

# Save to database
log_recommendations(
    user_ids=user_ids,
    recommendations=recommendations,
    exploration_flags=flags,
    timestamp=datetime.now()
)
```

### 3. Retraining
- Retrain models regularly with collected feedback
- Use off-policy correction if data was collected under different policy
- Monitor exploration rate and adjust epsilon

### 4. Combining with Rules
```python
# Pre-filter with business rules, then explore/exploit
valid_items = get_in_stock_items()
recommender.set_item_subset(valid_items)
recommendations = recommender.recommend(...)
```

### 5. Monitoring
- Track exploration rate: `flags.mean()`
- Monitor reward for explored vs exploited items
- A/B test different epsilon values

## Comparison with RankingRecommender

| Feature | RankingRecommender | ContextualBanditsRecommender |
|---------|----------------------|------------------------------|
| **Exploration** | Via `sampling_temperature` (gentle) | Explicit strategies (epsilon-greedy) |
| **Control** | Indirect (temperature) | Direct (epsilon) |
| **Tracking** | No built-in flags | `get_latest_strategy_flags()` |
| **Use Case** | Batch/deterministic recommendations | Online learning, A/B testing |
| **Complexity** | Simpler | Slightly more complex |

## When to Switch to RankingRecommender

Consider switching from Bandits to Propensity when:
- ✅ You have collected enough data (thousands of interactions per item)
- ✅ Item quality estimates are stable
- ✅ You want fully deterministic recommendations
- ✅ Exploration is no longer needed

## Common Issues

### Issue: Exploration rate doesn't match epsilon

**Solution**: This is expected. Epsilon is per-position, not per-user. Actual exploration rate ≈ epsilon × top_k / num_items.

### Issue: Static action ignores scorer

**Solution**: This is intentional. Static action is for control groups and doesn't use the model.

### Issue: Recommendations seem random

**Solution**: Check epsilon value. If too high (>0.5), most recommendations will be exploratory.

## Next Steps

- **[RankingRecommender](ranking.md)** - Switch when exploration is no longer needed
- **[Evaluation Guide](../user-guide/evaluation.md)** - Learn about off-policy evaluation
- **[Production Guide](../advanced/production.md)** - Deploy bandit systems to production
- **[HPO Guide](../advanced/hpo.md)** - Optimize epsilon and model hyperparameters

