# UpliftRecommender

The **UpliftRecommender** is designed for causal inference scenarios where you want to estimate the **incremental impact (uplift)** of recommending an item, rather than just predicting overall engagement.

## Overview

**Purpose**: Maximize the **causal effect** of recommendations by estimating how much more likely a user is to engage *because of* the recommendation, not just correlation.

**Key Insight**: Traditional recommenders might recommend items users would engage with anyway. Uplift modeling focuses on items where the recommendation makes the biggest difference.

## When to Use

✅ **Perfect For**:
- Maximizing incremental revenue (not total revenue)
- Understanding true treatment effects of recommendations
- Avoiding "preaching to the choir" (recommending what users would choose anyway)
- Marketing campaigns where you want to maximize lift
- Resource-constrained scenarios (limited promotion slots)

❌ **Not Ideal For**:
- Standard ranking tasks (use [RankingRecommender](ranking.md))
- When you don't have treatment/control data

## Causal Inference Strategies

### 1. T-Learner (Two-Model Approach)
- Trains separate models for treatment and control groups
- Uplift = Prediction(treatment) - Prediction(control)

### 2. S-Learner (Single-Model Approach)
- Trains one model with treatment indicator as a feature
- Uplift = Prediction(treated=1) - Prediction(treated=0)

## Basic Usage

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.recommender.uplift_model.uplift_t_learner import UpliftTLearnerRecommender
from skrec.scorer.universal import UniversalScorer

# Create T-Learner uplift recommender
estimator = XGBClassifierEstimator({"learning_rate": 0.1, "n_estimators": 100})
scorer = UniversalScorer(estimator)
recommender = UpliftTLearnerRecommender(scorer)

# Train (requires treatment indicator in interactions data)
recommender.train(
    interactions_ds=interactions_with_treatment_ds,
    users_ds=users_dataset,
    items_ds=items_dataset
)

# Recommend based on uplift scores
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_df,
    top_k=5
)
```

## Data Requirements

Your interactions dataset must include a **treatment indicator**:

```python
interactions_df = pd.DataFrame({
    "USER_ID": ["user_1", "user_2", "user_3"],
    "ITEM_ID": ["item_A", "item_B", "item_C"],
    "OUTCOME": [1.0, 0.0, 1.0],
    "treatment": [1, 0, 1]  # 1=treated (saw recommendation), 0=control
})
```

## Example: Incremental Revenue

```python
# Traditional recommender might recommend item_A
# because it has high overall conversion rate (80%)
# 
# But what if:
# - 75% of users buy item_A WITHOUT recommendation
# - Only 80% buy it WITH recommendation
# - Uplift = 80% - 75% = 5% (small incremental impact)
#
# Meanwhile item_B:
# - 10% of users buy item_B WITHOUT recommendation
# - 40% buy it WITH recommendation  
# - Uplift = 40% - 10% = 30% (large incremental impact!)
#
# Uplift recommender will prefer item_B!
```

## Best Practices

### 1. Data Collection
- Collect randomized treatment/control data
- Ensure sufficient samples in both groups
- Avoid selection bias

### 2. Model Selection
- **T-Learner**: Better when treatment effect is heterogeneous
- **S-Learner**: Simpler, works well with abundant data

### 3. Evaluation
- Measure incremental lift, not absolute performance
- Use A/B testing to validate uplift estimates
- Compare against baseline (no treatment)

### 4. Interpretation
- Negative uplift = Item hurts when recommended (avoid!)
- Zero uplift = Recommendation has no effect
- Positive uplift = Recommendation helps (prioritize!)

## Common Use Cases

### Marketing Campaigns
```python
# Recommend items with highest email campaign lift
# Don't waste email slots on items users would buy anyway
uplift_recommendations = recommender.recommend(...)
```

### Promotional Budgets
```python
# Allocate discounts to items with highest uplift from promotion
# Save money by not discounting items that would sell anyway
```

### Treatment Effect Heterogeneity
```python
# Estimate uplift varies by user segment
# Young users might respond differently to recommendations than older users
```

## Further Reading

- **[Evaluation Guide](../user-guide/evaluation.md)** - How to evaluate uplift models
- **[Training Guide](../user-guide/training.md)** - Training uplift recommenders

## Implementation Details

For implementation details and API reference, see:
- `skrec/recommender/uplift_model/uplift_recommender.py`
- `skrec/recommender/uplift_model/uplift_scorer_adapter.py`

