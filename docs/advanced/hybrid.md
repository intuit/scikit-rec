# Hybrid Recommendation Strategies

Combine multiple recommenders for more robust and effective recommendation systems.

## Overview

**Purpose**: Leverage strengths of different recommender types by combining them strategically.

**Common Patterns**:
- Ensemble (weighted combination)
- Fallback (primary with backup)
- Stage-based (filter then rank)
- Context-based (switch by scenario)

## Why Hybrid?

Different recommenders excel in different scenarios:

| Recommender | Strength | Weakness |
|------------|----------|----------|
| **RankingRecommender** | Warm items with interactions | Cold-start items |
| **ContextualBanditsRecommender** | Exploration | Requires interaction data |
| **`item_subset` / retriever** | Hard business constraints | Not a model — narrows candidates before ranking |

**Solution**: Combine them!

## Pattern 1: Fallback Strategy

Use a primary recommender, fall back to secondary when needed.

### Cold-Start Fallback

```python
def hybrid_recommend_with_fallback(user_id, item_id, user_features, top_k=5):
    """
    Use RankingRecommender for warm items,
    ContextualBanditsRecommender for cold items (exploration).
    """
    # Check if item has sufficient interaction data
    interaction_count = get_item_interaction_count(item_id)
    
    if interaction_count >= MIN_INTERACTIONS:
        # Warm item: use collaborative filtering
        recommender = ranking_recommender
    else:
        # Cold item: explore with bandits
        recommender = bandits_recommender
    
    interactions_df = pd.DataFrame({"USER_ID": [user_id]})
    users_df = pd.DataFrame({"USER_ID": [user_id], **user_features})
    
    return recommender.recommend(interactions_df, users_df, top_k=top_k)
```

### Error Fallback

```python
def safe_recommend_with_fallback(user_id, user_features, top_k=5):
    """
    Try ML recommender, fall back to popular items on failure.
    """
    try:
        # Primary: ML-based recommender
        recommendations = ml_recommender.recommend(
            interactions=interactions_df,
            users=users_df,
            top_k=top_k
        )
        return recommendations, 'ml'
        
    except Exception as e:
        logger.error(f"ML recommender failed: {e}")
        
        # Fallback: Popular items
        recommendations = get_popular_items(top_k)
        return recommendations, 'popular'
```

## Pattern 2: Ensemble (Weighted Combination)

Combine scores from multiple recommenders.

### Score-Level Ensemble

```python
def ensemble_recommend(user_id, user_features, top_k=5, weights=None):
    """
    Combine scores from multiple recommenders with weighted average.
    """
    if weights is None:
        weights = {'propensity': 0.6, 'text': 0.4}
    
    interactions_df = pd.DataFrame({"USER_ID": [user_id]})
    users_df = pd.DataFrame({"USER_ID": [user_id], **user_features})
    
    # Get scores from each recommender
    scores_propensity = ranking_recommender.score_items(interactions_df, users_df)
    scores_bandits = bandits_recommender.score_items(interactions_df, users_df)
    
    # Normalize scores (0-1 range)
    scores_propensity_norm = (scores_propensity - scores_propensity.min()) / (scores_propensity.max() - scores_propensity.min())
    scores_bandits_norm = (scores_bandits - scores_bandits.min()) / (scores_bandits.max() - scores_bandits.min())
    
    # Weighted combination
    ensemble_scores = (
        weights['propensity'] * scores_propensity_norm +
        weights['text'] * scores_bandits_norm
    )
    
    # Get top-k
    top_items = ensemble_scores.iloc[0].nlargest(top_k).index.tolist()
    return np.array([top_items])
```

### Rank-Level Ensemble

```python
def rank_fusion_recommend(user_id, user_features, top_k=5):
    """
    Combine rankings from multiple recommenders (Borda count).
    """
    interactions_df = pd.DataFrame({"USER_ID": [user_id]})
    users_df = pd.DataFrame({"USER_ID": [user_id], **user_features})
    
    # Get recommendations from each
    recs_a = ranking_recommender.recommend(interactions_df, users_df, top_k=20)
    recs_b = bandits_recommender.recommend(interactions_df, users_df, top_k=20)
    
    # Borda count: assign points based on rank
    item_scores = {}
    
    for rank, item in enumerate(recs_a[0]):
        item_scores[item] = item_scores.get(item, 0) + (20 - rank)
    
    for rank, item in enumerate(recs_b[0]):
        item_scores[item] = item_scores.get(item, 0) + (20 - rank)
    
    # Sort by combined score
    sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    top_items = [item for item, score in sorted_items[:top_k]]
    
    return np.array([top_items])
```

## Pattern 3: Stage-Based Pipeline

Multi-stage filtering and ranking.

### Filter → Rank

```python
def staged_recommend(user_id, user_features, top_k=5):
    """
    Stage 1: Business rules produce an allowed candidate set
    Stage 2: RankingRecommender ranks within that set
    """
    # Stage 1: Your policy / compliance / inventory filter
    candidate_items = filter_catalog_by_business_rules(
        user_id=user_id,
        user_location=user_features['location'],
        user_age=user_features['age'],
    )
    
    # Stage 2: Rank with ML
    ranking_recommender.set_item_subset(candidate_items)
    
    interactions_df = pd.DataFrame({"USER_ID": [user_id]})
    users_df = pd.DataFrame({"USER_ID": [user_id], **user_features})
    
    recommendations = ranking_recommender.recommend(
        interactions=interactions_df,
        users=users_df,
        top_k=top_k
    )
    
    return recommendations
```

### Coarse → Fine Ranking

```python
def two_stage_ranking(user_id, user_features, top_k=5):
    """
    Stage 1: Fast model retrieves top-100 candidates
    Stage 2: Expensive model re-ranks to top-k
    """
    # Stage 1: Fast retrieval (e.g., lightweight model)
    fast_recommender.set_item_subset(all_items)
    candidates_df = fast_recommender.score_items(interactions_df, users_df)
    top_100_items = candidates_df.iloc[0].nlargest(100).index.tolist()
    
    # Stage 2: Precise ranking (e.g., DeepFM)
    precise_recommender.set_item_subset(top_100_items)
    final_recommendations = precise_recommender.recommend(
        interactions=interactions_df,
        users=users_df,
        top_k=top_k
    )
    
    return final_recommendations
```

## Pattern 4: Context-Based Switching

Choose recommender based on context.

### User-Based Switching

```python
def context_aware_recommend(user_id, user_features, top_k=5):
    """
    Choose recommender based on user characteristics.
    """
    user_history_count = get_user_interaction_count(user_id)
    
    if user_history_count < 5:
        # New user: explore with bandits
        recommender = bandits_recommender
        strategy = 'bandits_cold'
    elif user_history_count < 20:
        # Medium history: use bandits for exploration
        recommender = bandits_recommender
        strategy = 'bandits'
    else:
        # Rich history: use collaborative filtering
        recommender = ranking_recommender
        strategy = 'collaborative'
    
    interactions_df = pd.DataFrame({"USER_ID": [user_id]})
    users_df = pd.DataFrame({"USER_ID": [user_id], **user_features})
    
    recommendations = recommender.recommend(interactions_df, users_df, top_k=top_k)
    
    # Log strategy used
    logger.info(f"Used {strategy} for user {user_id}")
    
    return recommendations
```

### Time-Based Switching

```python
def time_aware_recommend(user_id, user_features, current_hour, top_k=5):
    """
    Use different recommenders for different times of day.
    """
    if 9 <= current_hour <= 17:
        # Business hours: prioritize work-related items
        recommender = work_recommender
    elif 18 <= current_hour <= 22:
        # Evening: prioritize entertainment
        recommender = entertainment_recommender
    else:
        # Off-hours: use general recommender
        recommender = general_recommender
    
    recommendations = recommender.recommend(...)
    return recommendations
```

## Pattern 5: Diversity Enhancement

Combine recommenders to increase diversity.

```python
def diverse_recommend(user_id, user_features, top_k=5):
    """
    Mix recommendations from different sources for diversity.
    """
    # Get recommendations from each source
    collaborative_recs = ranking_recommender.recommend(
        interactions_df, users_df, top_k=10
    )[0]
    
    bandits_recs = bandits_recommender.recommend(
        interactions_df, users_df, top_k=10
    )[0]
    
    # Interleave for diversity
    diverse_recs = []
    for i in range(top_k):
        if i % 2 == 0 and len(collaborative_recs) > i//2:
            diverse_recs.append(collaborative_recs[i//2])
        elif len(bandits_recs) > i//2:
            diverse_recs.append(bandits_recs[i//2])
    
    return np.array([diverse_recs[:top_k]])
```

## Complete Example: Production Hybrid System

```python
class HybridRecommender:
    """
    Production-ready hybrid recommender combining multiple strategies.
    """
    def __init__(self, propensity_rec, bandits_rec, rule_rec):
        self.propensity_rec = propensity_rec
        self.bandits_rec = bandits_rec
        self.rule_rec = rule_rec
        
        self.min_item_interactions = 10
        self.min_user_interactions = 5
    
    def recommend(self, user_id, user_features, top_k=5):
        """
        Multi-strategy hybrid recommendation.
        """
        # Stage 1: Business rules filter
        valid_items = self.rule_rec.get_valid_items(user_id, user_features)
        
        # Stage 2: Choose primary recommender
        user_history = get_user_interaction_count(user_id)
        
        if user_history < self.min_user_interactions:
            # New user: explore with bandits
            primary_rec = self.bandits_rec
            strategy = 'bandits_cold'
        elif user_history < 20:
            # Growing user: explore with bandits
            primary_rec = self.bandits_rec
            strategy = 'bandits'
        else:
            # Established user: collaborative filtering
            primary_rec = self.propensity_rec
            strategy = 'collaborative'
        
        # Stage 3: Get primary recommendations
        primary_rec.set_item_subset(valid_items)
        
        interactions_df = pd.DataFrame({"USER_ID": [user_id]})
        users_df = pd.DataFrame({"USER_ID": [user_id], **user_features})
        
        try:
            primary_recs = primary_rec.recommend(
                interactions_df, users_df, top_k=top_k
            )
        except Exception as e:
            logger.error(f"Primary recommender failed: {e}")
            # Fallback to rule-based
            primary_recs = self.rule_rec.recommend(
                interactions_df, users_df, top_k=top_k
            )
            strategy = 'fallback_rules'
        
        # Stage 4: Boost cold items (diversification)
        final_recs = self._boost_cold_items(primary_recs[0], top_k)
        
        # Log strategy used
        statsd.increment('recommender.strategy', tags=[f'strategy:{strategy}'])
        
        return np.array([final_recs])
    
    def _boost_cold_items(self, recommendations, top_k):
        """
        Replace last 20% of recommendations with cold items for exploration.
        """
        num_cold = max(1, int(top_k * 0.2))
        warm_recs = recommendations[:-num_cold]
        
        # Get cold item recommendations via exploration
        cold_recs = self.bandits_rec.recommend(interactions_df, users_df, top_k=num_cold)[0]
        
        return list(warm_recs) + list(cold_recs)
```

## Best Practices

### 1. Weight Tuning

```python
# A/B test different weights
weights_configs = [
    {'propensity': 0.7, 'text': 0.3},
    {'propensity': 0.6, 'text': 0.4},
    {'propensity': 0.5, 'text': 0.5}
]

for weights in weights_configs:
    recommendations = ensemble_recommend(..., weights=weights)
    ndcg = evaluate(recommendations, ...)
    print(f"Weights {weights}: NDCG = {ndcg}")
```

### 2. Monitor Strategy Usage

```python
def monitored_hybrid_recommend(user_id, user_features, top_k=5):
    strategy = choose_strategy(user_id, user_features)
    
    # Log strategy usage
    statsd.increment('hybrid.strategy', tags=[f'strategy:{strategy}'])
    
    recommendations = execute_strategy(strategy, ...)
    return recommendations
```

### 3. Graceful Degradation

```python
recommender_priority = [
    ('primary', ml_recommender),
    ('secondary', bandits_recommender),
    ('tertiary', popular_items)
]

for name, recommender in recommender_priority:
    try:
        return recommender.recommend(...)
    except Exception as e:
        logger.warning(f"{name} failed: {e}")
        continue
```

## Next Steps

- **[RankingRecommender](../recommender-types/ranking.md)** - Learn about collaborative filtering
- **[Production Guide](production.md)** - Deploy hybrid systems
- **[Evaluation Guide](../user-guide/evaluation.md)** - Evaluate hybrid approaches

