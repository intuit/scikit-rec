# Production Deployment Guide

Best practices and patterns for deploying recommendation systems to production.

## Overview

This guide covers:
- Model serialization and deployment
- Real-time vs batch inference
- Performance optimization
- Monitoring and maintenance

## Model Serialization

### Saving Models

```python
import pickle

# Train your recommender
recommender.train(interactions_ds, users_ds, items_ds)

# Save to file
with open('recommender_model.pkl', 'wb') as f:
    pickle.dump(recommender, f)

# Or save to S3
import boto3
s3 = boto3.client('s3')
with open('recommender_model.pkl', 'rb') as f:
    s3.upload_fileobj(f, 'my-bucket', 'models/recommender_v1.pkl')
```

### Loading Models

```python
import pickle

# Load from file
with open('recommender_model.pkl', 'rb') as f:
    recommender = pickle.load(f)

# Or load from S3
import boto3
from io import BytesIO

s3 = boto3.client('s3')
buffer = BytesIO()
s3.download_fileobj('my-bucket', 'models/recommender_v1.pkl', buffer)
buffer.seek(0)
recommender = pickle.load(buffer)

# Make recommendations
recommendations = recommender.recommend(interactions_df, users_df, top_k=5)
```

## Deployment Patterns

### 1. Real-Time API (MXS, Lambda)

**Use Case**: User-facing recommendations with low latency requirements

```python
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model once at startup
with open('recommender_model.pkl', 'rb') as f:
    recommender = pickle.load(f)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    
    # Create DataFrames from request
    interactions_df = pd.DataFrame({
        "USER_ID": [data['user_id']]
    })
    
    users_df = pd.DataFrame({
        "USER_ID": [data['user_id']],
        **data['user_features']
    })
    
    # Get recommendations — use recommend_online() for single-user, no join overhead
    recommendations = recommender.recommend_online(
        interactions=interactions_df,
        users=users_df,
        top_k=data.get('top_k', 5),
    )
    
    return jsonify({
        'user_id': data['user_id'],
        'recommendations': recommendations[0].tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### 2. Batch Processing (Airflow, Spark)

**Use Case**: Pre-compute recommendations for all users offline

```python
from pyspark.sql import SparkSession
import pickle

def batch_recommendations(spark, model_path, users_path, output_path):
    # Load model
    with open(model_path, 'rb') as f:
        recommender = pickle.load(f)
    
    # Load users (large dataset)
    users_df = spark.read.parquet(users_path).toPandas()
    
    # Create interactions
    interactions_df = users_df[['USER_ID']].copy()
    
    # Batch recommend (process in chunks if needed)
    chunk_size = 10000
    all_recommendations = []
    
    for i in range(0, len(users_df), chunk_size):
        chunk_interactions = interactions_df.iloc[i:i+chunk_size]
        chunk_users = users_df.iloc[i:i+chunk_size]
        
        chunk_recs = recommender.recommend(
            interactions=chunk_interactions,
            users=chunk_users,
            top_k=10
        )
        all_recommendations.append(chunk_recs)
    
    # Save results
    # ... (save to database, S3, etc.)
```

### 3. Streaming (Kafka, Kinesis)

**Use Case**: Continuous recommendation updates

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import pickle

# Load model
with open('recommender_model.pkl', 'rb') as f:
    recommender = pickle.load(f)

consumer = KafkaConsumer(
    'user_events',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda m: json.dumps(m).encode('utf-8')
)

for message in consumer:
    user_data = message.value
    
    # Create DataFrames
    interactions_df = pd.DataFrame({"USER_ID": [user_data['user_id']]})
    users_df = pd.DataFrame({
        "USER_ID": [user_data['user_id']],
        **user_data['features']
    })
    
    # Get recommendations
    recommendations = recommender.recommend(
        interactions=interactions_df,
        users=users_df,
        top_k=5
    )
    
    # Send to output topic
    producer.send('recommendations', {
        'user_id': user_data['user_id'],
        'recommendations': recommendations[0].tolist(),
        'timestamp': time.time()
    })
```

## Performance Optimization

### 1. Single-User Optimization

For real-time APIs with one user per request, use `recommend_online()` which skips
the pandas join entirely:

```python
# No join overhead — designed for real-time serving
recommendations = recommender.recommend_online(
    interactions=single_user_interactions_df,
    users=single_user_df,
    top_k=5,
)
```

For scoring only (without ranking), call `scorer.score_fast()` directly with a
pre-merged single-row DataFrame (no `USER_ID`):

```python
features_df = pd.DataFrame({"feat1": [18], "feat2": [0]})  # no USER_ID
scores_df = recommender.scorer.score_fast(features_df)
# Returns: DataFrame with item names as columns
```

**Supported scorers**: `UniversalScorer`, `MulticlassScorer`, `MultioutputScorer`,
and `IndependentScorer`. Not supported for embedding-based estimators.

### 2. Caching Strategies

#### Cache Item Scores

```python
from functools import lru_cache
import hashlib

class CachedRecommender:
    def __init__(self, recommender):
        self.recommender = recommender
        self.cache = {}
    
    def recommend(self, user_id, user_features, top_k=5):
        # Create cache key from user features
        cache_key = hashlib.md5(
            json.dumps(user_features, sort_keys=True).encode()
        ).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key][:top_k]
        
        # Get recommendations
        interactions_df = pd.DataFrame({"USER_ID": [user_id]})
        users_df = pd.DataFrame({"USER_ID": [user_id], **user_features})
        
        recommendations = self.recommender.recommend(
            interactions=interactions_df,
            users=users_df,
            top_k=top_k
        )
        
        # Cache result
        self.cache[cache_key] = recommendations[0]
        return recommendations[0]
```

#### Cache with Redis

```python
import redis
import pickle
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_recommendations_cached(user_id, user_features, top_k=5):
    # Create cache key
    cache_key = f"recs:{user_id}:{hash(str(user_features))}"
    
    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return pickle.loads(cached)
    
    # Compute recommendations
    recommendations = recommender.recommend(...)
    
    # Cache with TTL (e.g., 1 hour)
    redis_client.setex(cache_key, 3600, pickle.dumps(recommendations))
    
    return recommendations
```

### 3. Model Optimization

#### Use Lighter Models

```python
# Development: Complex model
dev_config = {
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {"n_estimators": 200, "max_depth": 8}
    },
    "scorer_type": "universal",
    "recommender_type": "ranking"
}

# Production: Faster model
prod_config = {
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {"n_estimators": 100, "max_depth": 4}  # Lighter for low latency
    },
    "scorer_type": "universal",
    "recommender_type": "ranking"
}
```

#### Pre-filter Items

```python
# Filter items before scoring
def get_recommendations(user_id, user_features, top_k=5):
    # Apply business rules first
    eligible_items = get_in_stock_items(user_location)
    
    recommender.set_item_subset(eligible_items)
    
    recommendations = recommender.recommend(
        interactions=interactions_df,
        users=users_df,
        top_k=top_k
    )
    
    return recommendations
```

### 4. Parallel Processing

#### For Independent Scorer

```python
from skrec.scorer.independent import IndependentScorer

scorer = IndependentScorer(estimator)
scorer.set_parallel_inference(parallel_inference_status=True, num_cores=4)

recommender = RankingRecommender(scorer)
# Inference now parallelized
```

#### Batch Parallelization

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parallel_batch_recommend(users_df, batch_size=1000, n_workers=4):
    def process_batch(batch_idx):
        start = batch_idx * batch_size
        end = start + batch_size
        
        batch_df = users_df.iloc[start:end]
        batch_interactions = batch_df[['USER_ID']].copy()
        
        return recommender.recommend(
            interactions=batch_interactions,
            users=batch_df,
            top_k=10
        )
    
    n_batches = (len(users_df) + batch_size - 1) // batch_size
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_batch, range(n_batches)))
    
    return np.vstack(results)
```

## Monitoring

### 1. Latency Metrics

```python
import time
from datadog import statsd

def monitored_recommend(user_id, user_features, top_k=5):
    start_time = time.time()
    
    try:
        recommendations = recommender.recommend(...)
        
        # Log latency
        latency = (time.time() - start_time) * 1000  # ms
        statsd.timing('recommender.latency', latency)
        
        return recommendations
        
    except Exception as e:
        statsd.increment('recommender.errors')
        raise
```

### 2. Recommendation Quality

```python
def log_recommendation_metrics(user_id, recommendations, actual_click):
    # Log if top recommendation was clicked
    top_rec = recommendations[0]
    statsd.increment(
        'recommender.top1_accuracy',
        1 if top_rec == actual_click else 0
    )
    
    # Log if any recommendation was clicked
    hit = actual_click in recommendations
    statsd.increment('recommender.hit_rate', 1 if hit else 0)
```

### 3. Model Drift Detection

```python
from scipy.stats import ks_2samp

def check_feature_drift(current_data, baseline_data, feature_name):
    current_values = current_data[feature_name]
    baseline_values = baseline_data[feature_name]
    
    # KS test for distribution shift
    statistic, p_value = ks_2samp(current_values, baseline_values)
    
    if p_value < 0.05:
        print(f"WARNING: Drift detected in {feature_name}")
        statsd.increment('recommender.feature_drift', tags=[f'feature:{feature_name}'])
    
    return p_value
```

## A/B Testing

### Simple A/B Test

```python
import hashlib

def get_recommender_variant(user_id):
    # Consistent hashing for user assignment
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return 'A' if hash_val % 2 == 0 else 'B'

def recommend_with_ab_test(user_id, user_features, top_k=5):
    variant = get_recommender_variant(user_id)
    
    if variant == 'A':
        recommender = recommender_a  # Control
    else:
        recommender = recommender_b  # Treatment
    
    recommendations = recommender.recommend(...)
    
    # Log variant
    statsd.increment('recommender.variant', tags=[f'variant:{variant}'])
    
    return recommendations, variant
```

## Best Practices

### 1. Version Your Models

```python
import datetime

model_metadata = {
    'version': 'v1.2.0',
    'trained_at': datetime.datetime.now().isoformat(),
    'training_data_date': '2025-01-01',
    'config': config,
    'metrics': {
        'ndcg@5': 0.78,
        'precision@5': 0.65
    }
}

# Save metadata with model
with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f)
```

### 2. Graceful Degradation

```python
def safe_recommend(user_id, user_features, top_k=5):
    try:
        # Try ML model
        recommendations = recommender.recommend(...)
        return recommendations
        
    except Exception as e:
        logger.error(f"Recommender failed: {e}")
        statsd.increment('recommender.fallback')
        
        # Fallback to popular items
        return get_popular_items(top_k)
```

### 3. Feature Store Integration

```python
class FeatureStoreRecommender:
    def __init__(self, recommender, feature_store):
        self.recommender = recommender
        self.feature_store = feature_store
    
    def recommend(self, user_id, top_k=5):
        # Fetch features from feature store
        user_features = self.feature_store.get_user_features(user_id)
        
        interactions_df = pd.DataFrame({"USER_ID": [user_id]})
        users_df = pd.DataFrame({
            "USER_ID": [user_id],
            **user_features
        })
        
        return self.recommender.recommend(
            interactions=interactions_df,
            users=users_df,
            top_k=top_k
        )
```

### 4. Canary Deployments

```python
def canary_recommend(user_id, user_features, canary_percentage=10):
    # Route small percentage to new model
    if random.random() < canary_percentage / 100:
        recommender = new_model
        version = 'canary'
    else:
        recommender = stable_model
        version = 'stable'
    
    recommendations = recommender.recommend(...)
    
    # Log version used
    statsd.increment('recommender.version', tags=[f'version:{version}'])
    
    return recommendations
```

## Troubleshooting

### Issue: High latency in production

**Solutions**:
- Use `recommend_online()` for single-user requests (no join overhead)
- Implement caching (Redis)
- Pre-filter items
- Use lighter models
- Enable parallel inference

### Issue: Out of memory errors

**Solutions**:
- Process in smaller batches
- Use lighter models
- Reduce feature dimensionality
- Increase instance memory

### Issue: Stale recommendations

**Solutions**:
- Implement cache invalidation
- Reduce cache TTL
- Retrain models more frequently
- Use online learning approaches

## Next Steps

- **[Orchestration](orchestration.md)** - Config-driven deployment
- **[HPO Guide](hpo.md)** - Optimize for production
- **[Monitoring Guide]** - Production monitoring (coming soon)

