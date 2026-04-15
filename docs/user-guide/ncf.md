# Neural Collaborative Filtering (NCF)

Neural Collaborative Filtering (NCF) is a powerful deep learning approach for recommendation systems that learns user-item interactions through neural networks.

!!! tip "Alternative: Native matrix factorization"
    For collaborative filtering **without PyTorch** (NumPy-only, ALS or SGD), see [Collaborative Filtering (Matrix Factorization)](collaborative-filtering.md). Use NCF when you need neural architectures or user/item/context features.

## Overview

The `NCFEstimator` implements three variants of NCF as described in the seminal paper by He et al. (2017):

1. **GMF** (Generalized Matrix Factorization) - Neural implementation of matrix factorization using element-wise product
2. **MLP** (Multi-Layer Perceptron) - Deep network learning complex user-item interactions
3. **NeuMF** (Neural Matrix Factorization) - Ensemble of GMF and MLP for best performance

This implementation extends the original paper to support user features, item features, and interaction context.

## Quick Start

```python
from skrec.estimator.embedding.ncf_estimator import NCFEstimator
from skrec.scorer.universal import UniversalScorer
from skrec.recommender.ranking.ranking_recommender import RankingRecommender

# Create NCF estimator (NeuMF variant - recommended)
ncf_estimator = NCFEstimator(
    ncf_type="neumf",              # "gmf", "mlp", or "neumf"
    gmf_embedding_dim=32,           # GMF embedding size
    mlp_embedding_dim=32,           # MLP embedding size
    mlp_layers=[64, 32, 16, 8],    # Customizable MLP architecture
    dropout=0.2,                    # Dropout for regularization
    learning_rate=0.001,
    epochs=10,
    batch_size=256,
    random_state=42
)

# Create scorer and recommender
scorer = UniversalScorer(estimator=ncf_estimator)
recommender = RankingRecommender(scorer=scorer)

# Train
recommender.train(
    users_ds=users_dataset,
    items_ds=items_dataset,
    interactions_ds=interactions_dataset
)

# Recommend (batch mode - uses learned embeddings)
recommendations = recommender.recommend(
    interactions=pd.DataFrame({"USER_ID": ["user1", "user2"]}),
    users=None,  # Batch mode
    top_k=10
)
```

## NCF Variants

### GMF (Generalized Matrix Factorization)

Simple yet effective - learns user and item embeddings and combines them via element-wise product.

```python
ncf_estimator = NCFEstimator(
    ncf_type="gmf",
    gmf_embedding_dim=64,
    epochs=10
)
```

**Best for:**
- Simpler interaction patterns
- Faster training
- When you want interpretable embeddings

### MLP (Multi-Layer Perceptron)

Deep network that concatenates user and item embeddings and learns complex non-linear interactions.

```python
ncf_estimator = NCFEstimator(
    ncf_type="mlp",
    mlp_embedding_dim=32,
    mlp_layers=[128, 64, 32, 16],  # Fully customizable!
    dropout=0.3,
    epochs=15
)
```

**Best for:**
- Complex user-item interaction patterns
- Rich feature sets
- When you have sufficient training data

### NeuMF (Neural Matrix Factorization)

**Recommended** - Combines both GMF and MLP paths for best performance.

```python
ncf_estimator = NCFEstimator(
    ncf_type="neumf",
    gmf_embedding_dim=32,
    mlp_embedding_dim=32,
    mlp_layers=[64, 32, 16, 8],
    dropout=0.2,
    epochs=10
)
```

**Best for:**
- Production deployments
- Maximum recommendation quality
- Capturing both simple and complex patterns

## Customizable Architecture

### MLP Layer Configuration

The MLP architecture is **fully customizable**:

```python
# Shallow network (2 layers)
mlp_layers=[64, 32]

# Medium network (4 layers) - DEFAULT
mlp_layers=[64, 32, 16, 8]

# Deep network (5 layers)
mlp_layers=[256, 128, 64, 32, 16]

# Very deep network (6 layers)
mlp_layers=[512, 256, 128, 64, 32, 16]
```

The network dynamically builds:
```
Input → Linear(?, layers[0]) → ReLU → Dropout →
        Linear(layers[0], layers[1]) → ReLU → Dropout →
        ... → Output
```

### Embedding Dimensions

Control the capacity of your model:

```python
# Small embeddings (faster, less capacity)
gmf_embedding_dim=16
mlp_embedding_dim=16

# Medium embeddings (balanced) - RECOMMENDED
gmf_embedding_dim=32
mlp_embedding_dim=32

# Large embeddings (more capacity, slower)
gmf_embedding_dim=64
mlp_embedding_dim=64
```

### Regularization

Add dropout to prevent overfitting:

```python
dropout=0.0   # No dropout
dropout=0.2   # Light regularization
dropout=0.5   # Heavy regularization
```

## Advanced Features

### User and Item Features

NCF automatically incorporates user and item features:

```python
# Users DataFrame with features
users_df = pd.DataFrame({
    "USER_ID": ["u1", "u2", "u3"],
    "age": [25, 35, 45],
    "gender": [1, 0, 1],
    "premium": [True, False, True]
})

# Items DataFrame with features
items_df = pd.DataFrame({
    "ITEM_ID": ["i1", "i2", "i3"],
    "category": [1, 2, 1],
    "price": [10.0, 20.0, 15.0],
    "popularity": [0.8, 0.6, 0.9]
})

# Features are automatically projected and fused with embeddings
```

### Interaction Context

Add contextual features to interactions:

```python
interactions_df = pd.DataFrame({
    "USER_ID": ["u1", "u1", "u2"],
    "ITEM_ID": ["i1", "i2", "i1"],
    "OUTCOME": [1, 0, 1],
    "time_of_day": [14, 22, 9],    # Context features
    "device_type": [1, 2, 1],       # Context features
    "season": [3, 3, 2]             # Context features
})

# Context features are automatically incorporated into the MLP path
```

### Embedding Extraction for Production

Extract user embeddings for Feature Management Platform:

```python
# Train the model
recommender.train(users_ds, items_ds, interactions_ds)

# Extract user embeddings
user_embeddings_df = ncf_estimator.get_user_embeddings()
# Returns: DataFrame with USER_ID and EMBEDDING columns

# Save to embedding store (e.g., Redis, DynamoDB)
save_to_embedding_store(user_embeddings_df)

# Truncate model for lightweight deployment
ncf_estimator.truncate_user_data()

# Pickle truncated model
import pickle
with open("ncf_model.pkl", "wb") as f:
    pickle.dump(ncf_estimator, f)
```

### Real-time Inference with Embeddings

Use pre-computed embeddings for real-time inference:

```python
# Load pre-computed embeddings from store
users_with_embeddings = pd.DataFrame({
    "USER_ID": ["u1", "u2"],
    "EMBEDDING": [emb1_array, emb2_array],  # From embedding store
    "age": [25, 35],  # Optional: user features
})

# Real-time predictions
recommendations = recommender.recommend(
    interactions=interactions_df,
    users=users_with_embeddings,  # Provide embeddings
    top_k=10
)
```

## Hyperparameter Tuning

### Learning Rate

```python
learning_rate=0.0001  # Conservative (more epochs needed)
learning_rate=0.001   # Default (recommended)
learning_rate=0.01    # Aggressive (faster convergence, risk of instability)
```

### Epochs and Batch Size

```python
epochs=5              # Quick experimentation
epochs=10             # Default (usually sufficient)
epochs=20             # More thorough training

batch_size=64         # Smaller (more gradient updates, slower)
batch_size=256        # Default (good balance)
batch_size=1024       # Larger (faster, less frequent updates)
```

### Loss Functions

```python
loss_fn_name="bce"    # Binary Cross-Entropy (implicit feedback - clicks, views)
loss_fn_name="mse"    # Mean Squared Error (explicit ratings - 1-5 stars)
```

## Performance Tips

### 1. Start with NeuMF
```python
# Best overall performance
ncf_estimator = NCFEstimator(ncf_type="neumf")
```

### 2. Use Dropout for Large Models
```python
# Prevent overfitting on deep networks
mlp_layers=[256, 128, 64, 32, 16]
dropout=0.3
```

### 3. GPU Acceleration
```python
# Automatically uses GPU if available
device=None  # Auto-detect

# Or explicitly specify
device="cuda"  # Use GPU
device="cpu"   # Use CPU
```

### 4. Reproducibility
```python
# Set random seed for reproducible results
random_state=42
```

## Comparison with Other Estimators

| Feature | NCF | NeuralFactorization | TwoTower | XGBoost |
|---------|-----|---------------------|----------|---------|
| **Collaborative Filtering** | ✅ Excellent | ✅ Good | ✅ Good | ❌ No |
| **User/Item Embeddings** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| **Customizable Architecture** | ✅ **Fully customizable** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Feature Support** | ✅ User + Item + Context | ✅ User + Item + Context | ✅ User + Item | ✅ Any features |
| **Training Speed** | 🐢 Slow | 🐢 Slow | 🐢 Slow | 🚀 Fast |
| **Real-time Inference** | ✅ With embeddings | ✅ With embeddings | ✅ With embeddings | ✅ Yes |
| **Best For** | **Collaborative filtering** | Feature interactions | Retrieval | General-purpose |

## When to Use NCF

✅ **Use NCF when:**
- You have user-item interaction data (clicks, purchases, views)
- You want to learn collaborative patterns
- You need real-time inference with embedding stores
- You have sufficient training data (>10K interactions)
- Cold-start is not your primary concern

❌ **Don't use NCF when:**
- You have very sparse interaction data (<1K interactions)
- You need very fast training (use XGBoost/LightGBM)
- You need high interpretability (use LogisticRegression)

## Examples

### Example 1: Movie Recommendations

```python
from skrec.estimator.embedding.ncf_estimator import NCFEstimator
from skrec.scorer.universal import UniversalScorer
from skrec.recommender.ranking.ranking_recommender import RankingRecommender

# User features
users_df = pd.DataFrame({
    "USER_ID": ["u1", "u2", "u3"],
    "age_group": [1, 2, 3],
    "subscription_tier": [2, 1, 2]
})

# Movie features
items_df = pd.DataFrame({
    "ITEM_ID": ["m1", "m2", "m3"],
    "genre": [1, 2, 1],
    "release_year": [2020, 2019, 2021],
    "avg_rating": [4.5, 3.8, 4.2]
})

# Viewing history
interactions_df = pd.DataFrame({
    "USER_ID": ["u1", "u1", "u2", "u3"],
    "ITEM_ID": ["m1", "m2", "m1", "m3"],
    "OUTCOME": [1, 0, 1, 1],  # Watched or not
})

# Create NCF recommender
ncf = NCFEstimator(
    ncf_type="neumf",
    gmf_embedding_dim=32,
    mlp_embedding_dim=32,
    mlp_layers=[64, 32, 16],
    dropout=0.2,
    epochs=15
)

scorer = UniversalScorer(estimator=ncf)
recommender = RankingRecommender(scorer=scorer)

# Train and recommend
recommender.train(users_ds, items_ds, interactions_ds)
recommendations = recommender.recommend(
    interactions=pd.DataFrame({"USER_ID": ["u1"]}),
    users=None,
    top_k=5
)
```

### Example 2: E-commerce Recommendations

```python
# Configure for e-commerce (implicit feedback)
ncf = NCFEstimator(
    ncf_type="neumf",
    gmf_embedding_dim=64,      # Larger embeddings for many products
    mlp_embedding_dim=64,
    mlp_layers=[256, 128, 64, 32],  # Deeper network
    dropout=0.3,               # Higher dropout to prevent overfitting
    loss_fn_name="bce",        # Binary cross-entropy for clicks
    epochs=20,
    batch_size=512,            # Larger batch for efficiency
    random_state=42
)
```

### Example 3: Music Streaming with Context

```python
# Incorporate contextual features
interactions_df = pd.DataFrame({
    "USER_ID": ["u1", "u2", "u3"],
    "ITEM_ID": ["song1", "song2", "song3"],
    "OUTCOME": [1, 1, 0],
    "time_of_day": [8, 22, 14],      # Morning, night, afternoon
    "day_of_week": [1, 5, 3],        # Monday, Friday, Wednesday
    "listening_mode": [1, 2, 1]      # Active, background, active
})

ncf = NCFEstimator(
    ncf_type="mlp",            # MLP good for contextual features
    mlp_embedding_dim=48,
    mlp_layers=[128, 64, 32],  # Context features added to MLP input
    dropout=0.2,
    epochs=10
)
```

## Troubleshooting

### Model Not Learning

1. **Lower learning rate**: Try `learning_rate=0.0001`
2. **Increase epochs**: Try `epochs=20` or more
3. **Reduce complexity**: Start with simpler architecture `mlp_layers=[32, 16]`
4. **Check data quality**: Ensure sufficient positive interactions

### Overfitting

1. **Add dropout**: Increase `dropout` to 0.3-0.5
2. **Reduce model size**: Use smaller `mlp_layers` or embedding dimensions
3. **More data**: Add more training samples
4. **Early stopping**: Monitor validation loss

### Memory Issues

1. **Reduce batch size**: Try `batch_size=64` or `batch_size=32`
2. **Smaller embeddings**: Use `gmf_embedding_dim=16`, `mlp_embedding_dim=16`
3. **Shallower network**: Use fewer/smaller `mlp_layers`
4. **Use CPU**: Set `device="cpu"` if GPU memory is limited

## References

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). [Neural collaborative filtering](https://arxiv.org/abs/1708.05031). In WWW 2017.

## Next Steps

- **[Training Guide](training.md)** - Learn how to train NCF models
- **[Evaluation Guide](evaluation.md)** - Evaluate NCF performance
- **[Production Guide](../advanced/production.md)** - Deploy NCF to production
