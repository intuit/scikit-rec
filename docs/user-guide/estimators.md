# Estimator Guide

Estimators are the ML models at the core of the recommendation system. They predict engagement, conversion, or reward for user-item pairs.

## Available Estimators

### Classification Estimators

For binary outcomes (click/no-click, convert/no-convert):

- **`XGBClassifierEstimator`** - XGBoost classifier (most popular)
- **`LightGBMClassifierEstimator`** - LightGBM classifier (fast, memory efficient)
- **`LogisticRegressionClassifierEstimator`** - Logistic regression (simple baseline)
- **`SklearnUniversalClassifierEstimator`** - Wrap any sklearn-compatible classifier (e.g. `RandomForestClassifier`)
- **`DeepFMClassifier`** - DeepFM for feature interactions (requires `torch`)

### Regression Estimators

For continuous outcomes (revenue, time-spent, rating):

- **`XGBRegressorEstimator`** - XGBoost regressor
- **`LightGBMRegressorEstimator`** - LightGBM regressor
- **`SklearnUniversalRegressorEstimator`** - Wrap any sklearn-compatible regressor (e.g. `Ridge`, `RandomForestRegressor`)

### Multi-output Estimators

- **`MultiOutputClassifierEstimator`** - Wrapper for multioutput classification (multiple binary targets)

### Embedding Estimators

Specialized estimators for building models that learn user and item embeddings (e.g., two-tower, matrix factorization).

*   **Factorized Inputs:** Unlike general classifiers/regressors that take a single `X` matrix, embedding estimators are typically trained using separate DataFrames for users, items, and interactions. This is handled by the method `fit_embedding_model(users: Optional[DataFrame], items: Optional[DataFrame], interactions: DataFrame, ...)`

*   **Specialized Prediction:** Inference is made with `predict_proba_with_embeddings`, which supports two primary modes:
    *   **Batch Prediction Mode:** When called as `predict_proba_with_embeddings(interactions: DataFrame, users: None)`, the estimator uses its internally learned user embeddings and features (if any) to make predictions for the users specified in the `interactions` DataFrame. This is suitable for offline batch scoring.
    *   **Real-time Inference Mode:** When called as `predict_proba_with_embeddings(interactions: DataFrame, users: DataFrame)`, where the `users` DataFrame contains pre-computed user embeddings (under the `USER_EMBEDDING_NAME` column) and optionally other user features. This allows for efficient real-time predictions using externally managed user embeddings.

*   **Embedding Management:**
    *   `get_user_embeddings() -> DataFrame`: Extracts the learned user embeddings from the trained model into a DataFrame, typically containing `USER_ID_NAME` and `USER_EMBEDDING_NAME` columns.
    *   `truncate_user_data()`: Modifies the estimator's internal state to reduce its memory footprint, typically after user embeddings have been extracted for deployment. This involves removing most user-specific data from the model while preserving a placeholder embedding for unknown users. This makes the model more lightweight for pickling and deployment in real-time systems.

**Available Embedding Estimators:**

- **`MatrixFactorizationEstimator`** - **Native collaborative filtering** - ALS and SGD, continuous/binary/ordinal outcomes; NumPy-only, no PyTorch. See [Collaborative Filtering (Matrix Factorization)](collaborative-filtering.md).
- **`NCFEstimator`** - **Neural Collaborative Filtering** - GMF, MLP, and NeuMF variants for collaborative filtering
- **`NeuralFactorizationEstimator`** - Neural factorization with contextual interactions
- **`ContextualizedTwoTowerEstimator`** - Two-tower architecture with three selectable context modes (`user_tower`, `trilinear`, `scoring_layer`). See [Contextualized Two-Tower Guide](two-tower.md).
- **`DeepCrossNetworkEstimator`** - Deep cross network for feature interactions

### Sequential Estimators

Specialized estimators for modelling the **order** of user interactions. Unlike embedding estimators, these are trained on sequences of items (sorted by timestamp) rather than individual user-item pairs. Both support early stopping via `early_stopping_patience` + `restore_best_weights`.

| Class | Architecture | Loss | Use for |
|---|---|---|---|
| `SASRecClassifierEstimator` | Transformer (self-attention) | BCE | Implicit feedback, long histories |
| `SASRecRegressorEstimator` | Transformer (self-attention) | MSE | Explicit ratings |
| `HRNNClassifierEstimator` | GRU + GRU (two-level) | BCE | Session-structured data |
| `HRNNRegressorEstimator` | GRU + GRU (two-level) | MSE | Continuous outcomes with sessions |

See [SASRec Guide](sasrec.md) and [HRNN Guide](hrnn.md) for full documentation.

## Quick Start

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator

# Initialize with hyperparameters
estimator = XGBClassifierEstimator({
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 5,
    "subsample": 0.8
})
```

## Hyperparameter Tuning

### Manual Tuning
```python
estimator = XGBClassifierEstimator({
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 5
})
```

### Tuned Estimators {#tuned-estimators}

Each estimator type ships with a `Tuned*` variant that wraps sklearn's `GridSearchCV` or `RandomizedSearchCV`:

```python
from skrec.estimator.classification.xgb_classifier import TunedXGBClassifierEstimator
from skrec.estimator.datatypes import HPOType

param_space = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}

estimator = TunedXGBClassifierEstimator(
    hpo_method=HPOType.GRID_SEARCH_CV,
    param_space=param_space,
    optimizer_params={"cv": 5, "scoring": "roc_auc"},
)
estimator.fit(X_train, y_train)
proba = estimator.predict_proba(X_test)
```

`TunedEstimator` can also be used directly with any sklearn-compatible estimator:

```python
from sklearn.ensemble import RandomForestClassifier
from skrec.estimator.tuned_estimator import TunedEstimator

estimator = TunedEstimator(
    estimator_class=RandomForestClassifier,
    hpo_method=HPOType.GRID_SEARCH_CV,
    param_space={"n_estimators": [50, 100], "max_depth": [3, 5]},
    optimizer_params={"cv": 3},
)
estimator.fit(X_train, y_train)
proba = estimator.predict_proba(X_test)
```

**Learn more**: [HPO Guide](../advanced/hpo.md)

## Choosing an Estimator

### Decision Guide

1. **What type of outcome?**
   - Binary (0/1) → Classification estimators
   - Continuous (revenue, time) → Regression estimators
   - Multiple outcomes → MultiOutput estimators

2. **What's your priority?**
   - **Performance**: XGBoost or LightGBM
   - **Speed**: LightGBM or LogisticRegression
   - **Interpretability**: LogisticRegression or LinearRegression
   - **Feature interactions**: DeepFM
   - **Learned embeddings + real-time inference**: Embedding estimators

3. **How much data?**
   - Large datasets (>100K rows): XGBoost, LightGBM, DeepFM, Embedding estimators
   - Medium datasets: XGBoost, RandomForest
   - Small datasets: LogisticRegression, LinearRegression

4. **What's your deployment architecture?**
   - **Traditional batch prediction**: Any estimator
   - **Real-time with embedding store**: Embedding estimators (NeuralFactorization, TwoTower, DeepCrossNetwork)
   - **Cold-start scenarios**: Embedding estimators or content-based approaches

### Comparison Table

| Estimator | Speed | Performance | Interpretability | Data Needs | Use Case |
|-----------|-------|-------------|------------------|------------|----------|
| **XGBoost** | Medium | ⭐⭐⭐⭐⭐ | Medium | Medium-Large | General-purpose |
| **LightGBM** | Fast | ⭐⭐⭐⭐⭐ | Medium | Medium-Large | General-purpose |
| **LogisticRegression** | Very Fast | ⭐⭐⭐ | High | Any | Baseline/Simple |
| **RandomForest** | Slow | ⭐⭐⭐⭐ | Medium | Medium-Large | General-purpose |
| **DeepFMClassifier** | Slow | ⭐⭐⭐⭐⭐ | Low | Large | Feature interactions |
| **MatrixFactorization** | Fast | ⭐⭐⭐⭐ | Medium (latent factors) | Medium-Large | **Collaborative filtering (native)** |
| **NCF** | Slow | ⭐⭐⭐⭐⭐ | Low | Large | **Collaborative filtering (neural)** |
| **NeuralFactorization** | Slow | ⭐⭐⭐⭐⭐ | Low | Large | Embeddings + context |
| **TwoTower** | Slow | ⭐⭐⭐⭐⭐ | Low | Large | Real-time embeddings + context modes |
| **DeepCrossNetwork** | Slow | ⭐⭐⭐⭐⭐ | Low | Large | Cross-feature learning |

## Best Practices

### 1. Start Simple
```python
# Start with XGBoost + default parameters
estimator = XGBClassifierEstimator({"learning_rate": 0.1, "n_estimators": 100})
```

### 2. Use Cross-Validation
```python
# Use Tuned estimators with CV for robust parameter selection
estimator = TunedXGBClassifierEstimator(
    hpo_method=HPOType.GRID_SEARCH_CV,
    param_space=param_space,
    optimizer_params={"cv": 5},
)
```

### 3. Feature Engineering Matters
- Good features > complex models
- XGBoost/LightGBM handle raw features well
- Deep models benefit from feature engineering

### 4. Monitor Training
```python
# XGBoost early stopping
estimator = XGBClassifierEstimator({
    "n_estimators": 1000,
    "early_stopping_rounds": 50
})
```

## API Contract

The library has two estimator families with different calling conventions:

| Family | Training | Inference |
|---|---|---|
| **Tabular** (`BaseClassifier`, `BaseRegressor`) | `fit(X, y)` | `predict_proba(X)` or `predict(X)` |
| **Embedding / Sequential** (`BaseEmbeddingEstimator`, `SequentialEstimator`) | `fit_embedding_model(users, items, interactions)` | `predict_proba_with_embeddings(interactions, users)` |

Within each family, the method contract is **enforced by abstract base classes** — any concrete subclass that doesn't implement the required hook raises `TypeError` at instantiation, not at call time.

## Implementation Details

For implementation details and complete API, see:
- `skrec/estimator/classification/` - Classification estimators
- `skrec/estimator/regression/` - Regression estimators
- `skrec/estimator/embedding/` - Embedding estimators (MF, NCF, TwoTower, DCN)
- `skrec/estimator/sequential/` - Sequential estimators (SASRec, HRNN)
- `skrec/estimator/base_estimator.py` - Base estimator interface

## Next Steps

- **[Scorer Guide](scorers.md)** - Pair your estimator with the right scorer
- **[HPO Guide](../advanced/hpo.md)** - Optimize hyperparameters
- **[Training Guide](training.md)** - Train your complete pipeline

