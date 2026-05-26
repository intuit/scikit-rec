# Contributing to scikit-rec

Thank you for your interest in contributing to `scikit-rec`! We welcome bug reports, feature requests, and code contributions.

## How to contribute

1. Fork the repository and create a new branch.
2. Make your changes in a focused branch.
3. Add tests for bug fixes and new functionality.
4. Run the test suite locally:

```bash
git clone https://github.com/intuit/scikit-rec.git
cd scikit-rec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

For PyTorch-based estimators (NCF, Two-Tower, SASRec, HRNN, DeepFM), install with the torch extra:

```bash
pip install -e ".[dev,torch]"
```

5. Commit with a clear message and open a pull request against `main`.

## Guidelines

- Keep pull requests small and focused.
- Follow the existing code style and conventions.
- Add or update documentation when behavior changes.
- Use descriptive commit messages.

## Reporting issues

If you find a bug or have a feature idea, please open an issue at:

https://github.com/intuit/scikit-rec/issues

Include a clear description of the problem, a minimal reproduction example, and the expected behavior.

---

## Architecture guide for contributors

`scikit-rec` uses a 3-layer architecture: **Recommender → Scorer → Estimator**. Each layer has a well-defined abstract base class. Adding a new component means subclassing the right base, implementing a small set of abstract methods, and (for factory-accessible types) registering the new type in `skrec/orchestrator/factory.py`.

```
skrec/
  estimator/
    base_estimator.py          ← BaseEstimator (tabular)
    classification/            ← tabular classifiers
    regression/                ← tabular regressors
    embedding/                 ← BaseEmbeddingEstimator + embedding models
    sequential/                ← SequentialEstimator + SASRec, HRNN
  scorer/
    base_scorer.py             ← BaseScorer
    universal.py, independent.py, multiclass.py, multioutput.py, ...
  recommender/
    base_recommender.py        ← BaseRecommender
    ranking/, bandits/, sequential/, gcsl/, uplift_model/
  orchestrator/
    factory.py                 ← create_recommender_pipeline + capability_matrix
```

### Adding a new estimator

**Tabular classifier or regressor:**

1. Subclass `BaseClassifier` (in `skrec/estimator/classification/base_classifier.py`) or `BaseRegressor`.
2. Implement `fit(X, y)` and `predict_proba(X)` / `predict(X)`.
3. Add the file to `skrec/estimator/classification/` or `skrec/estimator/regression/`.
4. To expose it via `create_recommender_pipeline`, add a branch in `create_estimator()` in `factory.py` and extend `EstimatorConfig` with the new config key. Also add the new key to `capability_matrix()`.

**Embedding estimator:**

1. Subclass `BaseEmbeddingEstimator` (`skrec/estimator/embedding/base_embedding_estimator.py`).
2. Implement `fit_embedding_model(users, items, interactions, ...)` and `predict_proba_with_embeddings(...)`.
3. Add a lazy entry to `_EMBEDDING_ESTIMATOR_MAP` in `factory.py` — no eager import needed.

**Sequential estimator (SASRec-style):**

1. Subclass `SequentialEstimator` (`skrec/estimator/sequential/base_sequential_estimator.py`).
2. Add a lazy entry to `_SEQUENTIAL_ESTIMATOR_MAP` in `factory.py`.

### Adding a new scorer

1. Subclass `BaseScorer` (`skrec/scorer/base_scorer.py`).
2. Implement `score_items(interactions, users, items)` and `score_fast(features_df)` if supporting `recommend_online`.
3. Add it to `SCORER_TYPES` in `factory.py`, add a branch in `create_scorer()`, and add an entry (even an empty `frozenset`) to `_SCORER_CONFIG_ALLOWED`.

### Adding a new multi-target estimator family

`MixedTypeMultiTargetScorer` is polymorphic over estimator families via the runtime-checkable `MultiTargetEstimator` Protocol. To add a fourth family (e.g. a tree-based joint model, an MoE encoder, etc.):

1. Implement a class with the four attributes/methods on `skrec.estimator.classification._multi_target_protocol.MultiTargetEstimator`:
   - `target_specs: dict[str, TargetType | TargetGroupSpec]` attribute
   - `fit(X, y, X_valid=None, y_valid=None)`
   - `predict_proba_dict(X) -> dict[str, np.ndarray]` (multilabel groups fanned out)
   - `predict_targets_dict(X) -> dict[str, np.ndarray]` (multilabel groups fanned out)
2. The scorer's `__init__` Protocol check accepts your class automatically — no changes to `MixedTypeMultiTargetScorer` are required.
3. If the family fits the joint pattern (shared encoder + per-target heads), subclass `JointMultiTargetBaseEstimator` and supply an encoder via `_build_encoder(input_dim, label_input_dim)`. See `joint_multi_target_mlp.py` for the minimal template.
4. To expose via the factory, add a mode to `MULTI_TARGET_MODEL_TYPES` and a branch in `_create_multi_target_estimator()` in `factory.py`.
5. For `mode="independent"` extensions (adding a new sub-estimator type), extend `_INDEPENDENT_TARGET_COMPAT` and `_create_independent_sub_estimator()`.

Gate 1 (`tests/test_mixed_type_multi_target_gates.py`) asserts every family satisfies the Protocol; add your class to that test when contributing.

### Adding a new recommender

1. Subclass `BaseRecommender` (`skrec/recommender/base_recommender.py`).
2. Implement `train(...)`, `recommend(...)`, and optionally `score_items(...)`.
3. Add it to `RECOMMENDER_TYPES` in `factory.py` and add a branch in `create_recommender()`.

### Adding a new retriever

1. Subclass `BaseCandidateRetriever` (`skrec/retriever/base_retriever.py`).
2. Implement `retrieve(interactions, users, items)`.
3. Add an entry to `_RETRIEVER_MAP` in `factory.py`.

---

## Test conventions

Tests live in `tests/`. The main patterns:

- **Unit tests** (`test_base_*.py`, `test_*_scorer.py`, etc.) use small synthetic fixtures defined in `tests/conftest.py` and `tests/utils.py`.
- **Integration tests** (`test_*_integration.py`) run a full train → evaluate cycle on sample data and are the primary correctness check.
- **Smoke tests** (`test_estimator_smoke.py`) instantiate every estimator and verify they don't error on a tiny dataset — useful for catching import or API regressions.

When adding a new component, add at minimum:
- A unit test covering the abstract interface.
- An integration test that trains and evaluates end-to-end.
- If the component is factory-accessible, a test in `test_orchestrator_factory.py`.

S3-dependent tests (`test_s3.py`) use `moto` to mock AWS — no real credentials needed.

---

## Community

By contributing, you agree to abide by the project's Code of Conduct:

- `CODE_OF_CONDUCT.md`

We aim to make this project welcoming, inclusive, and respectful.
