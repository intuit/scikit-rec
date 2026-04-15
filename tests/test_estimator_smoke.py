"""Smoke tests for estimator classes that lack dedicated unit tests.

Each test instantiates the estimator, calls fit(), then predict/predict_proba,
and asserts the output shape and type are correct. These catch import errors,
signature mismatches, and basic runtime failures.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# --- Shared fixture ---


@pytest.fixture
def binary_data():
    """Simple binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n), "f3": rng.rand(n)})
    y = (rng.rand(n) > 0.5).astype(float)
    return X, y


@pytest.fixture
def regression_data():
    """Simple regression dataset."""
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n), "f3": rng.rand(n)})
    y = rng.randn(n)
    return X, y


# --- XGBClassifier ---


def test_xgb_classifier_smoke(binary_data):
    from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator

    X, y = binary_data
    est = XGBClassifierEstimator({"n_estimators": 5, "max_depth": 2})
    est.fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(np.isfinite(proba))


# --- XGBRegressor ---


def test_xgb_regressor_smoke(regression_data):
    from skrec.estimator.regression.xgb_regressor import XGBRegressorEstimator

    X, y = regression_data
    est = XGBRegressorEstimator({"n_estimators": 5, "max_depth": 2})
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


# --- WeightedXGBClassifier ---


def test_weighted_xgb_classifier_smoke(binary_data):
    from skrec.estimator.classification.xgb_classifier import (
        WeightedXGBClassifierEstimator,
    )

    X, y = binary_data
    est = WeightedXGBClassifierEstimator(params={"n_estimators": 5, "max_depth": 2})
    est.fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (len(X), 2)


# --- LightGBM Classifier ---


def test_lightgbm_classifier_smoke(binary_data):
    from skrec.estimator.classification.lightgbm_classifier import (
        LightGBMClassifierEstimator,
    )

    X, y = binary_data
    est = LightGBMClassifierEstimator(params={"n_estimators": 5, "max_depth": 2, "verbose": -1})
    est.fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_lightgbm_classifier_with_train_params(binary_data):
    from skrec.estimator.classification.lightgbm_classifier import (
        LightGBMClassifierEstimator,
    )

    X, y = binary_data
    X_valid, y_valid = X[:20], y[:20]
    est = LightGBMClassifierEstimator(
        params={"n_estimators": 10, "max_depth": 2, "verbose": -1},
        train_params={"callbacks": [lambda env: None]},
    )
    est.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    proba = est.predict_proba(X)
    assert proba.shape == (len(X), 2)


# --- LightGBM Regressor ---


def test_lightgbm_regressor_smoke(regression_data):
    from skrec.estimator.regression.lightgbm_regressor import LightGBMRegressorEstimator

    X, y = regression_data
    est = LightGBMRegressorEstimator(params={"n_estimators": 5, "max_depth": 2, "verbose": -1})
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (len(X),)


# --- SklearnUniversalClassifier ---


def test_sklearn_universal_classifier_smoke(binary_data):
    from skrec.estimator.classification.sklearn_universal_classifier import (
        SklearnUniversalClassifierEstimator,
    )

    X, y = binary_data
    est = SklearnUniversalClassifierEstimator(RandomForestClassifier, {"n_estimators": 5, "random_state": 42})
    est.fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (len(X), 2)


# --- SklearnUniversalRegressor ---


def test_sklearn_universal_regressor_smoke(regression_data):
    from skrec.estimator.regression.sklearn_universal_regressor import (
        SklearnUniversalRegressorEstimator,
    )

    X, y = regression_data
    est = SklearnUniversalRegressorEstimator(RandomForestRegressor, {"n_estimators": 5, "random_state": 42})
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (len(X),)


# --- TunedEstimator (via TunedXGBClassifier) ---


def test_tuned_estimator_smoke(binary_data):
    from skrec.estimator.classification.xgb_classifier import (
        TunedXGBClassifierEstimator,
    )
    from skrec.estimator.datatypes import HPOType

    X, y = binary_data
    param_space = {"max_depth": [2, 3], "n_estimators": [5, 10]}
    optimizer_params = {"n_iter": 2}
    est = TunedXGBClassifierEstimator(HPOType.RANDOMIZED_SEARCH_CV, param_space, optimizer_params)
    est.fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (len(X), 2)


# --- BatchXGBClassifier (fit should raise, batch path tested via integration) ---


def test_batch_xgb_classifier_fit_raises():
    from skrec.estimator.classification.xgb_classifier import (
        BatchXGBClassifierEstimator,
    )

    est = BatchXGBClassifierEstimator(params={"n_estimators": 5})
    with pytest.raises(RuntimeError, match="does not support single-pass fit"):
        est.fit(pd.DataFrame({"a": [1]}), np.array([0]))
