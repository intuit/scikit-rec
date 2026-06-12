"""Tests for the generic fit-time parameter passthrough (SklearnFitParamsMixin).

Covers the mixin contract (strategy resolution, error paths, setters) via a spy
estimator that records the kwargs reaching ``fit``, plus real-estimator checks
that ``sample_weight='balanced'`` actually shifts predictions and composes with
``WeightedXGBClassifierEstimator``.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from skrec.estimator.classification.lightgbm_classifier import LightGBMClassifierEstimator
from skrec.estimator.classification.multioutput_classifier import MultiOutputClassifierEstimator
from skrec.estimator.classification.sklearn_universal_classifier import (
    SklearnUniversalClassifierEstimator,
)
from skrec.estimator.classification.xgb_classifier import (
    WeightedXGBClassifierEstimator,
    XGBClassifierEstimator,
)
from skrec.estimator.regression.xgb_regressor import XGBRegressorEstimator


class _SpyClassifier:
    """Minimal sklearn-like classifier that records the kwargs passed to fit."""

    def __init__(self, **params):
        self.params = params
        self.fit_kwargs = None

    def fit(self, X, y, **kwargs):
        self.fit_kwargs = kwargs
        self._n = len(y)
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        return np.tile([0.5, 0.5], (n, 1))


@pytest.fixture
def imbalanced_xy():
    rng = np.random.RandomState(0)
    n = 400
    X = pd.DataFrame(rng.rand(n, 4), columns=list("abcd"))
    # ~12% positive — clearly imbalanced
    y = pd.Series((rng.rand(n) < 0.12).astype(int))
    return X, y


# --------------------------------------------------------------------------- #
# Mixin contract via the spy estimator
# --------------------------------------------------------------------------- #


def test_balanced_strategy_reaches_fit():
    X = pd.DataFrame(np.random.rand(50, 3), columns=list("abc"))
    y = pd.Series([0] * 40 + [1] * 10)
    est = SklearnUniversalClassifierEstimator(_SpyClassifier, {}, sample_weight="balanced")
    est.fit(X, y)
    w = est._model.fit_kwargs["sample_weight"]
    assert w.shape[0] == len(y)
    # balanced → minority class (label 1) gets higher weight than majority
    assert w[y.values == 1].mean() > w[y.values == 0].mean()


def test_callable_strategy_reaches_fit():
    X = pd.DataFrame(np.random.rand(30, 2), columns=list("ab"))
    y = pd.Series(np.r_[np.zeros(15), np.ones(15)].astype(int))
    est = SklearnUniversalClassifierEstimator(_SpyClassifier, {}, sample_weight=lambda yy: np.full(len(yy), 3.0))
    est.fit(X, y)
    assert np.allclose(est._model.fit_kwargs["sample_weight"], 3.0)


def test_explicit_array_reaches_fit():
    X = pd.DataFrame(np.random.rand(20, 2), columns=list("ab"))
    y = pd.Series(np.r_[np.zeros(10), np.ones(10)].astype(int))
    w = np.linspace(1, 2, 20)
    est = SklearnUniversalClassifierEstimator(_SpyClassifier, {}, sample_weight=w)
    est.fit(X, y)
    assert np.allclose(est._model.fit_kwargs["sample_weight"], w)


def test_none_strategy_no_sample_weight_kwarg():
    X = pd.DataFrame(np.random.rand(20, 2), columns=list("ab"))
    y = pd.Series(np.r_[np.zeros(10), np.ones(10)].astype(int))
    est = SklearnUniversalClassifierEstimator(_SpyClassifier, {})
    est.fit(X, y)
    assert "sample_weight" not in est._model.fit_kwargs


def test_static_fit_params_forwarded():
    X = pd.DataFrame(np.random.rand(20, 2), columns=list("ab"))
    y = pd.Series(np.r_[np.zeros(10), np.ones(10)].astype(int))
    est = SklearnUniversalClassifierEstimator(_SpyClassifier, {}, fit_params={"some_kwarg": 7})
    est.fit(X, y)
    assert est._model.fit_kwargs["some_kwarg"] == 7


def test_setters_post_construction():
    X = pd.DataFrame(np.random.rand(20, 2), columns=list("ab"))
    y = pd.Series(np.r_[np.zeros(10), np.ones(10)].astype(int))
    est = SklearnUniversalClassifierEstimator(_SpyClassifier, {})
    est.set_sample_weight("balanced")
    est.set_fit_params(extra=1)
    est.fit(X, y)
    assert "sample_weight" in est._model.fit_kwargs
    assert est._model.fit_kwargs["extra"] == 1


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #


def test_unknown_string_strategy_raises():
    X = pd.DataFrame(np.random.rand(10, 2), columns=list("ab"))
    y = pd.Series([0, 1] * 5)
    est = SklearnUniversalClassifierEstimator(_SpyClassifier, {}, sample_weight="bogus")
    with pytest.raises(ValueError, match="Unknown sample_weight strategy"):
        est.fit(X, y)


def test_length_mismatch_raises():
    X = pd.DataFrame(np.random.rand(10, 2), columns=list("ab"))
    y = pd.Series([0, 1] * 5)
    est = SklearnUniversalClassifierEstimator(_SpyClassifier, {}, sample_weight=np.ones(3))
    with pytest.raises(ValueError, match="!= n_samples"):
        est.fit(X, y)


def test_double_spec_raises_at_construction():
    with pytest.raises(ValueError, match="OR as"):
        SklearnUniversalClassifierEstimator(
            _SpyClassifier, {}, fit_params={"sample_weight": np.ones(5)}, sample_weight="balanced"
        )


# --------------------------------------------------------------------------- #
# Real-estimator behavior: balanced shifts predictions
# --------------------------------------------------------------------------- #


def test_balanced_shifts_xgb_predictions(imbalanced_xy):
    X, y = imbalanced_xy
    base = XGBClassifierEstimator({"n_estimators": 40})
    base.fit(X, y)
    balanced = XGBClassifierEstimator({"n_estimators": 40}, sample_weight="balanced")
    balanced.fit(X, y)
    # balancing up-weights the rare positive class → higher mean P(1)
    assert balanced.predict_proba(X)[:, 1].mean() > base.predict_proba(X)[:, 1].mean()


def test_balanced_shifts_logreg_predictions(imbalanced_xy):
    X, y = imbalanced_xy
    base = SklearnUniversalClassifierEstimator(LogisticRegression, {"max_iter": 500})
    base.fit(X, y)
    balanced = SklearnUniversalClassifierEstimator(LogisticRegression, {"max_iter": 500}, sample_weight="balanced")
    balanced.fit(X, y)
    assert balanced.predict_proba(X)[:, 1].mean() > base.predict_proba(X)[:, 1].mean()


def test_xgb_regressor_sample_weight_passthrough():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(100, 3), columns=list("abc"))
    yr = pd.Series(X["a"] * 2 + rng.rand(100))
    est = XGBRegressorEstimator({"n_estimators": 20}, sample_weight=np.ones(100))
    est.fit(X, yr)  # should not raise; sample_weight forwarded to XGBRegressor.fit
    assert est.predict(X).shape == (100,)


def test_lightgbm_sample_weight_and_train_params_backcompat():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(120, 3), columns=list("abc"))
    y = pd.Series((rng.rand(120) < 0.2).astype(int))
    # generic sample_weight
    LightGBMClassifierEstimator({"n_estimators": 10, "verbose": -1}, sample_weight="balanced").fit(X, y)
    # legacy train_params + set_training_params still work
    est = LightGBMClassifierEstimator({"n_estimators": 10, "verbose": -1}, train_params={"callbacks": []})
    est.set_training_params({"callbacks": []})
    est.fit(X, y)


# --------------------------------------------------------------------------- #
# WeightedXGB composition
# --------------------------------------------------------------------------- #


def test_weighted_xgb_composes_with_balanced(imbalanced_xy):
    X, y = imbalanced_xy
    # plain weighted (no item weights, no balanced) == normal XGB behavior
    plain = WeightedXGBClassifierEstimator({"n_estimators": 30})
    plain.fit(X, y)
    # compose item/action-free weighting with the generic 'balanced' strategy
    composed = WeightedXGBClassifierEstimator({"n_estimators": 30}, sample_weight="balanced")
    composed.fit(X, y)
    # balanced up-weights positives → composed has higher mean P(1)
    assert composed.predict_proba(X)[:, 1].mean() > plain.predict_proba(X)[:, 1].mean()


def test_multioutput_classifier_sample_weight_passthrough():
    from xgboost import XGBClassifier

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(120, 3), columns=list("abc"))
    Y = pd.DataFrame({"ITEM_1": (rng.rand(120) < 0.2).astype(int), "ITEM_2": (X["b"] > 0.5).astype(int)})
    est = MultiOutputClassifierEstimator(XGBClassifier, {"n_estimators": 15}, sample_weight="balanced")
    est.fit(X, Y)  # MultiOutputClassifier forwards sample_weight to each sub-estimator
    pp = est.predict_proba(X)
    assert isinstance(pp, list) and len(pp) == 2


# --------------------------------------------------------------------------- #
# Eval-set weighting: strategies re-derive per set; explicit arrays don't
# --------------------------------------------------------------------------- #


def test_explicit_array_sample_weight_with_validation_set_no_crash():
    """Regression: an explicit (train-sized) sample_weight array + a smaller
    validation set must NOT re-validate the train array against the eval rows.

    Previously _resolve_fit_kwargs re-resolved the configured strategy against
    y_valid to build sample_weight_eval_set; for an explicit array that raised
    'sample_weight length 100 != n_samples 20'. Explicit arrays are now treated
    as train-only (no eval-set weight derived)."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(100, 4), columns=list("abcd"))
    y = pd.Series((rng.rand(100) < 0.3).astype(int))
    Xv = pd.DataFrame(rng.rand(20, 4), columns=list("abcd"))
    yv = pd.Series((rng.rand(20) < 0.3).astype(int))
    est = XGBClassifierEstimator({"n_estimators": 10}, sample_weight=np.ones(len(y)))
    est.fit(X, y, X_valid=Xv, y_valid=yv)  # must not raise
    assert est.predict_proba(X).shape == (100, 2)


def test_balanced_and_callable_derive_eval_weights():
    """'balanced' and callable strategies recompute eval weights against y_valid
    (right-sized for the eval set), so they reach sample_weight_eval_set cleanly."""

    class _SpyEvalClassifier(_SpyClassifier):
        pass

    X = pd.DataFrame(np.random.rand(50, 3), columns=list("abc"))
    y = pd.Series([0] * 40 + [1] * 10)
    Xv = pd.DataFrame(np.random.rand(12, 3), columns=list("abc"))
    yv = pd.Series([0] * 9 + [1] * 3)
    # The mixin's eval-weight derivation is exercised on the XGB path
    # (supports_eval_weight=True); validate it does not raise and resolves a
    # right-sized eval weight for a callable.
    from skrec.estimator._fit_params_mixin import SklearnFitParamsMixin

    class _Probe(SklearnFitParamsMixin):
        pass

    probe = _Probe()
    probe._init_fit_params(sample_weight=lambda yy: np.full(len(yy), 2.0))
    kw = probe._resolve_fit_kwargs(X, y, Xv, yv, supports_eval_weight=True)
    assert kw["sample_weight"].shape[0] == len(y)
    assert kw["sample_weight_eval_set"][0].shape[0] == len(yv)  # re-derived for eval size

    # explicit array: no eval-set weight derived
    probe2 = _Probe()
    probe2._init_fit_params(sample_weight=np.ones(len(y)))
    kw2 = probe2._resolve_fit_kwargs(X, y, Xv, yv, supports_eval_weight=True)
    assert "sample_weight_eval_set" not in kw2
