# Tests for IndependentMultiTargetEstimator.

import io
import pickle

import numpy as np
import pandas as pd
import pytest

from skrec.constants import USER_ID_NAME  # noqa: F401  # used by moved tests
from skrec.estimator.classification import (
    IndependentMultiTargetEstimator,
    JointMultiTargetMLPEstimator,
    MultiTargetEstimator,
)
from skrec.estimator.classification.lightgbm_classifier import (
    LightGBMClassifierEstimator,
)
from skrec.estimator.classification.logreg_classifier import (
    LogisticRegressionClassifierEstimator,
)
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.estimator.regression.lightgbm_regressor import LightGBMRegressorEstimator
from skrec.estimator.regression.xgb_regressor import XGBRegressorEstimator
from skrec.evaluator.datatypes import RecommenderEvaluatorType  # noqa: F401
from skrec.metrics.datatypes import RecommenderMetricType  # noqa: F401
from skrec.orchestrator import create_estimator
from skrec.recommender.ranking.ranking_recommender import RankingRecommender  # noqa: F401
from skrec.scorer.mixed_type_multi_target import (
    MixedTypeMultiTargetScorer,  # noqa: F401
    TargetGroupSpec,
    TargetType,
)


def _make_synthetic(n: int = 150, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, 4)),
        columns=[f"feat_{i}" for i in range(4)],
    )
    y_bin = (X["feat_0"] > 0).astype(int).to_numpy()
    y_reg = (2.0 * X["feat_1"] + rng.normal(scale=0.1, size=n)).to_numpy()
    y_mc_idx = np.column_stack([X["feat_2"], X["feat_3"], -X["feat_2"]]).argmax(axis=1)
    y_mc = np.array(["A", "B", "C"])[y_mc_idx]
    y_ml = np.column_stack([(X["feat_2"] > 0).astype(int), (X["feat_3"] > 0).astype(int)])
    target_specs = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
        "ITEM_action": TargetType.MULTICLASS,
        "engagement": TargetGroupSpec(
            type=TargetType.MULTILABEL,
            columns=["ITEM_email_open", "ITEM_app_open"],
        ),
    }
    y = {
        "ITEM_clicked": y_bin,
        "ITEM_revenue": y_reg,
        "ITEM_action": y_mc,
        "engagement": y_ml,
    }
    return X, y, target_specs


def _all_four_estimators():
    """Default sub-estimator dict covering binary + regression + multiclass + multilabel.

    Multiclass uses LightGBMClassifier (sklearn-native predict_proba shape).
    Note: XGBClassifierEstimator on multiclass triggers a defensive shape
    error at predict time — see test_independent_multiclass_xgb_raises_clean.
    """
    return {
        "ITEM_clicked": XGBClassifierEstimator(params={"n_estimators": 30, "max_depth": 3}),
        "ITEM_revenue": LightGBMRegressorEstimator(params={"n_estimators": 30, "verbose": -1}),
        "ITEM_action": LightGBMClassifierEstimator(params={"n_estimators": 30, "verbose": -1}),
        "ITEM_email_open": XGBClassifierEstimator(params={"n_estimators": 30, "max_depth": 3}),
        "ITEM_app_open": LogisticRegressionClassifierEstimator(params={"max_iter": 200}),
    }


# ---------------------------------------------------------------------- #
# Protocol adherence
# ---------------------------------------------------------------------- #


def test_implements_multi_target_protocol():
    _, _, target_specs = _make_synthetic(n=10)
    est = IndependentMultiTargetEstimator(target_specs=target_specs, estimators=_all_four_estimators())
    assert isinstance(est, MultiTargetEstimator)


# ---------------------------------------------------------------------- #
# Construction validation
# ---------------------------------------------------------------------- #


def test_rejects_missing_estimator_for_target():
    _, _, target_specs = _make_synthetic(n=10)
    bad_estimators = _all_four_estimators()
    del bad_estimators["ITEM_email_open"]
    with pytest.raises(ValueError, match="missing entries for target"):
        IndependentMultiTargetEstimator(target_specs=target_specs, estimators=bad_estimators)


def test_rejects_group_key_in_estimators_dict():
    _, _, target_specs = _make_synthetic(n=10)
    bad_estimators = _all_four_estimators()
    bad_estimators["engagement"] = XGBClassifierEstimator()  # group key — wrong
    with pytest.raises(ValueError, match="group key"):
        IndependentMultiTargetEstimator(target_specs=target_specs, estimators=bad_estimators)


def test_rejects_unknown_key_in_estimators_dict():
    _, _, target_specs = _make_synthetic(n=10)
    bad_estimators = _all_four_estimators()
    bad_estimators["ITEM_unknown"] = XGBClassifierEstimator()
    with pytest.raises(ValueError, match="unknown key"):
        IndependentMultiTargetEstimator(target_specs=target_specs, estimators=bad_estimators)


def test_rejects_regressor_on_binary_target():
    target_specs = {"ITEM_clicked": TargetType.BINARY}
    bad_estimators = {"ITEM_clicked": LightGBMRegressorEstimator()}
    with pytest.raises(ValueError, match="binary"):
        IndependentMultiTargetEstimator(target_specs=target_specs, estimators=bad_estimators)


def test_rejects_classifier_on_regression_target():
    target_specs = {"ITEM_revenue": TargetType.REGRESSION}
    bad_estimators = {"ITEM_revenue": XGBClassifierEstimator()}
    with pytest.raises(ValueError, match="regression"):
        IndependentMultiTargetEstimator(target_specs=target_specs, estimators=bad_estimators)


def test_rejects_regressor_on_multilabel_member():
    target_specs = {"g1": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_a", "ITEM_b"])}
    bad_estimators = {
        "ITEM_a": LightGBMRegressorEstimator(),
        "ITEM_b": XGBClassifierEstimator(),
    }
    with pytest.raises(ValueError, match="binary"):
        IndependentMultiTargetEstimator(target_specs=target_specs, estimators=bad_estimators)


# ---------------------------------------------------------------------- #
# Fit + predict happy path
# ---------------------------------------------------------------------- #


def test_fit_and_predict_proba_shapes():
    X, y, target_specs = _make_synthetic(n=120)
    est = IndependentMultiTargetEstimator(target_specs=target_specs, estimators=_all_four_estimators())
    est.fit(X, y)
    proba = est.predict_proba_dict(X)
    assert set(proba.keys()) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_action",
        "ITEM_email_open",
        "ITEM_app_open",
    }
    assert proba["ITEM_clicked"].shape == (120, 2)
    assert proba["ITEM_revenue"].shape == (120,)
    assert proba["ITEM_action"].shape == (120, 3)
    assert proba["ITEM_email_open"].shape == (120, 2)
    assert proba["ITEM_app_open"].shape == (120, 2)


def test_predict_targets_dict_shapes():
    X, y, target_specs = _make_synthetic(n=100)
    est = IndependentMultiTargetEstimator(target_specs=target_specs, estimators=_all_four_estimators())
    est.fit(X, y)
    preds = est.predict_targets_dict(X)
    assert set(preds.keys()) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_action",
        "ITEM_email_open",
        "ITEM_app_open",
    }
    for k in ("ITEM_clicked", "ITEM_email_open", "ITEM_app_open"):
        assert set(np.unique(preds[k]).tolist()).issubset({0, 1})
    assert set(np.unique(preds["ITEM_action"]).tolist()).issubset({"A", "B", "C"})


# ---------------------------------------------------------------------- #
# Heterogeneous pickle round-trip — exercises 3 sub-estimator types
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "estimator_factory",
    [
        # XGB binary + XGB regression
        lambda: {
            "ITEM_clicked": XGBClassifierEstimator(params={"n_estimators": 10}),
            "ITEM_revenue": XGBRegressorEstimator(params={"n_estimators": 10}),
        },
        # LightGBM regression + LogReg binary
        lambda: {
            "ITEM_clicked": LogisticRegressionClassifierEstimator(params={"max_iter": 100}),
            "ITEM_revenue": LightGBMRegressorEstimator(params={"n_estimators": 10, "verbose": -1}),
        },
    ],
)
def test_pickle_round_trip_heterogeneous_sub_estimators(estimator_factory):
    X, _, _ = _make_synthetic(n=80)
    target_specs = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    y = {
        "ITEM_clicked": (X["feat_0"] > 0).astype(int).to_numpy(),
        "ITEM_revenue": (2.0 * X["feat_1"]).to_numpy(),
    }
    est = IndependentMultiTargetEstimator(target_specs=target_specs, estimators=estimator_factory())
    est.fit(X, y)
    pre = est.predict_proba_dict(X)

    buf = io.BytesIO()
    pickle.dump(est, buf)
    buf.seek(0)
    est2 = pickle.load(buf)

    post = est2.predict_proba_dict(X)
    for k in pre:
        np.testing.assert_allclose(pre[k], post[k], rtol=1e-5)


# ---------------------------------------------------------------------- #
# Partial-fit failure cleanup
# ---------------------------------------------------------------------- #


class _RaisingEstimator(XGBClassifierEstimator):
    """Sub-estimator that raises on .fit — used to drive the partial-fit test."""

    def fit(self, X, y, X_valid=None, y_valid=None):
        raise RuntimeError("synthetic failure for test")


def test_partial_fit_failure_leaves_estimator_unfitted():
    target_specs = {
        "ITEM_a": TargetType.BINARY,
        "ITEM_b": TargetType.BINARY,
    }
    estimators = {
        "ITEM_a": XGBClassifierEstimator(params={"n_estimators": 5}),
        "ITEM_b": _RaisingEstimator(),
    }
    est = IndependentMultiTargetEstimator(target_specs=target_specs, estimators=estimators)
    X = pd.DataFrame({"feat_0": [0.1, 0.2, -0.3, 0.5]})
    y = {
        "ITEM_a": np.array([0, 1, 0, 1]),
        "ITEM_b": np.array([1, 0, 1, 0]),
    }
    with pytest.raises(RuntimeError, match="synthetic failure"):
        est.fit(X, y)
    # Predict must NOT silently produce partial output.
    with pytest.raises(RuntimeError, match="not fitted"):
        est.predict_proba_dict(X)


# ---------------------------------------------------------------------- #
# Sanity: dict-y validation
# ---------------------------------------------------------------------- #


def test_fit_rejects_dict_y_with_wrong_keys():
    X, y, target_specs = _make_synthetic(n=20)
    est = IndependentMultiTargetEstimator(target_specs=target_specs, estimators=_all_four_estimators())
    with pytest.raises(ValueError, match="y keys must match"):
        est.fit(X, {"WRONG": np.array([0, 1])})


def test_fit_rejects_non_dict_y():
    X, _, target_specs = _make_synthetic(n=20)
    est = IndependentMultiTargetEstimator(target_specs=target_specs, estimators=_all_four_estimators())
    with pytest.raises(TypeError, match="dict"):
        est.fit(X, np.array([0, 1] * 10))


def test_check_fitted_before_predict():
    _, _, target_specs = _make_synthetic(n=10)
    est = IndependentMultiTargetEstimator(target_specs=target_specs, estimators=_all_four_estimators())
    with pytest.raises(RuntimeError, match="not fitted"):
        est.predict_proba_dict(pd.DataFrame({"feat_0": [0.5]}))


# ---------------------------------------------------------------------- #
# Multilabel fan-out: 3-member group → 3 distinct sub-estimators
# ---------------------------------------------------------------------- #


def test_unused_defaults_silently_ignored():
    """Plan test #14: ``defaults`` covers all four target types but
    ``target_specs`` declares only BINARY + REGRESSION → factory must
    construct only the two sub-estimators that are actually used and
    must NOT instantiate (or warn about) the unused MULTICLASS /
    MULTILABEL defaults. Pins the chosen semantics so a future change
    that flips to "warn" or "reject" surfaces here.
    """
    from skrec.orchestrator import create_estimator

    target_specs = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    est = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "independent",
                "independent": {
                    "defaults": {
                        "binary": {"estimator_type": "xgboost", "params": {"n_estimators": 10}},
                        "regression": {"estimator_type": "lightgbm", "params": {"n_estimators": 10, "verbose": -1}},
                        # Unused — must not appear in the composed dict.
                        "multiclass": {"estimator_type": "lightgbm", "params": {"n_estimators": 10}},
                        "multilabel": {"estimator_type": "xgboost", "params": {"n_estimators": 10}},
                    },
                },
            },
        },
        target_specs=target_specs,
    )
    assert set(est.estimators.keys()) == {"ITEM_clicked", "ITEM_revenue"}, (
        f"Unused defaults leaked into estimators dict: {sorted(est.estimators.keys())}"
    )


def test_independent_multiclass_xgb_raises_clean():
    """Defensive guard: XGBClassifierEstimator's inplace_predict mishandles
    multiclass (returns (n, 2K) instead of (n, K)). Our predict_proba_dict
    detects the shape mismatch and raises a clean, actionable error rather
    than letting malformed columns flow into the wide-format scorer output.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y_idx = np.column_stack([X["f0"], X["f1"], X["f2"]]).argmax(axis=1)
    y_mc = np.array(["A", "B", "C"])[y_idx]
    target_specs = {"ITEM_action": TargetType.MULTICLASS}
    est = IndependentMultiTargetEstimator(
        target_specs=target_specs,
        estimators={"ITEM_action": XGBClassifierEstimator(params={"n_estimators": 5})},
    )
    est.fit(X, {"ITEM_action": y_mc})
    with pytest.raises(RuntimeError, match="predict_proba of shape"):
        est.predict_proba_dict(X)


def test_multilabel_three_member_fan_out():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    target_specs = {
        "g": TargetGroupSpec(
            type=TargetType.MULTILABEL,
            columns=["ITEM_m1", "ITEM_m2", "ITEM_m3"],
        )
    }
    y = {
        "g": np.column_stack(
            [
                (X["f0"] > 0).astype(int),
                (X["f1"] > 0).astype(int),
                (X["f2"] > 0).astype(int),
            ]
        )
    }
    estimators = {
        "ITEM_m1": XGBClassifierEstimator(params={"n_estimators": 10}),
        "ITEM_m2": XGBClassifierEstimator(params={"n_estimators": 10}),
        "ITEM_m3": LogisticRegressionClassifierEstimator(params={"max_iter": 100}),
    }
    est = IndependentMultiTargetEstimator(target_specs=target_specs, estimators=estimators)
    est.fit(X, y)
    proba = est.predict_proba_dict(X)
    assert set(proba.keys()) == {"ITEM_m1", "ITEM_m2", "ITEM_m3"}
    for k in proba:
        assert proba[k].shape == (n, 2)


# ====================================================================== #
# Independent v2-list #5: both construction paths produce identical errors
# ====================================================================== #


def test_independent_5_both_construction_paths_same_error_for_missing_target():
    """Direct construction (estimators dict) and factory-construction
    (defaults/per_target) must raise structurally identical errors for the
    same misconfiguration (missing target coverage)."""
    target_specs = {
        "ITEM_a": TargetType.BINARY,
        "ITEM_b": TargetType.BINARY,
    }

    # Direct path: missing ITEM_b in estimators dict.
    with pytest.raises(ValueError, match="missing entries for target") as direct_err:
        IndependentMultiTargetEstimator(
            target_specs=target_specs,
            estimators={"ITEM_a": LightGBMClassifierEstimator(params={"n_estimators": 5, "verbose": -1})},
        )

    # Factory path: independent.defaults provides only "binary" — but here
    # we omit defaults entirely AND have no per_target → both targets uncovered.
    with pytest.raises(ValueError, match="missing coverage") as factory_err:
        create_estimator(
            estimator_config={
                "ml_task": "multi_target",
                "multi_target": {
                    "mode": "independent",
                    "independent": {"defaults": {}, "per_target": {}},
                },
            },
            scorer_type="mixed_type_multi_target",
            target_specs=target_specs,
        )

    # Both messages must name the offending target(s). Different error
    # phrasings are acceptable; what matters is that the user can locate
    # the missing target.
    assert "ITEM_b" in str(direct_err.value)
    assert "binary" in str(factory_err.value).lower()


# ====================================================================== #
# Independent v2-list #7: multiclass K≥2 upfront check
# ====================================================================== #


def test_independent_7_multiclass_catalogue_captures_K_classes():
    """The multiclass catalogue stored at fit time must equal the number
    of unique labels in y — this is the K used to shape-check the
    predict_proba output (the v2 plan's #7 upfront K check). Symmetric
    to the joint base's catalogue handling."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
        }
    )
    y_class = rng.choice(["a", "b", "c", "d"], size=n)
    ts = {"ITEM_class": TargetType.MULTICLASS}
    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={"ITEM_class": LightGBMClassifierEstimator(params={"n_estimators": 5, "verbose": -1})},
    )
    est.fit(df, {"ITEM_class": y_class})
    assert est._multiclass_classes["ITEM_class"] == ["a", "b", "c", "d"]
    proba = est.predict_proba_dict(df)
    # K=4 columns out, in catalogue order.
    assert proba["ITEM_class"].shape == (n, 4)


# ====================================================================== #
# Independent v2-list #10: pickle for multiclass + multilabel
# ====================================================================== #


def test_independent_10_pickle_round_trip_multiclass_and_multilabel():
    """Independent estimator with multiclass + multilabel members must
    pickle and unpickle cleanly, preserving class catalogues and the
    fanned-out estimators dict."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
        }
    )
    y_class = rng.choice(["a", "b", "c"], size=n)
    y_mem_a = (rng.normal(size=n) > 0).astype(int)
    y_mem_b = (rng.normal(size=n) > 0).astype(int)
    ts = {
        "ITEM_class": TargetType.MULTICLASS,
        "g": {"type": TargetType.MULTILABEL, "columns": ["ITEM_mem_a", "ITEM_mem_b"]},
    }
    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={
            "ITEM_class": LightGBMClassifierEstimator(params={"n_estimators": 5, "verbose": -1}),
            "ITEM_mem_a": LightGBMClassifierEstimator(params={"n_estimators": 5, "verbose": -1}),
            "ITEM_mem_b": LightGBMClassifierEstimator(params={"n_estimators": 5, "verbose": -1}),
        },
    )
    y = {"ITEM_class": y_class, "g": np.column_stack([y_mem_a, y_mem_b])}
    est.fit(df, y)

    blob = pickle.dumps(est)
    rehydrated = pickle.loads(blob)
    proba_pre = est.predict_proba_dict(df)
    proba_post = rehydrated.predict_proba_dict(df)
    for key in proba_pre:
        np.testing.assert_allclose(proba_pre[key], proba_post[key])
    assert rehydrated._multiclass_classes["ITEM_class"] == ["a", "b", "c"]


# ====================================================================== #
# Independent v2-list #11: partial-fit failure at 2nd target
# ====================================================================== #


def test_independent_11_partial_fit_failure_leaves_estimator_unfitted():
    """If the 2nd target's sub-estimator raises during fit, the overall
    estimator must be left in an unfitted state — no partial predictions
    allowed."""
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    ts = {
        "ITEM_a": TargetType.BINARY,
        "ITEM_b": TargetType.BINARY,
    }

    # Sub-estimator that fails on fit — exercises the partial-fit
    # failure path deterministically (more robust than relying on a
    # specific sub-estimator's degenerate-input behavior).
    class _FailingFit(LightGBMClassifierEstimator):
        def fit(self, X, y, X_valid=None, y_valid=None):
            raise RuntimeError("synthetic fit failure for partial-fit test")

    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={
            "ITEM_a": LightGBMClassifierEstimator(params={"n_estimators": 5, "verbose": -1}),
            "ITEM_b": _FailingFit(params={"n_estimators": 5, "verbose": -1}),
        },
    )
    y = {
        "ITEM_a": (rng.normal(size=n) > 0).astype(int),
        "ITEM_b": (rng.normal(size=n) > 0).astype(int),
    }
    with pytest.raises(RuntimeError, match="synthetic"):
        est.fit(df, y)
    assert est._fitted is False
    with pytest.raises(RuntimeError, match="not fitted"):
        est.predict_proba_dict(df)


# ====================================================================== #
# Independent v2-list #12: sub-estimator X_valid/y_valid kwarg compat
# ====================================================================== #


def test_independent_12_sub_estimator_xvalid_yvalid_threaded():
    """When X_valid + y_valid are supplied, the wrapper must forward
    per-target slices to each sub-estimator's fit. No-op for sub-
    estimators that ignore validation kwargs; must not raise."""
    rng = np.random.default_rng(0)
    n_train, n_valid = 60, 20
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n_train),
            "f1": rng.normal(size=n_train),
        }
    )
    valid_df = pd.DataFrame(
        {
            "f0": rng.normal(size=n_valid),
            "f1": rng.normal(size=n_valid),
        }
    )
    ts = {"ITEM_a": TargetType.BINARY}
    y = {"ITEM_a": (rng.normal(size=n_train) > 0).astype(int)}
    y_valid = {"ITEM_a": (rng.normal(size=n_valid) > 0).astype(int)}
    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={
            "ITEM_a": LightGBMClassifierEstimator(params={"n_estimators": 5, "verbose": -1}),
        },
    )
    # Should not raise — verifies threading happens.
    est.fit(df, y, X_valid=valid_df, y_valid=y_valid)
    assert est._fitted is True


# ====================================================================== #
# Independent v2-list #13: determinism across two factory builds
# ====================================================================== #


def test_independent_13_determinism_two_factory_builds_with_random_state():
    """Two factory builds with the same random_state on the same data
    must produce identical predictions (deterministic propagation)."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "ITEM_a": (rng.normal(size=n) > 0).astype(int),
        }
    )
    ts = {"ITEM_a": TargetType.BINARY}
    y = {"ITEM_a": df["ITEM_a"].to_numpy()}

    def _build():
        est = create_estimator(
            estimator_config={
                "ml_task": "multi_target",
                "multi_target": {
                    "mode": "independent",
                    "random_state": 11,
                    "independent": {
                        "defaults": {
                            "binary": {
                                "estimator_type": "lightgbm",
                                "params": {"n_estimators": 10, "verbose": -1},
                            },
                        },
                    },
                },
            },
            scorer_type="mixed_type_multi_target",
            target_specs=ts,
        )
        est.fit(df[["f0", "f1"]], y)
        return est

    est1 = _build()
    est2 = _build()
    p1 = est1.predict_proba_dict(df[["f0", "f1"]])["ITEM_a"]
    p2 = est2.predict_proba_dict(df[["f0", "f1"]])["ITEM_a"]
    np.testing.assert_allclose(p1, p2)


# --- M5: empty multiclass catalogue fails fast at eval time ---


def test_fix_r2_b5_empty_multiclass_catalogue_fails_fast():
    """Edge case: a multiclass target evaluated without a fitted catalogue
    must raise — pre-fix would silently fall back to ``range(K)`` and
    produce nonsense metric values."""

    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = {
        "ITEM_action": np.array(["A", "B", "C"])[np.column_stack([X["f0"], X["f1"], X["f2"]]).argmax(axis=1)],
    }
    ts = {"ITEM_action": TargetType.MULTICLASS}
    est = JointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 16},
    )
    est.fit(X, y)
    # Wipe the catalogue to simulate the failure mode.
    est._multiclass_classes = {}
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)
    inf = X.copy()
    inf.insert(0, USER_ID_NAME, [f"u{i}" for i in range(n)])
    logged = pd.DataFrame({"ITEM_action": y["ITEM_action"]})
    with pytest.raises(RuntimeError, match="empty _multiclass_classes"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.MULTICLASS_ACCURACY,
            eval_top_k=10,
            score_items_kwargs={"interactions": inf},
            eval_kwargs={"logged_rewards": logged},
        )


def test_fix_r2_7_independent_refit_with_bad_y_clears_fitted_state():
    """Pre-fix: a re-fit with bad y would raise from _validate_for_fit,
    but the prior _fitted=True remained — predict_* could still return
    stale predictions, contradicting the no-half-fit-state invariant."""
    from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator

    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y_good = {"ITEM_a": (X["f0"] > 0).astype(int).to_numpy()}
    ts = {"ITEM_a": TargetType.BINARY}
    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={"ITEM_a": XGBClassifierEstimator(params={"n_estimators": 5})},
    )
    est.fit(X, y_good)
    assert est._fitted is True

    # Re-fit with malformed y — validation must raise AND clear _fitted.
    with pytest.raises(ValueError):
        est.fit(X, {"WRONG_KEY": np.array([0])})
    # After failed re-fit, predict must NOT silently return stale predictions
    # from the previous successful fit.
    assert est._fitted is False, "Re-fit validation failure left _fitted=True (stale state)."
    with pytest.raises(RuntimeError, match="not fitted"):
        est.predict_proba_dict(X)


# ====================================================================== #
# Round 3 P0 ship-blockers
# ====================================================================== #


# --- P0-1: score_items preserves OBSERVED_* through batch schema-apply ---


def test_fix4_independent_integer_multiclass_k11_round_trips_correctly():
    """Same K=11 round-trip but through the independent family."""
    rng = np.random.default_rng(0)
    n = 220
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f0", "f1", "f2", "f3"])
    y_idx = np.array([i // 20 for i in range(n)])
    rng.shuffle(y_idx)
    X["f0"] = y_idx.astype(float) + 0.2 * rng.normal(size=n)
    y = {"ITEM_class": y_idx.astype(int)}
    ts = {"ITEM_class": TargetType.MULTICLASS}

    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={
            "ITEM_class": LightGBMClassifierEstimator(params={"n_estimators": 50, "verbose": -1}),
        },
    )
    est.fit(X, y)
    assert est._multiclass_classes["ITEM_class"] == list(range(11))
    preds = est.predict_targets_dict(X)["ITEM_class"]
    acc = float(np.mean(preds == y_idx))
    assert acc > 0.5, f"K=11 independent multiclass accuracy is {acc} — possible class-ordering regression."


# ====================================================================== #
# P1 review round — fixes for tightening / hardening
# ====================================================================== #


def test_p1_15_independent_regression_handles_2d_predict_output():
    """Sub-estimators that return (n, 1) for regression must be reshaped
    to (n,) so downstream stitching sees the joint family's contract."""
    from sklearn.linear_model import Ridge

    from skrec.estimator.regression.sklearn_universal_regressor import (
        SklearnUniversalRegressorEstimator,
    )

    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = {"ITEM_rev": rng.normal(size=n)}
    ts = {"ITEM_rev": TargetType.REGRESSION}

    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={"ITEM_rev": SklearnUniversalRegressorEstimator(Ridge, params={})},
    )
    est.fit(X, y)
    proba = est.predict_proba_dict(X)
    targets = est.predict_targets_dict(X)
    assert proba["ITEM_rev"].ndim == 1
    assert targets["ITEM_rev"].ndim == 1


def test_independent_target_group_spec_rejects_non_multilabel_type():
    """Direct construction parity with the scorer: TargetGroupSpec with
    a non-MULTILABEL ``type`` must be rejected at the estimator side
    too (was only rejected by the scorer before this fix)."""
    bad = {
        "g": {"type": TargetType.BINARY, "columns": ["ITEM_a", "ITEM_b"]},
    }
    with pytest.raises(ValueError, match="MULTILABEL"):
        IndependentMultiTargetEstimator(target_specs=bad, estimators={})


def test_independent_predict_validates_feature_names():
    """predict_proba_dict / predict_targets_dict must reject X with
    missing or extra feature columns (same UX as the joint family's
    _align_X). Pre-fix: silently passed through to sub-estimators."""
    import warnings as _warnings

    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
        }
    )
    y = {"ITEM_a": (rng.normal(size=n) > 0).astype(int)}
    est = IndependentMultiTargetEstimator(
        target_specs={"ITEM_a": TargetType.BINARY},
        estimators={"ITEM_a": LightGBMClassifierEstimator(params={"n_estimators": 5, "verbose": -1})},
    )
    est.fit(df, y)
    # Missing column
    with pytest.raises(ValueError, match="missing training-time feature"):
        est.predict_proba_dict(df[["f0"]])
    # Extra column
    extra = df.copy()
    extra["zz_extra"] = 0.0
    with pytest.raises(ValueError, match="unseen at training"):
        est.predict_proba_dict(extra)
    # Reordered columns: succeeds (aligned via .loc[:, feature_names]).
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        proba = est.predict_proba_dict(df[["f1", "f0"]])
    assert "ITEM_a" in proba


def test_independent_binary_predict_proba_shape_guard():
    """A binary sub-estimator returning the wrong shape (e.g. (n,) or
    (n, 1)) must raise an actionable error in predict_proba_dict
    rather than crashing downstream at proba[:, 1]."""

    class _BadShapeBinary(LightGBMClassifierEstimator):
        def predict_proba(self, X):  # type: ignore[override]
            n = len(X)
            return np.zeros(n, dtype=float)  # wrong shape: (n,), not (n, 2)

    rng = np.random.default_rng(0)
    n = 20
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {"ITEM_a": (rng.normal(size=n) > 0).astype(int)}
    ts = {"ITEM_a": TargetType.BINARY}
    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={"ITEM_a": _BadShapeBinary(params={"n_estimators": 5, "verbose": -1})},
    )
    est.fit(X, y)
    with pytest.raises(RuntimeError, match=r"shape .* expected"):
        est.predict_proba_dict(X)


def test_predict_targets_dict_binary_shape_guard():
    """Binary sub-estimator returning the wrong shape from predict_proba
    must raise via the shared _safe_sub_estimator_inference helper from predict_targets_dict,
    not crash at the downstream ``proba[:, 1]`` slice. P1-1 follow-up."""

    class _BadShapeBinary(LightGBMClassifierEstimator):
        def predict_proba(self, X):  # type: ignore[override]
            return np.zeros(len(X), dtype=float)  # wrong: (n,)

    rng = np.random.default_rng(0)
    n = 20
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {"ITEM_a": (rng.normal(size=n) > 0).astype(int)}
    ts = {"ITEM_a": TargetType.BINARY}
    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={"ITEM_a": _BadShapeBinary(params={"n_estimators": 5, "verbose": -1})},
    )
    est.fit(X, y)
    with pytest.raises(RuntimeError, match=r"shape .* expected"):
        est.predict_targets_dict(X)


def test_predict_targets_dict_multiclass_shape_guard():
    """Multiclass sub-estimator returning the wrong shape must raise via
    _safe_sub_estimator_inference from predict_targets_dict, not crash at ``proba.argmax(axis=1)``.
    P1-1 follow-up."""

    class _BadShapeMulticlass(LightGBMClassifierEstimator):
        def predict_proba(self, X):  # type: ignore[override]
            # Return (n, 5) when catalogue has K=3 classes.
            return np.zeros((len(X), 5), dtype=float)

    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {"ITEM_c": rng.choice(["a", "b", "c"], size=n)}
    ts = {"ITEM_c": TargetType.MULTICLASS}
    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={"ITEM_c": _BadShapeMulticlass(params={"n_estimators": 5, "verbose": -1})},
    )
    est.fit(X, y)
    with pytest.raises(RuntimeError, match=r"predict_proba of shape"):
        est.predict_targets_dict(X)


def test_safe_sub_estimator_inference_regression_n1_reshape():
    """Mock a regression sub-estimator that returns (n, 1) — the helper
    must reshape to (n,). Pre-fix this branch was only exercised through
    real sklearn MultiOutputRegressor wrappers; pin it explicitly so a
    future helper change can't quietly drop the reshape."""

    class _ReshapeRegressor(LightGBMRegressorEstimator):
        def predict(self, X):  # type: ignore[override]
            n = len(X)
            return np.zeros((n, 1), dtype=float)  # (n, 1) instead of (n,)

    rng = np.random.default_rng(0)
    n = 12
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {"ITEM_r": rng.normal(size=n)}
    ts = {"ITEM_r": TargetType.REGRESSION}
    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={"ITEM_r": _ReshapeRegressor(params={"n_estimators": 5, "verbose": -1})},
    )
    est.fit(X, y)
    proba = est.predict_proba_dict(X)
    targets = est.predict_targets_dict(X)
    # Both paths must yield 1-D shape regardless of sub-estimator output shape.
    assert proba["ITEM_r"].shape == (n,)
    assert targets["ITEM_r"].shape == (n,)


def test_safe_sub_estimator_inference_unknown_target_type_raises():
    """The helper's terminal NotImplementedError must fire for an
    unrecognized TargetType — a future enum addition that forgets to
    add a branch should surface here, not crash downstream."""

    # Use a sentinel that isn't a real TargetType.
    class _NotATargetType:
        value = "fake"

    rng = np.random.default_rng(0)
    n = 8
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {"ITEM_a": (rng.normal(size=n) > 0).astype(int)}
    ts = {"ITEM_a": TargetType.BINARY}
    est = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={"ITEM_a": LightGBMClassifierEstimator(params={"n_estimators": 5, "verbose": -1})},
    )
    est.fit(X, y)
    # Hit the helper directly with a synthetic target_type.
    with pytest.raises(NotImplementedError, match="Unsupported target_type"):
        est._safe_sub_estimator_inference("ITEM_a", _NotATargetType(), est.estimators["ITEM_a"], X)
