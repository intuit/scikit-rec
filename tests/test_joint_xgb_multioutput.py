"""Tests for the joint multi-output XGBoost estimators and the sklearn
tree-ensemble joint path, all through MultioutputScorer.

- ``JointXGBMultiOutputClassifierEstimator`` (binary multilabel, classifier mode)
- ``JointXGBMultiOutputRegressorEstimator`` (multi-output regression, regressor mode)
- ``SklearnUniversalClassifierEstimator(RandomForestClassifier)`` works joint with
  NO new estimator (RF multilabel predict_proba already returns the list contract)
"""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import skrec.constants as C
from skrec.estimator.classification.joint_xgb_multioutput import (
    JointXGBMultiOutputClassifierEstimator,
)
from skrec.estimator.classification.sklearn_universal_classifier import (
    SklearnUniversalClassifierEstimator,
)
from skrec.estimator.regression.joint_xgb_multioutput import (
    JointXGBMultiOutputRegressorEstimator,
)
from skrec.estimator.regression.sklearn_universal_regressor import (
    SklearnUniversalRegressorEstimator,
)
from skrec.scorer.multioutput import MultioutputScorer


def _wide_binary_frame(n=300, labels=("ITEM_a", "ITEM_b", "ITEM_c"), seed=0):
    rng = np.random.RandomState(seed)
    feats = pd.DataFrame(rng.rand(n, 4), columns=[f"f{i}" for i in range(4)])
    df = feats.copy()
    df[C.USER_ID_NAME] = np.arange(n).astype(str)
    # correlated-ish binary labels
    df["ITEM_a"] = ((feats.f0 + rng.rand(n) * 0.2) > 0.6).astype(int)
    df["ITEM_b"] = ((feats.f0 + feats.f1 + rng.rand(n) * 0.2) > 1.0).astype(int)
    df["ITEM_c"] = ((feats.f2 + rng.rand(n) * 0.2) > 0.6).astype(int)
    cols = [C.USER_ID_NAME] + list(labels) + [f"f{i}" for i in range(4)]
    return df[cols]


def _wide_continuous_frame(n=300, seed=0):
    rng = np.random.RandomState(seed)
    feats = pd.DataFrame(rng.rand(n, 4), columns=[f"f{i}" for i in range(4)])
    df = feats.copy()
    df[C.USER_ID_NAME] = np.arange(n).astype(str)
    df["ITEM_x"] = feats.f0 * 2 + rng.rand(n)
    df["ITEM_y"] = feats.f1 * 3 + rng.rand(n)
    return df[[C.USER_ID_NAME, "ITEM_x", "ITEM_y"] + [f"f{i}" for i in range(4)]]


# --------------------------------------------------------------------------- #
# Joint classifier
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("labels", [("ITEM_a", "ITEM_b"), ("ITEM_a", "ITEM_b", "ITEM_c")])
def test_joint_classifier_predict_proba_is_list_of_blocks(labels):
    df = _wide_binary_frame(labels=labels)
    est = JointXGBMultiOutputClassifierEstimator({"n_estimators": 30})
    est.fit(df[[f"f{i}" for i in range(4)]], df[list(labels)])
    pp = est.predict_proba(df[[f"f{i}" for i in range(4)]])
    assert isinstance(pp, list)
    assert len(pp) == len(labels)
    for block in pp:
        block = np.asarray(block)
        assert block.shape == (len(df), 2)
        assert np.allclose(block.sum(axis=1), 1.0)


def test_joint_classifier_end_to_end_through_scorer():
    df = _wide_binary_frame(labels=("ITEM_a", "ITEM_b", "ITEM_c"))
    scorer = MultioutputScorer(JointXGBMultiOutputClassifierEstimator({"n_estimators": 30}))
    X, y = scorer.process_datasets(interactions_df=df)
    assert np.asarray(y).shape[1] == 3  # joint 2-D y
    scorer.train_model(X, y)
    proba = scorer.score_items(interactions=df.head(5))
    assert list(proba.columns) == ["ITEM_a_0", "ITEM_a_1", "ITEM_b_0", "ITEM_b_1", "ITEM_c_0", "ITEM_c_1"]
    per_target = scorer.score_items_per_target(interactions=df.head(5))
    assert list(per_target.columns) == ["ITEM_a", "ITEM_b", "ITEM_c"]
    assert per_target.shape == (5, 3)


def test_joint_classifier_both_multi_strategies_run():
    df = _wide_binary_frame()
    feat = df[[f"f{i}" for i in range(4)]]
    y = df[["ITEM_a", "ITEM_b", "ITEM_c"]]
    for strat in ("one_output_per_tree", "multi_output_tree"):
        est = JointXGBMultiOutputClassifierEstimator({"n_estimators": 10, "multi_strategy": strat})
        est.fit(feat, y)
        assert len(est.predict_proba(feat)) == 3


def test_joint_classifier_sample_weight_passthrough():
    df = _wide_binary_frame()
    est = JointXGBMultiOutputClassifierEstimator({"n_estimators": 10}, sample_weight=np.ones(len(df)))
    est.fit(df[[f"f{i}" for i in range(4)]], df[["ITEM_a", "ITEM_b", "ITEM_c"]])
    assert len(est.predict_proba(df[[f"f{i}" for i in range(4)]])) == 3


def test_multi_output_tree_with_gpu_warns(caplog):
    with caplog.at_level(logging.WARNING):
        JointXGBMultiOutputClassifierEstimator(
            {"n_estimators": 5, "multi_strategy": "multi_output_tree", "device": "cuda:0"}
        )
    assert any("vector-leaf trees on" in r.message or "CPU" in r.message for r in caplog.records)


def test_joint_classifier_importable_from_package():
    from skrec.estimator.classification import JointXGBMultiOutputClassifierEstimator as Exported

    assert Exported is JointXGBMultiOutputClassifierEstimator


# --------------------------------------------------------------------------- #
# Joint regressor
# --------------------------------------------------------------------------- #


def test_joint_regressor_predict_shape():
    df = _wide_continuous_frame()
    est = JointXGBMultiOutputRegressorEstimator({"n_estimators": 30})
    est.fit(df[[f"f{i}" for i in range(4)]], df[["ITEM_x", "ITEM_y"]])
    out = np.asarray(est.predict(df[[f"f{i}" for i in range(4)]]))
    assert out.shape == (len(df), 2)


def test_joint_regressor_end_to_end_through_scorer():
    df = _wide_continuous_frame()
    scorer = MultioutputScorer(JointXGBMultiOutputRegressorEstimator({"n_estimators": 30}))
    X, y = scorer.process_datasets(interactions_df=df)
    scorer.train_model(X, y)
    out = scorer.predict_targets(interactions=df.head(5))
    assert list(out.columns) == ["ITEM_x", "ITEM_y"]
    assert out.shape == (5, 2)


def test_joint_regressor_importable_from_package():
    from skrec.estimator.regression import JointXGBMultiOutputRegressorEstimator as Exported

    assert Exported is JointXGBMultiOutputRegressorEstimator


# --------------------------------------------------------------------------- #
# Validation-set support (eval_set / early stopping / weighted eval)
# --------------------------------------------------------------------------- #


def test_joint_classifier_validation_set():
    df = _wide_binary_frame(n=400)
    feat = [f"f{i}" for i in range(4)]
    labels = ["ITEM_a", "ITEM_b", "ITEM_c"]
    Xtr, Xv = df[feat].iloc[:300], df[feat].iloc[300:]
    Ytr, Yv = df[labels].iloc[:300], df[labels].iloc[300:]
    est = JointXGBMultiOutputClassifierEstimator({"n_estimators": 20})
    est.fit(Xtr, Ytr, X_valid=Xv, y_valid=Yv)  # 2-D y_valid eval_set
    assert len(est.predict_proba(Xv)) == 3


def test_joint_classifier_validation_with_balanced_eval_weights():
    df = _wide_binary_frame(n=400)
    feat = [f"f{i}" for i in range(4)]
    labels = ["ITEM_a", "ITEM_b", "ITEM_c"]
    Xtr, Xv = df[feat].iloc[:300], df[feat].iloc[300:]
    Ytr, Yv = df[labels].iloc[:300], df[labels].iloc[300:]
    # 'balanced' also weights the eval set (sample_weight_eval_set) on 2-D y_valid
    est = JointXGBMultiOutputClassifierEstimator({"n_estimators": 20}, sample_weight="balanced")
    est.fit(Xtr, Ytr, X_valid=Xv, y_valid=Yv)
    assert len(est.predict_proba(Xv)) == 3


def test_joint_classifier_early_stopping():
    df = _wide_binary_frame(n=500)
    feat = [f"f{i}" for i in range(4)]
    labels = ["ITEM_a", "ITEM_b", "ITEM_c"]
    Xtr, Xv = df[feat].iloc[:380], df[feat].iloc[380:]
    Ytr, Yv = df[labels].iloc[:380], df[labels].iloc[380:]
    est = JointXGBMultiOutputClassifierEstimator(
        {"n_estimators": 200, "early_stopping_rounds": 5, "eval_metric": "logloss"}
    )
    est.fit(Xtr, Ytr, X_valid=Xv, y_valid=Yv)
    # early stopping pinned trees below the n_estimators ceiling
    assert est._model.best_iteration is not None
    assert est._model.best_iteration < 199


def test_joint_regressor_validation_set():
    df = _wide_continuous_frame(n=400)
    feat = [f"f{i}" for i in range(4)]
    targets = ["ITEM_x", "ITEM_y"]
    Xtr, Xv = df[feat].iloc[:300], df[feat].iloc[300:]
    Ytr, Yv = df[targets].iloc[:300], df[targets].iloc[300:]
    est = JointXGBMultiOutputRegressorEstimator({"n_estimators": 20})
    est.fit(Xtr, Ytr, X_valid=Xv, y_valid=Yv)
    assert np.asarray(est.predict(Xv)).shape == (len(Xv), 2)


def test_joint_classifier_validation_end_to_end_through_scorer():
    """Train/validation split both flow through MultioutputScorer.train_model."""
    df = _wide_binary_frame(n=500)
    train_df, valid_df = df.iloc[:380].copy(), df.iloc[380:].copy()
    scorer = MultioutputScorer(
        JointXGBMultiOutputClassifierEstimator(
            {"n_estimators": 50, "early_stopping_rounds": 5, "eval_metric": "logloss"}
        )
    )
    X, y = scorer.process_datasets(interactions_df=train_df)
    Xv, yv = scorer.process_datasets(interactions_df=valid_df)
    scorer.train_model(X, y, Xv, yv)
    out = scorer.score_items_per_target(interactions=valid_df.head(5))
    assert out.shape == (5, 3)


# --------------------------------------------------------------------------- #
# sklearn tree ensembles work joint with NO new estimator
# --------------------------------------------------------------------------- #


def test_random_forest_joint_multilabel_through_scorer():
    df = _wide_binary_frame(labels=("ITEM_a", "ITEM_b", "ITEM_c"))
    est = SklearnUniversalClassifierEstimator(RandomForestClassifier, {"n_estimators": 40, "random_state": 0})
    scorer = MultioutputScorer(est)
    X, y = scorer.process_datasets(interactions_df=df)
    scorer.train_model(X, y)
    # one forest jointly modeling all 3 labels (shared tree structure)
    assert est._model.n_outputs_ == 3
    # RF multilabel predict_proba already returns the list-of-blocks contract
    pp = est.predict_proba(X.head(5))
    assert isinstance(pp, list) and len(pp) == 3
    per_target = scorer.score_items_per_target(interactions=df.head(5))
    assert list(per_target.columns) == ["ITEM_a", "ITEM_b", "ITEM_c"]


def test_random_forest_joint_multioutput_regression_through_scorer():
    df = _wide_continuous_frame()
    est = SklearnUniversalRegressorEstimator(RandomForestRegressor, {"n_estimators": 40, "random_state": 0})
    scorer = MultioutputScorer(est)
    X, y = scorer.process_datasets(interactions_df=df)
    scorer.train_model(X, y)
    assert est._model.n_outputs_ == 2
    out = scorer.predict_targets(interactions=df.head(5))
    assert list(out.columns) == ["ITEM_x", "ITEM_y"]
    assert out.shape == (5, 2)
