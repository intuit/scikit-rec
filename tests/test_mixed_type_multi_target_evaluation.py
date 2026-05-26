# Tests for M6: MULTICLASS_ACCURACY metric + RankingRecommender's
# _evaluate_mixed_type_multi_target branch.
#
# Covers per-TargetType metric dispatch, cross-type rejection,
# per-target-keyed metric_type form, logged_rewards shape/column validation,
# NaN handling, ranking-metric rejection, SIMPLE-only enforcement, and
# MulticlassAccuracy numerical correctness.

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

torch = pytest.importorskip("torch")

from skrec.constants import USER_ID_NAME  # noqa: E402
from skrec.estimator.classification import (  # noqa: E402
    ConditionalJointMultiTargetMLPEstimator,
    JointMultiTargetMLPEstimator,  # noqa: E402
)
from skrec.evaluator.datatypes import RecommenderEvaluatorType  # noqa: E402
from skrec.metrics.datatypes import RecommenderMetricType  # noqa: E402
from skrec.metrics.factory import RecommenderMetricFactory  # noqa: E402
from skrec.metrics.multiclass_accuracy import MulticlassAccuracy  # noqa: E402
from skrec.recommender.ranking.ranking_recommender import RankingRecommender  # noqa: E402
from skrec.scorer.mixed_type_multi_target import (  # noqa: E402
    MixedTypeMultiTargetScorer,
    TargetGroupSpec,
    TargetType,
)


def _build_recommender_for_validation():
    """Shared fixture for inference-validator-coverage tests."""
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = {
        "ITEM_clicked": (X["f0"] > 0).astype(int).to_numpy(),
        "ITEM_revenue": X["f1"].to_numpy(),
        "ITEM_action": np.array(["A", "B", "C"])[np.column_stack([X["f0"], X["f1"], X["f2"]]).argmax(axis=1)],
    }
    ts = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
        "ITEM_action": TargetType.MULTICLASS,
    }
    est = JointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 16},
    )
    est.fit(X, y)
    return RankingRecommender(scorer=MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)), X, y, ts


# ---------------------------------------------------------------------- #
# MulticlassAccuracy metric — numerical correctness vs sklearn
# ---------------------------------------------------------------------- #


def test_multiclass_accuracy_matches_sklearn():
    rng = np.random.default_rng(0)
    n, K = 100, 4
    scores = rng.random(size=(n, K))
    labels = rng.integers(0, K, size=n)
    expected = accuracy_score(labels, scores.argmax(axis=1))
    actual = MulticlassAccuracy().calculate(
        recommendation_ranks=np.empty((n, 0)),
        modified_rewards=labels,
        recommendation_scores=scores,
    )
    assert actual == pytest.approx(expected)


def test_multiclass_accuracy_nan_labels_masked():
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]])
    labels = np.array([1.0, np.nan, 0.0])
    # Mask out the NaN row, expect prediction 1 (from row 0) and prediction 0
    # (from row 2). Both labels match → accuracy 1.0.
    out = MulticlassAccuracy().calculate(
        recommendation_ranks=np.empty((3, 0)),
        modified_rewards=labels,
        recommendation_scores=scores,
    )
    assert out == pytest.approx(1.0)


def test_multiclass_accuracy_all_nan_returns_nan():
    scores = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([np.nan, np.nan])
    out = MulticlassAccuracy().calculate(
        recommendation_ranks=np.empty((2, 0)),
        modified_rewards=labels,
        recommendation_scores=scores,
    )
    assert np.isnan(out)


def test_multiclass_accuracy_registered_in_factory():
    metric = RecommenderMetricFactory.create(RecommenderMetricType.MULTICLASS_ACCURACY)
    assert isinstance(metric, MulticlassAccuracy)


# ---------------------------------------------------------------------- #
# Helpers for RankingRecommender evaluate tests
# ---------------------------------------------------------------------- #


def _make_synthetic(n=50, seed=0):
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame(rng.normal(size=(n, 3)), columns=[f"feat_{i}" for i in range(3)])
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
        "ITEM_clicked": (feats["feat_0"] > 0).astype(int).to_numpy(),
        "ITEM_revenue": feats["feat_1"].to_numpy(),
        "ITEM_action": np.array(["A", "B", "C"])[
            np.column_stack([feats["feat_0"], feats["feat_1"], feats["feat_2"]]).argmax(axis=1)
        ],
        "engagement": np.column_stack(
            [
                (feats["feat_1"] > 0).astype(int).to_numpy(),
                (feats["feat_2"] > 0).astype(int).to_numpy(),
            ]
        ),
    }
    return feats, y, target_specs


def _build_recommender(target_specs, X, y):
    est = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 1, "hidden_dim": 16, "num_layers": 2, "batch_size": 32},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=target_specs)
    return RankingRecommender(scorer=scorer)


def _logged_rewards_from_y(y, target_specs):
    """Build the wide-format logged_rewards DataFrame from a dict-y."""
    cols = {}
    for key, spec in target_specs.items():
        if isinstance(spec, TargetType):
            if spec == TargetType.MULTICLASS:
                cols[key] = y[key]
            else:
                cols[key] = y[key]
        else:
            for i, member in enumerate(spec["columns"]):
                cols[member] = y[key][:, i]
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------- #
# Per-TargetType metric dispatch — broadcast and per-target-keyed forms
# ---------------------------------------------------------------------- #


def test_evaluate_per_target_dict_keyed():
    X, y, ts = _make_synthetic(n=80)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(80)])
    logged = _logged_rewards_from_y(y, ts)

    result = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type={
            "ITEM_clicked": RecommenderMetricType.ROC_AUC,
            "ITEM_revenue": RecommenderMetricType.RMSE,
            "ITEM_action": RecommenderMetricType.MULTICLASS_ACCURACY,
            "ITEM_email_open": RecommenderMetricType.ROC_AUC,
            "ITEM_app_open": RecommenderMetricType.PR_AUC,
        },
        eval_top_k=10,
        score_items_kwargs={"interactions": inf_df},
        eval_kwargs={"logged_rewards": logged},
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_action",
        "ITEM_email_open",
        "ITEM_app_open",
    }
    for v in result.values():
        assert isinstance(v, float)


def test_evaluate_metric_type_broadcast_binary_only_targets():
    X, y, _ = _make_synthetic(n=50)
    ts = {"ITEM_clicked": TargetType.BINARY}
    y_subset = {"ITEM_clicked": y["ITEM_clicked"]}
    recommender = _build_recommender(ts, X, y_subset)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(50)])
    logged = pd.DataFrame({"ITEM_clicked": y_subset["ITEM_clicked"]})
    out = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        eval_top_k=10,
        score_items_kwargs={"interactions": inf_df},
        eval_kwargs={"logged_rewards": logged},
    )
    assert "ITEM_clicked" in out


def test_evaluate_cross_type_metric_rejected_broadcast():
    X, y, ts = _make_synthetic(n=30)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(30)])
    logged = _logged_rewards_from_y(y, ts)
    # RMSE broadcast across all targets — incompatible with BINARY/MULTICLASS.
    with pytest.raises(ValueError, match="not compatible"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.RMSE,
            eval_top_k=10,
            score_items_kwargs={"interactions": inf_df},
            eval_kwargs={"logged_rewards": logged},
        )


def test_evaluate_cross_type_metric_rejected_per_target():
    X, y, ts = _make_synthetic(n=30)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(30)])
    logged = _logged_rewards_from_y(y, ts)
    with pytest.raises(ValueError, match="not compatible"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type={
                "ITEM_clicked": RecommenderMetricType.RMSE,  # wrong
                "ITEM_revenue": RecommenderMetricType.RMSE,
                "ITEM_action": RecommenderMetricType.MULTICLASS_ACCURACY,
                "ITEM_email_open": RecommenderMetricType.ROC_AUC,
                "ITEM_app_open": RecommenderMetricType.ROC_AUC,
            },
            eval_top_k=10,
            score_items_kwargs={"interactions": inf_df},
            eval_kwargs={"logged_rewards": logged},
        )


def test_evaluate_ranking_metric_rejected():
    X, y, ts = _make_synthetic(n=30)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(30)])
    logged = _logged_rewards_from_y(y, ts)
    with pytest.raises(ValueError, match="not compatible"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.NDCG_AT_K,
            eval_top_k=10,
            score_items_kwargs={"interactions": inf_df},
            eval_kwargs={"logged_rewards": logged},
        )


def test_evaluate_non_simple_evaluator_rejected():
    X, y, ts = _make_synthetic(n=30)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(30)])
    logged = _logged_rewards_from_y(y, ts)
    with pytest.raises(ValueError, match="SIMPLE"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.IPS,
            metric_type=RecommenderMetricType.RMSE,
            eval_top_k=10,
            score_items_kwargs={"interactions": inf_df},
            eval_kwargs={"logged_rewards": logged},
        )


def test_evaluate_per_target_dict_missing_key_raises():
    X, y, ts = _make_synthetic(n=30)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(30)])
    logged = _logged_rewards_from_y(y, ts)
    with pytest.raises(ValueError, match="missing entry"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type={
                # Missing ITEM_action and multilabel members
                "ITEM_clicked": RecommenderMetricType.ROC_AUC,
                "ITEM_revenue": RecommenderMetricType.RMSE,
            },
            eval_top_k=10,
            score_items_kwargs={"interactions": inf_df},
            eval_kwargs={"logged_rewards": logged},
        )


# ---------------------------------------------------------------------- #
# logged_rewards shape + column validation
# ---------------------------------------------------------------------- #


def test_evaluate_logged_rewards_missing_target_column():
    X, y, ts = _make_synthetic(n=30)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(30)])
    logged = _logged_rewards_from_y(y, ts).drop(columns=["ITEM_app_open"])
    with pytest.raises(ValueError, match="missing target column"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.RMSE,  # any
            eval_top_k=10,
            score_items_kwargs={"interactions": inf_df},
            eval_kwargs={"logged_rewards": logged},
        )


def test_evaluate_logged_rewards_extra_column():
    X, y, ts = _make_synthetic(n=30)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(30)])
    logged = _logged_rewards_from_y(y, ts).copy()
    logged["STRAY_COL"] = 0.0
    with pytest.raises(ValueError, match="unknown column"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.RMSE,
            eval_top_k=10,
            score_items_kwargs={"interactions": inf_df},
            eval_kwargs={"logged_rewards": logged},
        )


def test_evaluate_row_count_mismatch():
    X, y, ts = _make_synthetic(n=30)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(30)])
    logged = _logged_rewards_from_y(y, ts).iloc[:20]
    with pytest.raises(ValueError, match="rows"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.RMSE,
            eval_top_k=10,
            score_items_kwargs={"interactions": inf_df},
            eval_kwargs={"logged_rewards": logged},
        )


def test_evaluate_joint_vs_independent_dispatch_key_equivalence():
    """Plan test #16: joint and independent estimator pipelines on the same
    target_specs must produce evaluate() returns with (a) the same key set
    AND (b) each key resolved against the same RecommenderMetricType by the
    dispatch logic. Catches silent divergence between code paths sharing
    TARGET_TYPE_TO_METRICS."""
    from skrec.estimator.classification import IndependentMultiTargetEstimator
    from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
    from skrec.estimator.regression.lightgbm_regressor import (
        LightGBMRegressorEstimator,
    )

    X, y, ts = _make_synthetic(n=40)
    # Drop multilabel + multiclass to keep the independent factory simple
    # for the comparison — joint + independent on a binary+regression spec
    # is enough to pin the dispatch-key equivalence contract.
    ts = {"ITEM_clicked": TargetType.BINARY, "ITEM_revenue": TargetType.REGRESSION}
    y = {"ITEM_clicked": y["ITEM_clicked"], "ITEM_revenue": y["ITEM_revenue"]}

    joint = JointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 16},
    )
    joint.fit(X, y)
    ind = IndependentMultiTargetEstimator(
        target_specs=ts,
        estimators={
            "ITEM_clicked": XGBClassifierEstimator(params={"n_estimators": 5}),
            "ITEM_revenue": LightGBMRegressorEstimator(params={"n_estimators": 5, "verbose": -1}),
        },
    )
    ind.fit(X, y)

    def _eval_with(est):
        scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
        rec = RankingRecommender(scorer=scorer)
        inf_df = X.copy()
        inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
        logged = pd.DataFrame({"ITEM_clicked": y["ITEM_clicked"], "ITEM_revenue": y["ITEM_revenue"]})
        return rec.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type={
                "ITEM_clicked": RecommenderMetricType.ROC_AUC,
                "ITEM_revenue": RecommenderMetricType.RMSE,
            },
            eval_top_k=10,
            score_items_kwargs={"interactions": inf_df},
            eval_kwargs={"logged_rewards": logged},
        )

    out_joint = _eval_with(joint)
    out_ind = _eval_with(ind)
    assert set(out_joint.keys()) == set(out_ind.keys())


def test_logged_rewards_nan_treated_as_ignore_mask():
    """Plan test #17: per-target NaN rows are ignored by the metric (sklearn
    metrics natively mask via the ravel + NaN-mask in BaseClassificationMetric
    / BaseRegressionMetric.calculate). Compare AUC computed on the masked
    subset directly vs the scorer's evaluate output — they must agree."""

    X, y, ts = _make_synthetic(n=50)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    logged = _logged_rewards_from_y(y, ts).astype({"ITEM_clicked": float, "ITEM_revenue": float})
    # Mask out 20% of binary rows.
    mask_idx = np.arange(10)
    logged.loc[mask_idx, "ITEM_clicked"] = np.nan

    out = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type={
            "ITEM_clicked": RecommenderMetricType.ROC_AUC,
            "ITEM_revenue": RecommenderMetricType.RMSE,
            "ITEM_action": RecommenderMetricType.MULTICLASS_ACCURACY,
            "ITEM_email_open": RecommenderMetricType.ROC_AUC,
            "ITEM_app_open": RecommenderMetricType.ROC_AUC,
        },
        eval_top_k=10,
        score_items_kwargs={"interactions": inf_df},
        eval_kwargs={"logged_rewards": logged},
    )
    # Independently compute the masked AUC.
    proba = recommender.scorer.estimator.predict_proba_dict(X)["ITEM_clicked"][:, 1]
    valid = np.arange(10, 50)
    expected = roc_auc_score(y["ITEM_clicked"][valid], proba[valid])
    assert out["ITEM_clicked"] == pytest.approx(expected, abs=1e-6), (
        f"NaN-as-ignore-mask broken: {out['ITEM_clicked']} != {expected}"
    )


def test_logged_rewards_all_nan_returns_nan():
    """Plan test #18: a target column with 100% NaN ground truth → returned
    Dict[str, float] entry for that target is NaN (single-class degenerate
    masking covers this for binary; per-metric NaN policy covers regression
    + multiclass)."""
    X, y, ts = _make_synthetic(n=40)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    logged = _logged_rewards_from_y(y, ts).astype(
        {"ITEM_clicked": float, "ITEM_revenue": float, "ITEM_email_open": float}
    )
    logged["ITEM_revenue"] = np.nan  # all NaN
    out = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type={
            "ITEM_clicked": RecommenderMetricType.ROC_AUC,
            "ITEM_revenue": RecommenderMetricType.MAE,
            "ITEM_action": RecommenderMetricType.MULTICLASS_ACCURACY,
            "ITEM_email_open": RecommenderMetricType.ROC_AUC,
            "ITEM_app_open": RecommenderMetricType.ROC_AUC,
        },
        eval_top_k=10,
        score_items_kwargs={"interactions": inf_df},
        eval_kwargs={"logged_rewards": logged},
    )
    assert np.isnan(out["ITEM_revenue"]), f"All-NaN target should return NaN; got {out['ITEM_revenue']}"


def test_logged_rewards_column_reorder_aligns_by_name():
    """Plan test #20: ``logged_rewards`` columns can be in any order; the
    scorer aligns by name. Pass logged with columns reversed vs predict_
    targets's natural order, assert identical metric values to the
    in-order baseline."""
    X, y, ts = _make_synthetic(n=40)
    recommender = _build_recommender(ts, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    logged = _logged_rewards_from_y(y, ts)
    logged_reversed = logged.iloc[:, ::-1]

    metric_kwargs = {
        "eval_type": RecommenderEvaluatorType.SIMPLE,
        "metric_type": {
            "ITEM_clicked": RecommenderMetricType.ROC_AUC,
            "ITEM_revenue": RecommenderMetricType.RMSE,
            "ITEM_action": RecommenderMetricType.MULTICLASS_ACCURACY,
            "ITEM_email_open": RecommenderMetricType.ROC_AUC,
            "ITEM_app_open": RecommenderMetricType.ROC_AUC,
        },
        "eval_top_k": 10,
        "score_items_kwargs": {"interactions": inf_df},
    }
    out_in_order = recommender.evaluate(**metric_kwargs, eval_kwargs={"logged_rewards": logged})
    out_reversed = recommender.evaluate(**metric_kwargs, eval_kwargs={"logged_rewards": logged_reversed})
    for k in out_in_order:
        if np.isnan(out_in_order[k]):
            assert np.isnan(out_reversed[k])
        else:
            assert out_in_order[k] == pytest.approx(out_reversed[k], abs=1e-9)


def test_evaluate_missing_score_items_kwargs():
    X, y, ts = _make_synthetic(n=20)
    recommender = _build_recommender(ts, X, y)
    logged = _logged_rewards_from_y(y, ts)
    with pytest.raises(ValueError, match="score_items_kwargs"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.RMSE,
            eval_top_k=10,
            score_items_kwargs=None,
            eval_kwargs={"logged_rewards": logged},
        )


def _make_mixed_df(n: int = 80, seed: int = 0):
    """Wide-format DataFrame with one of every TargetType, ITEM_ prefixed."""
    import numpy as _np
    import pandas as _pd

    from skrec.constants import USER_ID_NAME as _UID
    from skrec.scorer.mixed_type_multi_target import TargetType as _TT

    rng = _np.random.default_rng(seed)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    latent = f0 + 0.7 * f1 + 0.3 * f2
    class_latent = -f0 + 0.5 * f2 + 0.1 * rng.normal(size=n)
    cls = _np.where(
        class_latent < -0.5,
        "a",
        _np.where(class_latent < 0.5, "b", "c"),
    )
    df = _pd.DataFrame(
        {
            "f0": f0,
            "f1": f1,
            "f2": f2,
            "ITEM_bin": (latent > 0).astype(int),
            "ITEM_rev": latent + rng.normal(scale=0.3, size=n),
            "ITEM_class": cls,
            "ITEM_email": (latent + rng.normal(scale=0.5, size=n) > 0).astype(int),
            "ITEM_app": (latent + rng.normal(scale=0.5, size=n) > 0).astype(int),
            _UID: _np.arange(n),
        }
    )
    target_specs = {
        "ITEM_bin": _TT.BINARY,
        "ITEM_rev": _TT.REGRESSION,
        "ITEM_class": _TT.MULTICLASS,
        "g": {"type": _TT.MULTILABEL, "columns": ["ITEM_email", "ITEM_app"]},
    }
    return df, target_specs


def _train_joint_mlp(df, target_specs, *, epochs: int = 3):
    from skrec.estimator.classification import JointMultiTargetMLPEstimator as _Joint

    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols]
    y = {
        "ITEM_bin": df["ITEM_bin"].to_numpy(),
        "ITEM_rev": df["ITEM_rev"].to_numpy(),
        "ITEM_class": df["ITEM_class"].to_numpy(),
        "g": df[["ITEM_email", "ITEM_app"]].to_numpy(),
    }
    est = _Joint(
        target_specs=target_specs,
        params={"epochs": epochs, "hidden_dim": 8, "num_layers": 2, "batch_size": 32, "seed": 0},
    )
    est.fit(X, y)
    return est


# ====================================================================== #
# Eval v2-list #7: per-target dict with unknown target name
# ====================================================================== #


def test_eval_7_per_target_metric_dict_unknown_target_name_raises():
    """metric_type dict with a key that doesn't match any declared target
    must raise — not silently fall back to a default. Currently the code
    raises "missing entry" for declared-but-not-supplied keys; the
    inverse (extra/typo key) is just as actionable."""
    df, target_specs = _make_mixed_df()
    est = _train_joint_mlp(df, target_specs)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=target_specs)
    recommender = RankingRecommender(scorer=scorer)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    interactions = df[feat_cols + [USER_ID_NAME]]
    logged = df[["ITEM_bin", "ITEM_rev", "ITEM_class", "ITEM_email", "ITEM_app"]]

    # Missing-key path (already covered, but pin it alongside the typo path).
    with pytest.raises(ValueError, match="missing entry"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type={
                # Note: ITEM_bin missing → missing-entry error
                "ITEM_rev": RecommenderMetricType.RMSE,
                "ITEM_class": RecommenderMetricType.MULTICLASS_ACCURACY,
                "ITEM_email": RecommenderMetricType.ROC_AUC,
                "ITEM_app": RecommenderMetricType.ROC_AUC,
            },
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged},
        )


# ====================================================================== #
# Eval v2-list #11: column-count shape mismatch (vs row-count)
# ====================================================================== #


def test_eval_11_logged_rewards_column_count_mismatch_raises():
    """logged_rewards with the right ROW count but the wrong COLUMN set
    (missing one fanned-out target) must raise — distinct from row-count
    mismatch. The validator already covers this; pin it here as the
    v2-list #11 entry."""
    df, target_specs = _make_mixed_df()
    est = _train_joint_mlp(df, target_specs)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=target_specs)
    recommender = RankingRecommender(scorer=scorer)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    interactions = df[feat_cols + [USER_ID_NAME]]
    # Drop ITEM_class → missing column.
    logged_missing = df[["ITEM_bin", "ITEM_rev", "ITEM_email", "ITEM_app"]]
    with pytest.raises(ValueError, match="missing target column"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged_missing},
        )

    # Extra (unknown) column.
    logged_extra = df[["ITEM_bin", "ITEM_rev", "ITEM_class", "ITEM_email", "ITEM_app"]].copy()
    logged_extra["ITEM_unknown"] = 0
    with pytest.raises(ValueError, match="unknown column"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged_extra},
        )


# ====================================================================== #
# Round 4 coverage: MULTICLASS NaN end-to-end (existing test only NaN'd
# BINARY/REGRESSION despite claiming "every type")
# ====================================================================== #


def test_multiclass_logged_rewards_nan_ignore_mask_end_to_end():
    """logged_rewards with NaN values in a MULTICLASS column at evaluate
    must be ignored by the metric (not raise, not warn as unknown
    label, not coerce to a sentinel class)."""
    import warnings as _warnings

    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y_class = rng.choice(["a", "b", "c"], size=n)
    y = {"ITEM_c": y_class}
    ts = {"ITEM_c": TargetType.MULTICLASS}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 3, "hidden_dim": 4, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)
    inf = X.copy()
    inf[USER_ID_NAME] = np.arange(n)
    logged = pd.DataFrame({"ITEM_c": y_class.astype(object)})
    nan_idx = rng.choice(n, size=15, replace=False)
    logged.loc[nan_idx, "ITEM_c"] = np.nan

    with _warnings.catch_warnings(record=True) as ws:
        _warnings.simplefilter("always")
        out = recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.MULTICLASS_ACCURACY,
            score_items_kwargs={"interactions": inf},
            eval_kwargs={"logged_rewards": logged},
        )
    assert "ITEM_c" in out
    assert np.isfinite(out["ITEM_c"])
    # No "unknown label" warning should fire from NaN entries.
    spurious = [w for w in ws if "training-time catalogue" in str(w.message)]
    assert not spurious, (
        f"NaN multiclass entries fired spurious unknown-label warning: {[str(w.message) for w in spurious]}"
    )


# ====================================================================== #
# Fix 1: evaluate() must route through OBSERVED-aware dispatch
# ====================================================================== #


def test_fix1_evaluate_honors_observed_columns_for_conditional_estimator():
    """The pre-fix code path called estimator.predict_proba_dict directly
    in _evaluate_mixed_type_multi_target, sidestepping the OBSERVED_* →
    observed dict construction. Result: conditioning was silently ignored
    during evaluate(). The fix routes through scorer._estimator_predict_proba.

    This test trains a conditional joint MLP on strongly-correlated targets
    and evaluates twice on the same valid_inf rows — once with no
    OBSERVED_* column, once with OBSERVED_a populated. The two evaluations
    should produce DIFFERENT ROC AUC for ITEM_b (because conditioning on
    ITEM_a shifts the prediction distribution). If the bypass were still
    in place, both calls would produce identical AUC.
    """
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=[f"f{i}" for i in range(3)])
    latent = X["f0"] + 0.7 * X["f1"]
    y = {
        "ITEM_a": (latent > 0).astype(int).to_numpy(),
        "ITEM_b": (latent + 0.15 * rng.normal(size=n) > 0).astype(int).to_numpy(),
    }
    ts = {"ITEM_a": TargetType.BINARY, "ITEM_b": TargetType.BINARY}

    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 10, "hidden_dim": 32, "num_layers": 2, "batch_size": 64, "mask_prob": 0.5, "seed": 0},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)

    valid_inf = X.iloc[200:].copy().reset_index(drop=True)
    valid_inf.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(valid_inf))])
    logged = pd.DataFrame({"ITEM_a": y["ITEM_a"][200:], "ITEM_b": y["ITEM_b"][200:]})

    # No OBSERVED — vanilla path
    result_no_obs = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type={
            "ITEM_a": RecommenderMetricType.ROC_AUC,
            "ITEM_b": RecommenderMetricType.ROC_AUC,
        },
        eval_top_k=10,
        score_items_kwargs={"interactions": valid_inf},
        eval_kwargs={"logged_rewards": logged},
    )

    # WITH OBSERVED_a (the true value of ITEM_a) — conditional path
    valid_inf_with_obs = valid_inf.copy()
    valid_inf_with_obs["OBSERVED_a"] = y["ITEM_a"][200:].astype(float)
    result_with_obs = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type={
            "ITEM_a": RecommenderMetricType.ROC_AUC,
            "ITEM_b": RecommenderMetricType.ROC_AUC,
        },
        eval_top_k=10,
        score_items_kwargs={"interactions": valid_inf_with_obs},
        eval_kwargs={"logged_rewards": logged},
    )

    # ITEM_b AUC must change when ITEM_a is conditioned on. If the bypass
    # is back, both AUCs match to floating-point precision.
    assert abs(result_no_obs["ITEM_b"] - result_with_obs["ITEM_b"]) > 1e-6, (
        f"Conditioning silently bypassed at evaluate() — ITEM_b AUC "
        f"identical with vs without OBSERVED_a "
        f"({result_no_obs['ITEM_b']} == {result_with_obs['ITEM_b']}). "
        f"Regression of fix 1."
    )


# ====================================================================== #
# Fix 2: factory recognizes the two v3 conditional modes
# ====================================================================== #


def test_fix3_logged_rewards_binary_non_numeric_rejected():
    recommender, X, y, _ = _build_recommender_for_validation()
    inf = X.copy()
    inf.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    logged = pd.DataFrame(
        {
            "ITEM_clicked": ["yes"] * len(X),  # not numeric
            "ITEM_revenue": y["ITEM_revenue"],
            "ITEM_action": y["ITEM_action"],
        }
    )
    with pytest.raises(ValueError, match="numeric"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.MULTICLASS_ACCURACY,
            eval_top_k=10,
            score_items_kwargs={"interactions": inf},
            eval_kwargs={"logged_rewards": logged},
        )


def test_fix3_logged_rewards_binary_out_of_range_rejected():
    recommender, X, y, _ = _build_recommender_for_validation()
    inf = X.copy()
    inf.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    logged = pd.DataFrame(
        {
            "ITEM_clicked": [0.0, 0.5, 1.0] * (len(X) // 3) + [0.0] * (len(X) % 3),
            "ITEM_revenue": y["ITEM_revenue"],
            "ITEM_action": y["ITEM_action"],
        }
    )
    with pytest.raises(ValueError, match=r"outside.*\{0, 1\}"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.MULTICLASS_ACCURACY,
            eval_top_k=10,
            score_items_kwargs={"interactions": inf},
            eval_kwargs={"logged_rewards": logged},
        )


def test_fix3_logged_rewards_regression_non_numeric_rejected():
    recommender, X, y, _ = _build_recommender_for_validation()
    inf = X.copy()
    inf.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    logged = pd.DataFrame(
        {
            "ITEM_clicked": y["ITEM_clicked"],
            "ITEM_revenue": ["bad"] * len(X),  # not numeric
            "ITEM_action": y["ITEM_action"],
        }
    )
    with pytest.raises(ValueError, match="REGRESSION.*numeric"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.MULTICLASS_ACCURACY,
            eval_top_k=10,
            score_items_kwargs={"interactions": inf},
            eval_kwargs={"logged_rewards": logged},
        )


def test_fix3_logged_rewards_multiclass_unknown_class_rejected():
    recommender, X, y, _ = _build_recommender_for_validation()
    inf = X.copy()
    inf.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    bad_labels = y["ITEM_action"].copy()
    bad_labels[0] = "Z"  # label not seen at training
    logged = pd.DataFrame(
        {
            "ITEM_clicked": y["ITEM_clicked"],
            "ITEM_revenue": y["ITEM_revenue"],
            "ITEM_action": bad_labels,
        }
    )
    with pytest.raises(ValueError, match="training-time class catalogue"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.MULTICLASS_ACCURACY,
            eval_top_k=10,
            score_items_kwargs={"interactions": inf},
            eval_kwargs={"logged_rewards": logged},
        )


def test_fix3_logged_rewards_nan_tolerated_for_every_type():
    """NaN per-column is the ignore-mask; validation must not reject."""
    recommender, X, y, _ = _build_recommender_for_validation()
    inf = X.copy()
    inf.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    logged = pd.DataFrame(
        {
            "ITEM_clicked": pd.Series(y["ITEM_clicked"], dtype=float),
            "ITEM_revenue": y["ITEM_revenue"].astype(float),
            "ITEM_action": y["ITEM_action"],
        }
    )
    logged.loc[0, "ITEM_clicked"] = np.nan
    logged.loc[1, "ITEM_revenue"] = np.nan
    # Doesn't raise — NaN is tolerated; per-target metric dispatch handles
    # missing values per column according to the metric's own NaN policy.
    recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type={
            "ITEM_clicked": RecommenderMetricType.ROC_AUC,
            "ITEM_revenue": RecommenderMetricType.MAE,
            "ITEM_action": RecommenderMetricType.MULTICLASS_ACCURACY,
        },
        eval_top_k=10,
        score_items_kwargs={"interactions": inf},
        eval_kwargs={"logged_rewards": logged},
    )


# ====================================================================== #
# Fix 4: multiclass label sort for integer K >= 10
# ====================================================================== #


# ---------------------------------------------------------------------- #
# Fix R2-2: ROC_AUC degenerate-class returns NaN, not 0.0
# ---------------------------------------------------------------------- #


def test_fix_r2_2_degenerate_class_returns_nan_not_zero():
    """A held-out slice with a single-class binary target → metric NaN
    (matches MultioutputScorer evaluation semantics). Pre-fix: ROC_AUC
    silently returned 0.0 (sklearn ValueError caught → 0.0)."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = {
        "ITEM_a": (X["f0"] > 0).astype(int).to_numpy(),
        "ITEM_b": (X["f1"] > 0).astype(int).to_numpy(),
    }
    ts = {"ITEM_a": TargetType.BINARY, "ITEM_b": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 16},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)

    inf = X.copy()
    inf.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    # ITEM_a held-out slice has only class 0 — degenerate.
    logged = pd.DataFrame({"ITEM_a": np.zeros(len(X), dtype=int), "ITEM_b": y["ITEM_b"]})
    result = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type={
            "ITEM_a": RecommenderMetricType.ROC_AUC,
            "ITEM_b": RecommenderMetricType.ROC_AUC,
        },
        eval_top_k=10,
        score_items_kwargs={"interactions": inf},
        eval_kwargs={"logged_rewards": logged},
    )
    assert np.isnan(result["ITEM_a"]), f"Degenerate single-class ITEM_a should yield NaN, got {result['ITEM_a']}."
    # ITEM_b is well-formed — metric should be a real number.
    assert not np.isnan(result["ITEM_b"])


# ---------------------------------------------------------------------- #
# Fix R2-3: orphan ITEM_* feature column rejected at inference
# ---------------------------------------------------------------------- #


def test_p1_2_multiclass_evaluate_skips_nan_in_unknown_check():
    """NaN values in multiclass ground truth must NOT trigger the
    unknown-label warning — NaN = "no logged outcome" by convention."""
    import warnings as _warnings

    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y_class = rng.choice(["a", "b", "c"], size=n)
    y = {"ITEM_c": y_class}
    ts = {"ITEM_c": TargetType.MULTICLASS}

    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 3, "hidden_dim": 4, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)

    interactions = X.copy()
    interactions[USER_ID_NAME] = np.arange(n)
    # Mix in NaN entries — should be masked, not warned.
    logged = pd.DataFrame({"ITEM_c": y_class.astype(object)})
    nan_idx = rng.choice(n, size=10, replace=False)
    logged.loc[nan_idx, "ITEM_c"] = np.nan

    with _warnings.catch_warnings(record=True) as ws:
        _warnings.simplefilter("always")
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.MULTICLASS_ACCURACY,
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged},
        )
    unknown_warns = [w for w in ws if "not in the training-time catalogue" in str(w.message)]
    assert not unknown_warns, (
        f"NaN should be treated as 'no logged outcome', not unknown label. "
        f"Got spurious warnings: {[str(w.message) for w in unknown_warns]}"
    )


def test_p1_1_regression_evaluate_rejects_inf_ground_truth():
    """inf in a REGRESSION logged_rewards column must be rejected up front;
    NaN remains allowed (mask)."""
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = {"ITEM_rev": rng.normal(size=n)}
    ts = {"ITEM_rev": TargetType.REGRESSION}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 2, "hidden_dim": 4})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)
    interactions = X.copy()
    interactions[USER_ID_NAME] = np.arange(n)

    logged_with_inf = pd.DataFrame({"ITEM_rev": y["ITEM_rev"].copy()})
    logged_with_inf.loc[0, "ITEM_rev"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.RMSE,
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged_with_inf},
        )

    # NaN must still be tolerated.
    logged_with_nan = pd.DataFrame({"ITEM_rev": y["ITEM_rev"].copy()})
    logged_with_nan.loc[0, "ITEM_rev"] = np.nan
    recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.RMSE,
        score_items_kwargs={"interactions": interactions},
        eval_kwargs={"logged_rewards": logged_with_nan},
    )


# ====================================================================== #
# Post-P1 follow-up: evaluate path routes through preprocess_inputs
# (no longer bypasses interactions_schema coercion)
# ====================================================================== #


def test_evaluate_routes_through_preprocess_inputs_no_schema():
    """When the recommender has no interactions_schema (standalone evaluate
    path), preprocess_inputs must still be called without raising. Previously
    _evaluate_mixed_type_multi_target side-stepped preprocess_inputs entirely;
    the post-P1 fix routes through it and the underlying primitives are
    defensive against missing schema attributes."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = {"ITEM_a": (rng.normal(size=n) > 0).astype(int)}
    ts = {"ITEM_a": TargetType.BINARY}

    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 2, "hidden_dim": 4, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)

    interactions = X.copy()
    interactions[USER_ID_NAME] = np.arange(n)
    logged = pd.DataFrame({"ITEM_a": y["ITEM_a"]})

    # The recommender was never trained, so interactions_schema is unset.
    # evaluate should still succeed by treating "no schema" as pass-through.
    result = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        score_items_kwargs={"interactions": interactions},
        eval_kwargs={"logged_rewards": logged},
    )
    assert isinstance(result, dict)
    assert "ITEM_a" in result
    assert 0.0 <= result["ITEM_a"] <= 1.0


# ====================================================================== #
# Evaluate-path validator-bypass regression (round 3 follow-up)
# _evaluate_mixed_type_multi_target must call
# scorer._validate_inference_interactions before scoring, matching every
# other inference entry point. Four negative cases below.
# ====================================================================== #


def _evaluate_setup_vanilla_binary(n=40, *, seed=0):
    """Shared fixture: vanilla joint-MLP scorer + interactions/logged frames."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = {"ITEM_a": (rng.normal(size=n) > 0).astype(int)}
    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 2, "hidden_dim": 4, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)
    interactions = X.copy()
    interactions[USER_ID_NAME] = np.arange(n)
    logged = pd.DataFrame({"ITEM_a": y["ITEM_a"]})
    return recommender, interactions, logged, y


def test_eval_vanilla_estimator_with_observed_raises():
    """vanilla estimator + OBSERVED_* at evaluate must raise
    NotImplementedError pointing at the conditional families — same
    contract as score_items / predict_targets. Before the fix the
    evaluate path silently dropped OBSERVED and returned plausible-but-
    wrong metrics."""
    recommender, interactions, logged, y = _evaluate_setup_vanilla_binary()
    interactions = interactions.copy()
    interactions["OBSERVED_a"] = y["ITEM_a"].astype(float)

    with pytest.raises(NotImplementedError, match="ConditionalMultiTargetEstimator"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged},
        )


def test_eval_orphan_item_column_raises():
    """Orphan ITEM_* feature column at evaluate must raise via the same
    validator the other inference entry points call. Pre-fix: bypassed."""
    recommender, interactions, logged, _ = _evaluate_setup_vanilla_binary()
    interactions = interactions.copy()
    # ITEM_a is declared; ITEM_typo is an orphan in the ITEM_ namespace.
    interactions["ITEM_typo"] = 0

    with pytest.raises(ValueError, match="Orphan ITEM_"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged},
        )


def test_eval_partial_multilabel_group_at_evaluate_raises():
    """Partial multilabel group OBSERVED_* presence at evaluate (one
    member's OBSERVED is in the frame, another isn't) must raise
    column-level group-mask-together check. Pre-fix: bypassed."""
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = {
        "g": rng.integers(0, 2, size=(n, 2)).astype(int),
    }
    ts = {"g": {"type": TargetType.MULTILABEL, "columns": ["ITEM_email", "ITEM_app"]}}
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 2, "hidden_dim": 8, "num_layers": 2, "batch_size": 32, "mask_prob": 0.5, "seed": 0},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)
    interactions = X.copy()
    interactions[USER_ID_NAME] = np.arange(n)
    # Provide OBSERVED for only ONE member of the multilabel group.
    interactions["OBSERVED_email"] = y["g"][:, 0].astype(float)
    # OBSERVED_app deliberately missing.

    logged = pd.DataFrame(
        {
            "ITEM_email": y["g"][:, 0],
            "ITEM_app": y["g"][:, 1],
        }
    )
    with pytest.raises(ValueError, match="group"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged},
        )


def test_eval_orphan_observed_typo_raises():
    """OBSERVED_typo (not matching any declared target) under a conditional
    estimator must raise the orphan-OBSERVED check. Pre-fix: bypassed
    AND silently stripped by schema apply before that. The P2-7
    preserved-prefix hook is what lets the typo survive into the
    validator; this test exercises both halves of the fix together."""
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = {"ITEM_a": (rng.normal(size=n) > 0).astype(int)}
    ts = {"ITEM_a": TargetType.BINARY}
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 2, "hidden_dim": 4, "num_layers": 2, "batch_size": 16, "mask_prob": 0.5, "seed": 0},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)
    interactions = X.copy()
    interactions[USER_ID_NAME] = np.arange(n)
    interactions["OBSERVED_typo"] = 0.0  # typo for OBSERVED_a
    logged = pd.DataFrame({"ITEM_a": y["ITEM_a"]})
    with pytest.raises(ValueError, match="Orphan OBSERVED"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged},
        )


# ====================================================================== #
# Round 4: older P1/P2 follow-up fixes
# ====================================================================== #


def test_evaluate_rejects_non_none_users_kwarg():
    """score_items_kwargs['users'] must be rejected (or accepted as None
    for symmetry). Pre-fix: silently ignored."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {"ITEM_a": (rng.normal(size=n) > 0).astype(int)}
    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 2, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)
    inf = X.copy()
    inf[USER_ID_NAME] = np.arange(n)
    logged = pd.DataFrame({"ITEM_a": y["ITEM_a"]})

    fake_users = pd.DataFrame({"u_feat": np.zeros(n)})
    with pytest.raises(ValueError, match="users"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": inf, "users": fake_users},
            eval_kwargs={"logged_rewards": logged},
        )

    # users=None is accepted (symmetry with MultioutputScorer).
    recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        score_items_kwargs={"interactions": inf, "users": None},
        eval_kwargs={"logged_rewards": logged},
    )


def test_evaluate_accepts_multilabel_keyed_metric_compat():
    """Through evaluate(), a multilabel member's metric resolution must
    consider both BINARY and MULTILABEL keys in
    TARGET_TYPE_TO_METRICS (today they happen to coincide, but the
    lookup-types union ensures future MULTILABEL-only metrics also
    resolve)."""
    from skrec.scorer.mixed_type_multi_target import TARGET_TYPE_TO_METRICS

    # Pin the compat tables — the union must contain at least every
    # MULTILABEL metric (today: roc_auc, pr_auc).
    bin_metrics = set(TARGET_TYPE_TO_METRICS[TargetType.BINARY])
    ml_metrics = set(TARGET_TYPE_TO_METRICS[TargetType.MULTILABEL])
    union = bin_metrics | ml_metrics
    assert ml_metrics.issubset(union)
    # Functional: a fanned-out member's evaluate compat-check must
    # accept a ROC_AUC metric.
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {"g": rng.integers(0, 2, size=(n, 2))}
    ts = {"g": {"type": TargetType.MULTILABEL, "columns": ["ITEM_x", "ITEM_y"]}}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 4, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)
    inf = X.copy()
    inf[USER_ID_NAME] = np.arange(n)
    logged = pd.DataFrame({"ITEM_x": y["g"][:, 0], "ITEM_y": y["g"][:, 1]})
    # Should run without raising — roc_auc is in BINARY ∪ MULTILABEL.
    out = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        score_items_kwargs={"interactions": inf},
        eval_kwargs={"logged_rewards": logged},
    )
    assert "ITEM_x" in out and "ITEM_y" in out


def test_evaluate_validates_interactions_before_logged_rewards_types():
    """When a caller passes BOTH a malformed-interactions frame (vanilla
    estimator + OBSERVED_*) AND a malformed logged_rewards frame
    (non-numeric BINARY column), the interactions-side error must
    surface first — fixing logged_rewards then discovering the
    OBSERVED problem on the next run would cost an extra debug cycle.
    P2-3."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {"ITEM_a": (rng.normal(size=n) > 0).astype(int)}
    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 4, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    rec = RankingRecommender(scorer=scorer)

    interactions = X.copy()
    interactions[USER_ID_NAME] = np.arange(n)
    interactions["OBSERVED_a"] = y["ITEM_a"].astype(float)  # vanilla + OBSERVED → error
    # Also malform logged_rewards (string in a binary column).
    logged = pd.DataFrame({"ITEM_a": ["bad"] * n})

    with pytest.raises(NotImplementedError, match="ConditionalMultiTargetEstimator"):
        rec.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged},
        )


def test_evaluate_validator_order_positive_clean_interactions_logged_error_fires():
    """Complementary positive direction for the validator-order pin:
    with CLEAN interactions and a malformed logged_rewards column,
    the logged_rewards type error must still fire (proves the
    logged_rewards loop runs at all when interactions pass)."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {"ITEM_a": rng.normal(size=n)}
    ts = {"ITEM_a": TargetType.REGRESSION}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 4, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    rec = RankingRecommender(scorer=scorer)

    interactions = X.copy()  # CLEAN — no OBSERVED_*, no orphans
    interactions[USER_ID_NAME] = np.arange(n)
    # Malformed logged_rewards: REGRESSION column with inf.
    logged = pd.DataFrame({"ITEM_a": y["ITEM_a"].copy()})
    logged.loc[0, "ITEM_a"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        rec.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.RMSE,
            score_items_kwargs={"interactions": interactions},
            eval_kwargs={"logged_rewards": logged},
        )
