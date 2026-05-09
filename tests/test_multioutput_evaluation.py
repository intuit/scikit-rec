"""End-to-end evaluation test for the wide-multioutput contract.

Trains a ``RankingRecommender`` wired to a ``MultioutputScorer`` over an
in-memory wide-format frame (one row per user, one ``ITEM_label_*`` column per
target), then runs ``evaluate()`` and computes per-label metrics.

This exercises the same code path that wide_multioutput evaluation uses in
practice: ``inference_input.build_trimmed_interactions_schema`` strips the
``ITEM_*`` target columns from the schema before scoring (because the trained
classifier's ``X`` was built without them), and the validation frame's
``ITEM_*`` columns are consumed separately to build ``logged_items`` /
``logged_rewards`` for the evaluator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from xgboost import XGBClassifier, XGBRegressor

from skrec.dataset.interactions_dataset import InteractionMultiOutputDataset
from skrec.estimator.classification.multioutput_classifier import (
    MultiOutputClassifierEstimator,
)
from skrec.estimator.regression.multioutput_regressor import (
    MultiOutputRegressorEstimator,
)
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.multioutput import DegenerateTargetPolicy, MultioutputScorer

LABEL_COLS = [
    "ITEM_label_workflow_automation",
    "ITEM_label_advance_reporting",
    "ITEM_label_dashboard",
    "ITEM_label_batch_invoices",
    "ITEM_label_custom_roles",
]
FEATURE_COLS = ["age", "income", "tenure_days"]


def _make_wide_multioutput_frame(n_users: int, seed: int) -> pd.DataFrame:
    """Build a wide-format frame where every label has both 0s and 1s in the slice.

    Per-target label rates are spread between 30% and 70% so a leave-N-out split
    is overwhelmingly unlikely to leave a target single-class on either side —
    that single-class case is the one that exposes the
    ``_create_proba_df`` shape mismatch and is out of scope for this test.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"USER_ID": [f"user_{i}" for i in range(n_users)]})
    df["age"] = rng.integers(18, 80, size=n_users).astype(float)
    df["income"] = rng.integers(20_000, 200_000, size=n_users).astype(float)
    df["tenure_days"] = rng.integers(1, 3650, size=n_users).astype(float)
    for i, label in enumerate(LABEL_COLS):
        rate = 0.3 + 0.1 * i
        df[label] = rng.binomial(1, rate, size=n_users).astype(float)
    return df


def _write_schema(df: pd.DataFrame, path: Path) -> None:
    schema = {
        "columns": [
            {"name": "USER_ID", "type": "str"},
            *[{"name": c, "type": "float"} for c in FEATURE_COLS],
            *[{"name": c, "type": "float"} for c in LABEL_COLS],
        ]
    }
    with open(path, "w") as f:
        yaml.safe_dump(schema, f, sort_keys=False)


@pytest.fixture
def wide_multioutput_recommender(tmp_path: Path) -> tuple[RankingRecommender, pd.DataFrame]:
    train_df = _make_wide_multioutput_frame(n_users=400, seed=0)
    valid_df = _make_wide_multioutput_frame(n_users=80, seed=1)

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "interactions_schema.yaml"
    train_df.to_csv(train_path, index=False)
    _write_schema(train_df, schema_path)

    train_ds = InteractionMultiOutputDataset(
        data_location=str(train_path),
        client_schema_path=str(schema_path),
    )

    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 20, "max_depth": 3, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator))
    recommender.train(interactions_ds=train_ds)
    return recommender, valid_df


def test_multioutput_evaluate_classifier_macro_roc_auc(wide_multioutput_recommender):
    """``evaluate(metric_type=ROC_AUC)`` returns a macro-averaged scalar in [0, 1].

    The MultioutputScorer-aware ``RankingRecommender.evaluate`` override
    routes per-label classification metrics: each target's positive-class
    probability is scored against its ground-truth column, and the default
    return (``per_label=False``) is the macro mean across labels.
    """
    recommender, valid_df = wide_multioutput_recommender

    n_users = len(valid_df)
    item_names = np.array(LABEL_COLS, dtype=object)
    logged_items = np.tile(item_names, (n_users, 1))
    logged_rewards = valid_df[LABEL_COLS].to_numpy(dtype=float)

    macro_roc_auc = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        score_items_kwargs={"interactions": valid_df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
    )
    assert isinstance(macro_roc_auc, float)
    assert np.isfinite(macro_roc_auc)
    assert 0.0 <= macro_roc_auc <= 1.0


def test_multioutput_evaluate_classifier_per_label(wide_multioutput_recommender):
    """``per_label=True`` returns a Dict[label, score] for diagnostics."""
    recommender, valid_df = wide_multioutput_recommender

    n_users = len(valid_df)
    logged_items = np.tile(np.array(LABEL_COLS, dtype=object), (n_users, 1))
    logged_rewards = valid_df[LABEL_COLS].to_numpy(dtype=float)

    per_label_pr_auc = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.PR_AUC,
        score_items_kwargs={"interactions": valid_df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        per_label=True,
    )
    assert isinstance(per_label_pr_auc, dict)
    assert set(per_label_pr_auc.keys()) == set(LABEL_COLS)
    for label, value in per_label_pr_auc.items():
        assert 0.0 <= value <= 1.0, f"{label}: {value}"


@pytest.mark.parametrize(
    "ranking_metric",
    [
        RecommenderMetricType.NDCG_AT_K,
        RecommenderMetricType.MRR_AT_K,
        RecommenderMetricType.MAP_AT_K,
        RecommenderMetricType.PRECISION_AT_K,
        RecommenderMetricType.RECALL_AT_K,
        RecommenderMetricType.AVERAGE_REWARD_AT_K,
    ],
)
def test_multioutput_evaluate_classifier_ranking_metric_returns_scalar(wide_multioutput_recommender, ranking_metric):
    """Binary classifier mode supports cross-target ranking metrics.

    Binary-only enforcement at fit time guarantees ``P(positive=1)`` per
    target gives a well-defined cross-target ordering — NDCG@K, Precision@K,
    MRR@K, MAP@K, Recall@K, AverageReward@K all evaluate that ranking
    against the binary ground truth.
    """
    recommender, valid_df = wide_multioutput_recommender
    n_users = len(valid_df)
    logged_items = np.tile(np.array(LABEL_COLS, dtype=object), (n_users, 1))
    logged_rewards = valid_df[LABEL_COLS].to_numpy(dtype=float)

    value = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=ranking_metric,
        eval_top_k=3,
        score_items_kwargs={"interactions": valid_df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
    )
    assert isinstance(value, float)
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_multioutput_evaluate_ranking_per_label_rejected(wide_multioutput_recommender):
    """Ranking metrics are inherently cross-target — ``per_label=True`` raises."""
    recommender, valid_df = wide_multioutput_recommender
    n_users = len(valid_df)
    logged_items = np.tile(np.array(LABEL_COLS, dtype=object), (n_users, 1))
    logged_rewards = valid_df[LABEL_COLS].to_numpy(dtype=float)

    with pytest.raises(ValueError, match="per_label=True is incompatible"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.NDCG_AT_K,
            eval_top_k=3,
            score_items_kwargs={"interactions": valid_df},
            eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
            per_label=True,
        )


def test_multioutput_evaluate_classifier_rejects_regression_metric(wide_multioutput_recommender):
    """Regression metrics on a classifier-mode scorer raise with a precise error."""
    recommender, valid_df = wide_multioutput_recommender
    n_users = len(valid_df)
    logged_items = np.tile(np.array(LABEL_COLS, dtype=object), (n_users, 1))
    logged_rewards = valid_df[LABEL_COLS].to_numpy(dtype=float)

    with pytest.raises(ValueError, match="regressor estimator"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.RMSE,
            score_items_kwargs={"interactions": valid_df},
            eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        )


def test_multioutput_per_label_metrics(wide_multioutput_recommender):
    """Compute per-label classification metrics from score_items output.

    `MultioutputScorer.score_items` returns a (n_users, sum_of_n_classes_per_item)
    DataFrame with columns named `<ITEM_label_X>_<class>`. For binary targets
    that's two columns per label; the "_1" column is the positive-class
    probability we want to score against the held-out target.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    recommender, valid_df = wide_multioutput_recommender
    proba = recommender.score_items(interactions=valid_df)

    assert proba.shape[0] == len(valid_df)
    expected_cols = [f"{label}_{c}" for label in LABEL_COLS for c in (0, 1)]
    assert list(proba.columns) == expected_cols

    per_label = {}
    for label in LABEL_COLS:
        y_true = valid_df[label].to_numpy(dtype=int)
        y_score = proba[f"{label}_1"].to_numpy(dtype=float)
        per_label[label] = {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "positive_rate": float(y_true.mean()),
        }

    assert set(per_label.keys()) == set(LABEL_COLS)
    for label, metrics in per_label.items():
        assert 0.0 <= metrics["roc_auc"] <= 1.0, label
        assert 0.0 <= metrics["pr_auc"] <= 1.0, label


def test_multioutput_single_class_target_constant_policy(tmp_path: Path):
    """``DegenerateTargetPolicy.CONSTANT`` makes single-class targets pass through.

    Previously this configuration crashed at ``_create_proba_df`` with
    ``ValueError: Shape of passed values is (N, 2), indices imply (N, 1)`` —
    the fix is policy-driven: under ``CONSTANT`` the column is recorded in
    ``degenerate_targets``, excluded from the underlying
    ``MultiOutputClassifier`` fit, and a constant ``(N, 1)`` prediction is
    reconstructed at score time so per-target shape stays uniform. Under
    ``RAISE`` (default) the same data raises at training — see
    :func:`test_multioutput_single_class_target_raise_policy_default`.
    """
    rng = np.random.default_rng(0)
    n_users = 200
    df = pd.DataFrame({"USER_ID": [f"user_{i}" for i in range(n_users)]})
    df["age"] = rng.integers(18, 80, size=n_users).astype(float)
    df["income"] = rng.integers(20_000, 200_000, size=n_users).astype(float)
    df["tenure_days"] = rng.integers(1, 3650, size=n_users).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n_users).astype(float)
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n_users).astype(float)
    df["ITEM_label_rare"] = 0.0  # all zeros — the degenerate target

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "age", "type": "float"},
                {"name": "income", "type": "float"},
                {"name": "tenure_days", "type": "float"},
                {"name": "ITEM_label_a", "type": "float"},
                {"name": "ITEM_label_b", "type": "float"},
                {"name": "ITEM_label_rare", "type": "float"},
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 10, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    proba = recommender.score_items(interactions=df)
    assert proba.shape[0] == n_users
    # The degenerate target was tracked and excluded from the underlying fit
    # but remains in the public catalogue and score-shape contract.
    assert "ITEM_label_rare" in recommender.scorer.item_names
    assert "ITEM_label_rare" in recommender.scorer.degenerate_targets
    # All proba columns are well-formed and finite.
    assert np.isfinite(proba.to_numpy()).all()


def test_multioutput_single_class_target_raise_policy_default(tmp_path: Path):
    """Default policy ``DegenerateTargetPolicy.RAISE`` rejects single-class targets at fit.

    The error message names every offending column in one shot (so a caller
    fixing data prep doesn't discover them one retry at a time) and points
    at the ``CONSTANT`` escape hatch.
    """
    rng = np.random.default_rng(0)
    n_users = 200
    df = pd.DataFrame({"USER_ID": [f"user_{i}" for i in range(n_users)]})
    df["age"] = rng.integers(18, 80, size=n_users).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n_users).astype(float)
    df["ITEM_label_rare1"] = 0.0
    df["ITEM_label_rare2"] = 1.0

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "age", "type": "float"},
                {"name": "ITEM_label_a", "type": "float"},
                {"name": "ITEM_label_rare1", "type": "float"},
                {"name": "ITEM_label_rare2", "type": "float"},
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 10, "max_depth": 2, "objective": "binary:logistic"},
    )
    # Default on_degenerate_target=RAISE.
    recommender = RankingRecommender(MultioutputScorer(estimator))
    with pytest.raises(ValueError) as exc_info:
        recommender.train(interactions_ds=train_ds)
    msg = str(exc_info.value)
    assert "ITEM_label_rare1" in msg
    assert "ITEM_label_rare2" in msg
    assert "ITEM_label_a" not in msg  # non-degenerate column should not be flagged
    assert "on_degenerate_target" in msg  # error message points at the escape hatch


def test_multioutput_on_degenerate_target_accepts_string():
    """``on_degenerate_target`` accepts the enum or its string value (str-Enum mixin)."""
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 10, "max_depth": 2, "objective": "binary:logistic"},
    )
    scorer_str = MultioutputScorer(estimator, on_degenerate_target="constant")
    scorer_enum = MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT)
    assert scorer_str.on_degenerate_target == scorer_enum.on_degenerate_target
    assert scorer_str.on_degenerate_target == DegenerateTargetPolicy.CONSTANT


@pytest.mark.parametrize("seed", list(range(10)))
def test_multioutput_evaluate_random_split_with_rare_target(tmp_path: Path, seed: int):
    """Parametrized seed sweep on a 1k-user × 13-target frame with one rare target.

    Holds out 100 users at random under each seed. The rare target has 3
    positives globally — depending on the seed, all 3 can land in valid,
    leaving the train slice single-class. Uses
    ``DegenerateTargetPolicy.CONSTANT`` so those seeds proceed via the
    constant-predictor fallback rather than raising at fit time, which lets
    the sweep verify ``evaluate()`` runs end-to-end across the full random
    split distribution.
    """
    rng = np.random.default_rng(seed)
    n_users = 1000
    user_ids = [f"user_{i}" for i in range(n_users)]
    df = pd.DataFrame({"USER_ID": user_ids})
    df["age"] = rng.integers(18, 80, size=n_users).astype(float)
    df["income"] = rng.integers(20_000, 200_000, size=n_users).astype(float)
    df["tenure_days"] = rng.integers(1, 3650, size=n_users).astype(float)

    label_cols = [f"ITEM_label_{i}" for i in range(12)] + ["ITEM_label_rare"]
    for i in range(12):
        df[f"ITEM_label_{i}"] = rng.binomial(1, 0.4 + 0.02 * i, size=n_users).astype(float)
    rare_positives = rng.choice(n_users, size=3, replace=False)
    df["ITEM_label_rare"] = 0.0
    df.loc[rare_positives, "ITEM_label_rare"] = 1.0

    perm = rng.permutation(n_users)
    valid_idx = perm[:100]
    train_idx = perm[100:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    # CONSTANT policy keeps fit-time green even on seeds where the rare
    # target ends up single-class in train.

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    train_df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "age", "type": "float"},
                {"name": "income", "type": "float"},
                {"name": "tenure_days", "type": "float"},
                *[{"name": c, "type": "float"} for c in label_cols],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 10, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    n_valid = len(valid_df)
    item_names_arr = np.array(label_cols, dtype=object)
    logged_items = np.tile(item_names_arr, (n_valid, 1))
    logged_rewards = valid_df[label_cols].to_numpy(dtype=float)

    metric = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        score_items_kwargs={"interactions": valid_df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
    )
    assert isinstance(metric, float)
    # ROC-AUC of degenerate single-class targets is NaN; the macro-mean
    # ignores NaNs but can still be finite if at least one label has signal.
    assert np.isnan(metric) or 0.0 <= metric <= 1.0


def test_multioutput_regressor_mode_end_to_end(tmp_path: Path):
    """`MultioutputScorer` accepts a ``BaseRegressor`` for continuous targets.

    Wide-format frame with three continuous ``ITEM_<name>`` targets — the
    regressor path passes continuous values through unchanged, returns ``(n_users, n_targets)``
    directly from ``score_items``, and exposes ``predict_targets`` for point
    estimates. ``predict_classes`` is unavailable in this mode.
    """
    rng = np.random.default_rng(0)
    n_users = 300
    df = pd.DataFrame({"USER_ID": [f"user_{i}" for i in range(n_users)]})
    df["age"] = rng.integers(18, 80, size=n_users).astype(float)
    df["income"] = rng.integers(20_000, 200_000, size=n_users).astype(float)
    target_cols = ["ITEM_revenue", "ITEM_minutes_engaged", "ITEM_clicks"]
    # Continuous targets correlated with features so the regressor has signal.
    df["ITEM_revenue"] = (df["income"] / 1000.0 + rng.normal(0, 5, n_users)).astype(float)
    df["ITEM_minutes_engaged"] = (df["age"] * 0.5 + rng.normal(0, 2, n_users)).astype(float)
    df["ITEM_clicks"] = (df["age"] + df["income"] / 5000.0 + rng.normal(0, 3, n_users)).astype(float)

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "age", "type": "float"},
                {"name": "income", "type": "float"},
                *[{"name": c, "type": "float"} for c in target_cols],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )

    train_ds = InteractionMultiOutputDataset(
        data_location=str(train_path),
        client_schema_path=str(schema_path),
    )
    estimator = MultiOutputRegressorEstimator(
        base_estimator=XGBRegressor,
        params={"n_estimators": 20, "max_depth": 3, "objective": "reg:squarederror"},
    )
    scorer = MultioutputScorer(estimator)
    assert scorer.is_classifier is False
    recommender = RankingRecommender(scorer)
    recommender.train(interactions_ds=train_ds)

    # score_items returns one column per target with predicted values.
    scores = recommender.score_items(interactions=df)
    assert scores.shape == (n_users, len(target_cols))
    assert list(scores.columns) == target_cols
    assert np.isfinite(scores.to_numpy()).all()

    # predict_targets is the regressor-mode equivalent of predict_classes.
    point_estimates = recommender.scorer.predict_targets(interactions=df)
    assert point_estimates.shape == scores.shape

    # predict_classes is unavailable in regressor mode.
    with pytest.raises(NotImplementedError, match="predict_classes is only defined for classifier"):
        recommender.scorer.predict_classes(interactions=df)

    # Regression metrics work end-to-end on the regressor mode.
    n = len(df)
    logged_items = np.tile(np.array(target_cols, dtype=object), (n, 1))
    logged_rewards = df[target_cols].to_numpy(dtype=float)

    macro_rmse = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.RMSE,
        score_items_kwargs={"interactions": df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
    )
    assert isinstance(macro_rmse, float)
    assert macro_rmse >= 0.0
    assert np.isfinite(macro_rmse)

    per_target_mae = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.MAE,
        score_items_kwargs={"interactions": df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        per_label=True,
    )
    assert isinstance(per_target_mae, dict)
    assert set(per_target_mae.keys()) == set(target_cols)
    for target, value in per_target_mae.items():
        assert value >= 0.0, f"{target}: {value}"

    # Classification metrics on the regressor-mode scorer raise.
    with pytest.raises(ValueError, match="classifier estimator"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": df},
            eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        )


def test_multioutput_rejects_multiclass_target_at_fit(tmp_path: Path):
    """Multi-class targets (3+ classes per ITEM_<name>) are rejected at fit time.

    Cross-target ranking and per-label classification metrics both assume
    binary targets. The error message names every offending column with
    its observed classes and points at all three migration paths
    (MulticlassScorer, one-hot encoding, mixed-type scorer).
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["age"] = rng.integers(18, 80, size=n).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)  # binary, OK
    df["ITEM_label_b"] = rng.integers(0, 4, size=n).astype(float)  # 4 classes, REJECT
    df["ITEM_label_c"] = rng.integers(0, 3, size=n).astype(float)  # 3 classes, REJECT

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "age", "type": "float"},
                {"name": "ITEM_label_a", "type": "float"},
                {"name": "ITEM_label_b", "type": "float"},
                {"name": "ITEM_label_c", "type": "float"},
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 10, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator))
    with pytest.raises(ValueError) as exc_info:
        recommender.train(interactions_ds=train_ds)
    msg = str(exc_info.value)
    assert "ITEM_label_b" in msg
    assert "ITEM_label_c" in msg
    assert "ITEM_label_a" not in msg  # binary target should not be flagged
    # Migration paths surfaced in the error.
    assert "MulticlassScorer" in msg
    assert "one-hot" in msg
    assert "mixed-type multi-target" in msg


def test_multioutput_recommend_returns_top_k_label_names(wide_multioutput_recommender):
    """``recommend(top_k=K)`` returns ``(n_users, K)`` of label names.

    Honors ``top_k`` (no longer ignored), ranks targets by per-target
    ``P(positive=1)`` per user, and returns the K highest-scoring label
    names from the catalogue — same shape contract as a long-format ranking
    recommender. Replaces the prior ``predict_classes`` shim.
    """
    recommender, valid_df = wide_multioutput_recommender
    top_k = 2
    recs = recommender.recommend(interactions=valid_df, top_k=top_k)
    assert recs.shape == (len(valid_df), top_k)
    # Every returned name must be from the catalogue.
    catalogue = set(recommender.scorer.item_names)
    assert set(recs.ravel().tolist()).issubset(catalogue)


def test_multioutput_rejects_non_classifier_non_regressor():
    """Constructor enforces the BaseClassifier-or-BaseRegressor contract."""

    class DummyEstimator:
        pass

    with pytest.raises(TypeError, match="BaseClassifier or BaseRegressor"):
        MultioutputScorer(DummyEstimator())


def test_multioutput_score_items_per_class_proba_contract(wide_multioutput_recommender):
    """Documents `MultioutputScorer.score_items` public shape contract.

    `score_items` in classifier mode returns a
    `(n_users, 2 * n_targets)` DataFrame — one column per (target, class)
    pair, uniformly named `ITEM_label_X_0` and `ITEM_label_X_1` regardless
    of input dtype. This is the contract for direct callers; the
    evaluator path no longer relies on this shape (the override in
    ``RankingRecommender.evaluate`` reads positive-class columns by name
    rather than treating ``score_items.shape[1]`` as ``n_items``).
    """
    recommender, valid_df = wide_multioutput_recommender
    scores = recommender.score_items(interactions=valid_df).to_numpy()

    n_targets = len(LABEL_COLS)
    n_classes_per_target = 2  # binary
    assert scores.shape == (len(valid_df), n_targets * n_classes_per_target)


# -------------------- C1: column correspondence -------------------------


def test_score_items_per_target_matches_score_items_positive_column(wide_multioutput_recommender):
    """``score_items_per_target[label]`` equals the positive-class column of ``score_items``.

    The two methods must agree on which column carries ``P(label = positive_class)``;
    otherwise downstream evaluation reads stale numbers. Pinning this contract
    catches drift between ``_create_proba_df``'s column-name format and
    ``positive_proba_column_name``'s lookup.
    """
    recommender, valid_df = wide_multioutput_recommender
    proba = recommender.score_items(interactions=valid_df)
    per_target = recommender.scorer.score_items_per_target(interactions=valid_df)

    assert list(per_target.columns) == LABEL_COLS
    assert per_target.shape == (len(valid_df), len(LABEL_COLS))
    for label in LABEL_COLS:
        positive_col = recommender.scorer.positive_proba_column_name(label)
        np.testing.assert_array_equal(
            per_target[label].to_numpy(),
            proba[positive_col].to_numpy(),
            err_msg=f"score_items_per_target[{label}] != score_items[{positive_col}]",
        )


# -------------------- C2: _score_items_np parity ------------------------


def test_score_items_np_returns_per_target_shape(wide_multioutput_recommender):
    """``MultioutputScorer._score_items_np`` returns ``(N, n_targets)`` not ``(N, sum_classes)``.

    Documents the deliberate departure from the BaseScorer default (which
    returns ``score_items().to_numpy()``). The override is what makes
    ``BaseRecommender.recommend`` work uniformly for the multioutput case.
    """
    recommender, valid_df = wide_multioutput_recommender
    arr = recommender.scorer._score_items_np(interactions=valid_df, users=None)
    assert arr.shape == (len(valid_df), len(LABEL_COLS))
    np.testing.assert_array_equal(
        arr,
        recommender.scorer.score_items_per_target(interactions=valid_df).to_numpy(),
    )


# -------------------- C3: recommend returns highest-scoring labels ------


def test_recommend_returns_actual_top_k_by_positive_proba(wide_multioutput_recommender):
    """``recommend(top_k=K)`` ranks labels by per-target ``P(positive=1)`` and
    returns the K highest. Asserts that, per user, the labels in the
    output dominate every label *not* in the output by predicted score.
    """
    recommender, valid_df = wide_multioutput_recommender
    top_k = 2
    recs = recommender.recommend(interactions=valid_df, top_k=top_k)
    per_target = recommender.scorer.score_items_per_target(interactions=valid_df)
    for u in range(len(valid_df)):
        chosen = list(recs[u])
        not_chosen = [c for c in LABEL_COLS if c not in chosen]
        # Every chosen label's score must be >= every non-chosen label's score for the same user.
        chosen_scores = per_target.iloc[u][chosen].to_numpy()
        not_chosen_scores = per_target.iloc[u][not_chosen].to_numpy()
        if len(not_chosen_scores) > 0:
            assert chosen_scores.min() >= not_chosen_scores.max(), (
                f"user {u}: chosen={chosen} (min={chosen_scores.min():.4f}) but a "
                f"non-chosen label scored higher (max={not_chosen_scores.max():.4f})"
            )


# -------------------- C4: oracle NDCG on a known answer -----------------


def test_evaluate_ndcg_oracle_value_on_perfect_ranking(tmp_path: Path):
    """Oracle test: when the model ranks targets perfectly per user,
    ``evaluate(NDCG_AT_K)`` returns the analytically known value.

    Setup: 3 binary targets, 4 users. Train slice has both classes for
    every target. We choose user features so the trained XGB classifier
    correctly ranks the targets. Then NDCG@3 against the user's true
    positives is 1.0 for users whose positives match the ordering the
    model learns. We don't pin a specific value (model-dependent), but
    we DO assert the computed NDCG against an independent sklearn-based
    reference computed on the same per-target positive-class scores —
    no two implementations should disagree.
    """
    from sklearn.metrics import ndcg_score

    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["age"] = rng.integers(18, 80, size=n).astype(float)
    df["income"] = rng.integers(20_000, 200_000, size=n).astype(float)
    targets = ["ITEM_label_a", "ITEM_label_b", "ITEM_label_c"]
    for i, c in enumerate(targets):
        df[c] = rng.binomial(1, 0.3 + 0.2 * i, size=n).astype(float)

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "age", "type": "float"},
                {"name": "income", "type": "float"},
                *[{"name": c, "type": "float"} for c in targets],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 20, "max_depth": 3, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator))
    recommender.train(interactions_ds=train_ds)

    logged_items = np.tile(np.array(targets, dtype=object), (n, 1))
    logged_rewards = df[targets].to_numpy(dtype=float)

    skrec_ndcg = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.NDCG_AT_K,
        eval_top_k=3,
        score_items_kwargs={"interactions": df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
    )

    # Reference: same per-target positive-class scores, scored by sklearn's NDCG.
    per_target = recommender.scorer.score_items_per_target(interactions=df).to_numpy()
    sklearn_ndcg = ndcg_score(logged_rewards, per_target, k=3)

    assert abs(skrec_ndcg - sklearn_ndcg) < 1e-6, (
        f"scikit-rec NDCG@3 ({skrec_ndcg:.6f}) disagrees with sklearn NDCG@3 "
        f"({sklearn_ndcg:.6f}) on the same scores and ground truth."
    )


# -------------------- B3: non-alphabetical subset ordering --------------


def test_recommend_with_non_alphabetical_item_subset(wide_multioutput_recommender):
    """When an ``item_subset`` is set whose canonical order (alphabetical)
    differs from the catalogue's insertion order, ``recommend()`` must still
    return correct label-name → score mappings.

    ``BaseScorer._process_item_subset`` sorts the subset alphabetically.
    The score-output builders must iterate in that sorted order so the
    per-row arg-sort indices map to the right label names. This test
    catches the bug where the score columns followed insertion order
    while ``_get_item_names`` returned the sorted subset.
    """
    recommender, valid_df = wide_multioutput_recommender
    # Pick a subset whose insertion order in the catalogue differs from
    # alphabetical order. LABEL_COLS is intentionally not alphabetical.
    subset = ["ITEM_label_workflow_automation", "ITEM_label_advance_reporting"]
    recommender.scorer.set_item_subset(subset)
    try:
        recs = recommender.recommend(interactions=valid_df, top_k=2)
        per_target = recommender.scorer.score_items_per_target(interactions=valid_df)
        # Output column names of score_items_per_target reflect the subset
        # in the canonical (sorted) order — same order the recommender's
        # _get_item_names() emits.
        assert sorted(subset) == list(per_target.columns)
        # Per-user, the recs must include only subset members and respect
        # ranking by score within subset.
        for u in range(len(valid_df)):
            chosen = list(recs[u])
            assert set(chosen).issubset(set(subset))
            scores = per_target.iloc[u][chosen].to_numpy()
            # With top_k = len(subset), chosen must equal sorted-by-score:
            assert list(scores) == sorted(scores, reverse=True)
    finally:
        recommender.scorer.clear_item_subset()


# -------------------- B4 / B5 / C7: evaluate-boundary validation --------


def test_evaluate_rejects_row_count_mismatch(wide_multioutput_recommender):
    """interactions.shape[0] != logged_rewards.shape[0] raises explicitly."""
    recommender, valid_df = wide_multioutput_recommender
    n = len(valid_df)
    logged_items = np.tile(np.array(LABEL_COLS, dtype=object), (n - 1, 1))
    logged_rewards = valid_df[LABEL_COLS].head(n - 1).to_numpy(dtype=float)
    with pytest.raises(ValueError, match="rows but logged_rewards has"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": valid_df},  # n rows
            eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},  # n-1 rows
        )


def test_evaluate_rejects_duplicate_target_names_in_logged_items(wide_multioutput_recommender):
    """Duplicate names in logged_items[0] raise — silently overwriting the
    name → column index would miscolumn the rewards."""
    recommender, valid_df = wide_multioutput_recommender
    n = len(valid_df)
    bad_names = list(LABEL_COLS)
    bad_names[0] = bad_names[1]  # duplicate one, drop a different one — same length
    logged_items = np.tile(np.array(bad_names, dtype=object), (n, 1))
    logged_rewards = valid_df[LABEL_COLS].to_numpy(dtype=float)
    with pytest.raises(ValueError, match="duplicate target name"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": valid_df},
            eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        )


def test_evaluate_rejects_users_in_score_items_kwargs(wide_multioutput_recommender):
    """Passing a `users` DataFrame surfaces a precise error at the eval boundary
    rather than an opaque scorer-internal one."""
    recommender, valid_df = wide_multioutput_recommender
    n = len(valid_df)
    logged_items = np.tile(np.array(LABEL_COLS, dtype=object), (n, 1))
    logged_rewards = valid_df[LABEL_COLS].to_numpy(dtype=float)
    with pytest.raises(ValueError, match="doesn't accept a `users`"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": valid_df, "users": valid_df},
            eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        )


# -------------------- B1 / B2: degenerate-target metric correctness -----


def test_evaluate_excludes_degenerate_targets_from_macro_classification(tmp_path: Path):
    """Macro ROC_AUC / PR_AUC ignore degenerate targets (per-label NaN +
    NaN-skipping mean), so single-class columns under CONSTANT policy
    don't contaminate the aggregate.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.integers(0, 100, size=n).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n).astype(float)
    df["ITEM_label_dead"] = 0.0  # globally degenerate — will be tracked under CONSTANT

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                {"name": "ITEM_label_a", "type": "float"},
                {"name": "ITEM_label_b", "type": "float"},
                {"name": "ITEM_label_dead", "type": "float"},
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 10, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    targets = ["ITEM_label_a", "ITEM_label_b", "ITEM_label_dead"]
    logged_items = np.tile(np.array(targets, dtype=object), (n, 1))
    logged_rewards = df[targets].to_numpy(dtype=float)

    per_label = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        score_items_kwargs={"interactions": df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        per_label=True,
    )
    assert np.isnan(per_label["ITEM_label_dead"]), (
        f"degenerate target should produce NaN per-label score, got {per_label['ITEM_label_dead']}"
    )
    assert np.isfinite(per_label["ITEM_label_a"])
    assert np.isfinite(per_label["ITEM_label_b"])

    # Macro-mean must equal the mean of the two informative labels — the
    # NaN on the degenerate target must be excluded, not coerced to 0.0.
    macro = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        score_items_kwargs={"interactions": df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
    )
    expected = float(np.mean([per_label["ITEM_label_a"], per_label["ITEM_label_b"]]))
    assert abs(macro - expected) < 1e-12


def test_evaluate_excludes_degenerate_targets_from_ranking_metric(tmp_path: Path):
    """Cross-target ranking metrics drop degenerate targets so constant
    predictions don't tie at the top of every per-user ordering.

    Sanity check: NDCG@K computed by skrec on a frame with one degenerate
    target should equal NDCG@K computed by sklearn on the *same frame
    minus the degenerate target column*.
    """
    from sklearn.metrics import ndcg_score

    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.integers(0, 100, size=n).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n).astype(float)
    df["ITEM_label_c"] = rng.binomial(1, 0.6, size=n).astype(float)
    df["ITEM_label_dead"] = 0.0

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                *[
                    {"name": c, "type": "float"}
                    for c in ["ITEM_label_a", "ITEM_label_b", "ITEM_label_c", "ITEM_label_dead"]
                ],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 10, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    informative = ["ITEM_label_a", "ITEM_label_b", "ITEM_label_c"]
    all_targets = informative + ["ITEM_label_dead"]
    logged_items = np.tile(np.array(all_targets, dtype=object), (n, 1))
    logged_rewards = df[all_targets].to_numpy(dtype=float)

    skrec_ndcg = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.NDCG_AT_K,
        eval_top_k=2,
        score_items_kwargs={"interactions": df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
    )
    # Reference: sklearn NDCG@2 over informative-only columns.
    per_target = recommender.scorer.score_items_per_target(interactions=df)
    sklearn_ndcg = ndcg_score(
        df[informative].to_numpy(dtype=float),
        per_target[informative].to_numpy(),
        k=2,
    )
    assert abs(skrec_ndcg - sklearn_ndcg) < 1e-6, (
        f"skrec NDCG@2 ({skrec_ndcg:.6f}) disagrees with sklearn over informative "
        f"targets ({sklearn_ndcg:.6f}) — degenerate target was probably not excluded"
    )


# -------------------- C9: column-name format stability ------------------


def test_recommend_filters_degenerate_targets_under_constant_policy(tmp_path: Path):
    """``recommend()`` must drop degenerate targets — constant predictions
    would tie at the top of every per-user ranking and dominate ``top_k``.

    Mirrors the eval-path filter. Uses ``DegenerateTargetPolicy.CONSTANT``
    so the scorer trains with a constant predictor for the dead column;
    asserts the recommended labels never include that column even when
    asking for ``top_k == n_targets`` (where unfiltered code would
    necessarily include it).
    """
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.integers(0, 100, size=n).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n).astype(float)
    df["ITEM_label_c"] = rng.binomial(1, 0.3, size=n).astype(float)
    df["ITEM_label_dead"] = 0.0

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                *[
                    {"name": c, "type": "float"}
                    for c in ["ITEM_label_a", "ITEM_label_b", "ITEM_label_c", "ITEM_label_dead"]
                ],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 10, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    # Ask for top_k equal to the total target count. Without filtering,
    # the dead column's constant 1.0 score would force it into every
    # user's top-K (it ties with whatever else hits 1.0). With filtering,
    # the dead column is excluded entirely.
    recs = recommender.recommend(interactions=df, top_k=3)
    flat = recs.ravel().tolist()
    assert "ITEM_label_dead" not in flat


def test_recommend_all_degenerate_raises_via_subset(tmp_path: Path):
    """``recommend()`` raises if the active item subset selects only degenerate
    targets — there's nothing rankable to return.

    Construction: train with one binary + one degenerate target under
    CONSTANT policy, then narrow the active subset to the degenerate one.
    """
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.integers(0, 100, size=n).astype(float)
    # ≥2 non-degenerate + ≥1 degenerate so train satisfies _validate_interactions's
    # "at least 2 ITEM columns" check after the CONSTANT policy drops the dead column.
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n).astype(float)
    df["ITEM_label_dead"] = 0.0

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                {"name": "ITEM_label_a", "type": "float"},
                {"name": "ITEM_label_b", "type": "float"},
                {"name": "ITEM_label_dead", "type": "float"},
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    recommender.scorer.set_item_subset(["ITEM_label_dead"])
    try:
        with pytest.raises(ValueError, match="All targets are degenerate"):
            recommender.recommend(interactions=df, top_k=1)
    finally:
        recommender.scorer.clear_item_subset()


def test_evaluate_validation_slice_single_class_emits_nan(tmp_path: Path):
    """A target binary in train but single-class in valid emits NaN per-label
    and is excluded from macro-mean. Without this, ROC_AUC/PR_AUC return
    ``0.0`` and silently drag the macro mean down.
    """
    rng = np.random.default_rng(7)
    n_train = 300
    train_df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n_train)]})
    train_df["x"] = rng.integers(0, 100, size=n_train).astype(float)
    train_df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n_train).astype(float)
    train_df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n_train).astype(float)

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    train_df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                {"name": "ITEM_label_a", "type": "float"},
                {"name": "ITEM_label_b", "type": "float"},
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator))
    recommender.train(interactions_ds=train_ds)

    # Construct a validation frame where ITEM_label_a is single-class
    # (all zeros) but ITEM_label_b has both classes.
    n_val = 30
    valid_df = pd.DataFrame(
        {
            "USER_ID": [f"v_{i}" for i in range(n_val)],
            "x": rng.integers(0, 100, size=n_val).astype(float),
            "ITEM_label_a": np.zeros(n_val, dtype=float),
            "ITEM_label_b": rng.binomial(1, 0.4, size=n_val).astype(float),
        }
    )
    targets = ["ITEM_label_a", "ITEM_label_b"]
    logged_items = np.tile(np.array(targets, dtype=object), (n_val, 1))
    logged_rewards = valid_df[targets].to_numpy(dtype=float)

    per_label = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        score_items_kwargs={"interactions": valid_df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        per_label=True,
    )
    assert np.isnan(per_label["ITEM_label_a"]), (
        f"validation-slice single-class target should produce NaN, got {per_label['ITEM_label_a']}"
    )
    assert np.isfinite(per_label["ITEM_label_b"])

    macro = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        score_items_kwargs={"interactions": valid_df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
    )
    # Macro must equal the only finite per-label value, not be averaged with 0.
    assert abs(macro - per_label["ITEM_label_b"]) < 1e-12


def test_evaluate_all_targets_undefined_raises(tmp_path: Path):
    """When every target's per-label metric is NaN (degenerate or single-class
    val), the macro path raises with a precise message rather than silently
    returning NaN. Mirrors the all-degenerate ranking branch behaviour.
    """
    rng = np.random.default_rng(0)
    n_train = 200
    train_df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n_train)]})
    train_df["x"] = rng.integers(0, 100, size=n_train).astype(float)
    train_df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n_train).astype(float)
    train_df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n_train).astype(float)

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    train_df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                {"name": "ITEM_label_a", "type": "float"},
                {"name": "ITEM_label_b", "type": "float"},
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator))
    recommender.train(interactions_ds=train_ds)

    # Validation slice: every target is single-class.
    n_val = 20
    valid_df = pd.DataFrame(
        {
            "USER_ID": [f"v_{i}" for i in range(n_val)],
            "x": rng.integers(0, 100, size=n_val).astype(float),
            "ITEM_label_a": np.zeros(n_val, dtype=float),
            "ITEM_label_b": np.zeros(n_val, dtype=float),
        }
    )
    targets = ["ITEM_label_a", "ITEM_label_b"]
    logged_items = np.tile(np.array(targets, dtype=object), (n_val, 1))
    logged_rewards = valid_df[targets].to_numpy(dtype=float)

    with pytest.raises(ValueError, match="All targets produced undefined"):
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": valid_df},
            eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        )


def test_factory_builds_multioutput_regressor():
    """The orchestrator factory wires a regressor-mode MultioutputScorer when
    ``ml_task='regression'`` and ``scorer_type='multioutput'``. Without this
    branch the factory silently builds a single-target ``XGBRegressorEstimator``
    even when the scorer expects a multi-target shape.
    """
    from skrec.estimator.regression.multioutput_regressor import MultiOutputRegressorEstimator
    from skrec.orchestrator.factory import create_estimator

    estimator_config = {
        "estimator_type": "tabular",
        "ml_task": "regression",
        "xgboost": {"n_estimators": 5, "max_depth": 2, "objective": "reg:squarederror"},
    }
    estimator = create_estimator(estimator_config, scorer_type="multioutput")
    assert isinstance(estimator, MultiOutputRegressorEstimator), (
        f"factory built {type(estimator).__name__} for regressor + multioutput; "
        f"expected MultiOutputRegressorEstimator. Without this wiring the regressor "
        f"mode is unreachable from the factory / orchestrator / agent path."
    )


def test_factory_builds_multioutput_classifier():
    """Companion to the regressor test: classifier path was already wired."""
    from skrec.estimator.classification.multioutput_classifier import MultiOutputClassifierEstimator
    from skrec.orchestrator.factory import create_estimator

    estimator_config = {
        "estimator_type": "tabular",
        "ml_task": "classification",
        "xgboost": {"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    }
    estimator = create_estimator(estimator_config, scorer_type="multioutput")
    assert isinstance(estimator, MultiOutputClassifierEstimator)


def test_rmse_oracle_against_sklearn(tmp_path: Path):
    """RMSE numerical value matches ``sklearn.metrics.root_mean_squared_error``
    on the same per-target predictions and ground truth.
    """
    from sklearn.metrics import mean_squared_error

    recommender, df, target_cols = _build_regressor_recommender(tmp_path, seed=0)
    n = len(df)
    logged_items = np.tile(np.array(target_cols, dtype=object), (n, 1))
    logged_rewards = df[target_cols].to_numpy(dtype=float)

    skrec_per_target = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.RMSE,
        score_items_kwargs={"interactions": df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        per_label=True,
    )
    per_target_pred = recommender.scorer.score_items_per_target(interactions=df)
    for label in target_cols:
        ref = float(np.sqrt(mean_squared_error(df[label].to_numpy(), per_target_pred[label].to_numpy())))
        assert abs(skrec_per_target[label] - ref) < 1e-9, (
            f"RMSE for {label}: skrec {skrec_per_target[label]} vs sklearn {ref}"
        )


def test_mae_oracle_against_sklearn(tmp_path: Path):
    """MAE numerical value matches ``sklearn.metrics.mean_absolute_error``."""
    from sklearn.metrics import mean_absolute_error

    recommender, df, target_cols = _build_regressor_recommender(tmp_path, seed=1)
    n = len(df)
    logged_items = np.tile(np.array(target_cols, dtype=object), (n, 1))
    logged_rewards = df[target_cols].to_numpy(dtype=float)

    skrec_per_target = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.MAE,
        score_items_kwargs={"interactions": df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        per_label=True,
    )
    per_target_pred = recommender.scorer.score_items_per_target(interactions=df)
    for label in target_cols:
        ref = float(mean_absolute_error(df[label].to_numpy(), per_target_pred[label].to_numpy()))
        assert abs(skrec_per_target[label] - ref) < 1e-9, (
            f"MAE for {label}: skrec {skrec_per_target[label]} vs sklearn {ref}"
        )


def _build_regressor_recommender(tmp_path: Path, seed: int) -> "tuple[RankingRecommender, pd.DataFrame, list[str]]":
    """Helper: build a small regressor-mode recommender for oracle metric tests."""
    rng = np.random.default_rng(seed)
    n = 100
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["age"] = rng.integers(18, 80, size=n).astype(float)
    df["income"] = rng.integers(20_000, 200_000, size=n).astype(float)
    target_cols = ["ITEM_revenue", "ITEM_minutes"]
    df["ITEM_revenue"] = (df["income"] / 1000.0 + rng.normal(0, 5, n)).astype(float)
    df["ITEM_minutes"] = (df["age"] * 0.5 + rng.normal(0, 2, n)).astype(float)

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "age", "type": "float"},
                {"name": "income", "type": "float"},
                *[{"name": c, "type": "float"} for c in target_cols],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputRegressorEstimator(
        base_estimator=XGBRegressor,
        params={"n_estimators": 10, "max_depth": 2, "objective": "reg:squarederror"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator))
    recommender.train(interactions_ds=train_ds)
    return recommender, df, target_cols


def test_multioutput_classifier_estimator_preflight_rejects_single_class_y():
    """Wrapper standalone use rejects single-class y as well as multi-class.

    Single-class y is handled upstream by ``DegenerateTargetPolicy`` when going
    through ``MultioutputScorer``; standalone callers (no scorer) get a
    consistent error rather than letting sklearn's behaviour vary by underlying
    classifier.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 2)), columns=["f0", "f1"])
    y_single = pd.DataFrame(
        {
            "t_bin": rng.integers(0, 2, size=30),
            "t_dead": np.zeros(30),  # single-class — REJECT
        }
    )
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    with pytest.raises(ValueError) as exc_info:
        estimator.fit(X, y_single)
    msg = str(exc_info.value)
    assert "single-class" in msg
    assert "DegenerateTargetPolicy" in msg


def test_multioutput_classifier_estimator_preflight_rejects_multiclass_y():
    """``MultiOutputClassifierEstimator._fit_model`` raises on non-binary y *before*
    the underlying sklearn fit runs.

    Defense-in-depth: even if the wrapper is used outside ``MultioutputScorer``
    (where the binary check happens at ``_validate_targets``), the wrapper
    itself enforces the binary-target contract — so the expensive sklearn
    ``MultiOutputClassifier.fit`` never runs on data the contract doesn't accept.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["f0", "f1", "f2"])
    y_bad = pd.DataFrame(
        {
            "t_bin": rng.integers(0, 2, size=50),
            "t_multi": rng.integers(0, 4, size=50),  # 4 classes, REJECT
        }
    )

    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    with pytest.raises(ValueError) as exc_info:
        estimator.fit(X, y_bad)
    msg = str(exc_info.value)
    # The wrapper's value check (rejecting anything outside {0, 1}) now
    # fires before the cardinality check, so multi-class values are
    # caught here as "non-binary" rather than as "3+ classes".
    assert "binary numeric" in msg
    assert "0, 1" in msg or "{0, 1}" in msg


def test_factory_regressor_pipeline_e2e(tmp_path: Path):
    """End-to-end: orchestrator factory builds a regressor MultioutputScorer
    that actually trains and produces RMSE numbers. The type-only assertion
    in `test_factory_builds_multioutput_regressor` is necessary but not
    sufficient — this test exercises factory + train + evaluate as a unit.
    """
    from skrec.orchestrator.factory import create_estimator

    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["age"] = rng.integers(18, 80, size=n).astype(float)
    target_cols = ["ITEM_revenue", "ITEM_minutes"]
    df["ITEM_revenue"] = (df["age"] * 0.5 + rng.normal(0, 2, n)).astype(float)
    df["ITEM_minutes"] = (df["age"] + rng.normal(0, 3, n)).astype(float)

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "age", "type": "float"},
                *[{"name": c, "type": "float"} for c in target_cols],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))

    # Factory wires the estimator class.
    estimator = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "regression",
            "xgboost": {"n_estimators": 10, "max_depth": 2, "objective": "reg:squarederror"},
        },
        scorer_type="multioutput",
    )
    scorer = MultioutputScorer(estimator)
    recommender = RankingRecommender(scorer)
    recommender.train(interactions_ds=train_ds)

    logged_items = np.tile(np.array(target_cols, dtype=object), (n, 1))
    logged_rewards = df[target_cols].to_numpy(dtype=float)
    rmse = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.RMSE,
        score_items_kwargs={"interactions": df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
    )
    assert isinstance(rmse, float)
    assert rmse >= 0.0
    assert np.isfinite(rmse)


def test_recommend_degenerate_warning_logs_once_per_instance(tmp_path: Path, caplog):
    """The degenerate-recommend warning must fire exactly once per recommender
    instance, not once per call. On a high-QPS serving path, an unthrottled
    log would generate megabytes of identical lines.
    """
    import logging

    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.integers(0, 100, size=n).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n).astype(float)
    df["ITEM_label_dead"] = 0.0

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                *[{"name": c, "type": "float"} for c in ["ITEM_label_a", "ITEM_label_b", "ITEM_label_dead"]],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    with caplog.at_level(logging.WARNING, logger="skrec.recommender.ranking.ranking_recommender"):
        recommender.recommend(interactions=df, top_k=1)
        recommender.recommend(interactions=df, top_k=1)
        recommender.recommend(interactions=df, top_k=1)

    matching = [
        rec for rec in caplog.records if "Excluding" in rec.getMessage() and "degenerate target" in rec.getMessage()
    ]
    assert len(matching) == 1, (
        f"expected the degenerate warning to fire exactly once across 3 recommend() "
        f"calls; got {len(matching)} occurrences"
    )


def test_recommend_with_retriever_rejected_at_construction():
    """``RankingRecommender(MultioutputScorer(...), retriever=...)`` raises at
    construction — MultioutputScorer doesn't have a retrieval phase since
    targets are wide-frame columns, not catalogue entries."""
    from skrec.retriever.popularity_retriever import PopularityRetriever

    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    scorer = MultioutputScorer(estimator)
    retriever = PopularityRetriever(top_k=10)

    with pytest.raises(ValueError, match="does not support a retriever"):
        RankingRecommender(scorer, retriever=retriever)


def test_score_items_column_format_is_uniformly_int_zero_one(wide_multioutput_recommender):
    """Column names in classifier mode are uniformly ``ITEM_<name>_0`` /
    ``ITEM_<name>_1`` regardless of input dtype.

    The no-LabelEncoder rework pinned the format: the binary-only contract
    guarantees every target has exactly two classes valued ``0`` and ``1``
    (or ``0.0`` / ``1.0``), so there's no per-target class-name lookup.
    Replaces the older parametrized test that swept input dtypes —
    non-numeric inputs are now rejected at fit time, see
    :func:`test_multioutput_rejects_non_binary_numeric_targets`.
    """
    recommender, valid_df = wide_multioutput_recommender
    proba = recommender.score_items(interactions=valid_df)
    expected = []
    for label in LABEL_COLS:
        expected.extend([f"{label}_0", f"{label}_1"])
    assert list(proba.columns) == expected
    for label in LABEL_COLS:
        # positive_proba_column_name is now a pure-format helper.
        assert recommender.scorer.positive_proba_column_name(label) == f"{label}_1"


@pytest.mark.parametrize(
    "bad_values",
    [
        # Non-numeric (strings) — must be pre-encoded
        ["no", "yes", "no", "yes", "no", "yes"] * 10,
        # Non-{0,1} integers
        [-1, 1, -1, 1, -1, 1] * 10,
        # Multi-class
        [0, 1, 2, 0, 1, 2] * 10,
        # Floats outside {0.0, 1.0}
        [0.0, 0.5, 1.0, 0.5, 0.0, 1.0] * 10,
    ],
)
def test_multioutput_rejects_non_binary_numeric_targets(tmp_path: Path, bad_values):
    """Classifier mode rejects any target whose values aren't strictly
    ``{0, 1}`` (or ``{0.0, 1.0}``).

    The no-LabelEncoder rework requires callers to pre-encode non-numeric
    binary labels themselves. The error message points at the one-liner
    pandas conversion pattern.
    """
    n = len(bad_values)
    df = pd.DataFrame(
        {
            "USER_ID": [f"u_{i}" for i in range(n)],
            "x": np.linspace(0, 1, n),
            "ITEM_a": bad_values,
            # Ensure the second target is valid so the rejection is
            # specifically about the bad column.
            "ITEM_b": ([0, 1] * (n // 2 + 1))[:n],
        }
    )
    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    schema_type = "str" if isinstance(bad_values[0], str) else "float"
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                {"name": "ITEM_a", "type": schema_type},
                {"name": "ITEM_b", "type": "float"},
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator))
    with pytest.raises(ValueError) as exc_info:
        recommender.train(interactions_ds=train_ds)
    msg = str(exc_info.value)
    assert "ITEM_a" in msg
    assert "binary numeric" in msg
    # Migration hint surfaces in the error.
    assert "pre-encode" in msg or "astype(float)" in msg


# ----------------- CR2: binary contract symmetric at eval ------------------


def test_evaluate_rejects_non_binary_logged_rewards_classifier_mode(wide_multioutput_recommender):
    """Classifier-mode evaluation rejects non-binary ``logged_rewards``.

    The binary contract is symmetric: training y was validated as ``{0, 1}``,
    so held-out ground truth must be too. Without this enforcement,
    ``ROCAUCMetric`` silently binarizes via ``> 0.5`` and returns ``0.0`` for
    slices like ``[0.0, 0.5, 0.0]`` — poisoning the macro mean.
    """
    recommender, valid_df = wide_multioutput_recommender
    n = len(valid_df)
    logged_items = np.tile(np.array(LABEL_COLS, dtype=object), (n, 1))
    # Inject non-binary values into the first target's logged_rewards.
    logged_rewards = valid_df[LABEL_COLS].to_numpy(dtype=float).copy()
    logged_rewards[:5, 0] = 0.5  # neither 0 nor 1
    with pytest.raises(ValueError) as exc_info:
        recommender.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.ROC_AUC,
            score_items_kwargs={"interactions": valid_df},
            eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
        )
    msg = str(exc_info.value)
    assert "binary numeric" in msg
    assert LABEL_COLS[0] in msg


def test_evaluate_accepts_nan_logged_rewards_as_ignore_mask(wide_multioutput_recommender):
    """NaN in ``logged_rewards`` is allowed — used as an ignore mask /
    not-observed marker, not flagged by the binary check."""
    recommender, valid_df = wide_multioutput_recommender
    n = len(valid_df)
    logged_items = np.tile(np.array(LABEL_COLS, dtype=object), (n, 1))
    logged_rewards = valid_df[LABEL_COLS].to_numpy(dtype=float).copy()
    logged_rewards[:3, 0] = np.nan
    # Should NOT raise.
    metric = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.ROC_AUC,
        score_items_kwargs={"interactions": valid_df},
        eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
    )
    assert isinstance(metric, float)


# ----------------- LG2: regressor wrapper standalone preflight -----------


def test_multioutput_regressor_estimator_preflight_rejects_1d_y():
    """``MultiOutputRegressorEstimator._fit_model`` rejects 1-D y.

    The wrapper expects ``(n_samples, n_targets)`` so the underlying
    ``MultiOutputRegressor`` fits one estimator per column. A 1-D y means
    the caller wanted a single-target regression and should use
    ``XGBRegressorEstimator`` (or any single-target ``BaseRegressor``)
    directly instead.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["f0", "f1", "f2"])
    y_1d = pd.Series(rng.normal(size=40))
    estimator = MultiOutputRegressorEstimator(
        base_estimator=XGBRegressor,
        params={"n_estimators": 5, "max_depth": 2, "objective": "reg:squarederror"},
    )
    with pytest.raises(ValueError) as exc_info:
        estimator.fit(X, y_1d)
    msg = str(exc_info.value)
    assert "2-D y" in msg
    assert "XGBRegressorEstimator" in msg


# ----------------- LG1: tuned variants fit-then-predict ------------------


def test_tuned_multioutput_classifier_fit_then_predict(tmp_path: Path):
    """``TunedMultiOutputClassifierEstimator`` fits via HPO and predicts.

    Construction-only coverage missed CR-class issues in earlier rounds —
    this test exercises the full HPO + fit + predict path so future
    refactors can't break the tuned variant silently.
    """
    from skrec.estimator.classification.multioutput_classifier import (
        TunedMultiOutputClassifierEstimator,
    )
    from skrec.estimator.datatypes import HPOType

    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = pd.DataFrame(
        {
            "t0": rng.integers(0, 2, size=n),
            "t1": rng.integers(0, 2, size=n),
        }
    )

    estimator = TunedMultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        hpo_method=HPOType.RANDOMIZED_SEARCH_CV,
        param_space={"estimator__n_estimators": [5, 10]},
        optimizer_params={"n_iter": 1, "cv": 2},
    )
    estimator.fit(X, y)
    proba = estimator.predict_proba(X)
    # MultiOutputClassifier emits a list-of-arrays per target.
    assert len(proba) == 2
    for arr in proba:
        assert arr.shape == (n, 2)


def test_partial_degeneracy_one_alive_two_dead_under_constant(tmp_path: Path):
    """``CONSTANT`` policy with 1 alive + N dead targets fits successfully.

    Catches the misleading "<2 ITEM columns" error that fired when the
    second ``_validate_interactions`` pass saw only the surviving column
    after CONSTANT dropped the dead ones. The structural ≥2 floor applies
    only to the user's input frame; once dead columns are dropped, sklearn's
    ``MultiOutputClassifier`` is happy with ≥1 fittable target.
    """
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.normal(size=n).astype(float)
    df["ITEM_label_alive"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_dead1"] = 0.0
    df["ITEM_label_dead2"] = 1.0

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                *[{"name": c, "type": "float"} for c in ["ITEM_label_alive", "ITEM_label_dead1", "ITEM_label_dead2"]],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    # Should NOT raise: structural ≥2 holds against the input (3 ITEM cols),
    # post-drop floor of ≥1 holds (1 surviving column).
    recommender.train(interactions_ds=train_ds)

    # Sanity-check the resulting model: the alive target produces a real
    # distribution, the dead targets produce constants.
    proba = recommender.score_items(interactions=df)
    assert "ITEM_label_alive_0" in proba.columns
    assert "ITEM_label_alive_1" in proba.columns
    alive_sum = proba["ITEM_label_alive_0"].to_numpy() + proba["ITEM_label_alive_1"].to_numpy()
    assert np.allclose(alive_sum, 1.0)
    # Dead targets are constants.
    assert np.allclose(proba["ITEM_label_dead1_0"].to_numpy(), 1.0)
    assert np.allclose(proba["ITEM_label_dead2_1"].to_numpy(), 1.0)


def test_caller_dataframe_not_mutated_on_classifier_constant(tmp_path: Path):
    """``train()`` under CONSTANT policy doesn't mutate the caller's
    ``interactions_df`` even when degenerate columns are dropped internally.

    Previously the in-place ``df.drop`` inside ``_validate_targets`` shrank
    the caller's frame silently. The ``process_datasets`` entry copy
    isolates the mutation.
    """
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.integers(0, 100, size=n).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n).astype(float)
    df["ITEM_label_dead"] = 0.0
    columns_before = list(df.columns)

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                *[{"name": c, "type": "float"} for c in ["ITEM_label_a", "ITEM_label_b", "ITEM_label_dead"]],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    # The fetched DataFrame held by the dataset object should still have
    # all original columns — _validate_targets didn't propagate its drop
    # back to the caller's frame.
    df_after = train_ds.fetch_data()
    assert list(df_after.columns) == columns_before


def test_warn_degenerate_resets_on_retrain(tmp_path: Path, caplog):
    """``train()`` resets ``_warned_degenerate_recommend`` so a second
    training (with a potentially different degenerate manifest) gets a
    fresh warning when recommend() next runs."""
    import logging

    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.integers(0, 100, size=n).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n).astype(float)
    df["ITEM_label_dead"] = 0.0

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                *[{"name": c, "type": "float"} for c in ["ITEM_label_a", "ITEM_label_b", "ITEM_label_dead"]],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    # First recommend — warn fires.
    with caplog.at_level(logging.WARNING, logger="skrec.recommender.ranking.ranking_recommender"):
        recommender.recommend(interactions=df, top_k=1)
    first_warns = [r for r in caplog.records if "degenerate target" in r.getMessage()]
    assert len(first_warns) == 1

    # Subsequent recommend on same instance — silenced.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="skrec.recommender.ranking.ranking_recommender"):
        recommender.recommend(interactions=df, top_k=1)
    silenced_warns = [r for r in caplog.records if "degenerate target" in r.getMessage()]
    assert len(silenced_warns) == 0

    # Re-train — flag should reset, next recommend warns again.
    recommender.train(interactions_ds=train_ds)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="skrec.recommender.ranking.ranking_recommender"):
        recommender.recommend(interactions=df, top_k=1)
    fresh_warns = [r for r in caplog.records if "degenerate target" in r.getMessage()]
    assert len(fresh_warns) == 1, (
        f"after re-train, expected the recommend-degenerate warning to fire once again "
        f"(flag reset); got {len(fresh_warns)} occurrences"
    )


def test_multioutput_scorer_pickle_round_trip(tmp_path: Path):
    """A trained ``MultioutputScorer`` survives a pickle round-trip and
    produces identical predictions before and after.

    Catches future state additions that aren't picklable by default
    (e.g. lambda closures, file handles, DataFrame columns with extension
    dtypes). Mirrors the existing UniversalScorer pickle-safety guarantee.
    """
    import pickle

    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.normal(size=n).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n).astype(float)
    df["ITEM_label_dead"] = 0.0

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                *[{"name": c, "type": "float"} for c in ["ITEM_label_a", "ITEM_label_b", "ITEM_label_dead"]],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    scorer_before = MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT)
    recommender = RankingRecommender(scorer_before)
    recommender.train(interactions_ds=train_ds)

    # Build the score-time input the same way the recommender's
    # preprocessing would (drop USER_ID, drop the ITEM_* targets — they're
    # the labels, not features). Calls scorer.score_items directly so the
    # test focuses on scorer pickling, not recommender plumbing.
    feature_df = df.drop(columns=["USER_ID", "ITEM_label_a", "ITEM_label_b", "ITEM_label_dead"])
    feature_df["USER_ID"] = df["USER_ID"]
    proba_before = scorer_before.score_items(interactions=feature_df).to_numpy()

    # Round-trip the scorer.
    blob = pickle.dumps(scorer_before)
    scorer_after = pickle.loads(blob)
    # Public state preserved.
    assert scorer_after.is_classifier == scorer_before.is_classifier
    assert scorer_after.on_degenerate_target == scorer_before.on_degenerate_target
    assert dict(scorer_after.degenerate_targets) == dict(scorer_before.degenerate_targets)
    assert list(scorer_after.item_names) == list(scorer_before.item_names)
    assert list(scorer_after._fitted_target_order) == list(scorer_before._fitted_target_order)
    # Predictions identical post round-trip.
    proba_after = scorer_after.score_items(interactions=feature_df).to_numpy()
    np.testing.assert_array_equal(proba_after, proba_before)


def test_fitted_target_order_pins_predict_proba_contract(wide_multioutput_recommender, tmp_path: Path):
    """``_fitted_target_order`` snapshots the columns the underlying classifier
    was fit on, in fit-time order. ``_calculate_scores`` uses it for
    name-keyed lookup against ``predict_proba``, which makes the implicit
    "predict_proba returns one entry per fitted column in fit-order"
    contract explicit in named state. Pin both the no-degenerate case and
    the with-degenerate case.
    """
    # No-degenerate case: fitted order equals item_names.
    recommender, _ = wide_multioutput_recommender
    assert list(recommender.scorer._fitted_target_order) == list(recommender.scorer.item_names)
    assert len(recommender.scorer._fitted_target_order) == len(LABEL_COLS)

    # With-degenerate case: fitted order is item_names with degenerates removed.
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.normal(size=n).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_dead"] = 0.0
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n).astype(float)

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                *[{"name": c, "type": "float"} for c in ["ITEM_label_a", "ITEM_label_dead", "ITEM_label_b"]],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    rec_with_dead = RankingRecommender(
        MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT)
    )
    rec_with_dead.train(interactions_ds=train_ds)
    # Fitted order is item_names minus the degenerate one, preserving
    # original insertion order of the surviving columns.
    assert list(rec_with_dead.scorer._fitted_target_order) == [
        "ITEM_label_a",
        "ITEM_label_b",
    ]
    # All targets remain in item_names (the public catalogue is preserved).
    assert list(rec_with_dead.scorer.item_names) == [
        "ITEM_label_a",
        "ITEM_label_dead",
        "ITEM_label_b",
    ]


def test_calculate_scores_interleave_correctness_with_mixed_degenerate(tmp_path: Path):
    """``score_items()`` returns correct per-target probabilities under a
    mix of degenerate and non-degenerate targets — exercises the
    positional interleave in ``_calculate_scores``.

    For degenerate targets, the proba columns ``_0`` / ``_1`` must reflect
    the seen value (1.0 in the seen-class column, 0.0 in the other).
    For non-degenerate targets, the proba must come from the actual model
    output, not from the constant block. Without this test, an off-by-one
    in the interleave (e.g. swapping which list entry corresponds to
    which target name) would silently misroute predictions and no shape
    error would catch it.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.normal(size=n).astype(float)
    # Mix: dead_first (0), alive_a (binary), dead_mid (1), alive_b (binary).
    # Two degenerate at positions 0 and 2, two non-degenerate at 1 and 3 —
    # forces the interleave to alternate. After CONSTANT drops the
    # degenerates, the underlying classifier sees alive_a + alive_b in that
    # order; the interleave must reconstruct the full 4-target output with
    # constants at positions 0 and 2.
    df["ITEM_label_dead_first"] = 0.0
    df["ITEM_label_alive_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_dead_mid"] = 1.0
    df["ITEM_label_alive_b"] = rng.binomial(1, 0.4, size=n).astype(float)

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                *[
                    {"name": c, "type": "float"}
                    for c in [
                        "ITEM_label_dead_first",
                        "ITEM_label_alive_a",
                        "ITEM_label_dead_mid",
                        "ITEM_label_alive_b",
                    ]
                ],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    proba = recommender.score_items(interactions=df)
    # Degenerate "all-0" target: P(class=0)=1, P(class=1)=0 for every user.
    assert np.allclose(proba["ITEM_label_dead_first_0"].to_numpy(), 1.0)
    assert np.allclose(proba["ITEM_label_dead_first_1"].to_numpy(), 0.0)
    # Degenerate "all-1" target: P(class=0)=0, P(class=1)=1 for every user.
    assert np.allclose(proba["ITEM_label_dead_mid_0"].to_numpy(), 0.0)
    assert np.allclose(proba["ITEM_label_dead_mid_1"].to_numpy(), 1.0)
    # Non-degenerate targets: proba columns must sum to 1 row-wise (real
    # model output, not constant blocks from a misrouted interleave).
    for label in ["ITEM_label_alive_a", "ITEM_label_alive_b"]:
        alive_sum = proba[f"{label}_0"].to_numpy() + proba[f"{label}_1"].to_numpy()
        assert np.allclose(alive_sum, 1.0), label
        # Neither column is identically 0 or 1 (real distribution).
        assert proba[f"{label}_0"].nunique() > 1, label
        assert proba[f"{label}_1"].nunique() > 1, label
    # The two non-degenerate targets should produce DIFFERENT distributions
    # (different positive rates → different model behaviour). If the
    # interleave swapped them, alive_a's predictions would equal alive_b's.
    a_pred = proba["ITEM_label_alive_a_1"].to_numpy()
    b_pred = proba["ITEM_label_alive_b_1"].to_numpy()
    assert not np.allclose(a_pred, b_pred), (
        "ITEM_label_alive_a and ITEM_label_alive_b produced identical predictions — "
        "the interleave may be swapping target identities"
    )


def test_multioutput_classifier_estimator_preflight_rejects_non_numeric_y_standalone():
    """Wrapper standalone rejects string-valued y.

    Mirrors :func:`test_multioutput_rejects_non_binary_numeric_targets`'s
    coverage of the scorer-side path. Without the wrapper-side value
    check, a y of `[["yes","no"], ...]` would pass the cardinality check
    (length 2) and fit silently via sklearn's internal label encoding —
    even though `MultioutputScorer` rejects the same input. This test
    asserts the wrapper's standalone-use contract matches the scorer's.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 2)), columns=["f0", "f1"])
    y_strings = pd.DataFrame(
        {
            "t_str": (["yes", "no"] * 20),
            "t_bin": rng.integers(0, 2, size=40),
        }
    )
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    with pytest.raises(ValueError) as exc_info:
        estimator.fit(X, y_strings)
    msg = str(exc_info.value)
    assert "binary numeric" in msg
    assert "t_str" in msg
    assert "t_bin" not in msg
    # Migration hint mirrors the scorer's.
    assert "pre-encode" in msg.lower() or "astype(float)" in msg


def test_evaluate_rejects_active_item_subset(wide_multioutput_recommender):
    """``evaluate()`` raises if an item_subset is set on the scorer.

    The eval path iterates the full ``scorer.item_names`` catalogue and
    indexes ``logged_rewards`` against it. ``score_items`` returns
    subset-only columns when a subset is active, so the per-label loop
    would KeyError on subset misses and the ranking branch's
    ``score_matrix_full[:, keep_indices]`` slicing would IndexError.
    Reject explicitly rather than silently swallowing the subset.
    """
    recommender, valid_df = wide_multioutput_recommender
    n = len(valid_df)
    logged_items = np.tile(np.array(LABEL_COLS, dtype=object), (n, 1))
    logged_rewards = valid_df[LABEL_COLS].to_numpy(dtype=float)

    recommender.scorer.set_item_subset([LABEL_COLS[0], LABEL_COLS[1]])
    try:
        with pytest.raises(ValueError, match="does not support an active item_subset"):
            recommender.evaluate(
                eval_type=RecommenderEvaluatorType.SIMPLE,
                metric_type=RecommenderMetricType.ROC_AUC,
                score_items_kwargs={"interactions": valid_df},
                eval_kwargs={"logged_items": logged_items, "logged_rewards": logged_rewards},
            )
    finally:
        recommender.scorer.clear_item_subset()


def test_recommend_with_partial_degenerate_subset(tmp_path: Path):
    """``recommend()`` with an item_subset containing a mix of degenerate
    and non-degenerate targets returns only the non-degenerate labels.

    Catches the gap between ``test_recommend_all_degenerate_raises_via_subset``
    (which exercises the full-degenerate edge) and the no-subset path.
    """
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({"USER_ID": [f"u_{i}" for i in range(n)]})
    df["x"] = rng.integers(0, 100, size=n).astype(float)
    df["ITEM_label_a"] = rng.binomial(1, 0.5, size=n).astype(float)
    df["ITEM_label_b"] = rng.binomial(1, 0.4, size=n).astype(float)
    df["ITEM_label_c"] = rng.binomial(1, 0.3, size=n).astype(float)
    df["ITEM_label_dead"] = 0.0

    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                *[
                    {"name": c, "type": "float"}
                    for c in ["ITEM_label_a", "ITEM_label_b", "ITEM_label_c", "ITEM_label_dead"]
                ],
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    recommender.train(interactions_ds=train_ds)

    # Subset includes one degenerate (ITEM_label_dead) and two non-degenerate.
    subset = ["ITEM_label_dead", "ITEM_label_a", "ITEM_label_b"]
    recommender.scorer.set_item_subset(subset)
    try:
        recs = recommender.recommend(interactions=df, top_k=2)
    finally:
        recommender.scorer.clear_item_subset()
    flat = set(recs.ravel().tolist())
    # Dead column must not appear, and only non-degenerate subset members are valid.
    assert "ITEM_label_dead" not in flat
    assert flat.issubset({"ITEM_label_a", "ITEM_label_b"})


def test_multioutput_classifier_estimator_preflight_rejects_nan_y():
    """Wrapper standalone rejects NaN-in-y with a precise error.

    Previously NaN was silently stripped via ``pd.isna`` before the
    binary check, so a column like ``[0, 1, NaN]`` passed the wrapper but
    crashed sklearn's ``MultiOutputClassifier.fit`` internally.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 2)), columns=["f0", "f1"])
    y_with_nan = pd.DataFrame(
        {
            "t_clean": rng.integers(0, 2, size=40).astype(float),
            "t_dirty": [0.0, 1.0, np.nan, 0.0, 1.0] * 8,
        }
    )
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    with pytest.raises(ValueError) as exc_info:
        estimator.fit(X, y_with_nan)
    msg = str(exc_info.value)
    assert "null/NaN" in msg or "null value" in msg
    assert "t_dirty" in msg
    assert "t_clean" not in msg
    assert "drop" in msg or "backfill" in msg


def test_constant_policy_all_degenerate_at_fit_raises_precise(tmp_path: Path):
    """``DegenerateTargetPolicy.CONSTANT`` with EVERY target degenerate at fit
    raises a precise "all degenerate" error rather than the structural
    "must contain at least 2 ITEM columns" surface error.
    """
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "USER_ID": [f"u_{i}" for i in range(n)],
            "x": rng.integers(0, 100, size=n).astype(float),
            "ITEM_label_dead1": 0.0,
            "ITEM_label_dead2": 1.0,
        }
    )
    train_path = tmp_path / "train.csv"
    schema_path = tmp_path / "schema.yaml"
    df.to_csv(train_path, index=False)
    yaml.safe_dump(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "x", "type": "float"},
                {"name": "ITEM_label_dead1", "type": "float"},
                {"name": "ITEM_label_dead2", "type": "float"},
            ]
        },
        open(schema_path, "w"),
        sort_keys=False,
    )
    train_ds = InteractionMultiOutputDataset(data_location=str(train_path), client_schema_path=str(schema_path))
    estimator = MultiOutputClassifierEstimator(
        base_estimator=XGBClassifier,
        params={"n_estimators": 5, "max_depth": 2, "objective": "binary:logistic"},
    )
    recommender = RankingRecommender(MultioutputScorer(estimator, on_degenerate_target=DegenerateTargetPolicy.CONSTANT))
    with pytest.raises(ValueError) as exc_info:
        recommender.train(interactions_ds=train_ds)
    msg = str(exc_info.value)
    # Precise diagnosis, not the structural "<2 ITEM" surface error.
    assert "every target column is degenerate" in msg
    assert "ITEM_label_dead1" in msg
    assert "ITEM_label_dead2" in msg


def test_tuned_multioutput_regressor_fit_then_predict():
    """``TunedMultiOutputRegressorEstimator`` fits via HPO and predicts."""
    from skrec.estimator.datatypes import HPOType
    from skrec.estimator.regression.multioutput_regressor import (
        TunedMultiOutputRegressorEstimator,
    )

    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = pd.DataFrame(
        {
            "t0": rng.normal(size=n),
            "t1": rng.normal(size=n),
        }
    )

    estimator = TunedMultiOutputRegressorEstimator(
        base_estimator=XGBRegressor,
        hpo_method=HPOType.RANDOMIZED_SEARCH_CV,
        param_space={"estimator__n_estimators": [5, 10]},
        optimizer_params={"n_iter": 1, "cv": 2},
    )
    estimator.fit(X, y)
    pred = estimator.predict(X)
    assert pred.shape == (n, 2)
    assert np.isfinite(pred).all()
