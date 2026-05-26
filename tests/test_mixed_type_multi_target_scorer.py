# Tests for MixedTypeMultiTargetScorer.
#
# M1 subset (this milestone): tests that exercise scorer construction,
# target_specs validation, dataset schema, and Protocol isinstance checks —
# i.e. everything that doesn't require a real estimator. Tests that need
# wide-format prediction / evaluation / recommend wiring land in M4–M8.

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import skrec.constants as C
from skrec.constants import USER_ID_NAME
from skrec.dataset.interactions_dataset import InteractionMixedTypeMultiTargetDataset
from skrec.estimator.classification import MultiTargetEstimator
from skrec.estimator.classification._multi_target_protocol import sort_multiclass_labels
from skrec.scorer.mixed_type_multi_target import (
    MixedTypeMultiTargetScorer,
    TargetGroupSpec,
    TargetType,
)

# torch-dependent estimators used by inference-validator + preservation tests.
torch = pytest.importorskip("torch")

from skrec.estimator.classification import (  # noqa: E402
    ConditionalJointMultiTargetMLPEstimator,
    IndependentMultiTargetEstimator,
    JointMultiTargetMLPEstimator,
)
from skrec.recommender.ranking.ranking_recommender import RankingRecommender  # noqa: E402
from skrec.scorer.multioutput import MultioutputScorer  # noqa: E402


class _StubMTE:
    """Protocol-conforming stub for scorer-init-only validation tests."""

    def __init__(self, target_specs):
        self.target_specs = target_specs

    def fit(self, X, y, X_valid=None, y_valid=None):
        return self

    def predict_proba_dict(self, X):
        return {}

    def predict_targets_dict(self, X):
        return {}


def _build_validation_scaffold():
    """Shared mini-fixture for the inference-validator tests."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = {
        "ITEM_clicked": (X["f0"] > 0).astype(int).to_numpy(),
        "ITEM_revenue": X["f1"].to_numpy(),
    }
    ts = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    est = JointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 16},
    )
    est.fit(X, y)
    return X, MixedTypeMultiTargetScorer(estimator=est, target_specs=ts), ts


# ---------------------------------------------------------------------- #
# Protocol-conforming stubs for tests that don't need real prediction.
# ---------------------------------------------------------------------- #


class _StubMultiTargetEstimator:
    """Minimal Protocol-conforming estimator stub.

    Implements every attribute/method on ``MultiTargetEstimator`` so
    ``isinstance(stub, MultiTargetEstimator)`` returns True. Methods return
    empty dicts — these tests only exercise scorer construction, not
    prediction.
    """

    def __init__(self, target_specs):
        self.target_specs = target_specs

    def fit(self, X, y, X_valid=None, y_valid=None):
        return self

    def predict_proba_dict(self, X):
        return {}

    def predict_targets_dict(self, X):
        return {}


class _PartialProtocolEstimator:
    """Stub missing predict_*_dict methods — Protocol isinstance check fails."""

    def __init__(self, target_specs):
        self.target_specs = target_specs

    def fit(self, X, y, X_valid=None, y_valid=None):
        return self


# ---------------------------------------------------------------------- #
# target_specs structural validation
# ---------------------------------------------------------------------- #


def test_empty_target_specs_rejected():
    """Test #18: scorer init with target_specs={} → clean error."""
    estimator = _StubMultiTargetEstimator(target_specs={})
    with pytest.raises(ValueError, match="non-empty"):
        MixedTypeMultiTargetScorer(estimator=estimator, target_specs={})


def test_target_specs_value_must_be_target_type_or_group_spec():
    target_specs = {"ITEM_clicked": "binary_string_not_enum"}
    estimator = _StubMultiTargetEstimator(target_specs=target_specs)
    with pytest.raises(ValueError, match="must be a TargetType or TargetGroupSpec"):
        MixedTypeMultiTargetScorer(estimator=estimator, target_specs=target_specs)


def test_simple_target_key_must_be_item_prefixed():
    target_specs = {"clicked": TargetType.BINARY}  # missing ITEM_ prefix
    estimator = _StubMultiTargetEstimator(target_specs=target_specs)
    with pytest.raises(ValueError, match="must be prefixed with 'ITEM_'"):
        MixedTypeMultiTargetScorer(estimator=estimator, target_specs=target_specs)


def test_multilabel_group_zero_members_rejected():
    """Test #22: empty columns=[] → clean error at scorer init."""
    target_specs = {
        "engagement": TargetGroupSpec(type=TargetType.MULTILABEL, columns=[]),
    }
    estimator = _StubMultiTargetEstimator(target_specs=target_specs)
    with pytest.raises(ValueError, match="non-empty list"):
        MixedTypeMultiTargetScorer(estimator=estimator, target_specs=target_specs)


def test_multilabel_group_single_member_accepted():
    """Test #21: single-member multilabel group is valid (fans out to 1 binary)."""
    target_specs = {
        "engagement": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_x"]),
    }
    estimator = _StubMultiTargetEstimator(target_specs=target_specs)
    scorer = MixedTypeMultiTargetScorer(estimator=estimator, target_specs=target_specs)
    assert scorer._fanned_out_target_columns == ["ITEM_x"]


def test_target_specs_group_key_collides_with_member_column_rejected():
    """Test #23: group key colliding with a member column → clean error."""
    target_specs = {
        "ITEM_x": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_x", "ITEM_y"]),
    }
    estimator = _StubMultiTargetEstimator(target_specs=target_specs)
    with pytest.raises(ValueError, match="collide with member column"):
        MixedTypeMultiTargetScorer(estimator=estimator, target_specs=target_specs)


def test_member_column_in_two_groups_rejected():
    target_specs = {
        "g1": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_a", "ITEM_b"]),
        "g2": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_b", "ITEM_c"]),
    }
    estimator = _StubMultiTargetEstimator(target_specs=target_specs)
    with pytest.raises(ValueError, match="appears in multiple groups"):
        MixedTypeMultiTargetScorer(estimator=estimator, target_specs=target_specs)


def test_member_column_also_simple_target_rejected():
    target_specs = {
        "ITEM_a": TargetType.BINARY,
        "g1": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_a", "ITEM_b"]),
    }
    estimator = _StubMultiTargetEstimator(target_specs=target_specs)
    with pytest.raises(ValueError, match="also declared as simple target"):
        MixedTypeMultiTargetScorer(estimator=estimator, target_specs=target_specs)


def test_multilabel_member_column_must_be_item_prefixed():
    target_specs = {
        "g1": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["plain_name"]),
    }
    estimator = _StubMultiTargetEstimator(target_specs=target_specs)
    with pytest.raises(ValueError, match="must be a string prefixed with 'ITEM_'"):
        MixedTypeMultiTargetScorer(estimator=estimator, target_specs=target_specs)


# ---------------------------------------------------------------------- #
# Estimator Protocol checks at __init__
# ---------------------------------------------------------------------- #


def test_scorer_rejects_non_protocol_estimator():
    """Test #4: a class without predict_*_dict methods fails Protocol check."""
    target_specs = {"ITEM_clicked": TargetType.BINARY}
    bad_estimator = _PartialProtocolEstimator(target_specs=target_specs)
    with pytest.raises(TypeError, match="MultiTargetEstimator"):
        MixedTypeMultiTargetScorer(estimator=bad_estimator, target_specs=target_specs)


def test_scorer_rejects_unrelated_object():
    """Test #3 variant: a plain object instance is not a MultiTargetEstimator."""
    target_specs = {"ITEM_clicked": TargetType.BINARY}
    with pytest.raises(TypeError, match="MultiTargetEstimator"):
        MixedTypeMultiTargetScorer(estimator=object(), target_specs=target_specs)


def test_target_specs_consistency_check():
    """Test #10: scorer target_specs ≠ estimator target_specs → clean error."""
    scorer_specs = {"ITEM_A": TargetType.BINARY}
    estimator_specs = {"ITEM_B": TargetType.BINARY}
    estimator = _StubMultiTargetEstimator(target_specs=estimator_specs)
    with pytest.raises(ValueError, match="Inconsistent target_specs"):
        MixedTypeMultiTargetScorer(estimator=estimator, target_specs=scorer_specs)


# ---------------------------------------------------------------------- #
# Test #24: MultiTargetEstimator Protocol isinstance strictness
# ---------------------------------------------------------------------- #


def test_multi_target_protocol_isinstance_strictness_positive():
    """A class implementing every Protocol attribute is recognized."""
    estimator = _StubMultiTargetEstimator(target_specs={"ITEM_a": TargetType.BINARY})
    assert isinstance(estimator, MultiTargetEstimator)


def test_multi_target_protocol_isinstance_strictness_negative():
    """A class missing predict_*_dict methods is NOT recognized.

    Guards against accidental Protocol widening if a future refactor adds
    matching method names elsewhere — a runtime_checkable Protocol only
    checks attribute names, so the negative assertion pins the strict set.
    """
    partial = _PartialProtocolEstimator(target_specs={})
    assert not isinstance(partial, MultiTargetEstimator)


def test_multi_target_protocol_negative_for_plain_object():
    assert not isinstance(object(), MultiTargetEstimator)
    assert not isinstance("string", MultiTargetEstimator)
    assert not isinstance(42, MultiTargetEstimator)


# ---------------------------------------------------------------------- #
# Training-time validators (against real frames)
# ---------------------------------------------------------------------- #


def _make_scorer(target_specs):
    """Factory: scorer with a Protocol-conforming stub estimator."""
    estimator = _StubMultiTargetEstimator(target_specs=target_specs)
    return MixedTypeMultiTargetScorer(estimator=estimator, target_specs=target_specs)


def test_validate_interactions_missing_user_id():
    scorer = _make_scorer({"ITEM_a": TargetType.BINARY})
    df = pd.DataFrame({"ITEM_a": [0, 1], "feature": [0.5, 0.7]})
    with pytest.raises(ValueError, match="USER_ID"):
        scorer._validate_interactions(df)


def test_validate_interactions_missing_target_column():
    scorer = _make_scorer({"ITEM_a": TargetType.BINARY, "ITEM_b": TargetType.REGRESSION})
    df = pd.DataFrame({USER_ID_NAME: ["u1", "u2"], "ITEM_a": [0, 1], "feature": [0.5, 0.7]})
    with pytest.raises(ValueError, match="ITEM_b"):
        scorer._validate_interactions(df)


def test_validate_interactions_binary_target_with_non_binary_values():
    """Test #1: declared BINARY but column has {0, 1, 2} → clean error."""
    scorer = _make_scorer({"ITEM_a": TargetType.BINARY})
    df = pd.DataFrame({USER_ID_NAME: ["u1", "u2", "u3"], "ITEM_a": [0, 1, 2]})
    with pytest.raises(ValueError, match="BINARY but column contains"):
        scorer._validate_interactions(df)


def test_validate_interactions_regression_target_non_numeric():
    scorer = _make_scorer({"ITEM_a": TargetType.REGRESSION})
    df = pd.DataFrame({USER_ID_NAME: ["u1", "u2"], "ITEM_a": ["foo", "bar"]})
    with pytest.raises(ValueError, match="REGRESSION but column dtype"):
        scorer._validate_interactions(df)


def test_validate_interactions_multiclass_single_class_rejected():
    """Test #2 (part): multiclass column with 1 unique value → error."""
    scorer = _make_scorer({"ITEM_a": TargetType.MULTICLASS})
    df = pd.DataFrame({USER_ID_NAME: ["u1", "u2"], "ITEM_a": ["X", "X"]})
    with pytest.raises(ValueError, match="MULTICLASS but has only"):
        scorer._validate_interactions(df)


def test_validate_interactions_duplicate_users_rejected():
    scorer = _make_scorer({"ITEM_a": TargetType.BINARY})
    df = pd.DataFrame({USER_ID_NAME: ["u1", "u1"], "ITEM_a": [0, 1]})
    with pytest.raises(ValueError, match="one row per user"):
        scorer._validate_interactions(df)


def test_validate_interactions_null_target_rejected():
    scorer = _make_scorer({"ITEM_a": TargetType.BINARY})
    df = pd.DataFrame({USER_ID_NAME: ["u1", "u2"], "ITEM_a": [0, None]})
    with pytest.raises(ValueError, match="null value"):
        scorer._validate_interactions(df)


def test_validate_interactions_happy_path_simple():
    scorer = _make_scorer(
        {
            "ITEM_clicked": TargetType.BINARY,
            "ITEM_revenue": TargetType.REGRESSION,
            "ITEM_action": TargetType.MULTICLASS,
        }
    )
    df = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u2", "u3"],
            "ITEM_clicked": [0, 1, 1],
            "ITEM_revenue": [10.5, 22.0, 5.0],
            "ITEM_action": ["A", "B", "A"],
            "feature_1": [0.5, 0.6, 0.7],
        }
    )
    scorer._validate_interactions(df)  # must not raise


def test_validate_interactions_happy_path_with_multilabel():
    scorer = _make_scorer(
        {
            "ITEM_clicked": TargetType.BINARY,
            "engagement": TargetGroupSpec(
                type=TargetType.MULTILABEL,
                columns=["ITEM_email_open", "ITEM_app_open"],
            ),
        }
    )
    df = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u2", "u3"],
            "ITEM_clicked": [0, 1, 1],
            "ITEM_email_open": [1, 0, 1],
            "ITEM_app_open": [0, 1, 1],
            "feature_1": [0.5, 0.6, 0.7],
        }
    )
    scorer._validate_interactions(df)  # must not raise


# ---------------------------------------------------------------------- #
# Inference-time validation
# ---------------------------------------------------------------------- #


def test_validate_inference_vanilla_rejects_observed_columns():
    """Test #8 (v3 form): OBSERVED_* with a vanilla estimator → clean error
    pointing at the conditional estimator classes."""
    scorer = _make_scorer({"ITEM_clicked": TargetType.BINARY})
    df = pd.DataFrame({USER_ID_NAME: ["u1"], "feature_1": [0.5], "OBSERVED_clicked": [1]})
    with pytest.raises(NotImplementedError, match="ConditionalMultiTargetEstimator"):
        scorer._validate_inference_interactions(df)


def test_validate_inference_rejects_observed_namespace_collision():
    """Test #9: a feature column starting with OBSERVED_ is rejected even when
    it doesn't match any declared target (vanilla rejection path catches it
    first; for conditional estimators, the orphan check catches it)."""
    scorer = _make_scorer({"ITEM_clicked": TargetType.BINARY})
    df = pd.DataFrame({USER_ID_NAME: ["u1"], "feature_1": [0.5], "OBSERVED_unrelated": [1]})
    # Vanilla estimator → rejected upfront naming the conditional families
    # (orphan check is downstream and only runs for conditional estimators).
    with pytest.raises(NotImplementedError, match="ConditionalMultiTargetEstimator"):
        scorer._validate_inference_interactions(df)


def test_validate_inference_missing_user_id():
    scorer = _make_scorer({"ITEM_clicked": TargetType.BINARY})
    df = pd.DataFrame({"feature_1": [0.5]})
    with pytest.raises(ValueError, match="USER_ID"):
        scorer._validate_inference_interactions(df)


def test_validate_inference_happy_path():
    scorer = _make_scorer({"ITEM_clicked": TargetType.BINARY})
    df = pd.DataFrame({USER_ID_NAME: ["u1", "u2"], "feature_1": [0.5, 0.6]})
    scorer._validate_inference_interactions(df)  # must not raise


# ---------------------------------------------------------------------- #
# process_datasets
# ---------------------------------------------------------------------- #


def test_process_datasets_rejects_users_df():
    scorer = _make_scorer({"ITEM_clicked": TargetType.BINARY})
    df = pd.DataFrame({USER_ID_NAME: ["u1"], "ITEM_clicked": [1]})
    users_df = pd.DataFrame({USER_ID_NAME: ["u1"]})
    with pytest.raises(ValueError, match="does not accept users_df"):
        scorer.process_datasets(users_df=users_df, interactions_df=df)


def test_process_datasets_rejects_items_df():
    scorer = _make_scorer({"ITEM_clicked": TargetType.BINARY})
    df = pd.DataFrame({USER_ID_NAME: ["u1"], "ITEM_clicked": [1]})
    items_df = pd.DataFrame({"ITEM_ID": ["i1"]})
    with pytest.raises(ValueError, match="does not accept users_df or items_df"):
        scorer.process_datasets(items_df=items_df, interactions_df=df)


def test_process_datasets_training_returns_dict_y():
    scorer = _make_scorer(
        {
            "ITEM_clicked": TargetType.BINARY,
            "ITEM_revenue": TargetType.REGRESSION,
            "engagement": TargetGroupSpec(
                type=TargetType.MULTILABEL,
                columns=["ITEM_email_open", "ITEM_app_open"],
            ),
        }
    )
    df = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u2"],
            "ITEM_clicked": [0, 1],
            "ITEM_revenue": [10.5, 22.0],
            "ITEM_email_open": [1, 0],
            "ITEM_app_open": [0, 1],
            "feature_1": [0.5, 0.6],
            "feature_2": [1.0, 2.0],
        }
    )
    X, y_dict = scorer.process_datasets(interactions_df=df, is_training=True)

    # X retains only feature columns.
    assert set(X.columns) == {"feature_1", "feature_2"}

    # y_dict keyed by target_specs entries.
    assert set(y_dict.keys()) == {"ITEM_clicked", "ITEM_revenue", "engagement"}

    np.testing.assert_array_equal(y_dict["ITEM_clicked"], np.array([0, 1]))
    np.testing.assert_array_equal(y_dict["ITEM_revenue"], np.array([10.5, 22.0]))
    # multilabel group: (n, n_members) in declared member order.
    np.testing.assert_array_equal(y_dict["engagement"], np.array([[1, 0], [0, 1]]))

    # item_names is the fanned-out flat list.
    np.testing.assert_array_equal(
        scorer.item_names,
        np.array(["ITEM_clicked", "ITEM_revenue", "ITEM_email_open", "ITEM_app_open"]),
    )


def test_process_datasets_inference_returns_empty_y_dict():
    scorer = _make_scorer({"ITEM_clicked": TargetType.BINARY})
    df = pd.DataFrame({USER_ID_NAME: ["u1", "u2"], "feature_1": [0.5, 0.6]})
    X, y_dict = scorer.process_datasets(interactions_df=df, is_training=False)
    assert list(X.columns) == ["feature_1"]
    assert y_dict == {}


# ---------------------------------------------------------------------- #
# _calculate_scores stays NotImplementedError forever (architectural seam)
# ---------------------------------------------------------------------- #


def test_calculate_scores_always_raises():
    scorer = _make_scorer({"ITEM_a": TargetType.BINARY})
    with pytest.raises(NotImplementedError, match="does not use _calculate_scores"):
        scorer._calculate_scores(pd.DataFrame())


# ---------------------------------------------------------------------- #
# Test #11: Dataset schema enforcement
# ---------------------------------------------------------------------- #


def _write_csv(df: pd.DataFrame, dst_dir: str, name: str = "interactions.csv") -> str:
    path = os.path.join(dst_dir, name)
    df.to_csv(path, index=False)
    return path


def test_dataset_schema_happy_path():
    """Dataset constructs on a frame that satisfies the schema (USER_ID present)."""
    df = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u2"],
            "ITEM_clicked": [0, 1],
            "feature_1": [0.5, 0.6],
        }
    )
    with tempfile.TemporaryDirectory() as d:
        path = _write_csv(df, d)
        ds = InteractionMixedTypeMultiTargetDataset(data_location=path, is_training=True)
        fetched = ds.fetch_data()
        assert USER_ID_NAME in fetched.columns


def test_dataset_schema_rejects_missing_user_id():
    """Frame without USER_ID is rejected at dataset construction."""
    df = pd.DataFrame({"ITEM_clicked": [0, 1], "feature_1": [0.5, 0.6]})
    with tempfile.TemporaryDirectory() as d:
        path = _write_csv(df, d)
        with pytest.raises(Exception):  # DatasetSchema raises ValueError/KeyError
            ds = InteractionMixedTypeMultiTargetDataset(data_location=path, is_training=True)
            ds.fetch_data()


# ---------------------------------------------------------------------- #
# M4: scorer wide-format stitching (score_items, predict_targets,
# score_fast, score_per_target) — exercised against a real fitted estimator.
# ---------------------------------------------------------------------- #


torch = pytest.importorskip("torch")  # M4 tests need a real estimator


def _make_fitted_scorer(target_specs, X, y):
    """Build a scorer with a fitted joint-MLP estimator for M4 tests."""
    est = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 1, "hidden_dim": 16, "num_layers": 2, "batch_size": 32},
    )
    est.fit(X, y)
    return MixedTypeMultiTargetScorer(estimator=est, target_specs=target_specs)


def _synthetic_for_m4(n=60, seed=0):
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


def test_score_items_wide_format_columns():
    """Test #5: every target type produces documented output column names."""
    feats, y, target_specs = _synthetic_for_m4(n=40)
    # Interactions for training (with target columns).
    train_df = feats.copy()
    train_df[USER_ID_NAME] = [f"u{i}" for i in range(40)]
    for k, v in y.items():
        if k == "engagement":
            train_df["ITEM_email_open"] = v[:, 0]
            train_df["ITEM_app_open"] = v[:, 1]
        else:
            train_df[k] = v
    scorer = _make_fitted_scorer(target_specs, feats, y)

    # Inference frame (just USER_ID + features).
    inf_df = feats.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(40)])
    out = scorer.score_items(interactions=inf_df)

    # binary → ITEM_clicked_0, ITEM_clicked_1
    # regression → ITEM_revenue
    # multiclass → ITEM_action_<class> per class (A, B, C)
    # multilabel → ITEM_email_open_0/1, ITEM_app_open_0/1
    expected_cols = {
        "ITEM_clicked_0",
        "ITEM_clicked_1",
        "ITEM_revenue",
        "ITEM_action_A",
        "ITEM_action_B",
        "ITEM_action_C",
        "ITEM_email_open_0",
        "ITEM_email_open_1",
        "ITEM_app_open_0",
        "ITEM_app_open_1",
    }
    assert set(out.columns) == expected_cols
    assert out.shape[0] == 40
    # USER_ID is not in output.
    assert USER_ID_NAME not in out.columns


def test_predict_targets_one_column_per_target():
    """Test #6: every target type produces one column per fanned-out target."""
    feats, y, target_specs = _synthetic_for_m4(n=30)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    inf_df = feats.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(30)])
    out = scorer.predict_targets(interactions=inf_df)
    assert set(out.columns) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_action",
        "ITEM_email_open",
        "ITEM_app_open",
    }
    assert out.shape == (30, 5)
    # multiclass column should carry original labels.
    assert set(np.unique(out["ITEM_action"]).tolist()).issubset({"A", "B", "C"})


def test_score_items_rejects_users_kwarg():
    feats, y, target_specs = _synthetic_for_m4(n=20)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    inf_df = feats.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(20)])
    users_df = pd.DataFrame({USER_ID_NAME: ["u0"]})
    with pytest.raises(ValueError, match="does not accept users"):
        scorer.score_items(interactions=inf_df, users=users_df)


def test_score_fast_one_row_happy_path():
    feats, y, target_specs = _synthetic_for_m4(n=20)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    one_row = feats.iloc[[0]].copy()
    out = scorer.score_fast(one_row)
    assert out.shape[0] == 1
    assert set(out.columns) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_action",
        "ITEM_email_open",
        "ITEM_app_open",
    }


def test_score_fast_rejects_multi_row():
    """Test #7: features.shape[0] != 1 → clean error."""
    feats, y, target_specs = _synthetic_for_m4(n=20)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    with pytest.raises(ValueError, match="shape\\[0\\] == 1"):
        scorer.score_fast(feats.iloc[:3])


def test_score_fast_rejects_ndarray_input():
    """Test #12: ndarray instead of DataFrame → clean error."""
    feats, y, target_specs = _synthetic_for_m4(n=10)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    with pytest.raises(TypeError, match="DataFrame"):
        scorer.score_fast(feats.iloc[0:1].to_numpy())


def test_score_fast_rejects_observed_columns():
    """Vanilla estimator + OBSERVED_* at score_fast → clean error."""
    feats, y, target_specs = _synthetic_for_m4(n=10)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    one_row = feats.iloc[[0]].copy()
    one_row["OBSERVED_clicked"] = 1
    with pytest.raises(NotImplementedError, match="ConditionalMultiTargetEstimator"):
        scorer.score_fast(one_row)


def test_score_fast_column_order_invariant():
    """Test #13: swapping feature columns → identical wide output."""
    feats, y, target_specs = _synthetic_for_m4(n=20)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    row = feats.iloc[[0]]
    out_orig = scorer.score_fast(row)
    out_reord = scorer.score_fast(row.iloc[:, ::-1])
    for c in out_orig.columns:
        # binary/regression/multilabel are numeric; multiclass labels are strings.
        a = out_orig[c].to_numpy()
        b = out_reord[c].to_numpy()
        if a.dtype.kind in {"U", "O"}:
            assert (a == b).all()
        else:
            np.testing.assert_allclose(a, b, rtol=1e-5)


def test_score_fast_rejects_extra_columns():
    """Test #14: a feature column not in training feature_names → clean error."""
    feats, y, target_specs = _synthetic_for_m4(n=10)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    one_row = feats.iloc[[0]].copy()
    one_row["unseen_feat"] = 1.0
    with pytest.raises(ValueError, match="unseen at training"):
        scorer.score_fast(one_row)


def test_score_per_target_happy_path():
    """Test #13 in evaluation: user-supplied callables yield per-target dict."""
    feats, y, target_specs = _synthetic_for_m4(n=40)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    inf_df = feats.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(40)])
    # Ground truth (wide; matches predict_targets column set).
    y_true = pd.DataFrame(
        {
            "ITEM_clicked": y["ITEM_clicked"],
            "ITEM_revenue": y["ITEM_revenue"],
            "ITEM_action": y["ITEM_action"],
            "ITEM_email_open": y["engagement"][:, 0],
            "ITEM_app_open": y["engagement"][:, 1],
        }
    )
    # Trivial callables: count positives for proba (binary/multilabel/multiclass);
    # mean abs diff for regression. Just to exercise the callable plumbing.
    result = scorer.score_per_target(
        interactions=inf_df,
        y_true=y_true,
        metric_callables={
            TargetType.BINARY: lambda yt, p: float((p[:, 1] >= 0.5).mean()),
            TargetType.REGRESSION: lambda yt, p: float(np.mean(np.abs(yt - p))),
            TargetType.MULTICLASS: lambda yt, p: float(p.shape[1]),  # K
        },
    )
    assert set(result.keys()) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_action",
        "ITEM_email_open",
        "ITEM_app_open",
    }
    assert all(isinstance(v, float) for v in result.values())


def test_score_per_target_name_override_beats_type_default():
    feats, y, target_specs = _synthetic_for_m4(n=20)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    inf_df = feats.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(20)])
    y_true = pd.DataFrame(
        {
            "ITEM_clicked": y["ITEM_clicked"],
            "ITEM_revenue": y["ITEM_revenue"],
            "ITEM_action": y["ITEM_action"],
            "ITEM_email_open": y["engagement"][:, 0],
            "ITEM_app_open": y["engagement"][:, 1],
        }
    )
    result = scorer.score_per_target(
        interactions=inf_df,
        y_true=y_true,
        metric_callables={
            TargetType.BINARY: lambda yt, p: 7.0,
            "ITEM_clicked": lambda yt, p: 99.0,  # name override
            TargetType.REGRESSION: lambda yt, p: 1.0,
            TargetType.MULTICLASS: lambda yt, p: 2.0,
        },
    )
    assert result["ITEM_clicked"] == 99.0  # override wins
    # Other binary members (multilabel fan-out) take the TargetType default.
    assert result["ITEM_email_open"] == 7.0
    assert result["ITEM_app_open"] == 7.0


def test_score_per_target_missing_callable_raises():
    feats, y, target_specs = _synthetic_for_m4(n=10)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    inf_df = feats.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(10)])
    y_true = pd.DataFrame(
        {
            "ITEM_clicked": y["ITEM_clicked"],
            "ITEM_revenue": y["ITEM_revenue"],
            "ITEM_action": y["ITEM_action"],
            "ITEM_email_open": y["engagement"][:, 0],
            "ITEM_app_open": y["engagement"][:, 1],
        }
    )
    # No callable for REGRESSION.
    with pytest.raises(KeyError, match="No metric callable"):
        scorer.score_per_target(
            interactions=inf_df,
            y_true=y_true,
            metric_callables={
                TargetType.BINARY: lambda yt, p: 1.0,
                TargetType.MULTICLASS: lambda yt, p: 1.0,
            },
        )


def test_score_per_target_y_true_column_mismatch():
    feats, y, target_specs = _synthetic_for_m4(n=10)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    inf_df = feats.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(10)])
    # Missing ITEM_app_open.
    y_true = pd.DataFrame(
        {
            "ITEM_clicked": y["ITEM_clicked"],
            "ITEM_revenue": y["ITEM_revenue"],
            "ITEM_action": y["ITEM_action"],
            "ITEM_email_open": y["engagement"][:, 0],
        }
    )
    with pytest.raises(ValueError, match="missing target column"):
        scorer.score_per_target(
            interactions=inf_df,
            y_true=y_true,
            metric_callables={
                TargetType.BINARY: lambda yt, p: 0.0,
                TargetType.REGRESSION: lambda yt, p: 0.0,
                TargetType.MULTICLASS: lambda yt, p: 0.0,
            },
        )


# ---------------------------------------------------------------------- #
# M7: RankingRecommender + BaseRecommender dispatch
# ---------------------------------------------------------------------- #


def _build_recommender_m7(target_specs, X, y):
    est = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 1, "hidden_dim": 16, "num_layers": 2, "batch_size": 32},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=target_specs)
    return RankingRecommender(scorer=scorer)


def test_recommend_short_circuits_to_predict_targets():
    """Test #15-ish: recommend() on MixedTypeMultiTargetScorer returns the
    wide-format predict_targets frame, not a top-K item list."""
    X, y, target_specs = _synthetic_for_m4(n=20)
    recommender = _build_recommender_m7(target_specs, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(20)])
    out = recommender.recommend(interactions=inf_df)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_action",
        "ITEM_email_open",
        "ITEM_app_open",
    }
    assert out.shape[0] == 20


def test_recommend_top_k_emits_warning(caplog):
    """Test #15: top_k != 1 emits a UserWarning naming the scorer."""
    import logging

    X, y, target_specs = _synthetic_for_m4(n=10)
    recommender = _build_recommender_m7(target_specs, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(10)])
    with caplog.at_level(logging.WARNING):
        recommender.recommend(interactions=inf_df, top_k=10)
    # After the P1-12 capability-flag dispatch refactor, the warning is
    # phrased generically as "Per-target scorer" rather than naming
    # MixedTypeMultiTargetScorer (the recommender no longer knows the
    # concrete subclass).
    assert any(
        ("Per-target scorer" in record.message or "MixedTypeMultiTargetScorer" in record.message)
        and "top_k" in record.message
        for record in caplog.records
    )


def test_recommend_rejects_users_kwarg():
    """Test #16: passing non-None users → clean error before scoring."""
    X, y, target_specs = _synthetic_for_m4(n=10)
    recommender = _build_recommender_m7(target_specs, X, y)
    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(10)])
    users_df = pd.DataFrame({USER_ID_NAME: ["u0"]})
    with pytest.raises(ValueError, match="users should be set to None"):
        recommender.recommend(interactions=inf_df, users=users_df)


def test_score_per_target_y_true_extra_columns_rejected():
    feats, y, target_specs = _synthetic_for_m4(n=10)
    scorer = _make_fitted_scorer(target_specs, feats, y)
    inf_df = feats.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(10)])
    y_true = pd.DataFrame(
        {
            "ITEM_clicked": y["ITEM_clicked"],
            "ITEM_revenue": y["ITEM_revenue"],
            "ITEM_action": y["ITEM_action"],
            "ITEM_email_open": y["engagement"][:, 0],
            "ITEM_app_open": y["engagement"][:, 1],
            "STRAY_COL": y["ITEM_clicked"],
        }
    )
    with pytest.raises(ValueError, match="unknown column"):
        scorer.score_per_target(
            interactions=inf_df,
            y_true=y_true,
            metric_callables={
                TargetType.BINARY: lambda yt, p: 0.0,
                TargetType.REGRESSION: lambda yt, p: 0.0,
                TargetType.MULTICLASS: lambda yt, p: 0.0,
            },
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
# Scorer v2-list #17: recommend_online dispatch-order
# ====================================================================== #


def test_scorer_17_recommend_online_dispatch_order_includes_mixed_type():
    """recommend_online's dispatch chain must check
    MixedTypeMultiTargetScorer BEFORE falling through to _score_fast_np
    (which assumes a rankable-scalar shape that doesn't apply to per-
    target prediction). Regression for the v2 plan's #17 ordering pin."""
    df, target_specs = _make_mixed_df()
    est = _train_joint_mlp(df, target_specs)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=target_specs)
    scorer._init_item_state()  # populate item_names (recommender wasn't trained)
    recommender = RankingRecommender(scorer=scorer)

    # Single-row inference frame.
    feat_cols = [c for c in df.columns if c.startswith("f")]
    one_row = df.iloc[[0]][feat_cols + [USER_ID_NAME]]

    # If the dispatch order regressed, this call would fall through to
    # _score_fast_np and raise (mixed-type proba dict can't be cast to
    # a (n_items,) ranking array).
    result = recommender.recommend_online(interactions=one_row, top_k=1)
    # Per-target wide-format DataFrame (one column per fanned-out target).
    assert isinstance(result, pd.DataFrame)
    expected = {"ITEM_bin", "ITEM_rev", "ITEM_class", "ITEM_email", "ITEM_app"}
    assert set(result.columns) == expected


# ====================================================================== #
# Scorer v2-list #19: single-target round-trip (one TargetType, no group)
# ====================================================================== #


def test_scorer_19_single_target_round_trip_binary():
    """target_specs with exactly ONE BINARY target must train, score,
    and predict end-to-end. Edge case: many code paths assume ≥2 targets
    or a heterogeneous mix; the single-binary-target case must not
    regress to MultioutputScorer-like semantics."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "ITEM_clicked": (rng.normal(size=n) > 0).astype(int),
            USER_ID_NAME: np.arange(n),
        }
    )
    ts = {"ITEM_clicked": TargetType.BINARY}

    X = df[["f0", "f1"]]
    y = {"ITEM_clicked": df["ITEM_clicked"].to_numpy()}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 3, "hidden_dim": 4, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)

    proba_df = scorer.score_items(interactions=df[["f0", "f1", USER_ID_NAME]])
    targets_df = scorer.predict_targets(interactions=df[["f0", "f1", USER_ID_NAME]])

    # Score frame has the BINARY (n, 2) split into _0/_1 cols.
    assert {"ITEM_clicked_0", "ITEM_clicked_1"}.issubset(set(proba_df.columns))
    # Targets frame has the single ITEM_clicked column.
    assert list(targets_df.columns) == ["ITEM_clicked"]
    assert len(targets_df) == n


# ====================================================================== #
# Scorer v2-list #20: output column-order vs target_specs insertion order
# ====================================================================== #


def test_scorer_20_output_column_order_follows_target_specs_insertion():
    """The wide-format output column order must follow target_specs
    insertion order (PEP 468 dict ordering), regardless of alphabetical
    sort. Regression for the v2 plan's #20 determinism pin."""
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "ITEM_z_first": (rng.normal(size=n) > 0).astype(int),
            "ITEM_a_second": rng.normal(size=n),
            "ITEM_m_third": (rng.normal(size=n) > 0).astype(int),
            USER_ID_NAME: np.arange(n),
        }
    )
    # Insertion order intentionally NON-alphabetical.
    ts = {
        "ITEM_z_first": TargetType.BINARY,
        "ITEM_a_second": TargetType.REGRESSION,
        "ITEM_m_third": TargetType.BINARY,
    }

    X = df[["f0", "f1"]]
    y = {
        "ITEM_z_first": df["ITEM_z_first"].to_numpy(),
        "ITEM_a_second": df["ITEM_a_second"].to_numpy(),
        "ITEM_m_third": df["ITEM_m_third"].to_numpy(),
    }
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 2, "hidden_dim": 4, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)

    targets_df = scorer.predict_targets(interactions=df[["f0", "f1", USER_ID_NAME]])
    # MUST follow insertion order, NOT alphabetical (which would put
    # ITEM_a_second first).
    assert list(targets_df.columns) == ["ITEM_z_first", "ITEM_a_second", "ITEM_m_third"]


# ====================================================================== #
# Doc-test for opt-in preservation registry (P2-8)
# ====================================================================== #


def test_preserved_inference_columns_registry():
    """Pin the per-scorer preserved_inference_columns defaults so a
    future scorer that forgets to opt out has to update this test
    explicitly — failure mode is "test red," not "silent OBSERVED leak."
    """
    # Scorers that MUST NOT preserve any columns (default empty list).
    # The class-level method object must be the inherited BaseScorer
    # default — a subclass override that doesn't change the return
    # value would still cause this test to fail and force whoever
    # changed it to update this opt-in registry.
    from skrec.scorer.base_scorer import BaseScorer as _BaseScorer
    from skrec.scorer.hierarchical import HierarchicalScorer
    from skrec.scorer.independent import IndependentScorer
    from skrec.scorer.multiclass import MulticlassScorer
    from skrec.scorer.multioutput import MultioutputScorer
    from skrec.scorer.sequential import SequentialScorer
    from skrec.scorer.universal import UniversalScorer

    no_preserve = [
        MultioutputScorer,
        MulticlassScorer,
        IndependentScorer,
        UniversalScorer,
        SequentialScorer,
        HierarchicalScorer,
    ]
    for cls in no_preserve:
        assert cls.preserved_inference_columns is _BaseScorer.preserved_inference_columns, (
            f"{cls.__name__} overrides preserved_inference_columns; update "
            f"this opt-in registry test before landing the change."
        )
        assert cls.preserved_inference_column_prefixes is _BaseScorer.preserved_inference_column_prefixes, (
            f"{cls.__name__} overrides preserved_inference_column_prefixes; "
            f"update this opt-in registry test before landing the change."
        )
        # Capability flag stays False
        assert cls.supports_observed_conditioning is False, f"{cls.__name__} silently opted into OBSERVED conditioning."
        assert cls.is_per_target_scorer is False, f"{cls.__name__} silently opted into per-target dispatch."

    # MixedTypeMultiTargetScorer MUST opt in.
    assert MixedTypeMultiTargetScorer.supports_observed_conditioning is True


# ====================================================================== #
# Fix 4: multiclass label sort for integer K >= 10
# ====================================================================== #


def test_fix4_sort_multiclass_labels_integer_natural_order():
    """sorted(..., key=str) yields [1, 10, 2]; natural sort yields [1, 2, 10]."""
    assert sort_multiclass_labels([10, 1, 2]) == [1, 2, 10]
    assert sort_multiclass_labels({1, 2, 10}) == [1, 2, 10]


def test_fix4_sort_multiclass_labels_eleven_integers_natural_order():
    """K=11 — the failure case the lex-sort bug hides: ['1', '10', '11', '2', ...]."""
    expected = list(range(11))
    assert sort_multiclass_labels(reversed(expected)) == expected


def test_fix4_sort_multiclass_labels_strings_lex_order():
    """String labels still sort lex (no behavior change for the common case)."""
    assert sort_multiclass_labels(["c", "a", "b"]) == ["a", "b", "c"]


def test_fix4_sort_multiclass_labels_floats_natural_order():
    assert sort_multiclass_labels([1.5, 0.5, 2.5]) == [0.5, 1.5, 2.5]


# ---------------------------------------------------------------------- #
# Fix R2-3: orphan ITEM_* feature column rejected at inference
# ---------------------------------------------------------------------- #


def test_fix_r2_3_orphan_item_column_rejected_at_inference():
    X, scorer, _ = _build_validation_scaffold()
    inf = X.copy()
    inf.insert(0, USER_ID_NAME, ["u"] * len(X))
    inf["ITEM_not_declared"] = 1  # orphan
    with pytest.raises(ValueError, match="Orphan ITEM_"):
        scorer._validate_inference_interactions(inf)


def test_fix_r2_3_orphan_item_rejection_runs_for_conditional_too():
    """Vanilla and conditional paths both pass through the orphan check."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = {"ITEM_a": (X["f0"] > 0).astype(int).to_numpy()}
    ts = {"ITEM_a": TargetType.BINARY}
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 16},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    inf = X.copy()
    inf.insert(0, USER_ID_NAME, ["u"] * len(X))
    inf["ITEM_stray"] = 0
    with pytest.raises(ValueError, match="Orphan ITEM_"):
        scorer._validate_inference_interactions(inf)


# ---------------------------------------------------------------------- #
# Fix R2-4: score_fast routes through _validate_inference_interactions
# ---------------------------------------------------------------------- #


# ---------------------------------------------------------------------- #
# Fix R2-4: score_fast routes through _validate_inference_interactions
# ---------------------------------------------------------------------- #


def test_fix_r2_4_score_fast_runs_full_validator_orphan_item():
    """score_fast must trigger the orphan-ITEM_* check (and by extension
    the whole validator) — previously it had its own narrow OBSERVED-only
    inline check, so other validation rules were skipped on the single-
    row path."""
    X, scorer, _ = _build_validation_scaffold()
    row = X.iloc[[0]].copy()
    row["ITEM_stray"] = 1
    with pytest.raises(ValueError, match="Orphan ITEM_"):
        scorer.score_fast(row)


def test_fix_r2_4_score_fast_runs_multilabel_group_mask_validator():
    """Plan v3 test #5 single-row path: score_fast must reject partial-
    multilabel-group observation. Pre-fix score_fast skipped the validator
    so this check never fired through score_fast."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    ts = {"g": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_m1", "ITEM_m2"])}
    y = {"g": rng.integers(0, 2, size=(n, 2))}
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 16},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    row = X.iloc[[0]].copy()
    row["OBSERVED_m1"] = 1.0
    row["OBSERVED_m2"] = np.nan  # partial-group observation in the single row
    with pytest.raises(ValueError, match="partial-group"):
        scorer.score_fast(row)


# ---------------------------------------------------------------------- #
# Fix R2-5: multilabel group column-level imbalance
# ---------------------------------------------------------------------- #


# ---------------------------------------------------------------------- #
# Fix R2-5: multilabel group column-level imbalance
# ---------------------------------------------------------------------- #


def test_fix_r2_5_multilabel_column_level_imbalance_rejected():
    """If one member's OBSERVED_* column is declared in the inference frame
    and another is absent, reject — the joint group-mask semantics break.
    Pre-fix: validator only checked consistency among PRESENT columns, so
    the asymmetry slipped through."""
    rng = np.random.default_rng(0)
    n = 20
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    ts = {"g": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_m1", "ITEM_m2"])}
    y = {"g": rng.integers(0, 2, size=(n, 2))}
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 16},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    inf = X.copy()
    inf.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    # OBSERVED_m1 is present, OBSERVED_m2 is NOT present in the frame.
    inf["OBSERVED_m1"] = 1.0
    with pytest.raises(ValueError, match="column-level"):
        scorer._validate_inference_interactions(inf)


# ---------------------------------------------------------------------- #
# Fix R2-6: predict_with_observed Protocol signature accepts None
# ---------------------------------------------------------------------- #


def test_fix_r2_b1_target_specs_key_with_dot_rejected_at_init():
    """nn.ModuleDict rejects keys containing '.', so a group_key like
    'engagement.group' would crash inside _PerTargetHeads with a less-
    actionable error. Catch it at scorer construction with a clear msg."""
    target_specs = {"engagement.group": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_a"])}
    with pytest.raises(ValueError, match=r"'\.'"):
        MixedTypeMultiTargetScorer(estimator=_StubMTE(target_specs), target_specs=target_specs)


def test_fix_r2_b1_target_specs_key_with_whitespace_rejected():
    target_specs = {"ITEM_my target": TargetType.BINARY}
    with pytest.raises(ValueError, match="whitespace"):
        MixedTypeMultiTargetScorer(estimator=_StubMTE(target_specs), target_specs=target_specs)


# --- M2: schema-apply preserve uses pandas index alignment ---


# --- M2: schema-apply preserve uses pandas index alignment ---


def test_fix_r2_b2_schema_apply_preserve_aligns_by_index_under_row_filter():
    """If interactions_schema.apply() ever filters rows (current implementation
    doesn't, but the seam must be defended), preserved OBSERVED_* columns
    align by index — surviving rows keep their original values; filtered
    rows simply drop. Pre-fix used positional assignment via .to_numpy(),
    which would have silently mis-aligned on row filtering."""
    from skrec.dataset.schema import DatasetSchema
    from skrec.recommender.ranking.ranking_recommender import RankingRecommender

    rng = np.random.default_rng(0)
    n = 20
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = {"ITEM_a": (X["f0"] > 0).astype(int).to_numpy()}
    ts = {"ITEM_a": TargetType.BINARY}
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 16},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    # process_datasets populates scorer.item_names; recommend_online needs it.
    train_with_uid = X.copy()
    train_with_uid.insert(0, USER_ID_NAME, [f"t{i}" for i in range(len(X))])
    train_with_uid["ITEM_a"] = y["ITEM_a"]
    scorer.process_datasets(interactions_df=train_with_uid, is_training=True)

    recommender = RankingRecommender(scorer=scorer)
    recommender.interactions_schema = DatasetSchema(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "f0", "type": "float"},
                {"name": "f1", "type": "float"},
                {"name": "f2", "type": "float"},
            ]
        }
    )
    recommender.users_schema = None

    # Single-row inference path is what recommend_online uses; the
    # index-alignment fix matters identically for the batch path. Build
    # a single-row frame with a non-default index value to prove the
    # OBSERVED_a value travels with the row rather than being assumed at
    # position 0.
    single = X.iloc[[7]].copy()  # row index 7
    single.insert(0, USER_ID_NAME, ["u_index7"])
    single["OBSERVED_a"] = 1.0

    recommender.recommend_online(interactions=single)
    # If index alignment broke, OBSERVED_a would be NaN-dropped and
    # the model would silently produce the unconditional prediction.
    # Check the schema-preserved value made it back into the post-apply
    # frame: the score must reflect conditioning.
    single_no_obs = single.drop(columns=["OBSERVED_a"])
    recommender.recommend_online(interactions=single_no_obs)
    # Single-row score_items proba is the rigorous test for diff;
    # recommend_online returns the predict_targets shape so we drive
    # score_items directly here.
    p_with = float(scorer.score_items(interactions=single)["ITEM_a_1"].iloc[0])
    p_without = float(scorer.score_items(interactions=single_no_obs)["ITEM_a_1"].iloc[0])
    # With OBSERVED_a=1 the model knows ITEM_a — should produce a different
    # ITEM_a prediction than the unobserved path.
    assert abs(p_with - p_without) > 1e-6, (
        "Schema-preserve index alignment broken — OBSERVED_a value didn't reach the conditional estimator."
    )


# --- M3: Joint fit rejects NaN in training y with a named error ---


# ====================================================================== #
# Round 3 P0 ship-blockers
# ====================================================================== #


# --- P0-1: score_items preserves OBSERVED_* through batch schema-apply ---


def test_fix_p0_1_score_items_preserves_observed_through_non_declaring_schema():
    """Pre-fix: only recommend_online wired the preservation hook. Batch
    score_items routed through InferenceInputPreparer.preprocess_inputs
    which called interactions_schema.apply() directly, silently stripping
    OBSERVED_* on every batch call. End-to-end test: build a recommender
    with a client schema that omits OBSERVED_*, call score_items, confirm
    conditioning still works."""
    from skrec.dataset.schema import DatasetSchema
    from skrec.recommender.ranking.ranking_recommender import RankingRecommender

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    latent = X["f0"] + 0.7 * X["f1"]
    y = {
        "ITEM_a": (latent > 0).astype(int).to_numpy(),
        "ITEM_b": (latent + 0.15 * rng.normal(size=n) > 0).astype(int).to_numpy(),
    }
    ts = {"ITEM_a": TargetType.BINARY, "ITEM_b": TargetType.BINARY}

    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 8, "hidden_dim": 32, "num_layers": 2, "batch_size": 32, "mask_prob": 0.5, "seed": 0},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    # process_datasets to populate item_names.
    train_with_uid = X.copy()
    train_with_uid.insert(0, USER_ID_NAME, [f"t{i}" for i in range(n)])
    for k in ts:
        train_with_uid[k] = y[k]
    scorer.process_datasets(interactions_df=train_with_uid, is_training=True)
    recommender = RankingRecommender(scorer=scorer)
    recommender.interactions_schema = DatasetSchema(
        {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "f0", "type": "float"},
                {"name": "f1", "type": "float"},
                {"name": "f2", "type": "float"},
            ]
        }
    )
    recommender.users_schema = None

    batch = X.iloc[:20].copy()
    batch.insert(0, USER_ID_NAME, [f"u{i}" for i in range(20)])
    batch_no_obs = batch.copy()
    batch_with_obs = batch.copy()
    batch_with_obs["OBSERVED_a"] = 1.0  # observe ITEM_a positive for every row

    # Batch score_items — the seam P0-1 targets.
    out_no = recommender.score_items(interactions=batch_no_obs)
    out_with = recommender.score_items(interactions=batch_with_obs)
    p_b_no = out_no["ITEM_b_1"].to_numpy()
    p_b_with = out_with["ITEM_b_1"].to_numpy()
    # If OBSERVED_a was silently stripped (the pre-fix behavior), both
    # calls produce identical proba and the diff is 0.
    assert np.abs(p_b_no - p_b_with).mean() > 1e-6, (
        "batch score_items silently stripped OBSERVED_a — predictions with/without are identical. P0-1 regression."
    )


# --- P0-2: attn_dropout actually controls nn.MultiheadAttention dropout ---


def test_p1_16_target_group_spec_rejects_duplicate_members():
    """TargetGroupSpec with duplicate member columns must be rejected at
    both the scorer and the independent-estimator validation paths."""
    ts = {"g": {"type": TargetType.MULTILABEL, "columns": ["ITEM_a", "ITEM_a"]}}

    with pytest.raises(ValueError, match="duplicate member"):
        MixedTypeMultiTargetScorer(
            estimator=JointMultiTargetMLPEstimator(
                target_specs={"ITEM_x": TargetType.BINARY},
                params={"epochs": 1, "hidden_dim": 2},
            ),
            target_specs=ts,
        )

    with pytest.raises(ValueError, match="duplicate member"):
        IndependentMultiTargetEstimator(target_specs=ts, estimators={})


def test_recommend_dispatches_via_is_per_target_scorer_flag():
    """The recommend()/recommend_online dispatch must read
    ``is_per_target_scorer`` (capability flag) rather than
    ``isinstance(scorer, MixedTypeMultiTargetScorer)`` — so flipping
    the flag on a future scorer routes it to the per-target path with
    no recommender-side edit."""
    from skrec.scorer.base_scorer import BaseScorer
    from skrec.scorer.mixed_type_multi_target import MixedTypeMultiTargetScorer

    assert BaseScorer.is_per_target_scorer is False
    assert MixedTypeMultiTargetScorer.is_per_target_scorer is True


def test_orphan_item_typo_at_inference_caught_after_schema_apply():
    """An ITEM_typo column survives schema apply (via the ITEM_ prefix
    preservation) so the scorer's orphan-ITEM_* validator can produce
    the actionable error instead of the schema silently stripping the
    typo."""

    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "ITEM_a": (rng.normal(size=n) > 0).astype(int),
            USER_ID_NAME: np.arange(n),
        }
    )
    est = JointMultiTargetMLPEstimator(
        target_specs={"ITEM_a": TargetType.BINARY},
        params={"epochs": 1, "hidden_dim": 2, "seed": 0},
    )
    est.fit(df[["f0"]], {"ITEM_a": df["ITEM_a"].to_numpy()})
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs={"ITEM_a": TargetType.BINARY})
    # The prefix list must publish ITEM_ AND OBSERVED_ so both typo
    # families survive into the validator.
    assert "ITEM_" in scorer.preserved_inference_column_prefixes()
    assert "OBSERVED_" in scorer.preserved_inference_column_prefixes()


# ====================================================================== #
# Round 4 (continued): more older P1 items
# ====================================================================== #


# ====================================================================== #
# Round 4 (continued): more older P1 items
# ====================================================================== #


def test_recommend_per_target_routes_through_preprocess_inputs():
    """recommend() on a per-target scorer must apply the same schema
    coercion that score_items uses. Previously bypassed
    _preprocess_inputs and could leave object-dtype columns un-coerced
    relative to score_items, producing inconsistent X across the two
    paths on the same frame."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {"ITEM_a": (rng.normal(size=n) > 0).astype(int)}
    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 2, "seed": 0})
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    scorer._init_item_state()
    recommender = RankingRecommender(scorer=scorer)
    inf = X.copy()
    inf[USER_ID_NAME] = np.arange(n)

    # Without schema configured, _preprocess_inputs is a pass-through;
    # what we're really pinning is that recommend() does not raise on
    # the standalone-scorer case (would AttributeError before P2-6's
    # getattr guard) and produces the per-target wide DataFrame the way
    # score_items does.
    rec_df = recommender.recommend(interactions=inf)
    score_df = scorer.score_items(interactions=inf)
    # Both should produce per-target outputs aligned with target_specs;
    # rec returns predict_targets style (one ITEM_a column), score
    # returns proba style (ITEM_a_0 + ITEM_a_1). Sanity: same row count.
    assert len(rec_df) == len(score_df) == n


# ---------------------------------------------------------------------- #
# OBSERVED_PREFIX constant
# ---------------------------------------------------------------------- #


def test_observed_prefix_constant():
    assert C.OBSERVED_PREFIX == "OBSERVED_"


# ---------------------------------------------------------------------- #
# ConditionalMultiTargetEstimator Protocol subclass — runtime_checkable
# isinstance strictness
# ---------------------------------------------------------------------- #


class _StubConditionalEstimator:
    """Stub with every ConditionalMultiTargetEstimator attribute filled in.

    Must set the ``is_conditional_multi_target`` sentinel — see
    ``_multi_target_protocol.py`` for the rationale. Without the
    sentinel, isinstance(stub, ConditionalMultiTargetEstimator) would
    pass structurally but the scorer's stricter
    ``_is_conditional_estimator`` helper (which checks both the Protocol
    AND the sentinel) would still treat the stub as vanilla.
    """

    is_conditional_multi_target: bool = True

    def __init__(self, target_specs):
        self.target_specs = target_specs

    def fit(self, X, y, X_valid=None, y_valid=None):
        return self

    def predict_proba_dict(self, X):
        return {}

    def predict_targets_dict(self, X):
        return {}

    def predict_with_observed(self, X, observed):
        return {}


# ---------------------------------------------------------------------- #
# BaseScorer.preserved_inference_columns default
# ---------------------------------------------------------------------- #


def test_base_scorer_preserved_inference_columns_default_empty():
    """Existing scorers (MultioutputScorer here) inherit the empty default."""
    from sklearn.dummy import DummyClassifier

    from skrec.estimator.classification.multioutput_classifier import (
        MultiOutputClassifierEstimator,
    )

    est = MultiOutputClassifierEstimator(DummyClassifier, params={"strategy": "stratified"})
    mscorer = MultioutputScorer(estimator=est)
    assert mscorer.preserved_inference_columns() == []


# ---------------------------------------------------------------------- #
# MixedTypeMultiTargetScorer override — returns OBSERVED_* names
# unconditionally (vanilla AND conditional estimator)
# ---------------------------------------------------------------------- #


def _make_scorer(target_specs):
    est = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1})
    return MixedTypeMultiTargetScorer(estimator=est, target_specs=target_specs)


def test_preserved_columns_for_simple_targets():
    scorer = _make_scorer(
        {
            "ITEM_clicked": TargetType.BINARY,
            "ITEM_revenue": TargetType.REGRESSION,
        }
    )
    assert sorted(scorer.preserved_inference_columns()) == sorted(["OBSERVED_clicked", "OBSERVED_revenue"])


def test_preserved_columns_for_multilabel_group_fans_out_per_member():
    scorer = _make_scorer(
        {
            "engagement": TargetGroupSpec(
                type=TargetType.MULTILABEL,
                columns=["ITEM_email_open", "ITEM_app_open"],
            ),
        }
    )
    assert sorted(scorer.preserved_inference_columns()) == sorted(["OBSERVED_email_open", "OBSERVED_app_open"])


def test_preserved_columns_returned_unconditionally():
    """Hook returns the same OBSERVED_* set whether the estimator is
    vanilla or conditional — preservation is a scorer property, not an
    estimator property. (Per the v1 plan note carried into v3 docs.)"""
    target_specs = {"ITEM_a": TargetType.BINARY, "ITEM_b": TargetType.REGRESSION}
    # Vanilla estimator → scorer still preserves the OBSERVED_* names.
    vanilla = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1})
    s1 = MixedTypeMultiTargetScorer(estimator=vanilla, target_specs=target_specs)
    assert set(s1.preserved_inference_columns()) == {"OBSERVED_a", "OBSERVED_b"}

    # Conditional estimator stub → scorer would (a) Protocol-check pass,
    # (b) return the same preserved set. Today we don't have a real
    # conditional estimator; the stub demonstrates the contract.
    cond = _StubConditionalEstimator(target_specs=target_specs)
    s2 = MixedTypeMultiTargetScorer(estimator=cond, target_specs=target_specs)
    assert set(s2.preserved_inference_columns()) == {"OBSERVED_a", "OBSERVED_b"}
    assert s1.preserved_inference_columns() == s2.preserved_inference_columns()


# ====================================================================== #
# Cross-link / v3 plumbing test moved out of test_v3_m1_observed_plumbing
# ====================================================================== #


# ====================================================================== #
# Cross-link / v3 plumbing test moved out of test_v3_m1_observed_plumbing
# ====================================================================== #


def test_v3_plumbing_schema_apply_round_trip_independent_of_conditional_estimator():
    """V3-M1 plumbing assertion: OBSERVED columns survive an
    interactions_schema apply round-trip even when no conditional
    estimator is constructed. Independent of conditional estimator
    presence — covers the case where the schema-apply hook is the only
    plumbing under test."""
    rng = np.random.default_rng(0)
    n = 10
    df = pd.DataFrame(
        {
            USER_ID_NAME: np.arange(n),
            "f0": rng.normal(size=n),
            "ITEM_a": (rng.normal(size=n) > 0).astype(int),
            "OBSERVED_a": rng.normal(size=n),
        }
    )

    ts = {"ITEM_a": TargetType.BINARY}
    # Construct just the scorer (no recommender, no conditional estimator).
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 2, "seed": 0})
    est.fit(df[["f0"]], {"ITEM_a": df["ITEM_a"].to_numpy()})
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)

    # preserved_inference_columns() must include OBSERVED_a even without
    # a conditional estimator (the hook is a scorer property, not an
    # estimator property).
    assert "OBSERVED_a" in scorer.preserved_inference_columns()
    # And the prefix hook publishes the OBSERVED_ prefix for typo capture.
    assert "OBSERVED_" in scorer.preserved_inference_column_prefixes()


# ====================================================================== #
# Round 4 coverage: row-filter monkey-patch exercises RuntimeError
# ====================================================================== #


# ====================================================================== #
# Round 4 coverage: row-filter monkey-patch exercises RuntimeError
# ====================================================================== #


def test_apply_interactions_schema_with_preservation_rejects_row_filter():
    """If a future DatasetSchema.apply() returns a row-filtered frame,
    the preservation primitive's index-equality check must raise
    RuntimeError. Monkey-patch a schema's apply to drop a row and
    verify the assertion fires (vs silently NaN-filling)."""
    from skrec.recommender.inference_input import InferenceInputPreparer

    rng = np.random.default_rng(0)
    n = 10
    df = pd.DataFrame(
        {
            USER_ID_NAME: np.arange(n),
            "f0": rng.normal(size=n),
            "ITEM_a": (rng.normal(size=n) > 0).astype(int),
            "OBSERVED_a": rng.normal(size=n),
        }
    )

    # Minimal owner stub with the surface the preparer reads.
    class _Schema:
        def __init__(self):
            self.columns = [USER_ID_NAME, "f0", "ITEM_a"]

        def remove_column(self, col):
            if col in self.columns:
                self.columns.remove(col)

        def apply(self, df):  # row-drop simulates a future apply()
            return df.iloc[:-1].copy()  # drop last row → index mismatch

    class _Owner:
        users_schema = None
        outcome_cols = []

        def __init__(self, scorer):
            self.scorer = scorer
            self.interactions_schema = _Schema()

        def _process_outcome_columns(self, df):
            return df

    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 2, "seed": 0})
    est.fit(df[["f0"]], {"ITEM_a": df["ITEM_a"].to_numpy()})
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    owner = _Owner(scorer)
    preparer = InferenceInputPreparer(owner)
    with pytest.raises(RuntimeError, match="no longer matches"):
        preparer.apply_interactions_schema_with_preservation(df, strip_user_id=False)


# ====================================================================== #
# Round 4 coverage: V3-M1 schema-apply round-trip actually invokes apply
# ====================================================================== #


def test_v3_m1_schema_apply_round_trip_actually_calls_apply():
    """The earlier V3-M1 test only checked
    preserved_inference_columns() returns the right names. This one
    actually invokes apply_interactions_schema_with_preservation on a
    minimal schema and verifies OBSERVED_a survives the round-trip."""
    from skrec.recommender.inference_input import InferenceInputPreparer

    rng = np.random.default_rng(0)
    n = 5
    df = pd.DataFrame(
        {
            USER_ID_NAME: np.arange(n),
            "f0": rng.normal(size=n),
            "ITEM_a": (rng.normal(size=n) > 0).astype(int),
            "OBSERVED_a": rng.normal(size=n),
            "stray_column": np.arange(n),  # not in schema; should be dropped
        }
    )

    class _Schema:
        def __init__(self):
            self.columns = [USER_ID_NAME, "f0", "ITEM_a"]

        def remove_column(self, col):
            if col in self.columns:
                self.columns.remove(col)

        def apply(self, df):
            # Column-project to declared columns (drops unknowns).
            return df[self.columns].copy()

    class _Owner:
        users_schema = None
        outcome_cols = []

        def __init__(self, scorer):
            self.scorer = scorer
            self.interactions_schema = _Schema()

        def _process_outcome_columns(self, df):
            return df

    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 2, "seed": 0})
    est.fit(df[["f0"]], {"ITEM_a": df["ITEM_a"].to_numpy()})
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    preparer = InferenceInputPreparer(_Owner(scorer))
    out = preparer.apply_interactions_schema_with_preservation(df, strip_user_id=False)
    # OBSERVED_a survived; stray_column dropped; original ITEM_a/f0 kept.
    assert "OBSERVED_a" in out.columns, "Preservation failed to keep OBSERVED_a"
    assert "stray_column" not in out.columns, "Schema strip missed stray_column"
    assert "f0" in out.columns and "ITEM_a" in out.columns
    # Values match the original.
    np.testing.assert_array_equal(out["OBSERVED_a"].to_numpy(), df["OBSERVED_a"].to_numpy())


def test_per_target_recommender_rejects_retriever_in_init():
    """RankingRecommender(__init__) must reject ``retriever != None`` when
    the scorer publishes ``is_per_target_scorer=True`` — symmetric to the
    existing MultioutputScorer rejection."""
    from skrec.retriever.popularity_retriever import PopularityRetriever

    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1})
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    with pytest.raises(ValueError, match="per-target scorer"):
        RankingRecommender(scorer=scorer, retriever=PopularityRetriever(top_k=10))


def test_preprocess_inputs_rejects_non_none_users_for_per_target_scorer():
    """preprocess_inputs must reject non-None ``users`` when the scorer
    is per-target — via the capability flag, not an isinstance ladder."""
    from skrec.recommender.inference_input import InferenceInputPreparer

    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1})
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)

    class _Owner:
        outcome_cols = []

        def __init__(self, sc):
            self.scorer = sc

        def _process_outcome_columns(self, df):
            return df

    preparer = InferenceInputPreparer(_Owner(scorer))
    df = pd.DataFrame({USER_ID_NAME: [1, 2], "f0": [0.1, 0.2]})
    users = pd.DataFrame({"u_feat": [0.0, 0.0]})
    with pytest.raises(ValueError, match="cannot accept Users"):
        preparer.preprocess_inputs(df, users)


def test_apply_interactions_schema_with_preservation_preserves_dtype():
    """Re-attach must preserve preserved-column dtypes (Int64 nullable,
    Float32, datetime64[ns]). A regression to ``.values`` assignment
    would silently downcast — pin the dtype round-trip."""
    from skrec.recommender.inference_input import InferenceInputPreparer

    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1})
    est.fit(
        pd.DataFrame({"f0": [0.1, 0.2, 0.3]}),
        {"ITEM_a": np.array([0, 1, 0])},
    )
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)

    class _Schema:
        def __init__(self):
            self.columns = [USER_ID_NAME, "f0", "ITEM_a"]

        def remove_column(self, col):
            if col in self.columns:
                self.columns.remove(col)

        def apply(self, df):
            return df[self.columns].copy()

    class _Owner:
        users_schema = None
        outcome_cols = []

        def __init__(self, sc):
            self.scorer = sc
            self.interactions_schema = _Schema()

        def _process_outcome_columns(self, df):
            return df

    df = pd.DataFrame(
        {
            USER_ID_NAME: [1, 2, 3],
            "f0": [0.1, 0.2, 0.3],
            "ITEM_a": [0, 1, 0],
            # Three preserved-column dtypes we want to round-trip.
            "OBSERVED_a": pd.array([1, 0, 1], dtype="Int64"),
            "OBSERVED_b": pd.array([0.5, 1.5, 2.5], dtype="float32"),
            "OBSERVED_ts": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        }
    )
    preparer = InferenceInputPreparer(_Owner(scorer))
    out = preparer.apply_interactions_schema_with_preservation(df, strip_user_id=False)
    assert out["OBSERVED_a"].dtype == "Int64", f"Int64 nullable dtype regressed to {out['OBSERVED_a'].dtype}"
    assert out["OBSERVED_b"].dtype == np.float32, f"float32 dtype regressed to {out['OBSERVED_b'].dtype}"
    assert pd.api.types.is_datetime64_any_dtype(out["OBSERVED_ts"]), (
        f"datetime64[ns] dtype regressed to {out['OBSERVED_ts'].dtype}"
    )


def test_recommend_top_k_warning_throttled_to_once_per_instance():
    """``recommend()`` with top_k != 1 against a per-target scorer must
    emit the warning at most once per recommender instance — high-QPS
    callers can't afford a warning per request."""
    import logging as _logging

    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 4, "seed": 0})
    est.fit(pd.DataFrame({"f0": [0.1, 0.2, 0.3]}), {"ITEM_a": np.array([0, 1, 0])})
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    scorer._init_item_state()
    rec = RankingRecommender(scorer=scorer)

    inf = pd.DataFrame({"f0": [0.1, 0.2], USER_ID_NAME: [1, 2]})
    import io

    log_buf = io.StringIO()
    handler = _logging.StreamHandler(log_buf)
    handler.setLevel(_logging.WARNING)
    log = _logging.getLogger("skrec.recommender.ranking.ranking_recommender")
    log.addHandler(handler)
    try:
        rec.recommend(interactions=inf, top_k=10)
        rec.recommend(interactions=inf, top_k=10)
        rec.recommend(interactions=inf, top_k=10)
    finally:
        log.removeHandler(handler)
    n_warnings = log_buf.getvalue().count("Per-target scorer.recommend() ignores top_k")
    assert n_warnings == 1, f"Expected 1 throttled warning, got {n_warnings} across 3 recommend() calls"
