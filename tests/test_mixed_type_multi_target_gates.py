# Verification gates 1-8 for the v2 mixed-type multi-target work.
#
# Inline tests in individual test files already exercise most contract
# surfaces; this file collects the verification gates from the v2 plan as
# named tests so a single pytest filter can confirm the gate set passes:
#
#   gate 1: Protocol gate (all 3 families satisfy MultiTargetEstimator)
#   gate 2: Encoder Protocol gate (both joint encoders satisfy
#           JointMultiTargetEncoder; same hidden_dim for same input)
#   gate 3: Default-sanity gate (each estimator with default params beats
#           random by a clear margin on a 4-target synthetic)
#   gate 5: Family-equivalence smoke (joint vs independent return-shape +
#           dispatch-key equivalence on the same target_specs)
#   gate 6: Evaluation correctness (per-target sklearn-equivalence)
#   gate 7: Dispatch-table consistency (in-code constant matches
#           capability_matrix and any human-canonical doc table)
#   gate 8: Agent-surface (capability_matrix() JSON-serializable, every
#           documented key present)
#
# Gate 4 (loss-curve visual) lands in the M9 notebook; not a unit test.
# Mandatory leakage test (gate 2 in v1) returns in v3 when conditional
# estimators land.

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

torch = pytest.importorskip("torch")

from skrec.constants import USER_ID_NAME  # noqa: E402
from skrec.estimator.classification import (  # noqa: E402
    IndependentMultiTargetEstimator,
    JointMultiTargetMLPEstimator,
    JointMultiTargetTransformerEstimator,
    MultiTargetEstimator,
)
from skrec.estimator.classification._joint_multi_target_base import (  # noqa: E402
    JointMultiTargetEncoder,
)
from skrec.estimator.classification._joint_multi_target_encoders import (  # noqa: E402
    MLPEncoder,
    TransformerEncoder,
)
from skrec.estimator.classification.lightgbm_classifier import (  # noqa: E402
    LightGBMClassifierEstimator,
)
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator  # noqa: E402
from skrec.estimator.regression.lightgbm_regressor import LightGBMRegressorEstimator  # noqa: E402
from skrec.evaluator.datatypes import RecommenderEvaluatorType  # noqa: E402
from skrec.metrics.datatypes import RecommenderMetricType  # noqa: E402
from skrec.orchestrator import capability_matrix  # noqa: E402
from skrec.orchestrator.factory import _INDEPENDENT_TARGET_COMPAT  # noqa: E402
from skrec.recommender.ranking.ranking_recommender import RankingRecommender  # noqa: E402
from skrec.scorer.mixed_type_multi_target import (  # noqa: E402
    TARGET_TYPE_TO_METRICS,
    MixedTypeMultiTargetScorer,
    TargetType,
)

SEED = 1234


def _make_separable(n=300, seed=SEED):
    """Synthetic with clear per-target signal — used by sanity gates."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"feat_{i}" for i in range(5)])
    # Binary target: monotone in feat_0
    y_bin = (X["feat_0"] > 0).astype(int).to_numpy()
    # Regression target: linear in feat_1 + small noise
    y_reg = (3.0 * X["feat_1"] + rng.normal(scale=0.05, size=n)).to_numpy()
    # Multiclass: argmax of three columns
    y_mc_idx = np.column_stack([X["feat_2"], X["feat_3"], X["feat_4"]]).argmax(axis=1)
    y_mc = np.array(["A", "B", "C"])[y_mc_idx]
    target_specs = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
        "ITEM_action": TargetType.MULTICLASS,
    }
    y = {
        "ITEM_clicked": y_bin,
        "ITEM_revenue": y_reg,
        "ITEM_action": y_mc,
    }
    return X, y, target_specs


# ---------------------------------------------------------------------- #
# Gate 1: Protocol gate — all 3 families satisfy MultiTargetEstimator
# ---------------------------------------------------------------------- #


def test_gate1_protocol_all_three_families():
    target_specs = {"ITEM_a": TargetType.BINARY}
    joint_mlp = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1})
    joint_tx = JointMultiTargetTransformerEstimator(target_specs=target_specs, params={"epochs": 1})
    independent = IndependentMultiTargetEstimator(
        target_specs=target_specs,
        estimators={"ITEM_a": XGBClassifierEstimator(params={"n_estimators": 5})},
    )
    for est in (joint_mlp, joint_tx, independent):
        assert isinstance(est, MultiTargetEstimator)


# ---------------------------------------------------------------------- #
# Gate 2: Encoder Protocol gate
# ---------------------------------------------------------------------- #


def test_gate2_encoder_protocol_both_satisfy():
    mlp_enc = MLPEncoder(n_features=4, hidden_dim=32, num_layers=2)
    tx_enc = TransformerEncoder(n_features=4, d_model=32, n_heads=4, n_layers=1, ffn_dim=32)
    assert isinstance(mlp_enc, JointMultiTargetEncoder)
    assert isinstance(tx_enc, JointMultiTargetEncoder)


def test_gate2_encoder_hidden_dim_consistent():
    mlp_enc = MLPEncoder(n_features=4, hidden_dim=32, num_layers=2)
    tx_enc = TransformerEncoder(n_features=4, d_model=32, n_heads=4, n_layers=1, ffn_dim=32)
    x = torch.randn(8, 4)
    h_mlp = mlp_enc(x)
    h_tx = tx_enc(x)
    # Both produce (batch, hidden_dim).
    assert h_mlp.shape == (8, 32)
    assert h_tx.shape == (8, 32)


# ---------------------------------------------------------------------- #
# Gate 3: Default-sanity gate — joint families train + beat random
# ---------------------------------------------------------------------- #


def test_gate3_joint_mlp_beats_random_on_separable_synthetic():
    X, y, target_specs = _make_separable(n=300)
    est = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 5, "hidden_dim": 32, "num_layers": 2, "batch_size": 64, "seed": SEED},
    )
    est.fit(X, y)
    preds = est.predict_targets_dict(X)
    proba = est.predict_proba_dict(X)
    # Binary AUC > 0.6
    assert roc_auc_score(y["ITEM_clicked"], proba["ITEM_clicked"][:, 1]) > 0.6
    # Regression: MAE should be a reasonable fraction of the target's scale.
    mae = mean_absolute_error(y["ITEM_revenue"], preds["ITEM_revenue"])
    assert mae < float(np.std(y["ITEM_revenue"]))
    # Multiclass top-1 > random (1/3 + margin)
    assert accuracy_score(y["ITEM_action"], preds["ITEM_action"]) > 0.45


def test_gate3_independent_beats_random_on_separable_synthetic():
    X, y, target_specs = _make_separable(n=300)
    est = IndependentMultiTargetEstimator(
        target_specs=target_specs,
        estimators={
            "ITEM_clicked": XGBClassifierEstimator(params={"n_estimators": 30, "max_depth": 3}),
            "ITEM_revenue": LightGBMRegressorEstimator(params={"n_estimators": 50, "verbose": -1}),
            "ITEM_action": LightGBMClassifierEstimator(params={"n_estimators": 30, "verbose": -1}),
        },
    )
    est.fit(X, y)
    preds = est.predict_targets_dict(X)
    proba = est.predict_proba_dict(X)
    assert roc_auc_score(y["ITEM_clicked"], proba["ITEM_clicked"][:, 1]) > 0.6
    mae = mean_absolute_error(y["ITEM_revenue"], preds["ITEM_revenue"])
    assert mae < float(np.std(y["ITEM_revenue"]))
    assert accuracy_score(y["ITEM_action"], preds["ITEM_action"]) > 0.45


# ---------------------------------------------------------------------- #
# Gate 5: Family-equivalence smoke — same target_specs → same output schema
# ---------------------------------------------------------------------- #


def test_gate5_joint_vs_independent_same_predict_targets_columns():
    X, y, target_specs = _make_separable(n=80)
    joint = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 1, "hidden_dim": 16, "num_layers": 1},
    )
    joint.fit(X, y)
    independent = IndependentMultiTargetEstimator(
        target_specs=target_specs,
        estimators={
            "ITEM_clicked": XGBClassifierEstimator(params={"n_estimators": 5}),
            "ITEM_revenue": LightGBMRegressorEstimator(params={"n_estimators": 5, "verbose": -1}),
            "ITEM_action": LightGBMClassifierEstimator(params={"n_estimators": 5, "verbose": -1}),
        },
    )
    independent.fit(X, y)
    pj = joint.predict_targets_dict(X)
    pi = independent.predict_targets_dict(X)
    assert set(pj.keys()) == set(pi.keys())
    pj_proba = joint.predict_proba_dict(X)
    pi_proba = independent.predict_proba_dict(X)
    assert set(pj_proba.keys()) == set(pi_proba.keys())


# ---------------------------------------------------------------------- #
# Gate 6: Evaluation correctness vs sklearn ground truth
# ---------------------------------------------------------------------- #


def test_gate6_eval_matches_sklearn_for_binary_and_regression():
    X, y, target_specs = _make_separable(n=120)
    est = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 2, "hidden_dim": 16, "num_layers": 1, "seed": SEED},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=target_specs)
    recommender = RankingRecommender(scorer=scorer)

    inf_df = X.copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(120)])
    logged = pd.DataFrame(
        {
            "ITEM_clicked": y["ITEM_clicked"],
            "ITEM_revenue": y["ITEM_revenue"],
            "ITEM_action": y["ITEM_action"],
        }
    )
    result = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type={
            "ITEM_clicked": RecommenderMetricType.ROC_AUC,
            "ITEM_revenue": RecommenderMetricType.MAE,
            "ITEM_action": RecommenderMetricType.MULTICLASS_ACCURACY,
        },
        eval_top_k=10,
        score_items_kwargs={"interactions": inf_df},
        eval_kwargs={"logged_rewards": logged},
    )

    # Independent sklearn computation against the scorer's predictions.
    proba_dict = est.predict_proba_dict(X)
    preds_dict = est.predict_targets_dict(X)
    expected_auc = roc_auc_score(y["ITEM_clicked"], proba_dict["ITEM_clicked"][:, 1])
    expected_mae = mean_absolute_error(y["ITEM_revenue"], preds_dict["ITEM_revenue"])
    # Multiclass uses the scorer's class catalogue mapping; reproduce that.
    classes = est._multiclass_classes["ITEM_action"]
    label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
    true_idx = np.array([label_to_idx[v] for v in y["ITEM_action"]])
    expected_acc = accuracy_score(true_idx, proba_dict["ITEM_action"].argmax(axis=1))

    assert result["ITEM_clicked"] == pytest.approx(expected_auc, abs=1e-6)
    assert result["ITEM_revenue"] == pytest.approx(expected_mae, abs=1e-6)
    assert result["ITEM_action"] == pytest.approx(expected_acc, abs=1e-6)


# ---------------------------------------------------------------------- #
# Gate 7: Dispatch-table consistency — in-code constant matches capability_matrix
# ---------------------------------------------------------------------- #


def test_gate7_target_type_metric_compat_matches_constant():
    cm = capability_matrix()
    for tt in TargetType:
        expected = TARGET_TYPE_TO_METRICS[tt]
        actual = cm["target_type_metric_compat"][tt.value]
        assert actual == expected, (
            f"capability_matrix['target_type_metric_compat'][{tt.value!r}] = "
            f"{actual}, but TARGET_TYPE_TO_METRICS[{tt}] = {expected}"
        )


def test_gate7_independent_target_compat_matches_constant():
    cm = capability_matrix()
    for tt in TargetType:
        expected = tuple(sorted(_INDEPENDENT_TARGET_COMPAT[tt]))
        actual = cm["independent_target_compat"][tt.value]
        assert actual == expected


def test_gate7_decision_rule_doc_table_matches_constant():
    """Third leg of the dispatch-table consistency gate (v2 plan).

    Parses the canonical metric-dispatch table out of
    ``docs/user-guide/decision-rule.md`` and asserts it agrees with the
    in-code ``TARGET_TYPE_TO_METRICS`` constant (and therefore, by gate-7's
    other two assertions, with ``capability_matrix()`` too). Catches doc
    drift — the table is the only human-canonical statement of the
    metric-dispatch rules, and silently letting it drift defeats the
    "single source of truth" guarantee.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    doc_path = repo_root / "docs" / "user-guide" / "decision-rule.md"
    assert doc_path.exists(), f"Expected canonical table at {doc_path}"
    text = doc_path.read_text(encoding="utf-8")

    # The table lives under a header; we find its block and parse rows
    # that look like:  | `BINARY` | `ROC_AUC`, `PR_AUC` | <notes> |
    # Each row maps a TargetType (uppercase, backticked) to a backticked
    # comma-separated list of RecommenderMetricType names (uppercase).
    # MULTILABEL row is suffixed " member" in the doc — strip and treat
    # as MULTILABEL.
    row_pattern = re.compile(
        r"^\|\s*`([A-Z_]+)`(?:\s+member)?\s*\|\s*((?:`[A-Z_]+`(?:\s*,\s*)?)+)\s*\|",
        re.MULTILINE,
    )
    parsed: dict[str, tuple[str, ...]] = {}
    for m in row_pattern.finditer(text):
        target_type_str = m.group(1)
        metrics_blob = m.group(2)
        metric_names = re.findall(r"`([A-Z_]+)`", metrics_blob)
        # RecommenderMetricType members are uppercase; values are lowercase.
        # The doc table uses enum-NAME notation (e.g. ROC_AUC) — map to
        # enum.value via RecommenderMetricType[NAME].value.
        from skrec.metrics.datatypes import RecommenderMetricType

        try:
            metric_values = tuple(RecommenderMetricType[n].value for n in metric_names)
        except KeyError as e:
            raise AssertionError(
                f"decision-rule.md references unknown RecommenderMetricType "
                f"name in row {target_type_str!r}: {e}. Either the doc or "
                f"the enum is stale."
            )
        # Map TargetType doc label → enum value string.
        try:
            tt_value = TargetType[target_type_str].value
        except KeyError as e:
            raise AssertionError(
                f"decision-rule.md references unknown TargetType: {e}. Either the doc or the enum is stale."
            )
        parsed[tt_value] = metric_values

    # All four TargetType values must be in the parsed table.
    missing = {tt.value for tt in TargetType} - set(parsed.keys())
    assert not missing, f"decision-rule.md table missing rows for: {sorted(missing)}. Parsed: {sorted(parsed.keys())}"

    # And each row must match TARGET_TYPE_TO_METRICS exactly.
    for tt in TargetType:
        expected = TARGET_TYPE_TO_METRICS[tt]
        actual = parsed[tt.value]
        assert actual == expected, (
            f"decision-rule.md metric dispatch table row for {tt.value!r} "
            f"has {actual}, but TARGET_TYPE_TO_METRICS[{tt}] = {expected}. "
            f"Doc drift — update the table OR the constant; gate 7 requires "
            f"both human-canonical and code sources to agree."
        )


# ---------------------------------------------------------------------- #
# Gate 8: Agent-surface — capability_matrix JSON-serializable
# ---------------------------------------------------------------------- #


def test_gate8_capability_matrix_json_serializable():
    """Gate 8: capability_matrix() must JSON-encode without custom encoders.

    The agent layer streams capability metadata through tool envelopes that
    use stdlib json.dumps — no tuple-to-list normalization, no default=
    fallback. Test the bare json.dumps(cm) so we catch any future addition
    of a non-JSON-serializable value (e.g. an enum member, a set, a Path).
    """
    cm = capability_matrix()
    encoded = json.dumps(cm)
    assert "mixed_type_multi_target" in encoded
    assert "target_type_metric_compat" in encoded
    assert "independent_target_compat" in encoded
    assert "scorer_supports_observed_conditioning" in encoded


def test_gate8_agent_surface_required_keys_present():
    cm = capability_matrix()
    required = {
        "scorer_types",
        "multi_target_model_types",
        "target_types",
        "target_type_metric_compat",
        "independent_target_compat",
        "scorer_supports_observed_conditioning",
        "scorer_config_keys",
        "metric_types",
        "evaluator_types",
    }
    missing = required - set(cm.keys())
    assert not missing, f"capability_matrix missing required keys: {missing}"
    # mixed_type_multi_target scorer entry is present and lists target_specs.
    assert "mixed_type_multi_target" in cm["scorer_config_keys"]
    assert "target_specs" in cm["scorer_config_keys"]["mixed_type_multi_target"]
    # MULTICLASS_ACCURACY metric is in the metric enum.
    assert "multiclass_accuracy" in cm["metric_types"]


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
# Joint MLP / Transformer v2-list #2: per-type metric sanity gates
# ====================================================================== #


@pytest.mark.parametrize(
    "estimator_cls,params",
    [
        (JointMultiTargetMLPEstimator, {"epochs": 30, "hidden_dim": 32, "num_layers": 2, "batch_size": 64, "seed": 0}),
        (
            JointMultiTargetTransformerEstimator,
            {"epochs": 30, "d_model": 32, "n_heads": 4, "n_layers": 2, "ffn_dim": 64, "batch_size": 64, "seed": 0},
        ),
    ],
    ids=["mlp", "transformer"],
)
def test_joint_2_per_type_metric_sanity_gate(estimator_cls, params):
    """Per-target metric thresholds the v2 plan asks for:
       BINARY ROC AUC > 0.6, REGRESSION RMSE drops vs predict-mean,
       MULTICLASS accuracy > random + 0.1, MULTILABEL ROC AUC > 0.6.
    Stronger than the monotonic-loss proxy already in the test suite."""
    from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

    df, target_specs = _make_mixed_df(n=400)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols]
    y = {
        "ITEM_bin": df["ITEM_bin"].to_numpy(),
        "ITEM_rev": df["ITEM_rev"].to_numpy(),
        "ITEM_class": df["ITEM_class"].to_numpy(),
        "g": df[["ITEM_email", "ITEM_app"]].to_numpy(),
    }

    est = estimator_cls(target_specs=target_specs, params=params)
    est.fit(X, y)
    proba = est.predict_proba_dict(X)
    targets = est.predict_targets_dict(X)

    # BINARY ROC AUC > 0.6 (loose floor; per-type sanity not perfection).
    bin_auc = roc_auc_score(y["ITEM_bin"], proba["ITEM_bin"][:, 1])
    assert bin_auc > 0.6, f"ITEM_bin AUC={bin_auc} below 0.6 sanity floor."

    # REGRESSION RMSE drops below predict-mean baseline.
    rmse_pred = float(np.sqrt(mean_squared_error(y["ITEM_rev"], proba["ITEM_rev"])))
    rmse_mean = float(np.sqrt(mean_squared_error(y["ITEM_rev"], np.full_like(y["ITEM_rev"], y["ITEM_rev"].mean()))))
    assert rmse_pred < rmse_mean * 0.95, (
        f"ITEM_rev RMSE={rmse_pred} not meaningfully below predict-mean baseline {rmse_mean}"
    )

    # MULTICLASS accuracy > 1/K + 0.1 (3 classes → 1/3 random; require > 0.43).
    acc = float(accuracy_score(y["ITEM_class"], targets["ITEM_class"]))
    assert acc > 1.0 / 3.0 + 0.1, f"ITEM_class accuracy={acc} not better than random+0.1."

    # MULTILABEL: each member's ROC AUC > 0.6.
    email_auc = roc_auc_score(df["ITEM_email"], proba["ITEM_email"][:, 1])
    app_auc = roc_auc_score(df["ITEM_app"], proba["ITEM_app"][:, 1])
    assert email_auc > 0.6 and app_auc > 0.6, f"Multilabel AUCs {email_auc=}, {app_auc=} below 0.6 floor."
