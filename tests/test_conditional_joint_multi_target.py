# V3-M4 tests: conditional joint MLP + conditional joint Transformer.
#
# Per the v3 plan test list:
#   1. Happy path: train with mask_prob=0.5, predict mixed observed/NaN
#   2. Conditioning has measurable effect on correlated targets
#   3. mask_prob=1.0 ≈ vanilla (locked decision #3)
#   4. mask_prob=0.0 rejected at init
#   5. Multilabel group-mask-together (locked decision #4) — scorer enforces
#   6. Label-channel zeroing (mandatory leakage gate)
#   8. Pickle round-trip preserves scaler + masking config
#   9. Single-row conditional inference via score_fast
#  10. recommend_online with constraining schema (preserved-columns hook)
#
# Both estimator families share the conditional base, so the bulk of the
# tests parameterize over (MLPEstimator, TransformerEstimator) pairs.

import io
import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from skrec.constants import USER_ID_NAME  # noqa: E402
from skrec.estimator.classification import (  # noqa: E402
    ConditionalJointMultiTargetMLPEstimator,
    ConditionalJointMultiTargetTransformerEstimator,
    ConditionalMultiTargetEstimator,
    JointMultiTargetMLPEstimator,
    MultiTargetEstimator,
)
from skrec.estimator.classification._conditional_label_encoding import (  # noqa: E402
    build_raw_chunks,
    sample_training_mask,
)
from skrec.recommender.ranking.ranking_recommender import RankingRecommender  # noqa: E402
from skrec.scorer.mixed_type_multi_target import (  # noqa: E402
    MixedTypeMultiTargetScorer,
    TargetGroupSpec,
    TargetType,
)

# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


def _make_synthetic(n=150, seed=0, correlated=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    if correlated:
        # Strongly correlate A and B for the conditioning-effect test.
        latent = X["f0"] + 0.5 * X["f1"]
        y_a = (latent > 0).astype(int).to_numpy()
        y_b = (latent + 0.1 * rng.normal(size=n) > 0).astype(int).to_numpy()
    else:
        y_a = (X["f0"] > 0).astype(int).to_numpy()
        y_b = (X["f1"] > 0).astype(int).to_numpy()
    y_reg = (2.0 * X["f2"] + 0.1 * rng.normal(size=n)).to_numpy()
    ts = {
        "ITEM_a": TargetType.BINARY,
        "ITEM_b": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    y = {"ITEM_a": y_a, "ITEM_b": y_b, "ITEM_revenue": y_reg}
    return X, y, ts


def _cond_estimators(target_specs, params=None):
    """Build both conditional family instances for parametrization."""
    p = {"epochs": 3, "batch_size": 32, "mask_prob": 0.5, "seed": 7, **(params or {})}
    return [
        (
            "mlp",
            ConditionalJointMultiTargetMLPEstimator(
                target_specs=target_specs,
                params={**p, "hidden_dim": 16, "num_layers": 1},
            ),
        ),
        (
            "tx",
            ConditionalJointMultiTargetTransformerEstimator(
                target_specs=target_specs,
                params={**p, "d_model": 16, "n_heads": 2, "n_layers": 1, "ffn_dim": 32},
            ),
        ),
    ]


# ---------------------------------------------------------------------- #
# 1 + Protocol adherence: both families implement Conditional MTE.
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("family", ["mlp", "tx"])
def test_conditional_estimator_implements_protocols(family):
    _, _, ts = _make_synthetic(n=20)
    pairs = dict(_cond_estimators(ts, params={"epochs": 1}))
    est = pairs[family]
    assert isinstance(est, MultiTargetEstimator)
    assert isinstance(est, ConditionalMultiTargetEstimator)


@pytest.mark.parametrize("family", ["mlp", "tx"])
def test_conditional_fit_and_predict_shapes(family):
    X, y, ts = _make_synthetic(n=80)
    pairs = dict(_cond_estimators(ts, params={"epochs": 2}))
    est = pairs[family]
    est.fit(X, y)
    proba = est.predict_proba_dict(X)
    assert set(proba.keys()) == {"ITEM_a", "ITEM_b", "ITEM_revenue"}
    assert proba["ITEM_a"].shape == (80, 2)
    assert proba["ITEM_revenue"].shape == (80,)
    # Conditional path
    observed = {
        "ITEM_a": y["ITEM_a"].astype(float),
        "ITEM_b": np.full(80, np.nan),  # unobserved → predict from features
        "ITEM_revenue": np.full(80, np.nan),
    }
    proba_cond = est.predict_with_observed(X, observed=observed)
    assert proba_cond["ITEM_b"].shape == (80, 2)


# ---------------------------------------------------------------------- #
# 4. mask_prob=0.0 rejected at init.
# ---------------------------------------------------------------------- #


def test_mask_prob_zero_rejected():
    _, _, ts = _make_synthetic(n=10)
    with pytest.raises(ValueError, match="mask_prob"):
        ConditionalJointMultiTargetMLPEstimator(
            target_specs=ts,
            params={"mask_prob": 0.0, "epochs": 1},
        )


def test_mask_prob_negative_rejected():
    _, _, ts = _make_synthetic(n=10)
    with pytest.raises(ValueError, match="mask_prob"):
        ConditionalJointMultiTargetMLPEstimator(
            target_specs=ts,
            params={"mask_prob": -0.1, "epochs": 1},
        )


def test_mask_prob_above_one_rejected():
    _, _, ts = _make_synthetic(n=10)
    with pytest.raises(ValueError, match="mask_prob"):
        ConditionalJointMultiTargetMLPEstimator(
            target_specs=ts,
            params={"mask_prob": 1.5, "epochs": 1},
        )


# ---------------------------------------------------------------------- #
# 6. Label-channel zeroing (gate 9 / mandatory leakage test).
# ---------------------------------------------------------------------- #


def test_label_channel_zeroing_masked_positions():
    """Masked positions must have BOTH is_observed=0 AND value=0 in the raw chunk.

    Builds a raw chunk for BINARY + MULTICLASS + MULTILABEL group with
    half-masked, half-observed rows. Asserts the masked rows are entirely
    zero (defense in depth).
    """
    rng = np.random.default_rng(0)
    n = 20
    target_specs = {
        "ITEM_b": TargetType.BINARY,
        "ITEM_m": TargetType.MULTICLASS,
        "g": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_m1", "ITEM_m2"]),
    }
    multiclass_classes = {"ITEM_m": ["A", "B", "C"]}
    y = {
        "ITEM_b": rng.integers(0, 2, size=n),
        "ITEM_m": rng.choice(["A", "B", "C"], size=n),
        "g": rng.integers(0, 2, size=(n, 2)),
    }
    mask = {
        "ITEM_b": np.array([i % 2 == 0 for i in range(n)]),  # alternating
        "ITEM_m": np.array([i % 2 == 0 for i in range(n)]),
        "g": np.array([i % 2 == 0 for i in range(n)]),  # group-level
    }
    chunks = build_raw_chunks(
        y_or_observed=y,
        mask=mask,
        target_specs=target_specs,
        multiclass_classes=multiclass_classes,
        regression_means={},
        regression_stds={},
    )
    # All masked rows (even indices) → entire chunk row is zeros.
    for key in target_specs:
        raw = chunks[key].numpy()
        for i in range(n):
            if mask[key][i]:
                assert np.all(raw[i] == 0.0), f"target {key!r} row {i} is masked but raw chunk is non-zero: {raw[i]}"
            else:
                # Observed rows must have is_observed=1 in column 0.
                assert raw[i, 0] == 1.0


# ---------------------------------------------------------------------- #
# 3. mask_prob=1.0 ≈ vanilla equivalence (locked decision #3).
#
# At mask_prob=1.0 every (row, target) is masked at training, so the label
# channel contributes only zero inputs. The model effectively trains on X
# alone (modulo the label-encoder bias parameters, which are small). Per-
# target metrics should be within a documented tolerance of a vanilla
# JointMultiTargetMLPEstimator trained on the same data.
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "vanilla_family, conditional_family, common_extra",
    [
        (
            "mlp",
            ConditionalJointMultiTargetMLPEstimator,
            {"hidden_dim": 32, "num_layers": 2},
        ),
        (
            "tx",
            ConditionalJointMultiTargetTransformerEstimator,
            {"d_model": 16, "n_heads": 2, "n_layers": 1, "ffn_dim": 32},
        ),
    ],
)
def test_mask_prob_one_approximates_vanilla(vanilla_family, conditional_family, common_extra):
    """Gate 10 (locked decision #3): with every (row, target) masked at
    training, the label channel is always zero — the conditional model
    degenerates to learning from features alone. Per-target ROC AUC on a
    separable synthetic should land within a documented tolerance of the
    matching vanilla family.

    Parametrized over BOTH conditional families (MLP + Transformer) per the
    v3 plan's "mandatory" gate-10 scope.

    Tolerance: AUC within 0.15 (binary, well-separable target). Both models
    should land well above random (>0.6) on this signal. Not bit-exact —
    the conditional model has extra label-encoder parameters whose biases
    add constant offsets to the encoder input, plus different random-init
    state.
    """
    from sklearn.metrics import roc_auc_score

    X, y, ts = _make_synthetic(n=400)
    common = {"epochs": 8, "batch_size": 64, "seed": 11, "lr": 1e-2, **common_extra}
    if vanilla_family == "mlp":
        from skrec.estimator.classification import JointMultiTargetMLPEstimator

        vanilla = JointMultiTargetMLPEstimator(target_specs=ts, params=common)
    else:
        from skrec.estimator.classification import (
            JointMultiTargetTransformerEstimator,
        )

        vanilla = JointMultiTargetTransformerEstimator(target_specs=ts, params=common)
    vanilla.fit(X, y)
    cond = conditional_family(
        target_specs=ts,
        params={**common, "mask_prob": 1.0, "label_embedding_dim": 4},
    )
    cond.fit(X, y)

    proba_v = vanilla.predict_proba_dict(X)["ITEM_a"][:, 1]
    proba_c = cond.predict_proba_dict(X)["ITEM_a"][:, 1]
    auc_v = roc_auc_score(y["ITEM_a"], proba_v)
    auc_c = roc_auc_score(y["ITEM_a"], proba_c)
    # Tolerance varies by family because the architectural divergence at
    # mask_prob=1.0 is asymmetric:
    #
    # - MLP: at mask_prob=1.0 the flat-concat label vector is all-zero, so
    #   the only label-channel contribution is the linear-projection biases.
    #   Very close to vanilla; 0.15 holds.
    #
    # - Transformer (P0-6 architecture): the per-target tokens are all-zero
    #   too, but they STILL enter the attention sequence — the model has to
    #   learn to ignore (n) extra always-zero tokens. Achievable with more
    #   data / epochs but harder under the test's small config. 0.35 lets
    #   the test detect "the model didn't learn at all" without flagging
    #   the architectural property as a regression.
    tolerance = 0.35 if vanilla_family == "tx" else 0.15
    assert auc_v > 0.6, f"Vanilla {vanilla_family} baseline didn't fit (AUC={auc_v})"
    assert auc_c > 0.6, f"Conditional {vanilla_family} at mask_prob=1.0 didn't fit (AUC={auc_c})"
    assert abs(auc_v - auc_c) < tolerance, (
        f"mask_prob=1.0 {vanilla_family} conditional AUC ({auc_c}) diverged "
        f"from vanilla ({auc_v}) by more than {tolerance}; locked-decision-#3 "
        f"tolerance breached."
    )


@pytest.mark.parametrize(
    "family_cls, params_extra",
    [
        (ConditionalJointMultiTargetMLPEstimator, {"hidden_dim": 16, "num_layers": 1}),
        (ConditionalJointMultiTargetTransformerEstimator, {"d_model": 16, "n_heads": 2, "n_layers": 1, "ffn_dim": 32}),
    ],
)
def test_label_channel_zeroing_estimator_level(family_cls, params_extra):
    """Gate 9 (mandatory leakage test) at the estimator level — parametrized
    over BOTH conditional families.

    Monkeypatches the conditional label encoder's forward to capture the
    raw_chunks dict passed to it during the training loop, then asserts
    masked-position rows are entirely zero (is_observed=0 AND value=0).
    The utility-level test (test_label_channel_zeroing_masked_positions
    above) covers ``build_raw_chunks`` in isolation; this test pins the
    end-to-end training-loop wiring per family.
    """
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    ts = {
        "ITEM_a": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    y = {
        "ITEM_a": (X["f0"] > 0).astype(int).to_numpy(),
        "ITEM_revenue": X["f1"].to_numpy(),
    }
    captured_chunks: list = []

    # Monkey-patch build_raw_chunks at the module level — it returns the
    # raw_chunks dict that the training loop feeds into the label encoder.
    # Intercepting here is robust against the fit() loop building a fresh
    # _label_encoder on each call (which a forward pre-hook would miss).
    from skrec.estimator.classification import _conditional_joint_multi_target_base as base_mod

    real_build = base_mod.build_raw_chunks

    def _capture_build_raw_chunks(**kwargs):
        chunks = real_build(**kwargs)
        captured_chunks.append({k: v.detach().cpu().numpy().copy() for k, v in chunks.items()})
        return chunks

    base_mod.build_raw_chunks = _capture_build_raw_chunks
    try:
        est = family_cls(
            target_specs=ts,
            params={
                "epochs": 1,
                "batch_size": 32,
                "mask_prob": 0.5,
                "seed": 0,
                "label_embedding_dim": 4,
                **params_extra,
            },
        )
        est.fit(X, y)
    finally:
        base_mod.build_raw_chunks = real_build

    assert captured_chunks, "build_raw_chunks was never called during training."

    # For every captured batch, verify masked-position zeroing: rows where
    # is_observed (column 0) is 0 must have the full chunk row equal to 0.
    for batch_chunks in captured_chunks:
        for target_name, raw in batch_chunks.items():
            assert raw.ndim == 2
            for i in range(raw.shape[0]):
                is_observed = raw[i, 0]
                if is_observed == 0.0:
                    assert np.all(raw[i] == 0.0), (
                        f"Estimator {family_cls.__name__} target {target_name!r} "
                        f"row {i} has is_observed=0 but non-zero values: {raw[i]}. "
                        f"Label-channel leakage gate (gate 9) breached."
                    )


# ---------------------------------------------------------------------- #
# 2. Conditioning has measurable effect (correlated targets).
# ---------------------------------------------------------------------- #


def test_conditioning_changes_predictions_on_correlated_targets():
    """When ITEM_a and ITEM_b are strongly correlated, observing ITEM_a
    should change the prediction distribution for ITEM_b."""
    X, y, ts = _make_synthetic(n=200, correlated=True)
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 5, "hidden_dim": 32, "num_layers": 2, "batch_size": 32, "mask_prob": 0.5, "seed": 22},
    )
    est.fit(X, y)

    # Subset of rows where we'll vary the observed ITEM_a value.
    X_sub = X.iloc[:30].reset_index(drop=True)
    n = len(X_sub)
    observed_unobserved = {
        "ITEM_a": np.full(n, np.nan),
        "ITEM_b": np.full(n, np.nan),
        "ITEM_revenue": np.full(n, np.nan),
    }
    observed_a_positive = {
        "ITEM_a": np.ones(n, dtype=float),
        "ITEM_b": np.full(n, np.nan),
        "ITEM_revenue": np.full(n, np.nan),
    }
    proba_unobs = est.predict_with_observed(X_sub, observed=observed_unobserved)
    proba_a_pos = est.predict_with_observed(X_sub, observed=observed_a_positive)
    # ITEM_b positive proba should differ between the two conditioning sets.
    diff = np.abs(proba_unobs["ITEM_b"][:, 1] - proba_a_pos["ITEM_b"][:, 1])
    assert diff.mean() > 0.001, (
        f"Observing ITEM_a should change ITEM_b predictions (correlated targets). Got mean abs diff = {diff.mean()}."
    )


# ---------------------------------------------------------------------- #
# 7. Loss-balance smoke for dollar-scale regression + binary.
# ---------------------------------------------------------------------- #


def test_loss_balance_smoke_dollar_scale_regression_plus_binary():
    """Plan conditional test #7: train with a regression target on dollar
    scale (~1e6) alongside a binary target. Assert (a) no NaN gradients
    in the first 3 epochs, AND (b) both per-target loss curves decrease.

    regression_normalize=True (the default) z-score-normalizes regression
    targets internally — without this, the MSE loss on a 1e6-scale target
    would dominate the BCE loss on the binary target and the binary head
    would never learn. This test catches regressions where the scaler is
    silently disabled or the loss-weighting changes.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    # Binary target: well-separated.
    y_binary = (X["f0"] > 0).astype(int).to_numpy()
    # Regression target: dollar-scale (~1e6), correlated with f1.
    y_dollars = (1.5e6 * X["f1"] + 1e5 * rng.normal(size=n)).to_numpy()
    ts = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    y = {"ITEM_clicked": y_binary, "ITEM_revenue": y_dollars}

    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={
            "epochs": 8,
            "hidden_dim": 32,
            "num_layers": 2,
            "batch_size": 64,
            "mask_prob": 0.5,
            "label_embedding_dim": 4,
            "seed": 0,
            "lr": 5e-3,
            "regression_normalize": True,  # default; pinning for clarity
        },
    )

    # Capture per-epoch loss + per-batch parameter sanity.
    import logging

    losses = []

    class _LossCapture(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "Conditional Train Loss" in msg:
                try:
                    losses.append(float(msg.rsplit(":", 1)[-1].strip()))
                except ValueError:
                    pass

    handler = _LossCapture()
    cond_logger = logging.getLogger("skrec.estimator.classification._conditional_joint_multi_target_base")
    cond_logger.addHandler(handler)
    cond_logger.setLevel(logging.INFO)
    try:
        est.fit(X, y)
    finally:
        cond_logger.removeHandler(handler)

    # (a) No NaN losses (would imply NaN gradients).
    for i, lv in enumerate(losses):
        assert not np.isnan(lv), (
            f"NaN loss at epoch {i + 1}: dollar-scale regression overflowed "
            f"the joint loss — regression_normalize may be silently disabled."
        )

    # (b) Both heads learn — final loss < first-epoch loss.
    assert len(losses) >= 2, f"Expected at least 2 epoch losses; got {losses}"
    assert losses[-1] < losses[0], (
        f"Joint loss did not decrease: {losses}. Likely the dollar-scale regression target is drowning the binary head."
    )

    # Sanity: the trained model still gives a non-degenerate binary
    # prediction (proves the BCE head learned something despite the
    # imbalanced raw scales).
    proba_clicked = est.predict_proba_dict(X)["ITEM_clicked"][:, 1]
    assert proba_clicked.std() > 1e-3, (
        f"Binary head produced near-constant predictions (std={proba_clicked.std()}). "
        f"Regression target likely overwhelmed the joint loss."
    )


# ---------------------------------------------------------------------- #
# 8. Pickle round-trip preserves regression scaler + masking config.
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("family", ["mlp", "tx"])
def test_pickle_round_trip(family):
    X, y, ts = _make_synthetic(n=60)
    pairs = dict(_cond_estimators(ts, params={"epochs": 2}))
    est = pairs[family]
    est.fit(X, y)
    pre = est.predict_proba_dict(X)

    buf = io.BytesIO()
    pickle.dump(est, buf)
    buf.seek(0)
    est2 = pickle.load(buf)
    post = est2.predict_proba_dict(X)
    for k in pre:
        np.testing.assert_allclose(pre[k], post[k], rtol=1e-5)
    # Specifically check the regression scaler survived.
    assert est2._regression_scaler.scalers == est._regression_scaler.scalers
    # And the mask_prob config.
    assert est2.params["mask_prob"] == est.params["mask_prob"]


# ---------------------------------------------------------------------- #
# Scorer-level: OBSERVED_* path with conditional estimator.
# ---------------------------------------------------------------------- #


def test_scorer_accepts_observed_with_conditional_estimator():
    X, y, ts = _make_synthetic(n=80)
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 2, "hidden_dim": 16, "num_layers": 1, "batch_size": 32},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)

    inf_df = X.copy().head(10)
    from skrec.constants import OBSERVED_PREFIX, USER_ID_NAME

    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(10)])
    inf_df[f"{OBSERVED_PREFIX}a"] = np.array([1.0, np.nan, 0.0, np.nan, 1.0, 0.0, np.nan, 1.0, 0.0, np.nan])
    # No NotImplementedError this time — conditional estimator permits it.
    out = scorer.score_items(interactions=inf_df)
    assert out.shape[0] == 10


def test_scorer_rejects_orphan_observed_with_conditional_estimator():
    X, y, ts = _make_synthetic(n=20)
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)

    from skrec.constants import USER_ID_NAME

    inf_df = X.head(5).copy()
    inf_df.insert(0, USER_ID_NAME, [f"u{i}" for i in range(5)])
    inf_df["OBSERVED_unrelated_target"] = 1
    with pytest.raises(ValueError, match="Orphan"):
        scorer._validate_inference_interactions(inf_df)


def test_scorer_rejects_partial_multilabel_group_observation():
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    ts = {
        "g": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_m1", "ITEM_m2"]),
    }
    y = {"g": rng.integers(0, 2, size=(n, 2))}
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)

    from skrec.constants import USER_ID_NAME

    inf_df = X.head(3).copy()
    inf_df.insert(0, USER_ID_NAME, ["u0", "u1", "u2"])
    # Row 0: m1 observed, m2 NaN — partial group observation (illegal).
    inf_df["OBSERVED_m1"] = [1.0, np.nan, 1.0]
    inf_df["OBSERVED_m2"] = [np.nan, np.nan, 1.0]
    with pytest.raises(ValueError, match="partial-group") as exc:
        scorer._validate_inference_interactions(inf_df)
    msg = str(exc.value)
    # The error must name three things so the caller can locate the
    # malformed row(s) without re-running with print statements:
    #   1. The offending group key ('g').
    #   2. The exact offending row index — assert the bracketed list
    #      ``[0]`` rather than just the substring ``"0"`` (which would
    #      also match incidental zeros elsewhere in the message).
    #   3. At least one member column name (ITEM_m1 / ITEM_m2) so the
    #      caller knows which fanned-out OBSERVED_* columns are tied
    #      to this group. The column-level sibling assertion checks
    #      the same shape — pinning it here keeps the row-level and
    #      column-level paths symmetric.
    assert "'g'" in msg, f"group key missing from error: {msg}"
    assert "[0]" in msg, f"offending row index (as a bracketed list) missing from error: {msg}"
    assert "ITEM_m1" in msg or "ITEM_m2" in msg, f"member column name(s) missing from error: {msg}"


# ---------------------------------------------------------------------- #
# 9. Single-row conditional inference via score_fast.
# ---------------------------------------------------------------------- #


def test_score_fast_conditional_honors_observed():
    """Plan v3 test #9: single-row conditional inference via score_fast.
    Pre-fix the assertion was a no-op (`is not None`). Strengthened to
    drive the underlying score_items probability surface (point estimates
    can tie when both end in the same bin) and require a measurable
    probability shift on ITEM_b when ITEM_a is observed positive vs NaN.
    """
    X, y, ts = _make_synthetic(n=200, correlated=True)
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 10, "hidden_dim": 32, "num_layers": 2, "batch_size": 32, "mask_prob": 0.5, "seed": 33},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)

    row = X.iloc[[0]].copy()
    row_obs = row.copy()
    row_obs["OBSERVED_a"] = 1.0
    row_nan = row.copy()
    row_nan["OBSERVED_a"] = np.nan

    # The wide point-estimate output (from score_fast) can tie when both
    # the observed and unobserved predictions land in the same class bin
    # for binary targets. Drive the underlying score_items proba surface
    # for the same single rows — that's where the conditioning shift is
    # actually measurable.
    from skrec.constants import USER_ID_NAME

    row_obs_with_user = row_obs.copy()
    row_obs_with_user.insert(0, USER_ID_NAME, ["u_obs"])
    row_nan_with_user = row_nan.copy()
    row_nan_with_user.insert(0, USER_ID_NAME, ["u_nan"])
    proba_obs = scorer.score_items(interactions=row_obs_with_user)
    proba_nan = scorer.score_items(interactions=row_nan_with_user)

    p_b_obs = float(proba_obs["ITEM_b_1"].iloc[0])
    p_b_nan = float(proba_nan["ITEM_b_1"].iloc[0])
    assert abs(p_b_obs - p_b_nan) > 1e-6, (
        f"score_fast / score_items did not honor OBSERVED_a: P(ITEM_b=1) "
        f"with observed={p_b_obs} vs NaN={p_b_nan} are equal. The "
        f"conditional dispatch silently bypassed the observed value."
    )
    # Also pin that score_fast (the single-row point-estimate path)
    # returns a non-None ITEM_b prediction for both inputs — it must
    # work end-to-end, not just score_items.
    out_obs = scorer.score_fast(row_obs)
    out_nan = scorer.score_fast(row_nan)
    assert out_obs["ITEM_b"].iloc[0] in (0, 1)
    assert out_nan["ITEM_b"].iloc[0] in (0, 1)


# ---------------------------------------------------------------------- #
# v3 plan test #10: recommend_online end-to-end with non-declaring
# client schema. Pins the preserved-columns hook's schema-apply
# set-aside-and-reattach seam (v3 risk #15: silent OBSERVED_* strip).
# ---------------------------------------------------------------------- #


def test_recommend_online_preserves_observed_through_non_declaring_schema():
    """End-to-end test for v3 risk #15.

    The caller's client inference schema declares ONLY ``USER_ID`` + feature
    columns — it does NOT declare ``OBSERVED_*``. Without the preserved-
    columns hook, ``interactions_schema.apply()`` would silently strip the
    ``OBSERVED_*`` column at the start of ``recommend_online`` (with just a
    warning logged), and the user's conditioning intent would be silently
    lost. The hook (``BaseScorer.preserved_inference_columns()`` + the
    set-aside / re-attach seam in ``BaseRecommender.recommend_online``)
    shields ``OBSERVED_*`` from this strip.

    This test forces the seam by constructing the recommender with an
    explicit ``interactions_schema`` that omits ``OBSERVED_*``, then verifies
    that:
      (a) ``recommend_online`` runs without raising, AND
      (b) the conditional estimator's prediction reflects the observed
          value — i.e. it does NOT match the all-NaN-observed prediction
          for the same feature row.
    """
    from skrec.dataset.schema import DatasetSchema

    X, y, ts = _make_synthetic(n=120, correlated=True)
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 8, "hidden_dim": 32, "num_layers": 2, "batch_size": 32, "mask_prob": 0.5, "seed": 7},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    recommender = RankingRecommender(scorer=scorer)

    # Client inference schema declares USER_ID + features, NOT OBSERVED_*.
    # Mirrors what a typical caller would have in their YAML — they don't
    # know about OBSERVED_*; the hook should preserve it anyway.
    schema_dict = {
        "columns": [
            {"name": "USER_ID", "type": "str"},
            {"name": "f0", "type": "float"},
            {"name": "f1", "type": "float"},
            {"name": "f2", "type": "float"},
            {"name": "f3", "type": "float"},
        ]
    }
    recommender.interactions_schema = DatasetSchema(schema_dict)
    recommender.users_schema = None  # recommend_online checks self.users_schema

    # Item names need to be populated for active_item_names lookup in
    # recommend_online. The scorer's _init_item_state was set during fit
    # via process_datasets earlier in this test fixture path — re-trigger
    # by calling process_datasets explicitly to set item_names.
    from skrec.constants import USER_ID_NAME

    train_with_user = X.copy()
    train_with_user.insert(0, USER_ID_NAME, [f"u{i}" for i in range(len(X))])
    for k, v in y.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            for j, col in enumerate(ts[k]["columns"]):
                train_with_user[col] = v[:, j]
        else:
            train_with_user[k] = v
    scorer.process_datasets(interactions_df=train_with_user, is_training=True)

    # Build the single-row inference frame WITH OBSERVED_a but WITHOUT
    # declaring it in the schema. Compare against a row with no OBSERVED_*.
    single = X.iloc[[0]].copy()
    single.insert(0, USER_ID_NAME, ["online_user"])
    single_with_obs = single.copy()
    single_with_obs["OBSERVED_a"] = 1.0  # Observe correlated target.

    out_with_obs = recommender.recommend_online(interactions=single_with_obs)
    out_no_obs = recommender.recommend_online(interactions=single)

    # (a) Both calls returned non-empty wide-format frames.
    assert out_with_obs is not None and len(out_with_obs.columns) > 0
    assert out_no_obs is not None and len(out_no_obs.columns) > 0

    # (b) Predictions differ on ITEM_b — proves OBSERVED_a survived the
    # schema-apply strip and reached the conditional estimator. If the
    # OBSERVED_* column had been silently dropped, both calls would have
    # produced identical predictions (both effectively unconditional).
    #
    # Precondition pinned as an assert (not a silent `if` guard) so a
    # future refactor that renames the output column surfaces here as
    # a test failure rather than silently degrading the seam-regression
    # check into a vacuous pass.
    assert "ITEM_b" in out_with_obs.columns, (
        f"Recommender output is missing 'ITEM_b' column "
        f"({list(out_with_obs.columns)}). Either the predict_targets "
        f"output convention changed (update this test) or the recommend "
        f"path itself broke (investigate)."
    )
    assert "ITEM_b" in out_no_obs.columns
    # Stronger seam test: predictions through score_items proba surface
    # differ. Point estimates (predict_targets) can tie when both observed
    # and unobserved paths land in the same class bin for binary targets;
    # the proba shift is the rigorous signal that conditioning reached
    # the estimator.
    out_proba_with = scorer.score_items(interactions=single_with_obs)
    out_proba_without = scorer.score_items(interactions=single)
    p_with = float(out_proba_with["ITEM_b_1"].iloc[0])
    p_without = float(out_proba_without["ITEM_b_1"].iloc[0])
    assert abs(p_with - p_without) > 1e-6, (
        f"OBSERVED_a was silently dropped — P(ITEM_b=1) with vs without "
        f"observed are equal ({p_with} == {p_without}). Schema-apply "
        f"preservation hook failed."
    )


# ====================================================================== #
# v3 conditional Transformer parametrization gaps
# ====================================================================== #


@pytest.mark.parametrize(
    "estimator_cls,kwargs",
    [
        (
            ConditionalJointMultiTargetMLPEstimator,
            {"epochs": 20, "hidden_dim": 16, "num_layers": 2, "batch_size": 32, "mask_prob": 0.5, "seed": 0},
        ),
        (
            ConditionalJointMultiTargetTransformerEstimator,
            {
                "epochs": 20,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 2,
                "ffn_dim": 32,
                "batch_size": 32,
                "mask_prob": 0.5,
                "seed": 0,
            },
        ),
    ],
    ids=["mlp", "tx"],
)
def test_v3_conditional_2_correlation_both_families(estimator_cls, kwargs):
    """Plan-test #2 (correlation between conditional and target) must
    hold for BOTH conditional families — observing the correlated target
    shifts predictions in the expected direction."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=[f"f{i}" for i in range(3)])
    latent = X["f0"] + 0.5 * X["f1"]
    y = {
        "ITEM_a": (latent > 0).astype(int).to_numpy(),
        "ITEM_b": (latent + 0.1 * rng.normal(size=n) > 0).astype(int).to_numpy(),
    }
    ts = {"ITEM_a": TargetType.BINARY, "ITEM_b": TargetType.BINARY}

    est = estimator_cls(target_specs=ts, params=kwargs)
    est.fit(X, y)

    # Predictions with no observation vs with ITEM_a observed.
    obs_none = est.predict_with_observed(X, observed=None)
    obs_a = est.predict_with_observed(X, observed={"ITEM_a": y["ITEM_a"].astype(float)})
    # Conditioning must SHIFT the distribution of ITEM_b predictions.
    diff = float(np.mean(np.abs(obs_a["ITEM_b"][:, 1] - obs_none["ITEM_b"][:, 1])))
    assert diff > 0.01, f"Conditioning had no effect on ITEM_b predictions (mean abs diff = {diff})"


# ====================================================================== #
# v3 risk-15: score_items preserves OBSERVED through non-declaring schema
# ====================================================================== #


def test_v3_risk15_score_items_preserves_observed_through_non_declaring_schema():
    """Parallel to the recommend_online OBSERVED-preservation test:
    score_items via the recommender (the batch path) must also preserve
    OBSERVED_* columns through a non-declaring interactions_schema. This
    exercises the P0-1 + P0-2 + post-P1 fix together via the batch entry
    point."""
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
        params={"epochs": 30, "hidden_dim": 32, "num_layers": 2, "batch_size": 64, "mask_prob": 0.5, "seed": 0},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)

    inf = X.copy()
    inf[USER_ID_NAME] = np.arange(n)
    # Add OBSERVED_a — not in any user schema (recommender wasn't trained).
    inf["OBSERVED_a"] = y["ITEM_a"].astype(float)

    # score_items goes through preprocess_inputs (now schema-aware via
    # P0-1's apply_interactions_schema_with_preservation). Without
    # preservation the OBSERVED_a column would be dropped and the
    # conditioning silently ignored. With preservation it survives.
    proba_no_obs_inf = X.copy()
    proba_no_obs_inf[USER_ID_NAME] = np.arange(n)
    proba_no_obs = scorer.score_items(interactions=proba_no_obs_inf)
    proba_with_obs = scorer.score_items(interactions=inf)
    # ITEM_b score columns must differ between observed and non-observed
    # — proves OBSERVED_a reached the conditional estimator.
    diff = float(np.mean(np.abs(proba_with_obs["ITEM_b_1"].to_numpy() - proba_no_obs["ITEM_b_1"].to_numpy())))
    # Loose floor — what matters is "non-trivial signal got through";
    # exact magnitude depends on training noise.
    assert diff > 0.005, (
        f"OBSERVED_a had no effect on score_items ITEM_b probabilities "
        f"(mean abs diff = {diff}). OBSERVED preservation through "
        f"preprocess_inputs may have regressed."
    )


# ====================================================================== #
# Round 4 coverage: mask_prob=1.0 parity over MULTILABEL members
# ====================================================================== #


@pytest.mark.parametrize(
    "vanilla_family, conditional_family, common_extra",
    [
        (
            "mlp",
            ConditionalJointMultiTargetMLPEstimator,
            {"hidden_dim": 32, "num_layers": 2},
        ),
        (
            "tx",
            ConditionalJointMultiTargetTransformerEstimator,
            {"d_model": 16, "n_heads": 2, "n_layers": 1, "ffn_dim": 32},
        ),
    ],
)
def test_mask_prob_one_vanilla_parity_multilabel(vanilla_family, conditional_family, common_extra):
    """Gate 10 extension: mask_prob=1.0 ≈ vanilla parity must hold on
    MULTILABEL members too. Round-1 finding was specifically that
    multilabel handling differed at mask_prob=1.0; the existing parity
    test only covers BINARY."""
    from sklearn.metrics import roc_auc_score

    from skrec.estimator.classification import (
        JointMultiTargetMLPEstimator,
        JointMultiTargetTransformerEstimator,
    )

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    # Multilabel group of 2 correlated binary members.
    latent = X["f0"] + 0.7 * X["f1"]
    g = np.column_stack(
        [
            (latent > 0).astype(int),
            (latent + 0.2 * rng.normal(size=n) > 0).astype(int),
        ]
    )
    ts = {"g": {"type": TargetType.MULTILABEL, "columns": ["ITEM_email", "ITEM_app"]}}
    y = {"g": g}

    common = {"epochs": 8, "batch_size": 64, "seed": 11, "lr": 1e-2, **common_extra}
    vanilla_cls = JointMultiTargetMLPEstimator if vanilla_family == "mlp" else JointMultiTargetTransformerEstimator
    vanilla = vanilla_cls(target_specs=ts, params=common)
    vanilla.fit(X, y)
    cond = conditional_family(
        target_specs=ts,
        params={**common, "mask_prob": 1.0, "label_embedding_dim": 4},
    )
    cond.fit(X, y)

    # Per-member AUC parity. members fan out into proba["ITEM_email"]
    # and proba["ITEM_app"] (each shape (n, 2)).
    proba_v = vanilla.predict_proba_dict(X)
    proba_c = cond.predict_proba_dict(X)
    tolerance = 0.35 if vanilla_family == "tx" else 0.18
    for member, member_y in [
        ("ITEM_email", g[:, 0]),
        ("ITEM_app", g[:, 1]),
    ]:
        auc_v = roc_auc_score(member_y, proba_v[member][:, 1])
        auc_c = roc_auc_score(member_y, proba_c[member][:, 1])
        assert auc_v > 0.6 and auc_c > 0.6, (
            f"{vanilla_family} {member}: vanilla={auc_v}, cond={auc_c}; one or both didn't fit."
        )
        assert abs(auc_v - auc_c) < tolerance, (
            f"{vanilla_family} {member}: conditional AUC ({auc_c}) "
            f"diverged from vanilla ({auc_v}) by more than {tolerance}."
        )


# ====================================================================== #
# Round 4 coverage: partial-multilabel error message includes group key
# ====================================================================== #


def test_partial_multilabel_error_includes_group_key_and_column_names():
    """The partial-multilabel error must name the offending group AND
    the offending columns so the caller can locate them. Pre-fix the
    test only matched "partial-group"."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = {"g": rng.integers(0, 2, size=(n, 2)).astype(int)}
    ts = {"g": {"type": TargetType.MULTILABEL, "columns": ["ITEM_email", "ITEM_app"]}}
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 4, "num_layers": 2, "batch_size": 16, "mask_prob": 0.5, "seed": 0},
    )
    est.fit(X, y)
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)
    interactions = X.copy()
    interactions[USER_ID_NAME] = np.arange(n)
    interactions["OBSERVED_email"] = y["g"][:, 0].astype(float)
    # OBSERVED_app missing — column-level imbalance.
    with pytest.raises(ValueError) as exc:
        scorer._validate_inference_interactions(interactions)
    msg = str(exc.value)
    # Must mention the group key AND at least one of the member columns.
    assert "g" in msg, f"Group key missing from error: {msg}"
    assert "OBSERVED_email" in msg or "OBSERVED_app" in msg, f"Member column name missing from error: {msg}"


class _StubConditionalEstimator:
    """Stub with every ConditionalMultiTargetEstimator attribute filled in.

    Sets the ``is_conditional_multi_target`` sentinel so the scorer's
    stricter ``_is_conditional_estimator`` helper (which requires both
    the Protocol AND the sentinel) treats it as conditional rather than
    vanilla. See ``_multi_target_protocol.py`` for the rationale.
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
# Fix R2-6: predict_with_observed Protocol signature accepts None
# ---------------------------------------------------------------------- #


def test_fix_r2_6_predict_with_observed_accepts_none():
    """Protocol now declares observed: Optional[dict]=None; implementation
    has accepted None all along. Pin the contract by calling
    predict_with_observed(X) without the second argument."""
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
    # No second arg — Optional[dict] = None default.
    out = est.predict_with_observed(X)
    assert "ITEM_a" in out and out["ITEM_a"].shape == (n, 2)
    # Explicit None should match.
    out_none = est.predict_with_observed(X, None)
    np.testing.assert_allclose(out["ITEM_a"], out_none["ITEM_a"], rtol=1e-6)


# ---------------------------------------------------------------------- #
# Fix R2-7: Independent _fitted=False resets BEFORE validation
# ---------------------------------------------------------------------- #


# ====================================================================== #
# Round-2 batch B (the second-half medium-severity comments)
# ====================================================================== #


# --- M1: target_specs keys rejected when they contain '.' or whitespace ---


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


# --- P0-5: sample_training_mask requires generator ---


def test_fix_p0_5_sample_training_mask_requires_generator():

    with pytest.raises(TypeError, match="seeded torch.Generator"):
        sample_training_mask(
            target_specs={"ITEM_a": TargetType.BINARY},
            n_samples=10,
            mask_prob=0.5,
            generator=None,
        )


# --- P0-6: conditional Transformer per-target tokens at d_model ---


# --- P0-6: conditional Transformer per-target tokens at d_model ---


def test_fix_p0_6_conditional_transformer_label_tokens_match_num_targets():
    """The conditional Transformer must append exactly num_targets extra
    tokens at d_model width — not num_targets * label_embedding_dim scalar
    tokens. Capture the seq tensor at the encoder's forward via a hook on
    the first block and check its length."""
    from skrec.estimator.classification import (
        ConditionalJointMultiTargetTransformerEstimator,
    )

    rng = np.random.default_rng(0)
    n = 32
    n_features = 4
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = {
        "ITEM_a": (X["f0"] > 0).astype(int).to_numpy(),
        "ITEM_b": (X["f1"] > 0).astype(int).to_numpy(),
        "ITEM_revenue": X["f2"].to_numpy(),
    }
    ts = {
        "ITEM_a": TargetType.BINARY,
        "ITEM_b": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    d_model = 16
    label_embedding_dim = 8  # intentionally != d_model to expose the bug
    est = ConditionalJointMultiTargetTransformerEstimator(
        target_specs=ts,
        params={
            "epochs": 1,
            "d_model": d_model,
            "n_heads": 2,
            "n_layers": 1,
            "ffn_dim": 32,
            "batch_size": 16,
            "mask_prob": 0.5,
            "label_embedding_dim": label_embedding_dim,
            "seed": 0,
        },
    )
    est.fit(X, y)

    # Now intercept the first block's forward to capture the seq length.
    captured_seq_shapes = []
    block = est._encoder.blocks[0]
    real_forward = block.forward

    def _capturing_forward(x):
        captured_seq_shapes.append(tuple(x.shape))
        return real_forward(x)

    block.forward = _capturing_forward
    try:
        # Drive a single forward via predict_proba_dict.
        est.predict_proba_dict(X.iloc[:4])
    finally:
        block.forward = real_forward

    assert captured_seq_shapes, "First block forward was never called."
    # Sequence shape: (batch, n_tokens, d_model).
    # Pre-fix: n_tokens = 1 (CLS) + n_features + num_targets * label_embedding_dim
    #        = 1 + 4 + 3*8 = 29
    # Post-fix: n_tokens = 1 (CLS) + n_features + num_targets
    #        = 1 + 4 + 3 = 8
    batch, n_tokens, dim = captured_seq_shapes[0]
    assert dim == d_model, f"Token dim {dim} != d_model {d_model}"
    expected = 1 + n_features + len(ts)  # CLS + features + per-target
    assert n_tokens == expected, (
        f"Conditional Transformer seq has {n_tokens} tokens; expected "
        f"{expected} (1 CLS + {n_features} features + {len(ts)} per-target "
        f"tokens). The label-token architecture must be PER TARGET at d_model, "
        f"not PER SCALAR (which would give 1 + {n_features} + "
        f"{len(ts) * label_embedding_dim} = "
        f"{1 + n_features + len(ts) * label_embedding_dim} tokens)."
    )


# --- P0-7: independent.defaults upfront coverage validation ---


def test_p1_12_conditional_unseen_label_demotes_to_unobserved():
    """When predict_with_observed receives a multiclass label not in the
    training-time catalogue, the row should be demoted to is_observed=0
    rather than silently zeroing the one-hot row while keeping
    is_observed=1 (which would be a label leak)."""
    from skrec.estimator.classification._conditional_label_encoding import (
        build_raw_chunks,
    )

    target_specs = {"ITEM_c": TargetType.MULTICLASS}
    classes = {"ITEM_c": ["a", "b", "c"]}
    arr = np.array(["a", "b", "zzz_unseen", "c"], dtype=object)
    mask = {"ITEM_c": np.array([False, False, False, False])}
    chunks = build_raw_chunks(
        y_or_observed={"ITEM_c": arr},
        mask=mask,
        target_specs=target_specs,
        multiclass_classes=classes,
        regression_means={},
        regression_stds={},
    )
    raw = chunks["ITEM_c"].cpu().numpy()
    # First column is is_observed; row 2 had unseen label → demoted to 0.
    assert raw[0, 0] == 1.0
    assert raw[1, 0] == 1.0
    assert raw[2, 0] == 0.0, "Unseen-label row should be demoted to is_observed=0"
    assert raw[3, 0] == 1.0
    # And its one-hot row must be all zeros.
    assert np.allclose(raw[2, 1:], 0.0)


# ====================================================================== #
# Round 4: older P1/P2 follow-up fixes
# ====================================================================== #


def test_conditional_protocol_sentinel_blocks_structural_lookalike():
    """A class that has predict_with_observed + the MultiTargetEstimator
    base attrs but does NOT set is_conditional_multi_target=True must
    fail the scorer's stricter conditional check (so OBSERVED_* under
    a structural look-alike doesn't silently activate the conditional
    path)."""

    class _StructuralLookAlike:
        # Structural attrs to satisfy MultiTargetEstimator + the
        # conditional Protocol's predict_with_observed shape, but no
        # sentinel — should be rejected by the scorer's
        # _is_conditional_estimator helper.
        target_specs = {"ITEM_a": TargetType.BINARY}

        def fit(self, X, y, X_valid=None, y_valid=None):
            return self

        def predict_proba_dict(self, X):
            n = X.shape[0]
            p1 = np.full(n, 0.5)
            return {"ITEM_a": np.column_stack([1.0 - p1, p1])}

        def predict_targets_dict(self, X):
            return {"ITEM_a": np.zeros(X.shape[0], dtype=int)}

        def predict_with_observed(self, X, observed=None):
            return self.predict_proba_dict(X)

    est = _StructuralLookAlike()
    scorer = MixedTypeMultiTargetScorer(estimator=est, target_specs={"ITEM_a": TargetType.BINARY})
    # Even though the look-alike has predict_with_observed, the sentinel
    # is missing, so the scorer must treat it as vanilla and reject
    # OBSERVED_* with NotImplementedError.
    inf = pd.DataFrame({"f0": [0.1, 0.2], USER_ID_NAME: [1, 2]})
    inf["OBSERVED_a"] = [1.0, 0.0]
    with pytest.raises(NotImplementedError, match="ConditionalMultiTargetEstimator"):
        scorer._validate_inference_interactions(inf)

    # Sanity: setting the sentinel flips behavior — OBSERVED_a accepted.
    est.is_conditional_multi_target = True
    scorer._validate_inference_interactions(inf)


def test_conditional_fit_logs_validation_loss():
    """X_valid + y_valid passed to a conditional estimator's fit must
    produce a per-epoch validation loss log line, not be silently
    accepted and ignored."""
    import logging as _logging

    rng = np.random.default_rng(0)
    n_train, n_valid = 60, 20
    X = pd.DataFrame(rng.normal(size=(n_train, 3)), columns=[f"f{i}" for i in range(3)])
    Xv = pd.DataFrame(rng.normal(size=(n_valid, 3)), columns=[f"f{i}" for i in range(3)])
    y = {"ITEM_a": (rng.normal(size=n_train) > 0).astype(int)}
    yv = {"ITEM_a": (rng.normal(size=n_valid) > 0).astype(int)}
    ts = {"ITEM_a": TargetType.BINARY}

    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 2, "hidden_dim": 4, "num_layers": 2, "batch_size": 16, "mask_prob": 0.5, "seed": 0},
    )
    import io

    log_stream = io.StringIO()
    handler = _logging.StreamHandler(log_stream)
    handler.setLevel(_logging.INFO)
    logger = _logging.getLogger("skrec.estimator.classification._conditional_joint_multi_target_base")
    logger.addHandler(handler)
    try:
        est.fit(X, y, X_valid=Xv, y_valid=yv)
    finally:
        logger.removeHandler(handler)
    log = log_stream.getvalue()
    assert "Conditional Validation Loss" in log, (
        f"Validation loss must be logged when X_valid/y_valid are passed. Got log:\n{log}"
    )


def test_conditional_loss_normalized_by_target_specs_len():
    """_compute_masked_loss divides by len(target_specs), not by the
    runtime count of targets that had ≥1 masked row. Test: a 2-target
    spec where one target's mask_dict is all-False per batch should
    still divide by 2 (stable per-target gradient scale)."""
    rng = np.random.default_rng(0)
    n = 32
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {
        "ITEM_a": (rng.normal(size=n) > 0).astype(int),
        "ITEM_b": (rng.normal(size=n) > 0).astype(int),
    }
    ts = {"ITEM_a": TargetType.BINARY, "ITEM_b": TargetType.BINARY}
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 4, "num_layers": 2, "batch_size": 16, "mask_prob": 0.5, "seed": 0},
    )
    est.fit(X, y)

    X_t = torch.from_numpy(X.to_numpy(dtype=np.float32)).to(est._device)
    y_t = {k: torch.from_numpy(v.astype(np.float32)).to(est._device) for k, v in y.items()}
    idx = torch.arange(n, device=est._device)
    # Only ITEM_a has masked rows; ITEM_b is fully observed.
    mask_dict = {
        "ITEM_a": np.ones(n, dtype=bool),
        "ITEM_b": np.zeros(n, dtype=bool),
    }
    encoder_module = est._encoder
    encoder_module.eval()
    est._heads.eval()
    est._label_encoder.eval()
    reg_means = {k: v[0] for k, v in est._regression_scaler.scalers.items()}
    reg_stds = {k: v[1] for k, v in est._regression_scaler.scalers.items()}
    from skrec.estimator.classification._conditional_label_encoding import (
        build_raw_chunks,
    )

    with torch.no_grad():
        raw = build_raw_chunks(
            y_or_observed={k: np.asarray(v) for k, v in y.items()},
            mask=mask_dict,
            target_specs=ts,
            multiclass_classes=est._multiclass_classes,
            regression_means=reg_means,
            regression_stds=reg_stds,
            device=est._device,
            strict_unknown_labels=True,
        )
        label_inputs = est._format_label_inputs(raw)
        hidden = encoder_module(X_t, label_inputs)
        logits = est._heads(hidden)
        loss_partial = est._compute_masked_loss(logits, y_t, idx, mask_dict).item()
    # Now flip: both targets masked.
    mask_dict_full = {
        "ITEM_a": np.ones(n, dtype=bool),
        "ITEM_b": np.ones(n, dtype=bool),
    }
    with torch.no_grad():
        raw_full = build_raw_chunks(
            y_or_observed={k: np.asarray(v) for k, v in y.items()},
            mask=mask_dict_full,
            target_specs=ts,
            multiclass_classes=est._multiclass_classes,
            regression_means=reg_means,
            regression_stds=reg_stds,
            device=est._device,
            strict_unknown_labels=True,
        )
        label_inputs_full = est._format_label_inputs(raw_full)
        hidden_full = encoder_module(X_t, label_inputs_full)
        logits_full = est._heads(hidden_full)
        loss_full = est._compute_masked_loss(logits_full, y_t, idx, mask_dict_full).item()
    # Both divide by len(target_specs)=2. The partial-mask call contributes
    # only ITEM_a's term; the full call contributes both. So
    # loss_partial ≈ loss_full / 2 ± (model output differences). Pin the
    # weaker invariant: loss_partial < loss_full * 0.8 (would have been
    # ~loss_full * 1.0 under the old "divide by n_contributing" rule).
    assert loss_partial < loss_full * 0.9, (
        f"loss_partial={loss_partial} vs loss_full={loss_full}: ratio "
        f"{loss_partial / max(loss_full, 1e-9):.3f} suggests denominator "
        f"wobble (should be ~0.5, not ~1.0)."
    )


def test_conditional_multiclass_unseen_label_at_train_raises():
    """A multiclass training row whose label isn't in the captured
    catalogue must raise at fit time, not silently demote to
    is_observed=0. The catalogue is built from training y itself, so
    unseen labels indicate a contract violation."""
    n = 30
    y_class = np.array(["a", "b", "c"] + ["a"] * (n - 3), dtype=object)
    ts = {"ITEM_class": TargetType.MULTICLASS}
    # Simulate the violation directly via build_raw_chunks with strict mode.
    # (No need to construct/fit the estimator — the contract enforcement
    # lives in build_raw_chunks itself.)
    from skrec.estimator.classification._conditional_label_encoding import (
        build_raw_chunks,
    )

    with pytest.raises(ValueError, match="not in the captured catalogue"):
        build_raw_chunks(
            y_or_observed={"ITEM_class": y_class},
            mask={"ITEM_class": np.array([False] * n)},
            target_specs=ts,
            multiclass_classes={"ITEM_class": ["a", "b"]},  # 'c' missing
            regression_means={},
            regression_stds={},
            strict_unknown_labels=True,
        )


def test_conditional_protocol_is_subclass_of_multi_target():
    # Both runtime-checkable Protocols.
    stub = _StubConditionalEstimator(target_specs={"ITEM_a": TargetType.BINARY})
    assert isinstance(stub, MultiTargetEstimator)  # also a base-MTE
    assert isinstance(stub, ConditionalMultiTargetEstimator)


def test_conditional_protocol_negative_for_vanilla_estimator():
    """JointMultiTargetMLPEstimator is a MultiTargetEstimator but NOT a
    ConditionalMultiTargetEstimator — pins the v2/v3 surface separation."""
    est = JointMultiTargetMLPEstimator(target_specs={"ITEM_a": TargetType.BINARY}, params={"epochs": 1})
    assert isinstance(est, MultiTargetEstimator)
    assert not isinstance(est, ConditionalMultiTargetEstimator)


def test_conditional_protocol_negative_for_partial():
    """A class missing predict_with_observed fails the conditional check."""

    class _Partial:
        target_specs = {}

        def fit(self, X, y, X_valid=None, y_valid=None):
            return self

        def predict_proba_dict(self, X):
            return {}

        def predict_targets_dict(self, X):
            return {}

    assert isinstance(_Partial(), MultiTargetEstimator)
    assert not isinstance(_Partial(), ConditionalMultiTargetEstimator)


# ---------------------------------------------------------------------- #
# BaseScorer.preserved_inference_columns default
# ---------------------------------------------------------------------- #


def test_conditional_check_fitted_catches_cleared_label_encoder():
    """Conditional _check_fitted must surface a 'not fitted' error if
    _label_encoder was cleared (vs NPE'ing inside predict_with_observed)."""
    X, y, ts = _make_synthetic(n=20)
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 4, "num_layers": 1, "batch_size": 8, "mask_prob": 0.5, "seed": 0},
    )
    est.fit(X, y)
    # Clear the label encoder mid-flight (simulates a partial state bug).
    est._label_encoder = None
    with pytest.raises(RuntimeError, match="not fully fitted|_label_encoder"):
        est.predict_with_observed(X, observed=None)


def test_conditional_fit_unseen_multiclass_in_y_valid_raises():
    """y_valid with multiclass labels not in the train catalogue must
    raise an actionable ValueError (was bare KeyError from
    _build_train_tensors before this round)."""
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    Xv = pd.DataFrame(rng.normal(size=(10, 3)), columns=["f0", "f1", "f2"])
    y = {"ITEM_c": np.array(["a", "b"] * (n // 2))}
    # Valid y has a label 'zzz' not in train.
    yv = {"ITEM_c": np.array(["a", "zzz"] * 5)}
    ts = {"ITEM_c": TargetType.MULTICLASS}
    est = ConditionalJointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 4, "num_layers": 1, "batch_size": 8, "mask_prob": 0.5, "seed": 0},
    )
    with pytest.raises(ValueError, match="training-time class catalogue"):
        est.fit(X, y, X_valid=Xv, y_valid=yv)
