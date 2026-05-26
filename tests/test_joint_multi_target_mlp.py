# Tests for JointMultiTargetMLPEstimator.
#
# Focused on the estimator-level Protocol contract: dict-y fit, dict-keyed
# predictions, output shapes per target type, basic determinism, and pickle
# round-trip. Full per-type metric sanity gates live in M8 (gate 3).

import io
import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from skrec.estimator.classification import (  # noqa: E402
    JointMultiTargetMLPEstimator,
    MultiTargetEstimator,
)
from skrec.scorer.mixed_type_multi_target import (  # noqa: E402
    MixedTypeMultiTargetScorer,
    TargetGroupSpec,
    TargetType,
)

SEED = 42


def _make_synthetic(n: int = 200, n_features: int = 5, seed: int = SEED):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # binary target ~ sign of feat_0
    y_binary = (X["feat_0"] > 0).astype(np.int64).to_numpy()
    # regression target ~ feat_1
    y_reg = (2.0 * X["feat_1"] + rng.normal(scale=0.1, size=n)).to_numpy()
    # multiclass target ~ argmax of three feature combos
    cls_logits = np.column_stack([X["feat_2"], X["feat_3"], X["feat_4"]])
    y_mc_idx = cls_logits.argmax(axis=1)
    y_mc = np.array(["A", "B", "C"])[y_mc_idx]
    # multilabel group: two binary members weakly correlated with features
    y_ml = np.column_stack(
        [
            (X["feat_2"] > 0).astype(np.int64).to_numpy(),
            (X["feat_3"] > 0).astype(np.int64).to_numpy(),
        ]
    )
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
        "ITEM_clicked": y_binary,
        "ITEM_revenue": y_reg,
        "ITEM_action": y_mc,
        "engagement": y_ml,
    }
    return X, y, target_specs


def test_implements_multi_target_protocol():
    _, _, target_specs = _make_synthetic(n=10)
    est = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1})
    assert isinstance(est, MultiTargetEstimator)


def test_fit_happy_path_all_four_target_types():
    X, y, target_specs = _make_synthetic(n=100)
    est = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 2, "hidden_dim": 32, "num_layers": 2, "batch_size": 32},
    )
    est.fit(X, y)
    # Returns probabilities keyed by fanned-out target.
    proba = est.predict_proba_dict(X)
    assert set(proba.keys()) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_action",
        "ITEM_email_open",
        "ITEM_app_open",
    }
    # binary → (n, 2)
    assert proba["ITEM_clicked"].shape == (100, 2)
    # regression → (n,)
    assert proba["ITEM_revenue"].shape == (100,)
    # multiclass → (n, K=3)
    assert proba["ITEM_action"].shape == (100, 3)
    # multilabel members → (n, 2) per member
    assert proba["ITEM_email_open"].shape == (100, 2)
    assert proba["ITEM_app_open"].shape == (100, 2)


def test_predict_targets_dict_shapes_and_dtypes():
    X, y, target_specs = _make_synthetic(n=80)
    est = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 1, "hidden_dim": 16, "num_layers": 2, "batch_size": 32},
    )
    est.fit(X, y)
    preds = est.predict_targets_dict(X)
    assert set(preds.keys()) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_action",
        "ITEM_email_open",
        "ITEM_app_open",
    }
    # binary / multilabel-member → 0/1 ints
    for k in ("ITEM_clicked", "ITEM_email_open", "ITEM_app_open"):
        vals = preds[k]
        assert vals.shape == (80,)
        assert set(np.unique(vals).tolist()).issubset({0, 1})
    # regression → continuous values, de-normalized to original scale
    assert preds["ITEM_revenue"].shape == (80,)
    # multiclass → original labels (strings, preserved)
    assert preds["ITEM_action"].shape == (80,)
    assert set(np.unique(preds["ITEM_action"]).tolist()).issubset({"A", "B", "C"})


def test_fit_rejects_dict_y_with_wrong_keys():
    X, y, target_specs = _make_synthetic(n=20)
    bad_y = {"WRONG_KEY": y["ITEM_clicked"]}
    est = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1})
    with pytest.raises(ValueError, match="y keys must match target_specs"):
        est.fit(X, bad_y)


def test_fit_rejects_non_dict_y():
    X, _, target_specs = _make_synthetic(n=20)
    est = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1})
    with pytest.raises(TypeError, match="dict"):
        est.fit(X, np.array([0, 1] * 10))


def test_predict_alignment_rejects_missing_features():
    X, y, target_specs = _make_synthetic(n=30)
    est = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1, "hidden_dim": 16})
    est.fit(X, y)
    X_bad = X.drop(columns=["feat_0"])
    with pytest.raises(ValueError, match="missing training-time feature columns"):
        est.predict_proba_dict(X_bad)


def test_predict_alignment_rejects_extra_features():
    X, y, target_specs = _make_synthetic(n=30)
    est = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1, "hidden_dim": 16})
    est.fit(X, y)
    X_bad = X.copy()
    X_bad["new_feat"] = 1.0
    with pytest.raises(ValueError, match="unseen at training"):
        est.predict_proba_dict(X_bad)


def test_predict_column_order_invariant():
    X, y, target_specs = _make_synthetic(n=40)
    est = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1, "hidden_dim": 16, "seed": 0})
    est.fit(X, y)
    proba_orig = est.predict_proba_dict(X)
    X_reordered = X.iloc[:, ::-1]
    proba_reordered = est.predict_proba_dict(X_reordered)
    for k in proba_orig:
        np.testing.assert_allclose(proba_orig[k], proba_reordered[k], rtol=1e-5)


def test_pickle_round_trip_predictions_match():
    X, y, target_specs = _make_synthetic(n=50)
    est = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 2, "hidden_dim": 16, "num_layers": 2, "seed": 7},
    )
    est.fit(X, y)
    pre = est.predict_proba_dict(X)

    buf = io.BytesIO()
    pickle.dump(est, buf)
    buf.seek(0)
    est2 = pickle.load(buf)

    post = est2.predict_proba_dict(X)
    for k in pre:
        np.testing.assert_allclose(pre[k], post[k], rtol=1e-5)


def test_determinism_same_seed_same_predictions():
    """Test #4 (joint MLP, M8 verification).

    Determinism contract (v2): single-process CPU + ``torch.randperm`` with
    a seeded ``torch.Generator`` for batch shuffling (no DataLoader, so no
    ``num_workers`` plumbing is required or applicable).
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` is set in
    ``_set_seed``; ``torch.manual_seed`` covers module init,
    ``np.random.seed`` covers any numpy randomness. GPU determinism is
    NOT guaranteed in v2 (warn_only degrades gracefully); this test pins
    CPU. See ``_set_seed`` in
    ``skrec/estimator/classification/_joint_multi_target_base.py`` for
    the full plumbing contract.
    """
    X, y, target_specs = _make_synthetic(n=60)

    def fit_and_predict():
        est = JointMultiTargetMLPEstimator(
            target_specs=target_specs,
            params={
                "epochs": 2,
                "hidden_dim": 16,
                "num_layers": 2,
                "seed": 1234,
                "batch_size": 16,
            },
        )
        est.fit(X, y)
        return est.predict_proba_dict(X)

    a = fit_and_predict()
    b = fit_and_predict()
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-5)


def test_validation_loss_runs():
    X, y, target_specs = _make_synthetic(n=80)
    X_train, X_valid = X.iloc[:60], X.iloc[60:].reset_index(drop=True)
    y_train = {k: v[:60] for k, v in y.items()}
    y_valid = {k: v[60:] for k, v in y.items()}
    est = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 1, "hidden_dim": 16, "num_layers": 2},
    )
    est.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)  # must not raise


def test_loss_decreases_monotonically_over_epochs():
    """Plan test #5: loss should monotonically decrease over 5 epochs on
    a separable synthetic. Captures the training-loss log emitted by the
    base estimator and asserts the per-epoch sequence is non-increasing
    (allowing tiny float-noise upticks)."""
    import logging

    X, y, target_specs = _make_synthetic(n=300)
    est = JointMultiTargetMLPEstimator(
        target_specs=target_specs,
        params={"epochs": 5, "hidden_dim": 32, "num_layers": 2, "batch_size": 64, "lr": 1e-2, "seed": 0},
    )
    captured = []

    class _LossCapture(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "Train Loss" in msg:
                # Format: "Epoch [k/N] - Train Loss: %.4f"
                try:
                    captured.append(float(msg.rsplit(":", 1)[-1].strip()))
                except ValueError:
                    pass

    handler = _LossCapture()
    logger = logging.getLogger("skrec.estimator.classification._joint_multi_target_base")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        est.fit(X, y)
    finally:
        logger.removeHandler(handler)

    assert len(captured) == 5, f"Expected 5 epoch losses, got {captured}"
    # Allow tiny float-noise upticks but the trajectory should clearly
    # decrease end-to-end.
    assert captured[-1] < captured[0] * 0.85, (
        f"Loss didn't decrease meaningfully: {captured}. Either training is broken or the synthetic isn't separable."
    )
    # Non-strict monotonicity: each epoch shouldn't be much worse than the
    # previous (tolerance 10% relative).
    for i in range(1, len(captured)):
        assert captured[i] <= captured[i - 1] * 1.1, f"Loss spiked at epoch {i + 1}: {captured}"


def test_check_fitted_before_predict():
    _, _, target_specs = _make_synthetic(n=10)
    est = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1})
    X = pd.DataFrame({"feat_0": [0.5]})
    with pytest.raises(RuntimeError, match="not fitted"):
        est.predict_proba_dict(X)


def test_fix4_joint_mlp_integer_multiclass_k11_round_trips_correctly():
    """End-to-end: train joint MLP on a K=11 integer multiclass target,
    confirm predictions land in the right class ID (not the lex-shuffled
    position the pre-fix code would have produced)."""
    rng = np.random.default_rng(0)
    n = 220  # enough rows to see all 11 classes
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f0", "f1", "f2", "f3"])
    # Construct y so each class has a distinct mean on f0 — gives the
    # trained model some signal to learn correct argmax.
    y_idx = np.array([i // 20 for i in range(n)])  # 11 classes
    rng.shuffle(y_idx)
    X["f0"] = y_idx.astype(float) + 0.2 * rng.normal(size=n)
    y = {"ITEM_class": y_idx.astype(int)}
    ts = {"ITEM_class": TargetType.MULTICLASS}

    est = JointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 30, "hidden_dim": 64, "num_layers": 2, "batch_size": 64, "lr": 1e-2, "seed": 0},
    )
    est.fit(X, y)
    # Class catalogue must be in natural integer order, NOT lex of str.
    assert est._multiclass_classes["ITEM_class"] == list(range(11)), (
        f"Class catalogue is in wrong order: {est._multiclass_classes['ITEM_class']}. "
        f"Expected [0, 1, ..., 10]. The pre-fix sort(key=str) would yield "
        f"[0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]."
    )
    preds = est.predict_targets_dict(X)["ITEM_class"]
    # Sanity: at least some predictions should agree with truth (the model
    # has trained on a separable signal). If the class catalogue ordering
    # were wrong, accuracy would be capped at ~1/K (random) even after
    # training, because the argmax index maps to the wrong label.
    acc = float(np.mean(preds == y_idx))
    assert acc > 0.3, (
        f"K=11 multiclass accuracy is {acc} — the model trained but "
        f"predictions don't match. Possible class-ordering regression."
    )


# ====================================================================== #
# Round 2 review fixes
# ====================================================================== #


def _build_validation_scaffold():
    """Shared mini-fixture for R2-3/4/5 inference-validator tests."""
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
# Fix R2-2: ROC_AUC degenerate-class returns NaN, not 0.0
# ---------------------------------------------------------------------- #


# --- M3: Joint fit rejects NaN in training y with a named error ---


def test_fix_r2_b3_joint_fit_rejects_nan_in_y():
    X = pd.DataFrame({"f0": [0.1, 0.2, -0.3]})
    y = {"ITEM_a": np.array([1.0, np.nan, 0.0])}
    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 4, "num_layers": 1, "batch_size": 4},
    )
    with pytest.raises(ValueError, match="NaN"):
        est.fit(X, y)


# --- M5: empty multiclass catalogue fails fast at eval time ---


def test_fix_p0_4_mlp_default_no_grad_clip():
    """Symmetric counter-test: vanilla joint MLP defaults to grad_clip_norm=None,
    so clip_grad_norm_ must NOT be called."""
    import torch.nn.utils as torch_nn_utils

    rng = np.random.default_rng(0)
    n = 16
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = {"ITEM_a": (X["f0"] > 0).astype(int).to_numpy()}
    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 8},
    )

    real_clip = torch_nn_utils.clip_grad_norm_
    captured = []

    def _capture(params, max_norm, *args, **kwargs):
        captured.append(max_norm)
        return real_clip(params, max_norm, *args, **kwargs)

    torch_nn_utils.clip_grad_norm_ = _capture
    try:
        est.fit(X, y)
    finally:
        torch_nn_utils.clip_grad_norm_ = real_clip
    assert not captured, f"MLP default should not clip but did with max_norm={captured}."


# --- P0-5: sample_training_mask requires generator ---


# ====================================================================== #
# P1 review round — fixes for tightening / hardening
# ====================================================================== #


def test_p1_7_regression_scaler_rejects_nan_input():
    """_RegressionScaler.fit must raise on all-NaN input rather than
    silently producing NaN scalers that propagate to NaN gradients."""
    from skrec.estimator.classification._joint_multi_target_base import _RegressionScaler

    scaler = _RegressionScaler()
    with pytest.raises(ValueError, match="non-finite"):
        scaler.fit("ITEM_rev", np.array([np.nan, np.nan, np.nan]))
    with pytest.raises(ValueError, match="non-finite"):
        scaler.fit("ITEM_rev", np.array([1.0, np.inf, 3.0]))
    # Sanity: normal finite input still works.
    scaler.fit("ITEM_rev", np.array([1.0, 2.0, 3.0]))
    assert "ITEM_rev" in scaler.scalers


def test_p1_8_reject_zero_or_negative_epochs():
    """epochs=0 / negative must raise at fit-time, not silently skip
    training and leave the model at random init."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
    y = {"ITEM_x": (rng.normal(size=20) > 0).astype(int)}
    ts = {"ITEM_x": TargetType.BINARY}

    for bad_epochs in (0, -1):
        est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": bad_epochs, "hidden_dim": 4})
        with pytest.raises(ValueError, match="epochs"):
            est.fit(X, y)

    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 5, "batch_size": 0, "hidden_dim": 4})
    with pytest.raises(ValueError, match="batch_size"):
        est.fit(X, y)


def test_p1_9_object_dtype_nan_in_multiclass_y_detected():
    """Object-dtype y with None / np.nan sprinkled in must trigger the NaN
    guard via the pd.isna fallback (not the float-cast path that raises)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
    # Object-dtype multiclass with one None
    y_class = np.array(["a", "b", None] + ["a"] * 17, dtype=object)
    y = {"ITEM_c": y_class}
    ts = {"ITEM_c": TargetType.MULTICLASS}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 2, "hidden_dim": 4})
    with pytest.raises(ValueError, match="NaN"):
        est.fit(X, y)


def test_p1_6_loss_target_count_invariant():
    """Joint loss reduction is mean-across-targets, not sum. The total
    loss for a 2-target problem should be roughly half the SUMMED-loss
    a sum-reduction would give — i.e., adding targets doesn't blow up
    the effective learning rate."""
    rng = np.random.default_rng(0)
    n = 32
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = {
        "ITEM_a": (rng.normal(size=n) > 0).astype(int),
        "ITEM_b": (rng.normal(size=n) > 0).astype(int),
    }
    ts = {"ITEM_a": TargetType.BINARY, "ITEM_b": TargetType.BINARY}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 4, "seed": 0})
    est.fit(X, y)

    # Reach into _compute_loss directly to verify the reduction.
    X_t = torch.from_numpy(X.to_numpy(dtype=np.float32)).to(est._device)
    y_t = {k: torch.from_numpy(v.astype(np.float32)).to(est._device) for k, v in y.items()}
    with torch.no_grad():
        hidden = est._encoder.forward(X_t)
        logits = est._heads(hidden)
        idx = torch.arange(n, device=est._device)
        loss = est._compute_loss(logits, y_t, idx).item()
    # Two BCE means averaged → roughly the magnitude of a single BCE,
    # not double. Loose bound: well under what summing-2-BCEs would give
    # (each BCE on random init ≈ ln(2) ≈ 0.69; sum would be ~1.4).
    assert loss < 1.2, f"Loss={loss}; expected mean-of-targets (~0.69), not sum (~1.4)."


def test_vanilla_validation_loss_unseen_multiclass_label_raises():
    """_build_train_tensors must raise an actionable ValueError when
    y_valid carries a MULTICLASS label not in the train catalogue.
    Vanilla counterpart to the conditional version — same fragility
    surfaces through the vanilla _validation_loss path."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["f0", "f1", "f2"])
    Xv = pd.DataFrame(rng.normal(size=(10, 3)), columns=["f0", "f1", "f2"])
    y = {"ITEM_c": np.array(["a", "b"] * 20)}
    yv = {"ITEM_c": np.array(["a", "zzz"] * 5)}
    ts = {"ITEM_c": TargetType.MULTICLASS}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 4, "seed": 0})
    with pytest.raises(ValueError, match="training-time class catalogue"):
        est.fit(X, y, X_valid=Xv, y_valid=yv)


def test_vanilla_y_valid_nan_raises_upfront():
    """y_valid carrying NaN must be rejected by _validate_for_fit just
    like y is — symmetric guard added in round 4. Previously NaN in
    y_valid reached _compute_loss via _validation_loss and contaminated
    logged per-epoch val numbers."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 2)), columns=["f0", "f1"])
    Xv = pd.DataFrame(rng.normal(size=(5, 2)), columns=["f0", "f1"])
    y = {"ITEM_r": rng.normal(size=20)}
    yv = {"ITEM_r": np.array([1.0, np.nan, 2.0, 3.0, 4.0])}
    ts = {"ITEM_r": TargetType.REGRESSION}
    est = JointMultiTargetMLPEstimator(target_specs=ts, params={"epochs": 1, "hidden_dim": 4, "seed": 0})
    with pytest.raises(ValueError, match="y_valid.*contains NaN"):
        est.fit(X, y, X_valid=Xv, y_valid=yv)


def test_empty_target_specs_rejected_at_init():
    """Empty target_specs would divide-by-zero in _compute_loss. Reject
    at __init__ so the misconfiguration surfaces before fit."""
    with pytest.raises(ValueError, match="non-empty"):
        JointMultiTargetMLPEstimator(target_specs={}, params={"epochs": 1})


@pytest.mark.parametrize(
    "estimator_cls,extra_params",
    [
        ("mlp", {"hidden_dim": 4, "num_layers": 1}),
        ("tx", {"d_model": 8, "n_heads": 2, "n_layers": 1, "ffn_dim": 16}),
    ],
)
def test_joint_loss_target_count_invariant_both_families(estimator_cls, extra_params):
    """Parametrize the loss-normalization invariant over Transformer
    too — _compute_loss is on the shared base, so a future override on
    one subclass that drops the mean-across-targets reduction would
    silently break the equivalent of p1_6."""
    import torch as _torch

    from skrec.estimator.classification import (
        JointMultiTargetMLPEstimator,
        JointMultiTargetTransformerEstimator,
    )

    cls = JointMultiTargetMLPEstimator if estimator_cls == "mlp" else JointMultiTargetTransformerEstimator
    rng = np.random.default_rng(0)
    n = 32
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = {
        "ITEM_a": (rng.normal(size=n) > 0).astype(int),
        "ITEM_b": (rng.normal(size=n) > 0).astype(int),
    }
    ts = {"ITEM_a": TargetType.BINARY, "ITEM_b": TargetType.BINARY}
    est = cls(
        target_specs=ts,
        params={"epochs": 1, "batch_size": 16, "seed": 0, **extra_params},
    )
    est.fit(X, y)
    X_t = _torch.from_numpy(X.to_numpy(dtype=np.float32)).to(est._device)
    y_t = {k: _torch.from_numpy(v.astype(np.float32)).to(est._device) for k, v in y.items()}
    with _torch.no_grad():
        hidden = est._encoder.forward(X_t)
        logits = est._heads(hidden)
        idx = _torch.arange(n, device=est._device)
        loss = est._compute_loss(logits, y_t, idx).item()
    # Two BCEs averaged ≈ one BCE (≈ ln 2 ≈ 0.69), NOT summed (~1.4).
    assert loss < 1.2, (
        f"{estimator_cls} loss={loss}; expected mean-of-targets (~0.69), "
        f"not sum (~1.4) — _compute_loss normalization regression?"
    )
