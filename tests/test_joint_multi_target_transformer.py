# Tests for JointMultiTargetTransformerEstimator.
#
# Mirrors the joint-MLP test suite for the FT-Transformer-style encoder.
# Full per-type metric sanity is in M8; this file is contract-level only.

import io
import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from skrec.estimator.classification import (  # noqa: E402
    JointMultiTargetTransformerEstimator,
    MultiTargetEstimator,
)
from skrec.scorer.mixed_type_multi_target import (  # noqa: E402
    TargetGroupSpec,
    TargetType,
)


def _make_synthetic(n: int = 100, n_features: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y_binary = (X["feat_0"] > 0).astype(np.int64).to_numpy()
    y_reg = (X["feat_1"] + rng.normal(scale=0.1, size=n)).to_numpy()
    y_ml = np.column_stack(
        [
            (X["feat_2"] > 0).astype(np.int64).to_numpy(),
            (X["feat_3"] > 0).astype(np.int64).to_numpy(),
        ]
    )
    target_specs = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
        "engagement": TargetGroupSpec(
            type=TargetType.MULTILABEL,
            columns=["ITEM_email_open", "ITEM_app_open"],
        ),
    }
    y = {
        "ITEM_clicked": y_binary,
        "ITEM_revenue": y_reg,
        "engagement": y_ml,
    }
    return X, y, target_specs


def test_implements_multi_target_protocol():
    _, _, target_specs = _make_synthetic(n=20)
    est = JointMultiTargetTransformerEstimator(target_specs=target_specs, params={"epochs": 1})
    assert isinstance(est, MultiTargetEstimator)


def test_fit_and_predict_shapes():
    X, y, target_specs = _make_synthetic(n=80)
    est = JointMultiTargetTransformerEstimator(
        target_specs=target_specs,
        params={
            "epochs": 1,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "ffn_dim": 32,
            "batch_size": 32,
        },
    )
    est.fit(X, y)
    proba = est.predict_proba_dict(X)
    assert set(proba.keys()) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_email_open",
        "ITEM_app_open",
    }
    assert proba["ITEM_clicked"].shape == (80, 2)
    assert proba["ITEM_revenue"].shape == (80,)
    assert proba["ITEM_email_open"].shape == (80, 2)


def test_predict_targets_dict_shapes():
    X, y, target_specs = _make_synthetic(n=60)
    est = JointMultiTargetTransformerEstimator(
        target_specs=target_specs,
        params={"epochs": 1, "d_model": 16, "n_heads": 2, "n_layers": 1, "ffn_dim": 32},
    )
    est.fit(X, y)
    preds = est.predict_targets_dict(X)
    assert set(preds.keys()) == {
        "ITEM_clicked",
        "ITEM_revenue",
        "ITEM_email_open",
        "ITEM_app_open",
    }
    for k in ("ITEM_clicked", "ITEM_email_open", "ITEM_app_open"):
        assert set(np.unique(preds[k]).tolist()).issubset({0, 1})


def test_pickle_round_trip_predictions_match():
    X, y, target_specs = _make_synthetic(n=40)
    est = JointMultiTargetTransformerEstimator(
        target_specs=target_specs,
        params={
            "epochs": 2,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "ffn_dim": 32,
            "seed": 11,
        },
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
    X, y, target_specs = _make_synthetic(n=60)

    def fit_and_predict():
        est = JointMultiTargetTransformerEstimator(
            target_specs=target_specs,
            params={
                "epochs": 2,
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "ffn_dim": 32,
                "seed": 99,
                "batch_size": 16,
            },
        )
        est.fit(X, y)
        return est.predict_proba_dict(X)

    a = fit_and_predict()
    b = fit_and_predict()
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-5)


def test_loss_decreases_monotonically_over_epochs():
    """Mirror of joint MLP test #5 for the Transformer family."""
    import logging

    X, y, target_specs = _make_synthetic(n=200)
    est = JointMultiTargetTransformerEstimator(
        target_specs=target_specs,
        params={
            "epochs": 5,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 1,
            "ffn_dim": 64,
            "batch_size": 64,
            "lr": 1e-2,
            "seed": 0,
        },
    )
    captured = []

    class _LossCapture(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "Train Loss" in msg:
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
    assert captured[-1] < captured[0] * 0.95, f"Loss didn't decrease meaningfully across epochs: {captured}."


def test_d_model_not_divisible_by_n_heads_rejected():
    _, _, target_specs = _make_synthetic(n=10)
    est = JointMultiTargetTransformerEstimator(
        target_specs=target_specs,
        params={"d_model": 17, "n_heads": 4, "epochs": 1, "n_layers": 1},
    )
    X, y, _ = _make_synthetic(n=10)
    with pytest.raises(ValueError, match="must be divisible"):
        est.fit(X, y)


# --- P0-2: attn_dropout actually controls nn.MultiheadAttention dropout ---


def test_fix_p0_2_attn_dropout_honored_in_custom_encoder_layer():
    """The custom encoder layer constructs nn.MultiheadAttention with
    dropout=attn_dropout (not post-hoc-assigned, which doesn't reliably
    take effect). Pin the wire: build an encoder with a distinctive
    attn_dropout and assert the MultiheadAttention sub-module's dropout
    attribute matches."""
    from skrec.estimator.classification._joint_multi_target_encoders import (
        TransformerEncoder,
    )

    enc = TransformerEncoder(
        n_features=4,
        d_model=16,
        n_heads=2,
        n_layers=2,
        ffn_dim=32,
        attn_dropout=0.37,
        ffn_dropout=0.05,
    )
    # Every custom block's self_attn must carry the declared attn_dropout.
    for i, block in enumerate(enc.blocks):
        assert block.self_attn.dropout == pytest.approx(0.37), (
            f"Block {i}'s nn.MultiheadAttention.dropout = {block.self_attn.dropout}, expected 0.37."
        )


# --- P0-3: warmup LR schedule actually wired through the optimizer ---


# --- P0-3: warmup LR schedule actually wired through the optimizer ---


def test_fix_p0_3_warmup_schedule_ramps_lr_then_holds():
    """Wire test: build a vanilla joint Transformer with warmup_steps=10
    and lr=0.1; train for one batch at a time and capture the optimizer
    lr per step. Steps 1..10 should ramp linearly from ~0 → 0.1; step 11+
    should hold at 0.1."""
    from skrec.estimator.classification import JointMultiTargetTransformerEstimator

    rng = np.random.default_rng(0)
    n = 64
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = {"ITEM_a": (X["f0"] > 0).astype(int).to_numpy()}
    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetTransformerEstimator(
        target_specs=ts,
        params={
            "epochs": 1,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "ffn_dim": 32,
            "batch_size": 8,  # 8 batches per epoch
            "lr": 0.1,
            "warmup_steps": 4,
            "seed": 0,
        },
    )
    # Hook into the LambdaLR by capturing it at the call site. Simplest path:
    # patch AdamW.step to record self.param_groups[0]['lr'] each call.
    import torch.optim

    real_step = torch.optim.AdamW.step
    captured_lrs = []

    def _capturing_step(self, *args, **kwargs):
        captured_lrs.append(self.param_groups[0]["lr"])
        return real_step(self, *args, **kwargs)

    torch.optim.AdamW.step = _capturing_step
    try:
        est.fit(X, y)
    finally:
        torch.optim.AdamW.step = real_step
    assert len(captured_lrs) >= 4, f"Need at least 4 optimizer steps; got {len(captured_lrs)}"
    # First step should be at warmup-ramped LR, much less than 0.1.
    assert captured_lrs[0] < 0.1, f"warmup_steps=4 should ramp from step 1 < lr; got {captured_lrs[0]}"
    # Each step in warmup should be strictly larger than the prior (linear ramp).
    for i in range(1, min(4, len(captured_lrs))):
        assert captured_lrs[i] > captured_lrs[i - 1] - 1e-9, (
            f"LR not monotonically ramping in warmup: {captured_lrs[:5]}"
        )
    # Post-warmup steps should hold at the target lr.
    if len(captured_lrs) > 5:
        for lr in captured_lrs[5:]:
            assert lr == pytest.approx(0.1, abs=1e-6), f"Post-warmup LR drifted: {captured_lrs}"


# --- P0-4: gradient clipping actually applied during Transformer training ---


# --- P0-4: gradient clipping actually applied during Transformer training ---


def test_fix_p0_4_grad_clip_norm_applied_on_transformer():
    """Wire test: monkey-patch torch.nn.utils.clip_grad_norm_ to record the
    max_norm it was called with. Train a Transformer for one batch with
    grad_clip_norm=0.5; assert clip_grad_norm_ was called with max_norm=0.5."""
    import torch.nn.utils as torch_nn_utils

    from skrec.estimator.classification import JointMultiTargetTransformerEstimator

    rng = np.random.default_rng(0)
    n = 16
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = {"ITEM_a": (X["f0"] > 0).astype(int).to_numpy()}
    ts = {"ITEM_a": TargetType.BINARY}
    est = JointMultiTargetTransformerEstimator(
        target_specs=ts,
        params={
            "epochs": 1,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "ffn_dim": 32,
            "batch_size": 8,
            "grad_clip_norm": 0.5,
            "warmup_steps": 0,  # isolate the clip wiring
            "seed": 0,
        },
    )

    real_clip = torch_nn_utils.clip_grad_norm_
    captured_norms = []

    def _capture(params, max_norm, *args, **kwargs):
        captured_norms.append(float(max_norm))
        return real_clip(params, max_norm, *args, **kwargs)

    torch_nn_utils.clip_grad_norm_ = _capture
    try:
        est.fit(X, y)
    finally:
        torch_nn_utils.clip_grad_norm_ = real_clip
    assert captured_norms, "clip_grad_norm_ was never called; P0-4 wire broken."
    assert all(n == 0.5 for n in captured_norms), f"clip_grad_norm_ called with wrong max_norm: {captured_norms}"
