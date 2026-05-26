# Shared base for v3 conditional joint multi-target estimators.
#
# Extends JointMultiTargetBaseEstimator with three additions:
#
#   1. A label encoder (ConditionalLabelEncoder) sitting next to the
#      feature encoder. Per (row, target) raw chunks
#      (is_observed flag + value/one-hot/binary-vector) are projected to
#      label_embedding_dim and concatenated → fed as label_inputs to the
#      feature encoder's forward (label_input_dim is non-zero here, where
#      the v2 encoder Protocol always set it to 0).
#
#   2. Per-(row, target_specs-entry) Bernoulli masking at training time.
#      Loss is computed only on masked positions — the unmasked targets
#      are given to the model via label_inputs, so loss against them
#      would test memory, not generalization. Multilabel groups have a
#      single mask flag per row (members mask together — v3 locked
#      decision #4).
#
#   3. ``predict_with_observed(X, observed)`` for inference-time
#      conditioning. ``predict_proba_dict(X)`` falls out as the special
#      case of all-NaN observed (every target masked, predictions
#      from features alone — equivalent in spirit to the v2 vanilla path
#      modulo the label-channel contributing biases, see the
#      ``mask_prob=1.0 ≈ vanilla`` equivalence gate).
#
# Defense-in-depth zeroing on masked positions (gate 9 / mandatory
# leakage test): both the is_observed flag AND the value are zeroed,
# preventing any network path from picking up the actual label when
# supposedly masked. The build_raw_chunks utility in
# _conditional_label_encoding.py implements this.
#
# Implements the ConditionalMultiTargetEstimator runtime-checkable
# Protocol (the four MultiTargetEstimator attrs + predict_with_observed).

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    AdamW = None  # type: ignore[assignment]

from skrec.estimator.classification._conditional_label_encoding import (
    ConditionalLabelEncoder,
    build_mask_from_observed_dict,
    build_raw_chunks,
    sample_training_mask,
)
from skrec.estimator.classification._joint_multi_target_base import (
    JointMultiTargetBaseEstimator,
    _PerTargetHeads,
)
from skrec.estimator.classification._multi_target_protocol import (
    TargetGroupSpec,
    TargetType,
)
from skrec.util.logger import get_logger
from skrec.util.torch_device import select_torch_device

logger = get_logger(__name__)


class ConditionalJointMultiTargetBaseEstimator(JointMultiTargetBaseEstimator, ABC):
    """Abstract base for conditional joint estimators (MLP + Transformer).

    Subclasses inherit ``_build_encoder`` from
    :class:`JointMultiTargetBaseEstimator`; the encoder receives a non-zero
    ``label_input_dim`` that activates the label-channel path in
    ``MLPEncoder`` / ``TransformerEncoder``.
    """

    # Explicit opt-in sentinel for the ConditionalMultiTargetEstimator
    # Protocol. Required because @runtime_checkable Protocol isinstance
    # checks are structural — without this sentinel, any class that
    # happens to have a predict_with_observed method (correct contract or
    # not) would silently pass isinstance and activate OBSERVED handling
    # in the scorer. See _multi_target_protocol.py for the rationale.
    is_conditional_multi_target: bool = True

    DEFAULT_PARAMS = {
        **JointMultiTargetBaseEstimator.DEFAULT_PARAMS,
        "mask_prob": 0.5,
        "label_embedding_dim": 8,
    }

    def __init__(
        self,
        target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(target_specs=target_specs, params=params)
        mask_prob = float(self.params["mask_prob"])
        if not (0.0 < mask_prob <= 1.0):
            raise ValueError(
                f"mask_prob must be in (0, 1]; got {mask_prob}. "
                f"mask_prob=0 has no learning signal (every target is observed; "
                f"the model never has to predict anything)."
            )
        self._label_encoder: Optional[ConditionalLabelEncoder] = None

    # ------------------------------------------------------------------ #
    # Subclass hooks for label-channel shape.
    #
    # MLP-style consumers (the default) flat-concat per-target projections
    # into a single (batch, num_targets * embedding_dim) vector and feed it
    # alongside X. Transformer-style consumers project to ``d_model`` and
    # feed per-target tokens — one transformer token per declared target
    # — appended directly to the feature-token sequence. The hooks let the
    # Transformer subclass override both knobs in one place.
    # ------------------------------------------------------------------ #

    def _label_token_dim(self) -> int:
        """Per-target label embedding width.

        MLP default: ``params['label_embedding_dim']`` (8). Transformer
        subclass overrides to ``d_model`` so each per-target token slots
        directly into the FT-Transformer-style sequence at the right
        width.
        """
        return int(self.params["label_embedding_dim"])

    def _encoder_label_input_dim(self) -> int:
        """``label_input_dim`` to construct the feature encoder with.

        MLP default: ``num_targets * label_token_dim`` — the flat-concat
        vector is treated as extra scalar features by the MLP. Transformer
        subclass returns 0 because the per-target tokens are appended to
        the sequence in ``forward`` and don't need a scalar-token path.
        """
        if self._label_encoder is None:
            raise RuntimeError("_label_encoder must be built before this call.")
        return self._label_encoder.total_flat_dim

    def _format_label_inputs(self, raw_chunks: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        """Build the label_inputs tensor passed to the feature encoder.

        MLP default: ``(batch, num_targets * label_token_dim)`` flat tensor.
        Transformer subclass overrides to return
        ``(batch, num_targets, d_model)`` — 3-D, one token per target.
        """
        if self._label_encoder is None:
            raise RuntimeError("_label_encoder must be built before this call.")
        return self._label_encoder(raw_chunks)

    # ------------------------------------------------------------------ #
    # Fit (override)
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: pd.DataFrame,
        y: Dict[str, NDArray],
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[Dict[str, NDArray]] = None,
    ) -> "ConditionalJointMultiTargetBaseEstimator":
        self._validate_for_fit(X, y, X_valid, y_valid)
        self._feature_names = X.columns.tolist()
        self._set_seed(int(self.params["seed"]))
        self._device = select_torch_device(self.params["device"])

        # Catalogues + scalers (same as vanilla base).
        self._prepare_target_catalogues(y)

        # Build the label encoder + the feature encoder. Each per-target
        # raw chunk is projected to ``_label_token_dim()`` width. Two
        # consumer shapes:
        #   - MLP-style: flat concat → (batch, num_targets * dim);
        #     encoder built with label_input_dim = total_flat_dim.
        #   - Transformer-style: stacked per-target tokens →
        #     (batch, num_targets, dim); encoder receives them as 3-D
        #     label_inputs (one extra token per target appended to the
        #     feature-token sequence), so label_input_dim=0 — there's
        #     no scalar-token path to build.
        # Subclasses override the hooks; MLP uses the defaults.
        multiclass_counts = {k: len(v) for k, v in self._multiclass_classes.items()}
        per_target_dim = self._label_token_dim()
        self._label_encoder = ConditionalLabelEncoder(
            target_specs=self.target_specs,
            multiclass_class_counts=multiclass_counts,
            label_embedding_dim=per_target_dim,
        ).to(self._device)
        encoder_label_input_dim = self._encoder_label_input_dim()

        input_dim = X.shape[1]
        self._encoder = self._build_encoder(input_dim=input_dim, label_input_dim=encoder_label_input_dim)
        encoder_module: nn.Module = self._encoder  # type: ignore[assignment]
        encoder_module.to(self._device)
        self._heads = _PerTargetHeads(
            hidden_dim=self._encoder.hidden_dim,
            target_specs=self.target_specs,
            multiclass_class_counts=multiclass_counts,
        ).to(self._device)

        optimizer = AdamW(
            list(encoder_module.parameters()) + list(self._heads.parameters()) + list(self._label_encoder.parameters()),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )

        # Optional warmup scheduler + gradient clipping — same plumbing as
        # the vanilla joint base, via the shared _build_scheduler_and_clip
        # helper. Conditional Transformer inherits the
        # warmup_steps=100 / grad_clip_norm=1.0 defaults from its subclass
        # DEFAULT_PARAMS; conditional MLP leaves them off.
        scheduler, clip_norm = self._build_scheduler_and_clip(optimizer)
        all_params = (
            list(encoder_module.parameters()) + list(self._heads.parameters()) + list(self._label_encoder.parameters())
        )

        # Tensorize features + y (y_tensors are the un-encoded targets used
        # for loss; raw label chunks are computed per-batch from y too).
        X_t, y_tensors = self._build_train_tensors(X, y)

        # Regression scaler params for raw-chunk building (z-score normalize).
        reg_means = {k: v[0] for k, v in self._regression_scaler.scalers.items()}
        reg_stds = {k: v[1] for k, v in self._regression_scaler.scalers.items()}

        batch_size = int(self.params["batch_size"])
        epochs = int(self.params["epochs"])
        mask_prob = float(self.params["mask_prob"])
        n = X_t.shape[0]

        for epoch in range(epochs):
            encoder_module.train()
            self._heads.train()
            self._label_encoder.train()

            shuffle_gen = torch.Generator().manual_seed(int(self.params["seed"]) + epoch)
            perm = torch.randperm(n, generator=shuffle_gen)

            total_loss = 0.0
            n_contributing_batches = 0

            # Per-epoch torch.Generator for mask sampling — deterministic
            # given (seed, epoch).
            mask_gen = torch.Generator().manual_seed(int(self.params["seed"]) * 31 + epoch + 7)

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch_X = X_t[idx]
                bsz = idx.shape[0]

                # Per-batch mask: True = MASKED (unobserved at training).
                mask_dict = sample_training_mask(self.target_specs, bsz, mask_prob, generator=mask_gen)

                # Slice y to this batch (np), then build raw chunks with
                # the mask applied (defense-in-depth zero on masked).
                batch_y_np: Dict[str, NDArray] = {}
                for key, spec in self.target_specs.items():
                    arr = np.asarray(y[key])
                    if isinstance(spec, TargetType):
                        batch_y_np[key] = arr[idx.numpy()]
                    else:
                        batch_y_np[key] = arr[idx.numpy(), :]

                raw_chunks = build_raw_chunks(
                    y_or_observed=batch_y_np,
                    mask=mask_dict,
                    target_specs=self.target_specs,
                    multiclass_classes=self._multiclass_classes,
                    regression_means=reg_means,
                    regression_stds=reg_stds,
                    device=self._device,
                    # Training-time strictness: unseen multiclass labels
                    # must raise rather than silently demote (the
                    # catalogue is built from y itself, so unseen-at-
                    # train is a contract violation).
                    strict_unknown_labels=True,
                )
                label_inputs = self._format_label_inputs(raw_chunks)

                hidden = encoder_module(batch_X, label_inputs)
                per_target_logits = self._heads(hidden)

                loss = self._compute_masked_loss(per_target_logits, y_tensors, idx, mask_dict)

                # Edge case: when every target's mask is all-False for this
                # batch (no row was masked for any target), loss stays as
                # the zero-initialised tensor with no grad and backward()
                # would crash with "element 0 ... does not require grad."
                # Skip the optimizer step in that case — there's no signal
                # to learn from this batch anyway. Also skip the
                # total_loss bookkeeping: all-observed batches contribute
                # loss.item() == 0.0, which used to bias the logged
                # avg_loss downward (the denominator counted every batch
                # while the numerator only accumulated learning batches).
                # Track contributing batches separately so the logged
                # mean reflects the actual training signal.
                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    if clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=clip_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    total_loss += loss.item()
                    n_contributing_batches += 1

            avg_loss = total_loss / max(1, n_contributing_batches)
            logger.info(
                "Epoch [%d/%d] - Conditional Train Loss: %.4f",
                epoch + 1,
                epochs,
                avg_loss,
            )

            # Validation loss (conditional variant): evaluate with EVERY
            # target masked (observed=None / all-True mask) so the loss
            # measures vanilla-style prediction quality on the held-out
            # split. Without this, X_valid/y_valid were accepted by
            # _validate_for_fit but the per-epoch loop was a silent
            # no-op — no logged val loss, no early-stopping signal.
            #
            # _validate_for_fit upfront-rejects the asymmetric cases
            # (one of X_valid / y_valid None, the other not), so by the
            # time we reach here either BOTH are None (skip val loss) or
            # BOTH are not None (compute val loss). Assert the
            # invariant so a future refactor that bypasses the upfront
            # validator surfaces here rather than silently no-op'ing.
            assert (X_valid is None) == (y_valid is None), (
                "Internal: X_valid/y_valid asymmetry should have been "
                "rejected by _validate_for_fit before fit reached the "
                "per-epoch loop."
            )
            if X_valid is not None:
                val_loss = self._conditional_validation_loss(X_valid, y_valid)
                logger.info("  Conditional Validation Loss: %.4f", val_loss)

        return self

    def _conditional_validation_loss(self, X_valid: pd.DataFrame, y_valid: Dict[str, NDArray]) -> float:
        """One-pass validation loss with all targets masked.

        Mirrors :meth:`JointMultiTargetBaseEstimator._validation_loss`,
        but threads through the label-channel + masked-loss machinery so
        the number is comparable to the per-epoch train loss above. We
        use an all-True mask (every target unobserved) because validation
        time has no observed-label channel; this matches the inference-
        time predict_proba_dict(X) path that conditional callers actually
        hit in production.
        """
        encoder_module: nn.Module = self._encoder  # type: ignore[assignment]
        encoder_module.eval()
        self._heads.eval()  # type: ignore[union-attr]
        self._label_encoder.eval()  # type: ignore[union-attr]

        X_t, y_tensors = self._build_train_tensors(X_valid, y_valid)
        n_valid = X_t.shape[0]
        all_masked = {key: np.ones(n_valid, dtype=bool) for key in self.target_specs}
        reg_means = {k: v[0] for k, v in self._regression_scaler.scalers.items()}
        reg_stds = {k: v[1] for k, v in self._regression_scaler.scalers.items()}
        y_valid_np: Dict[str, NDArray] = {}
        for key, spec in self.target_specs.items():
            arr = np.asarray(y_valid[key])
            y_valid_np[key] = arr  # already correct shape per _validate_for_fit
        with torch.inference_mode():
            raw_chunks = build_raw_chunks(
                y_or_observed=y_valid_np,
                mask=all_masked,
                target_specs=self.target_specs,
                multiclass_classes=self._multiclass_classes,
                regression_means=reg_means,
                regression_stds=reg_stds,
                device=self._device,
                # Validation y is held-out training-like data; same
                # strictness as the train path — unseen labels indicate
                # a contract violation, not graceful inference noise.
                strict_unknown_labels=True,
            )
            label_inputs = self._format_label_inputs(raw_chunks)
            hidden = encoder_module(X_t, label_inputs)
            per_target_logits = self._heads(hidden)  # type: ignore[misc]
            all_idx = torch.arange(n_valid, device=self._device)
            loss = self._compute_masked_loss(per_target_logits, y_tensors, all_idx, all_masked)
        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Loss with mask
    # ------------------------------------------------------------------ #

    def _compute_masked_loss(
        self,
        per_target_logits: Dict[str, "torch.Tensor"],
        y_tensors: Dict[str, "torch.Tensor"],
        idx: "torch.Tensor",
        mask_dict: Dict[str, NDArray],
    ) -> "torch.Tensor":
        """Mean of per-target masked losses, applied only on MASKED rows.

        Mask semantics: True = MASKED = unobserved → loss IS computed on
        these positions (the network must predict them from features +
        unmasked context). False = OBSERVED → loss is NOT computed (the
        true value was provided to the network via label_inputs; scoring
        memory).

        Normalization: divide the summed per-target loss by
        ``len(target_specs)`` — the FULL declared target count, not the
        number of targets that happened to have ≥ 1 masked row in this
        batch. The earlier ``n_contributing`` denominator drifted batch
        to batch at low ``mask_prob`` (between 1/1 and 1/N), giving each
        target an effective LR that wobbled with mask sparsity. Vanilla
        always divides by ``len(target_specs)``; matching that here keeps
        the per-target gradient scale stable across batches and keeps
        the ``mask_prob=1.0 ≈ vanilla`` equivalence gate honest.
        """
        loss = torch.zeros((), device=self._device)
        any_target_contributed = False
        for key, spec in self.target_specs.items():
            # Empty-mask check on numpy BEFORE touching the device.
            # mask_t.sum() on a device tensor forces an implicit .item()
            # and stalls the GPU stream every batch — moving the check
            # to ``mask_dict[key].any()`` keeps the per-target skip free
            # of H↔D sync. (CPU runs are unaffected.)
            if not mask_dict[key].any():
                continue  # nothing to learn from in this batch for this target

            logits = per_target_logits[key]
            target = y_tensors[key][idx]
            mask_t = torch.from_numpy(mask_dict[key]).to(
                device=self._device, dtype=torch.float32
            )  # (bsz,) — 1 where masked
            # The empty-mask short-circuit happens on numpy above (host
            # side). The DENOMINATOR ``mask_t.sum()`` below stays on
            # device intentionally — it participates in the autograd
            # graph through the ``(per_row * mask_t).sum() / mask_t.sum()``
            # expression. Moving the denominator to numpy / .item()
            # would detach it from the graph and break gradients flowing
            # back through the masking. Do NOT "harmonize" by hoisting
            # the denominator off-device.

            if isinstance(spec, TargetType):
                if spec == TargetType.BINARY:
                    per_row = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), target, reduction="none")
                    loss = loss + (per_row * mask_t).sum() / mask_t.sum()
                elif spec == TargetType.REGRESSION:
                    per_row = nn.functional.mse_loss(logits.view(-1), target, reduction="none")
                    loss = loss + (per_row * mask_t).sum() / mask_t.sum()
                elif spec == TargetType.MULTICLASS:
                    per_row = nn.functional.cross_entropy(logits, target, reduction="none")
                    loss = loss + (per_row * mask_t).sum() / mask_t.sum()
            else:
                # Multilabel: per-member BCE; mask is per-row (group-level),
                # so all members share the same row mask.
                #
                # Note on parity with vanilla: the masked form
                # ``per_dim.mean(dim=1)`` followed by
                # ``(per_row * mask_t).sum() / mask_t.sum()`` matches
                # the vanilla ``reduction='mean'`` over the full
                # ``(bsz, n_members)`` tensor ONLY when every row is
                # masked. At partial masking they differ by exactly the
                # expected amount — vanilla averages over all rows×
                # members, masked averages over masked-rows then over
                # members. This asymmetry is intentional (masked rows
                # are the only learning signal in the conditional loop)
                # and the ``mask_prob=1.0 ≈ vanilla`` gate stays honest
                # because at mask_prob=1.0 every row is masked and the
                # two reductions agree. Do not "harmonize" without
                # re-checking that gate.
                per_dim = nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )  # (bsz, n_members)
                per_row = per_dim.mean(dim=1)
                loss = loss + (per_row * mask_t).sum() / mask_t.sum()
            any_target_contributed = True
        if not any_target_contributed:
            # Every target's mask was all-False for this batch — there is
            # nothing to learn from. Return a zero-grad tensor so the
            # caller's loss.requires_grad guard skips backward()/step().
            return loss
        # Normalize by the FULL declared target count, not by however
        # many targets had a masked row in this batch. See docstring.
        return loss / float(len(self.target_specs))

    # ------------------------------------------------------------------ #
    # _check_fitted override — the conditional path needs the label
    # encoder in addition to the base estimator's (_encoder, _heads,
    # _feature_names). Without this override, predict_with_observed
    # would NPE on a label-encoder that was cleared / never built
    # rather than producing the standard "not fitted yet" error.
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        super()._check_fitted()
        if self._label_encoder is None:
            raise RuntimeError(
                "Conditional estimator is not fully fitted yet: _label_encoder is None. Call fit(...) first."
            )

    # ------------------------------------------------------------------ #
    # Inference: predict_with_observed + predict_proba_dict override
    # ------------------------------------------------------------------ #

    def predict_with_observed(
        self,
        X: pd.DataFrame,
        observed: Optional[Dict[str, NDArray]] = None,
    ) -> Dict[str, NDArray]:
        """Per-target probabilities/values conditioned on ``observed`` ground truth.

        Args:
            X: Feature matrix.
            observed: Dict keyed by ``target_specs`` entries; per-row NaN
                marks "not observed for this row." Multilabel groups must
                mask together; the scorer's inference validator enforces
                this on the caller's side. Missing keys = fully unobserved
                for that target (every row masked).

        Returns:
            Same shape as :meth:`predict_proba_dict` — dict keyed by
            fanned-out target name.
        """
        self._check_fitted()
        X_aligned = self._align_X(X)
        X_t = self._to_tensor(X_aligned.to_numpy(dtype=np.float32))
        bsz = X_t.shape[0]

        # Build mask from observed dict (NaN means masked).
        observed = observed or {}
        mask_dict = build_mask_from_observed_dict(observed, self.target_specs, n_samples=bsz)

        # For missing target_specs entries (no observed key at all), the
        # build_raw_chunks call still needs a value array — supply a
        # zero/NaN placeholder of the right shape for the value chunk to
        # work; the mask is all-True so the values are zeroed anyway.
        #
        # Perf note (deferred from round 3): we allocate ``np.full(bsz,
        # nan)`` for every missing key then ``nan_to_num`` it to zero
        # inside build_raw_chunks. A future micro-optimization could
        # skip the alloc + nan-fill by passing a "fully masked" sentinel
        # straight to build_raw_chunks. Not done because (a) the alloc
        # is bsz floats per missing key — sub-millisecond at production
        # bsz; (b) the placeholder shape is asymmetric for multilabel
        # groups (2-D) and the sentinel path would need to mirror that
        # shape logic, adding a parallel code path to maintain. Revisit
        # if profiling flags this at very wide target_specs.
        observed_filled: Dict[str, NDArray] = {}
        for key, spec in self.target_specs.items():
            if key in observed:
                observed_filled[key] = np.asarray(observed[key])
            else:
                if isinstance(spec, TargetType):
                    observed_filled[key] = np.full(bsz, np.nan)
                else:
                    observed_filled[key] = np.full((bsz, len(spec["columns"])), np.nan)

        reg_means = {k: v[0] for k, v in self._regression_scaler.scalers.items()}
        reg_stds = {k: v[1] for k, v in self._regression_scaler.scalers.items()}

        encoder_module: nn.Module = self._encoder  # type: ignore[assignment]
        encoder_module.eval()
        self._heads.eval()  # type: ignore[union-attr]
        self._label_encoder.eval()  # type: ignore[union-attr]

        with torch.inference_mode():
            raw_chunks = build_raw_chunks(
                y_or_observed=observed_filled,
                mask=mask_dict,
                target_specs=self.target_specs,
                multiclass_classes=self._multiclass_classes,
                regression_means=reg_means,
                regression_stds=reg_stds,
                device=self._device,
            )
            label_inputs = self._format_label_inputs(raw_chunks)
            hidden = encoder_module(X_t, label_inputs)
            per_target_logits = self._heads(hidden)  # type: ignore[misc]

        # Reuse the vanilla decoder so the per-type logit→proba switch
        # lives in exactly one place. Before P2-4 this loop was duplicated
        # between the two predict paths; a new TargetType or activation
        # tweak would have to be applied twice.
        return self._decode_logits_to_proba(per_target_logits)

    def predict_proba_dict(self, X: pd.DataFrame) -> Dict[str, NDArray]:
        """Vanilla path: equivalent to ``predict_with_observed(X, {})``.

        Falls back through the conditional code path with an empty
        observed dict — every target is masked, so the raw label chunks
        for every target are all-zero (defense-in-depth zeroing on
        masked positions). The encoder then sees the
        ``ConditionalLabelEncoder.projections[key](zeros)`` output for
        each target, which is NOT exactly zero: each projection is an
        ``nn.Linear(raw_dim, embedding_dim)`` with a learned bias, so
        the masked-input projection collapses to the per-target bias
        vector. The vanilla-equivalence gate
        (``mask_prob=1.0 ≈ vanilla``) tolerates this through a wide
        AUC tolerance — the predictions match in shape and direction
        but not bit-for-bit.
        """
        return self.predict_with_observed(X, observed=None)
