# Shared base for joint multi-target estimators (MLP + Transformer).
#
# Architecture:
#   - JointMultiTargetEncoder (internal Protocol): the encoder produces a
#     fixed-width hidden representation from input features (and optionally
#     from label inputs — v3 hook only; v2 always passes label_input_dim=0).
#   - _PerTargetHeads (nn.Module): one head per declared target. Each head
#     is a linear projection from hidden_dim to the target's output size,
#     followed by the appropriate loss in the training loop:
#         BINARY                → Linear(hidden_dim, 1)   + BCEWithLogitsLoss
#         REGRESSION            → Linear(hidden_dim, 1)   + MSELoss
#         MULTICLASS            → Linear(hidden_dim, K_k) + CrossEntropyLoss
#         MULTILABEL group      → Linear(hidden_dim, K_k) + per-dim BCE
#   - JointMultiTargetBaseEstimator: holds the encoder + heads, runs the
#     training loop, exposes the MultiTargetEstimator Protocol contract
#     (predict_proba_dict / predict_targets_dict). Subclasses provide the
#     encoder via _build_encoder(input_dim, label_input_dim).
#
# Dict-y is a structural choice (not a sklearn-y array): each target has its
# own shape, loss, and output decoder. Sklearn's CV utilities don't apply.
# The Protocol-conforming fit method override is on this base; v2 callers
# go directly through estimator.fit(X, y_dict).
#
# Note: v3 hook — label_input_dim is plumbed through every encoder
# constructor with a default of 0. v3 conditional estimators pass non-zero
# values to feed per-target label encodings (is_observed flag + value) as
# extra inputs. v2 leaves it at 0; the encoder code path treats >0 as a
# concatenation gate. Do NOT remove label_input_dim from the constructor
# signature even though v2 doesn't use it.
#
# Regression target z-score normalization: regression targets are scaled to
# mean=0, std=1 per target before training (helps loss balance across
# heterogeneous target scales). Scalers are stored on the estimator and
# applied inversely at predict time, so callers always see the original
# value scale.
#
# Multiclass target class catalogue: at fit time, each multiclass target's
# unique labels are captured in declared sort order and frozen. Predictions
# are mapped back to original labels at predict time (matches the v2
# "preserve input label dtype" output contract).

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

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

from skrec.estimator.classification._multi_target_protocol import (
    TargetGroupSpec,
    TargetType,
    sort_multiclass_labels,
)
from skrec.util.logger import get_logger
from skrec.util.torch_device import select_torch_device

logger = get_logger(__name__)


_nn_module = nn.Module if nn is not None else object


# ---------------------------------------------------------------------- #
# Encoder Protocol (internal — joint family only)
# ---------------------------------------------------------------------- #


@runtime_checkable
class JointMultiTargetEncoder(Protocol):
    """Internal Protocol for the joint family's feature encoder.

    Implementations: MLPEncoder, TransformerEncoder. Both produce a 2-D
    tensor ``(batch, hidden_dim)`` from input features (and optionally from
    label inputs — v3 hook).

    Attributes:
        hidden_dim: Dimensionality of the encoder's output representation.
            Heads project from this dimension to per-target output sizes.

    Note: v3 hook — label_input_dim > 0 activates the label-input path on
    conditional estimators. v2 always passes label_input_dim=0; encoders
    must treat that case as feature-only.
    """

    hidden_dim: int

    def forward(self, X: "torch.Tensor", label_inputs: Optional["torch.Tensor"] = None) -> "torch.Tensor": ...


# ---------------------------------------------------------------------- #
# Per-target heads
# ---------------------------------------------------------------------- #


class _PerTargetHeads(_nn_module):
    """One linear head per declared target.

    Naming: for simple targets, the head's key is the target_specs key
    (which IS the column name, ITEM_-prefixed). For multilabel groups, the
    head's key is the group key (one head per group; the head's output
    dimension is the number of group members).
    """

    def __init__(
        self,
        hidden_dim: int,
        target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
        multiclass_class_counts: Dict[str, int],
    ) -> None:
        super().__init__()
        self.target_specs = target_specs
        self.multiclass_class_counts = multiclass_class_counts
        heads = {}
        for key, spec in target_specs.items():
            if isinstance(spec, TargetType):
                if spec == TargetType.BINARY:
                    heads[key] = nn.Linear(hidden_dim, 1)
                elif spec == TargetType.REGRESSION:
                    heads[key] = nn.Linear(hidden_dim, 1)
                elif spec == TargetType.MULTICLASS:
                    k = multiclass_class_counts[key]
                    heads[key] = nn.Linear(hidden_dim, k)
            else:  # TargetGroupSpec (multilabel)
                n_members = len(spec["columns"])
                heads[key] = nn.Linear(hidden_dim, n_members)
        # nn.ModuleDict requires PyTorch-safe keys (no dots etc.) — target
        # names are ITEM_<name> or plain identifiers so this is safe.
        self.heads = nn.ModuleDict(heads)

    def forward(self, hidden: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        """Run every head against the shared hidden representation.

        Returns:
            Dict keyed by target_specs entries:
              BINARY     → (batch, 1)
              REGRESSION → (batch, 1)
              MULTICLASS → (batch, K_k)
              MULTILABEL → (batch, n_members)
        """
        return {key: head(hidden) for key, head in self.heads.items()}


# ---------------------------------------------------------------------- #
# Regression z-score scaler
# ---------------------------------------------------------------------- #


class _RegressionScaler:
    """Per-regression-target z-score normalization."""

    def __init__(self) -> None:
        self.scalers: Dict[str, Tuple[float, float]] = {}  # name -> (mean, std)

    def fit(self, name: str, values: NDArray) -> None:
        mean = float(np.mean(values))
        std = float(np.std(values))
        # NaN/inf in (mean, std) would silently propagate through training as
        # NaN gradients. All-NaN input → np.mean/std return NaN; inf-
        # contaminated input → either is inf. Reject explicitly so the caller
        # sees the offending target name, not "loss is nan" 10 epochs later.
        if not (np.isfinite(mean) and np.isfinite(std)):
            raise ValueError(
                f"Regression target {name!r} produced non-finite "
                f"statistics (mean={mean}, std={std}). Clean NaN/inf out of "
                f"y[{name!r}] before fitting."
            )
        if std < 1e-9:
            std = 1.0  # avoid div-by-zero on constant targets
        self.scalers[name] = (mean, std)

    def transform(self, name: str, values: NDArray) -> NDArray:
        if name not in self.scalers:
            return values
        mean, std = self.scalers[name]
        return (values - mean) / std

    def inverse_transform(self, name: str, values: NDArray) -> NDArray:
        if name not in self.scalers:
            return values
        mean, std = self.scalers[name]
        return values * std + mean


# ---------------------------------------------------------------------- #
# Joint multi-target base estimator
# ---------------------------------------------------------------------- #


class JointMultiTargetBaseEstimator(ABC):
    """Abstract base for joint multi-target torch estimators.

    Concrete subclasses (joint MLP, joint Transformer) supply an encoder via
    :meth:`_build_encoder`. The rest of the training / prediction loop lives
    here.

    Attributes:
        target_specs: Per-target schema (must match the scorer's).
        params: Flat dict of hyperparameters (subclasses provide
            ``DEFAULT_PARAMS``).
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "batch_size": 1024,
        "epochs": 10,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "regression_normalize": True,
        "device": None,
        "seed": 42,
        # Family-agnostic training-stability knobs. Subclasses override
        # DEFAULT_PARAMS to set Transformer-appropriate values; the MLP
        # subclass keeps them at the conservative defaults below.
        # grad_clip_norm: None → no clipping; float → torch.nn.utils.clip_
        # grad_norm_. Plan risk #3 ("Transformer training instability")
        # names gradient clipping as the mitigation; MLPs train stably
        # without it so the default is off.
        "grad_clip_norm": None,
        # warmup_steps: 0 → no warmup; positive int → linear warmup from
        # 0 → lr over the first ``warmup_steps`` optimizer steps via a
        # LambdaLR scheduler. Transformer subclass turns this on by
        # default; MLP leaves it at 0.
        "warmup_steps": 0,
    }

    def __init__(
        self,
        target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "Joint multi-target estimators require PyTorch. Install with `pip install scikit-rec[torch]`."
            )
        # ``_compute_loss`` divides the summed per-target loss by
        # ``len(target_specs)``; an empty dict here produces a
        # divide-by-zero → NaN gradient → silent training failure.
        # ``_compute_masked_loss`` in the conditional base has the
        # same precondition. Reject upfront so the misconfiguration
        # surfaces at construction.
        if not target_specs:
            raise ValueError(
                "target_specs must be non-empty. An empty target_specs "
                "would divide-by-zero in _compute_loss and silently "
                "produce NaN gradients."
            )
        self.target_specs = target_specs
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        # State populated by .fit()
        self._encoder: Optional[JointMultiTargetEncoder] = None
        self._heads: Optional[_PerTargetHeads] = None
        self._regression_scaler = _RegressionScaler()
        # name -> ordered class labels (preserves original dtype on round-trip)
        self._multiclass_classes: Dict[str, List[Any]] = {}
        self._feature_names: Optional[List[str]] = None
        self._device: Optional[str] = None

    # ----- Subclass hook --------------------------------------------- #

    @abstractmethod
    def _build_encoder(self, input_dim: int, label_input_dim: int = 0) -> JointMultiTargetEncoder:
        """Instantiate and return the encoder for this estimator family.

        Args:
            input_dim: Number of feature columns in X.
            label_input_dim: v3 hook; v2 always 0.

        Returns:
            An ``nn.Module`` satisfying :class:`JointMultiTargetEncoder`.
        """

    # ----- Public Protocol contract ---------------------------------- #

    def fit(
        self,
        X: pd.DataFrame,
        y: Dict[str, NDArray],
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[Dict[str, NDArray]] = None,
    ) -> "JointMultiTargetBaseEstimator":
        """Train the joint network on ``(X, y_dict)`` for ``epochs`` epochs.

        Args:
            X: Feature DataFrame.
            y: Dict keyed by ``target_specs`` entries. Simple targets: 1-D.
                Multilabel groups: 2-D ``(n_samples, n_members)``.
            X_valid: Optional validation features.
            y_valid: Optional validation labels in the same dict shape.

        Returns:
            self.
        """
        self._validate_for_fit(X, y, X_valid, y_valid)
        self._feature_names = X.columns.tolist()

        self._set_seed(int(self.params["seed"]))

        self._device = select_torch_device(self.params["device"])

        # Capture multiclass class catalogues + regression scalers.
        self._prepare_target_catalogues(y)

        # Build encoder + heads.
        input_dim = X.shape[1]
        self._encoder = self._build_encoder(input_dim=input_dim, label_input_dim=0)
        # mypy: _encoder is an nn.Module via our subclass return.
        encoder_module: nn.Module = self._encoder  # type: ignore[assignment]
        encoder_module.to(self._device)

        multiclass_counts = {k: len(v) for k, v in self._multiclass_classes.items()}
        self._heads = _PerTargetHeads(
            hidden_dim=self._encoder.hidden_dim,
            target_specs=self.target_specs,
            multiclass_class_counts=multiclass_counts,
        ).to(self._device)

        # Optimizer covers both encoder + heads.
        optimizer = AdamW(
            list(encoder_module.parameters()) + list(self._heads.parameters()),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )

        # Prepare training tensors.
        X_t, y_tensors = self._build_train_tensors(X, y)

        batch_size = int(self.params["batch_size"])
        epochs = int(self.params["epochs"])
        n = X_t.shape[0]

        scheduler, clip_norm = self._build_scheduler_and_clip(optimizer)

        for epoch in range(epochs):
            encoder_module.train()
            self._heads.train()
            # Deterministic shuffle keyed by (seed, epoch). The generator
            # always lives on CPU because torch.randperm with a seeded
            # ``torch.Generator()`` only works on CPU in stable PyTorch
            # (the CUDA generator API doesn't accept randperm). The
            # resulting CPU ``perm`` tensor is then used to fancy-index
            # ``X_t`` (which may be on GPU); PyTorch promotes the index
            # tensor with an implicit H→D transfer every batch. For the
            # batch sizes we run (≤ 4096 int64s per batch on CPU side)
            # this transfer is microseconds and stays well below the
            # encoder forward cost. We accept the trade in exchange for
            # cross-device reproducibility — bit-identical shuffles on
            # CPU + GPU make the determinism gate apply to both. Drop the
            # generator path (or move it to ``perm.to(device)`` once
            # before slicing) if a future profile shows the H→D promote
            # is non-trivial for large batch sizes on a fast GPU.
            generator = torch.Generator().manual_seed(int(self.params["seed"]) + epoch)
            perm = torch.randperm(n, generator=generator)

            total_loss = 0.0
            n_batches = max(1, (n + batch_size - 1) // batch_size)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch_X = X_t[idx]
                hidden = encoder_module(batch_X)
                per_target_logits = self._heads(hidden)
                loss = self._compute_loss(per_target_logits, y_tensors, idx)
                optimizer.zero_grad()
                loss.backward()
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder_module.parameters()) + list(self._heads.parameters()),
                        max_norm=clip_norm,
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / n_batches
            logger.info("Epoch [%d/%d] - Train Loss: %.4f", epoch + 1, epochs, avg_loss)

            if X_valid is not None and y_valid is not None:
                val_loss = self._validation_loss(X_valid, y_valid)
                logger.info("  Validation Loss: %.4f", val_loss)

        return self

    def predict_proba_dict(self, X: pd.DataFrame) -> Dict[str, NDArray]:
        """Per-target probabilities/values, multilabel groups fanned out."""
        self._check_fitted()
        X_aligned = self._align_X(X)
        X_t = self._to_tensor(X_aligned.to_numpy(dtype=np.float32))

        encoder_module: nn.Module = self._encoder  # type: ignore[assignment]
        encoder_module.eval()
        self._heads.eval()  # type: ignore[union-attr]

        with torch.inference_mode():
            hidden = encoder_module(X_t)
            per_target_logits = self._heads(hidden)  # type: ignore[misc]

        return self._decode_logits_to_proba(per_target_logits)

    def _build_scheduler_and_clip(
        self, optimizer: "torch.optim.Optimizer"
    ) -> Tuple[Optional["torch.optim.lr_scheduler.LambdaLR"], Optional[float]]:
        """Build the optional warmup LambdaLR scheduler + parse grad-clip.

        Shared between the vanilla joint base and the conditional joint
        base so the two ``fit`` loops can't drift on warmup / clip
        plumbing. Reads:
          - ``params['warmup_steps']``: 0 → no scheduler; positive int →
            linear warmup from 0 → lr over the first N optimizer steps,
            then constant. Plan risk #3 mitigation (Transformer training
            stability); MLP defaults to 0, Transformer subclass to 100.
          - ``params['grad_clip_norm']``: ``None`` → no clipping; float
            → max-norm cap for torch.nn.utils.clip_grad_norm_. MLP
            default ``None``; Transformer subclass default 1.0.
        """
        warmup_steps = int(self.params.get("warmup_steps", 0) or 0)
        scheduler = None
        if warmup_steps > 0:
            from torch.optim.lr_scheduler import LambdaLR

            def _lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                return 1.0

            scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
        grad_clip = self.params.get("grad_clip_norm")
        clip_norm = float(grad_clip) if grad_clip is not None else None
        return scheduler, clip_norm

    def _decode_logits_to_proba(self, per_target_logits: Dict[str, "torch.Tensor"]) -> Dict[str, NDArray]:
        """Convert per-target head logits → user-facing proba/value dict.

        Pulled out of ``predict_proba_dict`` so the conditional subclass's
        ``predict_with_observed`` reuses the same decoder rather than
        copy-pasting the per-target type switch (which is exactly where a
        new TargetType or activation tweak would otherwise drift).

        Per target type:
          - BINARY:     logits → sigmoid → (n, 2) (1 - p1, p1) stack.
          - REGRESSION: logits → inverse z-score → (n,).
          - MULTICLASS: logits → softmax → (n, K).
          - MULTILABEL: logits → per-member sigmoid → fan out into
                        one (n, 2) entry per member column.
        """
        out: Dict[str, NDArray] = {}
        for key, spec in self.target_specs.items():
            logits = per_target_logits[key].detach().cpu().numpy()
            if isinstance(spec, TargetType):
                if spec == TargetType.BINARY:
                    p1 = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
                    out[key] = np.column_stack([1.0 - p1, p1])
                elif spec == TargetType.REGRESSION:
                    values = logits.reshape(-1)
                    out[key] = self._regression_scaler.inverse_transform(key, values)
                elif spec == TargetType.MULTICLASS:
                    # softmax over K
                    logits_2d = logits  # (n, K)
                    e = np.exp(logits_2d - logits_2d.max(axis=1, keepdims=True))
                    out[key] = e / e.sum(axis=1, keepdims=True)
            else:  # multilabel group — fan out per member
                # logits shape (n, n_members). Each member becomes (n, 2).
                p1_per_member = 1.0 / (1.0 + np.exp(-logits))
                for i, member_col in enumerate(spec["columns"]):
                    p1 = p1_per_member[:, i]
                    out[member_col] = np.column_stack([1.0 - p1, p1])
        return out

    def predict_targets_dict(self, X: pd.DataFrame) -> Dict[str, NDArray]:
        """Per-target point estimates, multilabel groups fanned out."""
        proba = self.predict_proba_dict(X)
        out: Dict[str, NDArray] = {}
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                if spec == TargetType.BINARY:
                    out[key] = (proba[key][:, 1] >= 0.5).astype(np.int64)
                elif spec == TargetType.REGRESSION:
                    out[key] = proba[key]  # already de-normalized values
                elif spec == TargetType.MULTICLASS:
                    class_labels = self._multiclass_classes[key]
                    arg = proba[key].argmax(axis=1)
                    # Preserve original label dtype.
                    out[key] = np.array([class_labels[i] for i in arg])
            else:  # multilabel — per member
                for member_col in spec["columns"]:
                    out[member_col] = (proba[member_col][:, 1] >= 0.5).astype(np.int64)
        return out

    # ----- Internals ------------------------------------------------- #

    def _check_y_dict_shapes_and_nan(
        self,
        y_dict: Dict[str, NDArray],
        n: int,
        y_label: str = "y",
    ) -> None:
        """Per-target shape + NaN guard for a dict-y at fit / validation time.

        Shared by the training-y check and the validation-y check (a
        symmetric guard added in round 4 to stop NaN in ``y_valid``
        from contaminating logged per-epoch validation loss). Raises a
        named ValueError per offending target so the caller can locate
        the problem.

        Args:
            y_dict: Dict to check, keyed by target_specs entries.
            n: Expected row count (X.shape[0] or X_valid.shape[0]).
            y_label: Used in error messages so the caller knows whether
                ``y`` or ``y_valid`` is at fault.
        """
        if set(y_dict.keys()) != set(self.target_specs.keys()):
            raise ValueError(
                f"{y_label} keys must match target_specs keys exactly. "
                f"{y_label}={sorted(y_dict.keys())} "
                f"target_specs={sorted(self.target_specs.keys())}."
            )
        for key, spec in self.target_specs.items():
            arr = np.asarray(y_dict[key])
            if isinstance(spec, TargetType):
                if arr.ndim != 1 or arr.shape[0] != n:
                    raise ValueError(f"{y_label}[{key!r}] must be 1-D with {n} samples; got shape {arr.shape}.")
            else:
                expected_members = len(spec["columns"])
                if arr.ndim != 2 or arr.shape != (n, expected_members):
                    raise ValueError(
                        f"{y_label}[{key!r}] (multilabel group) must be 2-D "
                        f"with shape ({n}, {expected_members}); got {arr.shape}."
                    )
            # NaN guard. For MULTICLASS the labels may be non-numeric
            # (e.g. 'class_A' strings) — np.isnan would TypeError. Try a
            # numeric cast first; fall back to pd.isna which handles
            # object dtype + mixed types + Python None correctly.
            try:
                has_nan = bool(np.isnan(arr.astype(np.float64)).any())
            except (TypeError, ValueError):
                has_nan = bool(pd.isna(arr).any())
            if has_nan:
                raise ValueError(
                    f"{y_label}[{key!r}] contains NaN value(s). Training "
                    f"y / validation y must be NaN-free (the scorer's "
                    f"validator enforces this upstream; if you're fitting "
                    f"the estimator directly, clean y first)."
                )

    def _validate_for_fit(
        self,
        X: pd.DataFrame,
        y: Dict[str, NDArray],
        X_valid: Optional[pd.DataFrame],
        y_valid: Optional[Dict[str, NDArray]],
    ) -> None:
        """Dict-y aware validation.

        BaseEstimator._validate_for_fit assumes y has ``.shape``; that breaks
        on dict y. Override on the estimator side (this method) is the
        single point where the dict shape is enforced.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame; got {type(X).__name__}.")
        if not isinstance(y, dict):
            raise TypeError(f"y must be a dict keyed by target_specs entries; got {type(y).__name__}.")
        if set(y.keys()) != set(self.target_specs.keys()):
            raise ValueError(
                f"y keys must match target_specs keys exactly. "
                f"y={sorted(y.keys())} target_specs={sorted(self.target_specs.keys())}."
            )
        # Per-target shape checks + NaN guard. The scorer's training-time
        # validator rejects null targets upstream, but when callers fit the
        # estimator directly (bypassing the scorer), NaN would propagate
        # silently into gradients via _compute_loss. Fail fast here with a
        # named error per offending target.
        self._check_y_dict_shapes_and_nan(y, n=X.shape[0], y_label="y")

        if X_valid is not None and y_valid is None:
            raise ValueError("X_valid provided but y_valid is None.")
        if y_valid is not None and X_valid is None:
            raise ValueError("y_valid provided but X_valid is None.")

        # Symmetric NaN/shape guard for y_valid. Pre-fix only ``y`` was
        # checked; NaN in ``y_valid`` reached _compute_loss via
        # _validation_loss (and _conditional_validation_loss) and
        # contaminated the logged per-epoch val numbers with NaN from
        # epoch 1 onward. Fail fast here, mirroring the y guard.
        if y_valid is not None and X_valid is not None:
            self._check_y_dict_shapes_and_nan(y_valid, n=X_valid.shape[0], y_label="y_valid")

        # epochs=0 or negative would silently skip training entirely and
        # leave the encoder + heads at random init — predict_proba then
        # returns garbage. Fail upfront so the misconfiguration surfaces.
        epochs = int(self.params["epochs"])
        if epochs < 1:
            raise ValueError(
                f"epochs must be a positive integer; got {epochs}. "
                f"A value of 0 would skip training and leave the model at "
                f"random initialization."
            )
        batch_size = int(self.params["batch_size"])
        if batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer; got {batch_size}.")

    def _set_seed(self, seed: int) -> None:
        """Plumb the seed through every RNG source the training loop touches.

        Determinism contract (v2):
          - Single-process CPU + ``torch.randperm`` + seeded ``torch.Generator``
            for batch shuffling (NOT DataLoader / num_workers — keeps the
            wire as flat as possible and sidesteps worker-fork RNG issues).
          - ``torch.use_deterministic_algorithms(True, warn_only=True)`` for
            ops that have deterministic implementations; ``warn_only`` so
            ops that don't (e.g. some CUDA kernels) degrade to a warning
            instead of crashing the run.
          - ``torch.manual_seed`` for module init (Linear, etc.).
          - ``np.random.seed`` for any numpy randomness on the data path.

        GPU + multi-process is NOT a guaranteed-deterministic surface in v2;
        the determinism tests pin CPU and rely on the above.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
            # Surface the GPU-determinism contract gap once per fit so a
            # caller running on CUDA isn't surprised when bit-for-bit
            # reproducibility doesn't hold across runs. v2's determinism
            # tests pin CPU only; some CUDA kernels lack deterministic
            # implementations and downgrade to a warning under
            # ``use_deterministic_algorithms(warn_only=True)`` below.
            logger.warning(
                "CUDA detected: scikit-rec joint multi-target estimators "
                "guarantee deterministic training only on CPU. GPU runs "
                "may differ across executions on ops without deterministic "
                "kernels. Set device='cpu' for bit-reproducibility."
            )
        # warn_only=True: don't blow up if a CUDA op lacks a deterministic
        # path; surface the warning so users on non-CPU paths know.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (TypeError, RuntimeError):  # pragma: no cover
            # Older torch versions may not support warn_only or the API at all.
            pass
        np.random.seed(seed)

    def _prepare_target_catalogues(self, y: Dict[str, NDArray]) -> None:
        """Snapshot multiclass class labels + fit regression z-score scalers."""
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                if spec == TargetType.MULTICLASS:
                    # Sort for determinism; preserve dtype. Natural sort
                    # (not lex-of-str) so integer labels K≥10 land in
                    # numeric order — see sort_multiclass_labels docstring.
                    labels = sort_multiclass_labels(set(np.asarray(y[key]).tolist()))
                    self._multiclass_classes[key] = labels
                elif spec == TargetType.REGRESSION and bool(self.params["regression_normalize"]):
                    self._regression_scaler.fit(key, np.asarray(y[key]))

    def _build_train_tensors(
        self, X: pd.DataFrame, y: Dict[str, NDArray]
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        X_t = self._to_tensor(X.to_numpy(dtype=np.float32))

        y_tensors: Dict[str, "torch.Tensor"] = {}
        for key, spec in self.target_specs.items():
            arr = np.asarray(y[key])
            if isinstance(spec, TargetType):
                if spec == TargetType.BINARY:
                    y_tensors[key] = torch.from_numpy(arr.astype(np.float32)).to(self._device)
                elif spec == TargetType.REGRESSION:
                    transformed = self._regression_scaler.transform(key, arr.astype(np.float64))
                    y_tensors[key] = torch.from_numpy(transformed.astype(np.float32)).to(self._device)
                elif spec == TargetType.MULTICLASS:
                    # Map labels → integer class indices. Surface unseen
                    # labels (present in y but not in the catalogue
                    # captured at fit time) with an actionable error
                    # naming the offenders — earlier this raised a bare
                    # ``KeyError`` via the dict lookup, which was a
                    # latent fragility exposed publicly when
                    # ``_conditional_validation_loss`` started calling
                    # _build_train_tensors on a held-out y_valid that
                    # may contain labels the catalogue (built from
                    # training y) doesn't know about.
                    classes = self._multiclass_classes[key]
                    label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
                    values = arr.tolist()
                    unknown = sorted(
                        {v for v in values if v not in label_to_idx},
                        key=str,
                    )
                    if unknown:
                        raise ValueError(
                            f"Multiclass target {key!r}: label(s) {unknown} "
                            f"not in the training-time class catalogue "
                            f"{classes}. The catalogue is built from "
                            f"training y at fit time, so unseen labels in "
                            f"y_valid (or in y on a re-fit) are a contract "
                            f"violation. Clean the offending rows or "
                            f"include all expected labels in the training "
                            f"split."
                        )
                    indices = np.array([label_to_idx[v] for v in values])
                    y_tensors[key] = torch.from_numpy(indices.astype(np.int64)).to(self._device)
            else:  # multilabel
                y_tensors[key] = torch.from_numpy(arr.astype(np.float32)).to(self._device)

        return X_t, y_tensors

    def _compute_loss(
        self,
        per_target_logits: Dict[str, "torch.Tensor"],
        y_tensors: Dict[str, "torch.Tensor"],
        idx: "torch.Tensor",
    ) -> "torch.Tensor":
        """Mean of per-target losses for the batch (target-count invariant).

        Each per-target loss is itself a sample-mean (the torch loss
        ``reduction='mean'`` default), so the per-target term is on the
        same scale regardless of batch size. We then average ACROSS targets
        rather than summing: with sum, adding a new declared target would
        proportionally inflate the total loss and the effective learning
        rate per target would shrink. Mean keeps each per-target term's
        contribution invariant to ``len(target_specs)`` — a one-line change
        but it's what lets users add targets without re-tuning ``lr``.

        Contract: y is required to be NaN-free at this point. The scorer's
        ``_validate_interactions`` rejects null target values at training
        time (per-column ``null_count`` check), so NaN here would mean
        a caller bypassed the scorer and fit the estimator directly with
        unclean y. We do NOT mask NaN inside the loss — masking would
        silently hide the upstream contract violation; instead any NaN
        will propagate to gradients, train_loss → NaN, and the estimator's
        own determinism gate would catch the failure. Document, don't
        absorb.
        """
        loss = torch.zeros((), device=self._device)
        n_terms = 0
        for key, spec in self.target_specs.items():
            logits = per_target_logits[key]
            target = y_tensors[key][idx]
            if isinstance(spec, TargetType):
                if spec == TargetType.BINARY:
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(logits.view(-1), target)
                elif spec == TargetType.REGRESSION:
                    loss = loss + nn.functional.mse_loss(logits.view(-1), target)
                elif spec == TargetType.MULTICLASS:
                    loss = loss + nn.functional.cross_entropy(logits, target)
            else:  # multilabel
                loss = loss + nn.functional.binary_cross_entropy_with_logits(logits, target)
            n_terms += 1
        # Average across targets. ``target_specs`` is required to be
        # non-empty by __init__ validation, so n_terms ≥ 1 here.
        return loss / float(n_terms)

    def _validation_loss(self, X_valid: pd.DataFrame, y_valid: Dict[str, NDArray]) -> float:
        """One-pass validation loss in eval mode."""
        encoder_module: nn.Module = self._encoder  # type: ignore[assignment]
        encoder_module.eval()
        self._heads.eval()  # type: ignore[union-attr]
        X_t, y_tensors = self._build_train_tensors(X_valid, y_valid)
        with torch.inference_mode():
            hidden = encoder_module(X_t)
            per_target_logits = self._heads(hidden)  # type: ignore[misc]
            all_idx = torch.arange(X_t.shape[0], device=self._device)
            loss = self._compute_loss(per_target_logits, y_tensors, all_idx)
        return float(loss.item())

    def _align_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder X columns to match the column order seen at training time.

        Raises if any training-time column is missing.
        """
        if self._feature_names is None:
            raise RuntimeError("Estimator is not fitted yet.")
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing training-time feature columns: {missing}")
        extra = [c for c in X.columns if c not in self._feature_names]
        if extra:
            raise ValueError(f"X contains feature columns unseen at training: {extra}")
        return X.loc[:, self._feature_names]

    def _to_tensor(self, arr: NDArray) -> "torch.Tensor":
        return torch.from_numpy(arr).to(self._device, dtype=torch.float32)

    def _check_fitted(self) -> None:
        if self._encoder is None or self._heads is None or self._feature_names is None:
            raise RuntimeError("Estimator is not fitted yet. Call fit(...) first.")
