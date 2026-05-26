# Per-target label encoding for v3 conditional joint estimators.
#
# Each declared target contributes one "label chunk" of raw dimensions:
#
#   TargetType         Raw dims      Layout
#   -----------------  ------------  ------------------------------------------
#   BINARY             2             [is_observed, value]
#   REGRESSION         2             [is_observed, z-score normalized value]
#   MULTICLASS (K)     1 + K         [is_observed, one-hot over K]
#   MULTILABEL (K_m)   1 + K_m       [is_observed_group, binary vector per member]
#
# Each chunk is linearly projected to label_embedding_dim (uniform width)
# and concatenated across target_specs entries → total label_input_dim =
# num_entries * label_embedding_dim. The encoder (MLP / Transformer) sees
# this as an extra input alongside the feature matrix X.
#
# Defense-in-depth zeroing (gate 9, mandatory leakage test): masked
# positions have BOTH is_observed=0 AND value=0. is_observed=0 is the
# learned signal, but zeroing the value too prevents any path through the
# network from picking up the actual label when it's supposedly masked —
# label-channel leakage is the canonical conditional-model bug class.
#
# Multilabel group-mask-together (v3 locked decision #4): a multilabel
# group has ONE is_observed flag covering the whole group. Members are
# either all observed together (flag=1, member values present) or all
# masked together (flag=0, all member values zeroed). The scorer's
# inference-side validator enforces this on user-supplied observed data;
# at training, the per-(row, group) Bernoulli draws produce the same
# semantics.

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from skrec.estimator.classification._multi_target_protocol import (
    TargetGroupSpec,
    TargetType,
)

_nn_module = nn.Module if nn is not None else object


def per_target_raw_dim(
    target_spec: Union[TargetType, TargetGroupSpec],
    multiclass_n_classes: Optional[int] = None,
) -> int:
    """Width of one target's raw label chunk before projection.

    Args:
        target_spec: The declared spec for this target.
        multiclass_n_classes: Required for ``TargetType.MULTICLASS``; ignored
            otherwise.

    Returns:
        Number of raw dimensions: 2 for binary/regression,
        ``1 + multiclass_n_classes`` for multiclass, ``1 + n_members`` for
        multilabel groups.
    """
    if isinstance(target_spec, TargetType):
        if target_spec == TargetType.BINARY:
            return 2
        if target_spec == TargetType.REGRESSION:
            return 2
        if target_spec == TargetType.MULTICLASS:
            if multiclass_n_classes is None:
                raise ValueError("multiclass_n_classes is required for MULTICLASS target encoding.")
            return 1 + multiclass_n_classes
        raise ValueError(f"Unsupported TargetType: {target_spec}")
    # TargetGroupSpec (multilabel)
    return 1 + len(target_spec["columns"])


class ConditionalLabelEncoder(_nn_module):
    """Module that projects raw per-target chunks → ``embedding_dim`` per target.

    Returns either a flat-concat tensor (for MLP-style encoders) or a stacked
    per-target token tensor (for FT-Transformer-style encoders). Construction
    is the same for both; the consumer picks the output shape at forward time.

    Attributes:
        embedding_dim: Width of each projected per-target chunk.
        num_targets: Count of ``target_specs`` entries (one chunk / token per
            entry; multilabel groups contribute one chunk for the whole group,
            NOT per member — matches the v3 locked-decision-#4 group-mask-
            together semantics).
        total_flat_dim: ``num_targets * embedding_dim`` — width of the
            ``forward()`` flat-concat output (used by the MLP path).
    """

    def __init__(
        self,
        target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
        multiclass_class_counts: Dict[str, int],
        label_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.target_specs = target_specs
        self.embedding_dim = label_embedding_dim
        self.num_targets = len(target_specs)
        # nn.ModuleDict keyed by target_specs entries — multilabel groups
        # use the group key (one projection per group, not per member).
        projections = {}
        for key, spec in target_specs.items():
            if isinstance(spec, TargetType) and spec == TargetType.MULTICLASS:
                k = multiclass_class_counts[key]
                raw_dim = per_target_raw_dim(spec, multiclass_n_classes=k)
            else:
                raw_dim = per_target_raw_dim(spec)
            projections[key] = nn.Linear(raw_dim, label_embedding_dim)
        self.projections = nn.ModuleDict(projections)
        self.total_flat_dim = self.num_targets * label_embedding_dim
        # Back-compat alias for callers expecting the old name.
        self.total_dim = self.total_flat_dim

    def encode_per_target(self, raw_chunks: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        """Project each raw chunk; stack along a new token axis.

        Returns:
            ``(batch, num_targets, embedding_dim)`` tensor — one
            ``embedding_dim``-wide token per declared target_specs entry,
            in declaration order. Used by the FT-Transformer-style
            conditional encoder, which appends these directly to the
            feature-token sequence (one transformer token per target).
        """
        parts = [self.projections[key](raw_chunks[key]) for key in self.target_specs]
        return torch.stack(parts, dim=1)

    def forward(self, raw_chunks: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        """Flat-concat path (MLP-style encoders).

        Returns:
            ``(batch, total_flat_dim)`` projected + concatenated tensor.
        """
        per_target = self.encode_per_target(raw_chunks)
        batch = per_target.shape[0]
        return per_target.reshape(batch, self.num_targets * self.embedding_dim)


def build_raw_chunks(
    y_or_observed: Dict[str, NDArray],
    mask: Dict[str, NDArray],
    target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
    multiclass_classes: Dict[str, List],
    regression_means: Dict[str, float],
    regression_stds: Dict[str, float],
    device: Optional["torch.device"] = None,
    strict_unknown_labels: bool = False,
) -> Dict[str, "torch.Tensor"]:
    """Build per-target raw label chunks with defense-in-depth zeroing on masked positions.

    Args:
        y_or_observed: Dict keyed by target_specs entries. Same shapes as the
            estimator's training y (BINARY/REGRESSION 1-D, MULTICLASS 1-D of
            labels, MULTILABEL 2-D of binary vectors). At inference, ``np.nan``
            values mark "not observed for this row" — interpreted as masked
            with value zeroed.
        mask: Dict keyed by target_specs entries; per-row bool array
            (True = MASKED = unobserved). For multilabel groups, mask is a
            single per-row flag covering the whole group.
        target_specs, multiclass_classes, regression_means, regression_stds:
            Per-target catalogue / scaler info captured at fit time.
        device: Optional torch device; defaults to CPU.
        strict_unknown_labels: When True, multiclass values not in the
            captured catalogue raise ``ValueError`` instead of being
            demoted to ``is_observed=0``. Training callers MUST pass True
            (the catalogue was built from training y itself — an unseen
            label there is a contract violation, not a missing
            observation). Inference callers leave it False so
            inference-time noise (e.g. a new category not seen at fit)
            degrades gracefully to "this row is unobserved for the
            multiclass channel" instead of crashing the request.

    Returns:
        Dict keyed by target_specs entries; each value is a
        ``(n_samples, raw_dim_for_target)`` float32 tensor.
    """
    chunks: Dict[str, "torch.Tensor"] = {}
    for key, spec in target_specs.items():
        m = mask[key].astype(bool)  # True = masked = unobserved
        # is_observed starts from the explicit mask; multiclass may further
        # demote rows to "masked" if they carry an unseen-at-train label.
        is_observed_bool = ~m
        # Cache the np.asarray once per target — both the simple-target and
        # multilabel branches reach for the underlying ndarray.
        arr = np.asarray(y_or_observed[key])
        if isinstance(spec, TargetType):
            if spec == TargetType.BINARY:
                value = np.where(m, 0.0, np.nan_to_num(arr, nan=0.0).astype(np.float32))
                value = value.reshape(-1, 1)
                is_observed = is_observed_bool.astype(np.float32).reshape(-1, 1)
                raw = np.concatenate([is_observed, value], axis=1)
            elif spec == TargetType.REGRESSION:
                mean = regression_means.get(key, 0.0)
                std = regression_stds.get(key, 1.0) or 1.0
                # Compute the z-score normalization only on unmasked
                # rows; masked rows are about to be zeroed anyway. The
                # previous form ran ``nan_to_num + subtract + divide``
                # over EVERY row and then threw the result away with
                # np.where — wasted FLOPs at large batch × many
                # regression targets. Allocating the output buffer at
                # zero up front + filling only the unmasked positions
                # avoids that wasted work.
                value = np.zeros(arr.shape, dtype=np.float32)
                unmasked = ~m
                if unmasked.any():
                    arr_unmasked = np.asarray(arr[unmasked], dtype=np.float64)
                    # nan_to_num is still needed inside ``unmasked`` for
                    # the (rare) case of an observed-row NaN coming in
                    # from a caller that bypassed the scorer's NaN guard.
                    value[unmasked] = ((np.nan_to_num(arr_unmasked, nan=0.0) - mean) / std).astype(np.float32)
                value = value.reshape(-1, 1)
                is_observed = is_observed_bool.astype(np.float32).reshape(-1, 1)
                raw = np.concatenate([is_observed, value], axis=1)
            elif spec == TargetType.MULTICLASS:
                classes = multiclass_classes[key]
                K = len(classes)
                label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
                n = arr.shape[0]
                one_hot = np.zeros((n, K), dtype=np.float32)
                # Vectorized index lookup. Build a pandas Series view of
                # ``arr`` so .map(label_to_idx) dispatches in C — NaN /
                # None / unseen-label entries all become NaN in the
                # resulting Series (Series.map returns NaN for missing
                # keys, same as Python dict.get). Then:
                #   - valid_mask = ~isna(mapped) & ~m   → rows with a
                #     known label AND not explicitly masked.
                #   - unseen rows (mapped is NaN but m was False) get
                #     demoted to is_observed=0 below; their one_hot row
                #     stays at zero. Without the demotion, the network
                #     would learn "all-zeros one-hot ⇒ observed" — a
                #     label-leak waiting to happen.
                # pandas is already a hard module-level dep (imported
                # at the top) — using ``pd.Series(...).map`` here gives
                # us the same vectorized lookup without re-paying the
                # ``import`` dict-lookup per multiclass target per batch.
                mapped = pd.Series(arr).map(label_to_idx)
                known_mask = ~mapped.isna()
                valid_mask = known_mask.to_numpy() & ~m
                # Rows whose label wasn't in the catalogue AND weren't
                # already explicitly masked.
                unknown_unmasked = ~known_mask.to_numpy() & ~m
                if strict_unknown_labels and unknown_unmasked.any():
                    # Training-time path: the catalogue was built from
                    # training y itself, so any label not in it is a
                    # contract violation (somebody passed unclean y or
                    # the catalogue was wiped between fit and a re-call).
                    # Surface the offending labels so the caller can
                    # locate them.
                    offenders = sorted(
                        {arr[i] for i in np.where(unknown_unmasked)[0]},
                        key=str,
                    )
                    raise ValueError(
                        f"Multiclass target {key!r}: training-time "
                        f"label(s) {offenders} not in the captured "
                        f"catalogue {classes}. The catalogue is built "
                        f"from training y at fit time, so unseen labels "
                        f"here indicate either unclean y or a mid-fit "
                        f"catalogue wipe."
                    )
                # Inference-time path: demote unseen-label rows to
                # is_observed=0 (no embedding for the unseen label).
                is_observed_bool[unknown_unmasked] = False
                if valid_mask.any():
                    row_idx = np.where(valid_mask)[0]
                    col_idx = mapped[valid_mask].to_numpy().astype(np.int64)
                    one_hot[row_idx, col_idx] = 1.0
                is_observed = is_observed_bool.astype(np.float32).reshape(-1, 1)
                raw = np.concatenate([is_observed, one_hot], axis=1)
            else:
                raise ValueError(f"Unsupported TargetType in raw-chunk builder: {spec}")
        else:
            # MULTILABEL group: y shape (n, n_members). Per-row group mask.
            arr_f = arr.astype(np.float64)
            arr_f = np.nan_to_num(arr_f, nan=0.0).astype(np.float32)
            # Zero member values where the group is masked.
            zeroed = np.where(m[:, None], 0.0, arr_f)
            is_observed = is_observed_bool.astype(np.float32).reshape(-1, 1)
            raw = np.concatenate([is_observed, zeroed], axis=1)

        tensor = torch.from_numpy(raw).to(device=device, dtype=torch.float32)
        chunks[key] = tensor
    return chunks


def build_mask_from_observed_dict(
    observed: Dict[str, NDArray],
    target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
    n_samples: int,
) -> Dict[str, NDArray]:
    """Infer the per-(row, entry) mask from an inference-time observed dict.

    For each target_specs entry:
      - Missing from the dict → fully masked (mask=True for every row).
      - Present but value is NaN per row → that row is masked.
      - Multilabel group: members must mask together per the v3 locked rule;
        the scorer's validator enforces this on the caller's side, so here
        we conservatively read row-wise NaN from any member to mark the
        whole group as masked.

    Returns:
        Dict keyed by target_specs entries; values are ``(n_samples,)``
        bool arrays (True = masked = unobserved).
    """
    out: Dict[str, NDArray] = {}
    for key, spec in target_specs.items():
        if key not in observed:
            out[key] = np.ones(n_samples, dtype=bool)
            continue
        arr = np.asarray(observed[key])
        if isinstance(spec, TargetType):
            # 1-D array; NaN ⇒ masked.
            try:
                m = np.isnan(arr.astype(np.float64))
            except (TypeError, ValueError):
                # Non-numeric multiclass labels: treat None as masked.
                m = np.array([v is None for v in arr.tolist()])
            out[key] = m
        else:
            # 2-D (n, n_members); any-member NaN ⇒ whole group masked
            # (members-mask-together convention).
            try:
                m_per_member = np.isnan(arr.astype(np.float64))
            except (TypeError, ValueError):
                m_per_member = np.zeros(arr.shape, dtype=bool)
            out[key] = m_per_member.any(axis=1)
    return out


def sample_training_mask(
    target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
    n_samples: int,
    mask_prob: float,
    generator: "torch.Generator",
) -> Dict[str, NDArray]:
    """Bernoulli(mask_prob) per (row, target_specs entry).

    For multilabel groups, the draw is per-row at the GROUP level (not per
    member) — matches the v3 locked decision #4 "members mask together."

    Args:
        target_specs: Declared per-target schema.
        n_samples: Number of rows.
        mask_prob: Probability a (row, entry) is MASKED (unobserved at train).
        generator: Required seeded ``torch.Generator``. Pre-fix this was
            optional with an unseeded ``np.random.default_rng()`` fallback,
            which would silently produce different masks every run despite
            the estimator's ``seed`` argument. Direct callers must supply
            their own generator — the conditional base estimator wires one
            keyed off ``(seed, epoch)``.

    Returns:
        Dict keyed by target_specs entries; values are ``(n_samples,)`` bool
        arrays (True = masked = unobserved for that row).

    Raises:
        TypeError: If ``generator`` is ``None``. Reproducibility requires
            an explicit seeded generator at the call site.
    """
    if generator is None:
        raise TypeError(
            "sample_training_mask requires a seeded torch.Generator. The "
            "previous fallback to np.random.default_rng() silently broke "
            "reproducibility — pass an explicit generator (e.g. "
            "torch.Generator().manual_seed(...)) instead."
        )
    out: Dict[str, NDArray] = {}
    for key in target_specs:
        samples = torch.rand(n_samples, generator=generator).numpy()
        out[key] = samples < mask_prob
    return out
