# Wide-format scorer for heterogeneous per-target supervised learning.
#
# Sibling of MultioutputScorer for the "one row per user, multiple targets in
# the same row" contract, but with three structural extensions:
#
#   1. Heterogeneous target types per column. MultioutputScorer requires every
#      ITEM_<name> column to share a mode (all binary OR all continuous).
#      MixedTypeMultiTargetScorer accepts per-target declarations via
#      target_specs — one column may be BINARY, the next REGRESSION, the next
#      MULTICLASS, and a multilabel group of K members may live alongside.
#
#   2. Estimator family pluralism via a runtime-checkable Protocol. The scorer
#      accepts any MultiTargetEstimator: joint MLP, joint Transformer, or
#      independent (a dict of per-target sub-estimators). The scorer's I/O
#      contract is decoupled from the estimator's training machinery.
#
#   3. Optional real-time-label conditioning at inference. v2 ships
#      vanilla-only — OBSERVED_* columns at inference are rejected with a
#      pointer to the conditional estimator classes added in v3. v2's
#      _validate_inference_interactions is structured so v3 can flip
#      "reject" → "delegate to estimator" purely additively.
#
# Why one scorer with three families (and not three scorers):
#   - The wide-format I/O contract (ITEM_* column unpacking, dict-y stitching,
#     output column conventions, recommend() short-circuit, recommend_online
#     wiring) is identical across families. A single scorer means one
#     capability matrix row, one set of recommender wiring, one decision-rule
#     branch. The estimator Protocol does the polymorphism cleanly.
#
# Why no `preserved_inference_columns()` hook in v2:
#   - The hook exists to shield OBSERVED_* columns from interactions_schema's
#     silent unknown-column strip. Without OBSERVED_* support in v2, no
#     columns need shielding. v3 adds the hook on BaseScorer and overrides
#     it here. (See v3 plan.)
#
# Why two validators (training-time + inference-time) rather than one with
# an is_training flag:
#   - Note: v3 hook — _validate_inference_interactions will flip from
#     "reject all OBSERVED_*" to OBSERVED-aware dispatch (vanilla rejects;
#     conditional permits). Keeping training and inference as separate named
#     methods means external callers like BatchTrainingDataset (which calls
#     the 1-arg base signature) can never accidentally route to the inference
#     path. A flag-based design risked exactly that, since defaulting wrong
#     is silent.
#
# Why dict-shaped y (and not a wide 2-D matrix):
#   - Targets have different shapes per type. binary/regression columns are
#     1-D, multiclass column is 1-D but resolves to (n, K) at the heads,
#     multilabel groups are (n, n_members). A dict keyed by target name lets
#     each estimator family fan out into per-target loss heads cleanly.
#     The downside is that BaseEstimator._validate_for_fit assumes y has
#     .shape — concrete estimator families override _validate_for_fit on
#     the estimator side. BaseScorer.train_model is a passthrough that
#     doesn't inspect y, so no override is needed here.
#
# Notes for future maintainers (especially v3):
#   - target_col = None is a marker, not a real column name. Code paths that
#     read self.target_col must be overridden here (we do not inherit any
#     target_col-reading logic from BaseScorer).
#   - score_items / predict_targets / score_fast are fresh implementations
#     that do NOT route through _calculate_scores. _calculate_scores raises
#     NotImplementedError so a future change that wires it back into the
#     standard path fails loudly rather than silently producing wrong shapes.
#   - v3 hook — score_items / predict_targets / score_fast will check
#     isinstance(estimator, ConditionalMultiTargetEstimator) and build an
#     `observed` dict from OBSERVED_* columns when conditional. v2's code
#     paths take only the X branch.

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import skrec.constants as C
from skrec.estimator.classification._multi_target_protocol import (
    ConditionalMultiTargetEstimator,
    MultiTargetEstimator,
    TargetGroupSpec,
    TargetType,
)
from skrec.scorer.base_scorer import BaseScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)

# Re-exported here so user-facing imports
# `from skrec.scorer.mixed_type_multi_target import TargetType` keep working —
# the canonical home is the Protocol module to avoid scorer↔estimator circular
# imports.
__all__ = ["MixedTypeMultiTargetScorer", "TargetGroupSpec", "TargetType"]


# Valid set of values a binary or multilabel-member column may carry at
# training time. Matches MultioutputScorer's binary contract for symmetry.
_BINARY_VALUES: frozenset = frozenset({0, 1, 0.0, 1.0, True, False})


# Per-TargetType → compatible metric names (lowercase, matching
# RecommenderMetricType.value strings). Single source of truth for:
#   - RankingRecommender._evaluate_mixed_type_multi_target dispatch (M6),
#   - capability_matrix()["target_type_metric_compat"] (M5),
#   - docs/user-guide/decision-rule.md (M9; gate 7 asserts agreement).
# Note: MULTICLASS_ACCURACY enum value is added to RecommenderMetricType in M6.
# The original init-fn + global-rebind was a workaround for a circular-import
# edge case that no longer exists (TargetType moved out to
# _multi_target_protocol, breaking the cycle). Module-level dict literal is
# the simplest correct form now.
TARGET_TYPE_TO_METRICS: dict["TargetType", tuple[str, ...]] = {
    TargetType.BINARY: ("roc_auc", "pr_auc"),
    TargetType.REGRESSION: ("rmse", "mae"),
    TargetType.MULTICLASS: ("multiclass_accuracy",),
    TargetType.MULTILABEL: ("roc_auc", "pr_auc"),  # per member
}


class MixedTypeMultiTargetScorer(BaseScorer):
    """Wide-format scorer for heterogeneous per-target supervised learning.

    Each entry in ``target_specs`` declares one target — either a simple
    ``ITEM_<name>`` column with a single ``TargetType``, or a multilabel
    group of ITEM_-prefixed member columns. The scorer routes the wide
    interactions frame through any estimator implementing the
    ``MultiTargetEstimator`` Protocol (joint MLP, joint Transformer, or
    independent).

    Args:
        estimator: Any estimator implementing ``MultiTargetEstimator``.
            Embedding estimators and non-Protocol estimators are rejected
            with a ``TypeError`` at construction. The estimator's
            ``target_specs`` must equal this scorer's ``target_specs``.
        target_specs: Per-target declaration. Keys are target names (for
            simple targets, the key is the ITEM_-prefixed column name; for
            multilabel groups, the key is any non-ITEM_ identifier). Values
            are either ``TargetType`` enums or ``TargetGroupSpec`` dicts.
            Must be non-empty.

    Raises:
        TypeError: If ``estimator`` does not implement ``MultiTargetEstimator``.
        ValueError: If ``target_specs`` is empty, contains malformed entries,
            or has structural collisions (group key matching a member column
            or simple target key; member column appearing in multiple groups
            or also as a simple target).
        ValueError: If the estimator's ``target_specs`` differs from the
            scorer's ``target_specs`` (consistency check).

    .. warning::
        **Not thread-safe.** Like :class:`MultioutputScorer`, this scorer
        carries shared mutable state (``item_names``, ``item_subset``). Use
        one instance per thread or serialize calls.

    **score_per_target callable contract** (per :class:`TargetType`):

    +-------------------+--------------------------------------------+--------------------------------+
    | Target type       | ``y_true`` shape / dtype                   | ``predictions`` shape          |
    +===================+============================================+================================+
    | BINARY            | ``(n,)`` of 0/1                            | ``(n, 2)`` class probabilities |
    +-------------------+--------------------------------------------+--------------------------------+
    | REGRESSION        | ``(n,)`` of continuous values              | ``(n,)`` predicted values      |
    +-------------------+--------------------------------------------+--------------------------------+
    | MULTICLASS        | ``(n,)`` of class labels (orig. dtype)     | ``(n, K)`` class probabilities |
    +-------------------+--------------------------------------------+--------------------------------+
    | MULTILABEL member | ``(n,)`` of 0/1                            | ``(n, 2)`` class probabilities |
    +-------------------+--------------------------------------------+--------------------------------+

    Multilabel groups fan out: each member column gets one call with the
    binary contract above. Lookup precedence for ``metric_callables``:
    target-name key beats :class:`TargetType`-keyed default. See
    :meth:`score_per_target` for the full method docstring.
    """

    # Marker — full _validate_interactions override; we do NOT inherit
    # the target_col-based validation from BaseScorer.
    target_col: Optional[str] = None  # type: ignore[assignment]

    # v3 capability flag — opt into ``OBSERVED_*`` columns at inference for
    # real-time-label conditioning. Read by capability_matrix() so the
    # published "scorer_supports_observed_conditioning" tuple is derived
    # from this class attribute, not a hand-maintained list. Note: vanilla
    # estimators paired with this scorer still reject OBSERVED_* at the
    # inference validator — the flag advertises scorer-side capability, not
    # estimator-side acceptance.
    supports_observed_conditioning: bool = True

    # Per-target scorer flag — recommend / recommend_online / evaluate
    # route through the per-target wide-format path when this is True,
    # rather than the top-k ranking path. Avoids isinstance(scorer,
    # MixedTypeMultiTargetScorer) ladders at the three dispatch sites
    # (which would each need editing every time a new per-target scorer
    # lands).
    is_per_target_scorer: bool = True

    def __init__(
        self,
        estimator: MultiTargetEstimator,
        target_specs: dict[str, Union[TargetType, TargetGroupSpec]],
    ) -> None:
        # Protocol check first — gives the most actionable error if the
        # caller hands in something structurally unrelated (e.g., a raw
        # XGBClassifierEstimator instead of the IndependentMultiTargetEstimator
        # wrapper around it).
        if not isinstance(estimator, MultiTargetEstimator):
            raise TypeError(
                f"MixedTypeMultiTargetScorer requires an estimator implementing "
                f"MultiTargetEstimator (target_specs, fit, predict_proba_dict, "
                f"predict_targets_dict). Got {type(estimator).__name__}. "
                f"Use JointMultiTargetMLPEstimator, JointMultiTargetTransformerEstimator, "
                f"or IndependentMultiTargetEstimator."
            )

        # target_specs structural validation. Catches malformed input before
        # any estimator-side validation runs.
        self._validate_target_specs(target_specs)

        # Consistency check between scorer's target_specs and the estimator's.
        # Both are sources of truth — they must agree.
        estimator_specs = getattr(estimator, "target_specs", None)
        if estimator_specs is not None and estimator_specs != target_specs:
            raise ValueError(
                "Inconsistent target_specs between scorer and estimator. "
                f"Scorer target_specs keys: {sorted(target_specs.keys())}. "
                f"Estimator target_specs keys: {sorted(estimator_specs.keys())}. "
                "Construct the estimator with the same target_specs you pass "
                "to the scorer."
            )

        super().__init__(estimator)
        self.target_specs: dict[str, Union[TargetType, TargetGroupSpec]] = target_specs

        # Pre-compute the fanned-out flat column list (simple-target columns
        # plus per-member columns from each group) in declared insertion
        # order. This is the canonical column order for predict_targets /
        # score_items output.
        self._fanned_out_target_columns: list[str] = self._compute_fanned_out_columns()

    # ------------------------------------------------------------------ #
    # target_specs validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_target_specs(
        target_specs: dict[str, Union[TargetType, TargetGroupSpec]],
    ) -> None:
        """Structural validation of the declared per-target schema.

        Catches: empty dict; malformed values; group key colliding with a
        simple-target key or any member column; member column appearing in
        multiple groups; member column also declared as a simple target;
        multilabel group with no member columns; non-MULTILABEL group type
        (reserved for future use).
        """
        if not target_specs:
            raise ValueError("target_specs must be non-empty. Declare at least one target.")

        simple_target_names: set[str] = set()
        group_keys: set[str] = set()
        all_member_columns: list[str] = []
        member_to_group: dict[str, str] = {}

        for key, spec in target_specs.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"target_specs keys must be non-empty strings. Got {key!r}.")
            # nn.ModuleDict (used downstream by joint estimators' per-target
            # heads + the conditional label encoder) rejects keys containing
            # '.' — "module name can't contain '.', got: ...". Validate
            # upfront so the failure surfaces at scorer construction with an
            # actionable message rather than mid-fit inside the torch
            # internals. Whitespace and other unicode are also unfriendly
            # downstream; restrict to a conservative identifier-ish set
            # (allows '_' and ITEM_/group identifiers, rejects '.', ' ',
            # '/', etc.).
            if "." in key or any(c.isspace() for c in key):
                raise ValueError(
                    f"target_specs key {key!r} contains '.' or whitespace. "
                    f"Keys are used as nn.ModuleDict keys downstream "
                    f"(per-target heads + conditional label encoder), which "
                    f"rejects these characters. Use underscores instead "
                    f"(e.g. 'engagement_group' not 'engagement.group')."
                )

            if isinstance(spec, TargetType):
                # Simple target: key IS the column name; must be ITEM_-prefixed
                # for symmetry with the wide-format contract.
                if not key.startswith(C.ITEM_PREFIX):
                    raise ValueError(
                        f"Simple-target key {key!r} must be prefixed with "
                        f"'{C.ITEM_PREFIX}'. For multilabel groups, use a "
                        f"non-ITEM_ identifier and declare members under "
                        f"TargetGroupSpec(columns=[...])."
                    )
                simple_target_names.add(key)

            elif isinstance(spec, dict):
                # Group spec. Validate shape.
                if "type" not in spec or "columns" not in spec:
                    raise ValueError(
                        f"TargetGroupSpec for key {key!r} must have 'type' and 'columns' fields. Got {spec!r}."
                    )
                group_type = spec["type"]
                if not isinstance(group_type, TargetType):
                    # Accept the string form too (matches the TargetType enum value).
                    try:
                        group_type = TargetType(group_type)
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"TargetGroupSpec for key {key!r}: 'type' must be a TargetType. Got {spec['type']!r}."
                        )
                if group_type != TargetType.MULTILABEL:
                    raise ValueError(
                        f"TargetGroupSpec for key {key!r}: only "
                        f"TargetType.MULTILABEL is supported in v2. Got {group_type}."
                    )
                columns = spec["columns"]
                if not isinstance(columns, (list, tuple)) or len(columns) == 0:
                    raise ValueError(
                        f"TargetGroupSpec for key {key!r}: 'columns' must be a non-empty list. Got {columns!r}."
                    )
                # Within-group duplicates would silently collapse fanned-out
                # columns / per-member heads / metric lookups. The cross-group
                # check below catches the same name appearing in two groups,
                # but a same-name-twice-in-one-group typo (e.g.,
                # columns=["ITEM_a", "ITEM_a"]) would also hit that branch
                # with a confusing "appears in multiple groups: g and g"
                # message. Catch within-group dupes first with a clearer
                # error.
                seen_in_group: dict = {}
                for col in columns:
                    seen_in_group[col] = seen_in_group.get(col, 0) + 1
                within_dups = sorted(c for c, n in seen_in_group.items() if n > 1)
                if within_dups:
                    raise ValueError(
                        f"TargetGroupSpec for key {key!r}: duplicate member "
                        f"column(s) {within_dups} within the same group. "
                        f"Multilabel groups require unique members."
                    )
                for col in columns:
                    if not isinstance(col, str) or not col.startswith(C.ITEM_PREFIX):
                        raise ValueError(
                            f"TargetGroupSpec for key {key!r}: every member "
                            f"column must be a string prefixed with "
                            f"'{C.ITEM_PREFIX}'. Got {col!r}."
                        )
                    if col in member_to_group:
                        raise ValueError(
                            f"Member column {col!r} appears in multiple groups: {member_to_group[col]!r} and {key!r}."
                        )
                    member_to_group[col] = key
                    all_member_columns.append(col)
                group_keys.add(key)

            else:
                raise ValueError(
                    f"target_specs value for key {key!r} must be a TargetType "
                    f"or TargetGroupSpec dict. Got {type(spec).__name__}: {spec!r}."
                )

        # Cross-check: group keys must not collide with simple-target keys
        # (dict already prevents at the structural level but value-level
        # collision check is worth pinning).
        collisions = group_keys & simple_target_names
        if collisions:
            raise ValueError(
                f"Group key(s) collide with simple-target key(s): {sorted(collisions)}. "
                f"Use a non-ITEM_ identifier for group keys."
            )

        # Group keys must not equal any member column either.
        key_member_collisions = group_keys & set(all_member_columns)
        if key_member_collisions:
            raise ValueError(
                f"Group key(s) collide with member column name(s): "
                f"{sorted(key_member_collisions)}. Use a distinct identifier for "
                f"group keys (typically a non-ITEM_-prefixed name)."
            )

        # Member columns must not also be declared as simple targets.
        member_simple_collisions = set(all_member_columns) & simple_target_names
        if member_simple_collisions:
            raise ValueError(
                f"Member column(s) also declared as simple target(s): "
                f"{sorted(member_simple_collisions)}. A column can be EITHER a "
                f"simple target OR a member of a multilabel group, not both."
            )

    def _compute_fanned_out_columns(self) -> list[str]:
        """Flat ordered list of ITEM_-prefixed columns this scorer predicts.

        For simple targets, the key IS the column name. For multilabel groups,
        each member column appears in declared order. The result drives the
        wide-format output column order in ``score_items`` /
        ``predict_targets`` (deterministic regardless of dict insertion-order
        iteration internals).
        """
        out: list[str] = []
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                out.append(key)
            else:  # TargetGroupSpec
                out.extend(spec["columns"])
        return out

    def _simple_target_columns(self) -> list[str]:
        """Subset of ``_fanned_out_target_columns`` covering simple targets only."""
        return [k for k, v in self.target_specs.items() if isinstance(v, TargetType)]

    def _all_member_columns(self) -> list[str]:
        """Flat list of all multilabel member columns across every group."""
        out: list[str] = []
        for spec in self.target_specs.values():
            if isinstance(spec, dict):  # TargetGroupSpec
                out.extend(spec["columns"])
        return out

    # ------------------------------------------------------------------ #
    # Training-time validation
    # ------------------------------------------------------------------ #

    def _validate_interactions(self, interactions_df: Optional[pd.DataFrame]) -> None:
        """Training-time wide-format validator.

        Matches the 1-arg base signature so external callers like
        :class:`BatchTrainingDataset` (which call this directly) route to the
        right validator. Inference-time validation lives in
        :meth:`_validate_inference_interactions` — they are deliberately
        separate methods (no is_training flag) so a wrong default can't
        silently route a training call through the looser inference path.
        """
        self._validate_interactions_base(interactions_df)
        assert interactions_df is not None  # narrowed by _validate_interactions_base

        if C.USER_ID_NAME not in interactions_df.columns:
            raise ValueError(f"'{C.USER_ID_NAME}' column must exist in interactions_df.")

        # Every declared target column must be present at training time.
        missing: list[str] = [col for col in self._fanned_out_target_columns if col not in interactions_df.columns]
        if missing:
            raise ValueError(
                f"Declared target columns missing from interactions_df: {missing}. "
                f"target_specs requires: {self._fanned_out_target_columns}."
            )

        # No duplicate users — wide format is one row per user.
        if interactions_df[C.USER_ID_NAME].duplicated().any():
            raise ValueError(
                f"MixedTypeMultiTargetScorer accepts only one row per user; '{C.USER_ID_NAME}' contains duplicates."
            )

        # Null check on every target column.
        for col in self._fanned_out_target_columns:
            null_count = interactions_df[col].isnull().sum()
            if null_count > 0:
                raise ValueError(
                    f"Target column '{col}' contains {null_count} null value(s). Remove or impute before training."
                )

        # Per-target-type value validation.
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                self._validate_simple_target_values(interactions_df, key, spec)
            else:  # TargetGroupSpec (multilabel)
                for member_col in spec["columns"]:
                    self._validate_simple_target_values(interactions_df, member_col, TargetType.BINARY)

    @staticmethod
    def _validate_simple_target_values(interactions_df: pd.DataFrame, col: str, target_type: TargetType) -> None:
        """Per-column value-domain validation.

        - BINARY: values must be in {0, 1, 0.0, 1.0, True, False}.
        - REGRESSION: column must be numeric dtype.
        - MULTICLASS: column must have at least 2 unique values.
        - MULTILABEL (member): handled as BINARY at the per-member level.
        """
        series = interactions_df[col]

        if target_type == TargetType.BINARY:
            unique_vals = set(series.unique().tolist())
            if not unique_vals.issubset(_BINARY_VALUES):
                raise ValueError(
                    f"Target {col!r} declared as BINARY but column contains "
                    f"values outside {{0, 1}}: {sorted(unique_vals, key=str)}. "
                    f"Pre-encode at the caller (e.g. df[col] = "
                    f"(df[col] == 'yes').astype(float))."
                )

        elif target_type == TargetType.REGRESSION:
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(
                    f"Target {col!r} declared as REGRESSION but column dtype is {series.dtype}, not numeric."
                )

        elif target_type == TargetType.MULTICLASS:
            n_unique = series.nunique(dropna=True)
            if n_unique < 2:
                raise ValueError(
                    f"Target {col!r} declared as MULTICLASS but has only "
                    f"{n_unique} unique value(s). Multiclass requires at least 2."
                )

    # ------------------------------------------------------------------ #
    # Inference-time validation
    # ------------------------------------------------------------------ #

    def _validate_inference_interactions(self, interactions_df: pd.DataFrame) -> None:
        """Inference-time validator with OBSERVED-aware dispatch (v3).

        Vanilla estimators (``MultiTargetEstimator`` only): any ``OBSERVED_*``
        column is rejected with a clean error pointing at the conditional
        estimator classes.

        Conditional estimators (``ConditionalMultiTargetEstimator``):
        ``OBSERVED_*`` columns are permitted. Validates:
          - No orphan ``OBSERVED_*`` columns (must match a declared target).
          - For multilabel groups: members must mask together per row, AND
            the whole group's ``OBSERVED_*`` columns must be either all
            present or none present at the schema level (column-level
            imbalance is rejected — a frame with one member's OBSERVED_*
            present and another absent silently weakens the group-mask
            rule). Partial-group observation within a row → clean error.

        Orphan ITEM_* feature columns (columns that look like targets but
        aren't declared in ``target_specs``) are also rejected — the
        ITEM_ namespace belongs to declared targets at training time, and
        a stray ITEM_* feature at inference almost always indicates a
        caller bug.
        """
        if interactions_df is None:
            raise TypeError("interactions_df must not be None at inference.")

        if C.USER_ID_NAME not in interactions_df.columns:
            raise ValueError(f"'{C.USER_ID_NAME}' column must exist in interactions_df at inference time.")

        # Orphan ITEM_* feature-column rejection. The ITEM_ namespace is
        # reserved for declared targets. At inference, target columns are
        # optional (the scorer drops them in _extract_X_inference), but
        # any ITEM_* column that isn't a declared target is suspicious.
        declared_target_cols = set(self._fanned_out_target_columns)
        observed_cols = [c for c in interactions_df.columns if c.startswith(C.OBSERVED_PREFIX)]
        orphan_item_cols = [
            c for c in interactions_df.columns if c.startswith(C.ITEM_PREFIX) and c not in declared_target_cols
        ]
        if orphan_item_cols:
            raise ValueError(
                f"Orphan ITEM_* column(s) at inference: {sorted(orphan_item_cols)}. "
                f"Each ITEM_<name> column must either be a declared target "
                f"(in target_specs) or not start with the ITEM_ prefix. "
                f"Declared targets: {sorted(declared_target_cols)}."
            )

        if not observed_cols:
            return  # No OBSERVED_*; nothing more to validate.

        # Vanilla path: reject with a clear pointer to the conditional families.
        # Use the sentinel-checked helper so a structural look-alike that
        # forgot the opt-in attribute still routes here (and rejects),
        # rather than silently activating OBSERVED handling.
        if not self._is_conditional_estimator():
            raise NotImplementedError(
                f"OBSERVED_* columns require a ConditionalMultiTargetEstimator "
                f"(e.g. ConditionalJointMultiTargetMLPEstimator or "
                f"ConditionalJointMultiTargetTransformerEstimator). Got "
                f"vanilla {type(self.estimator).__name__}. "
                f"Offending columns: {observed_cols}. Remove them or rebuild "
                f"the scorer with a conditional estimator."
            )

        # Conditional path: OBSERVED_* permitted; validate shape.
        # The expected OBSERVED_* set is whatever preserved_inference_columns
        # returns (the mapping is "ITEM_<suffix> ↔ OBSERVED_<suffix>").
        expected = set(self.preserved_inference_columns())
        orphans = set(observed_cols) - expected
        if orphans:
            raise ValueError(
                f"Orphan OBSERVED_* column(s) at inference: {sorted(orphans)}. "
                f"Each OBSERVED_<suffix> must match a declared target "
                f"ITEM_<suffix>. Declared targets: "
                f"{sorted(self._fanned_out_target_columns)}."
            )

        # Multilabel group-mask-together check (v3 locked decision #4).
        # Two rules:
        #   (a) Column-level: a group's OBSERVED_* columns must be either
        #       ALL present or NONE present. If one member's OBSERVED_*
        #       is declared and another isn't, the per-row "members must
        #       mask together" rule is silently weakened (the missing
        #       column is treated as fully unobserved per build_mask_from_
        #       observed_dict, but the present member can still be observed
        #       on individual rows — breaking the joint contract).
        #   (b) Row-level: within rows where the group's OBSERVED_* columns
        #       are present, NaN-per-row must agree across members.
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                continue
            member_obs_cols = [f"{C.OBSERVED_PREFIX}{m[len(C.ITEM_PREFIX) :]}" for m in spec["columns"]]
            present = [c for c in member_obs_cols if c in interactions_df.columns]
            absent = [c for c in member_obs_cols if c not in interactions_df.columns]
            if present and absent:
                # Column-level imbalance — rule (a).
                raise ValueError(
                    f"Multilabel group {key!r} has column-level OBSERVED_* "
                    f"imbalance: members {present} are present in the inference "
                    f"frame but {absent} are absent. Per the v3 group-mask-"
                    f"together rule, a multilabel group's OBSERVED_* columns "
                    f"must be either all present (members mask together per row) "
                    f"or all absent (fully unobserved). Mixing breaks the joint "
                    f"semantics. Members: {spec['columns']}."
                )
            if not present:
                continue  # entire group is fully unobserved at inference — fine.
            # Row-level: NaN-per-row across all members must agree — rule (b).
            nan_mask = interactions_df[present].isna().to_numpy()
            row_mixed = nan_mask.any(axis=1) & (~nan_mask.all(axis=1))
            if row_mixed.any():
                offending_rows = np.flatnonzero(row_mixed)[:5].tolist()
                raise ValueError(
                    f"Multilabel group {key!r} has partial-group observation in "
                    f"row(s) {offending_rows} (showing up to 5). Per the v3 "
                    f"group-mask-together rule, all members of a multilabel "
                    f"group must be either all observed OR all NaN per row. "
                    f"Members: {spec['columns']}."
                )

    # ------------------------------------------------------------------ #
    # Dataset processing
    # ------------------------------------------------------------------ #

    def process_datasets(
        self,
        users_df: Optional[pd.DataFrame] = None,
        items_df: Optional[pd.DataFrame] = None,
        interactions_df: Optional[pd.DataFrame] = None,
        is_training: Optional[bool] = True,
    ) -> tuple[pd.DataFrame, dict[str, NDArray]]:
        """Validate, split features from targets, and produce (X, y_dict).

        Args:
            users_df: Must be ``None`` (wide-format only).
            items_df: Must be ``None`` (wide-format only).
            interactions_df: Wide-format DataFrame with ``USER_ID``, declared
                target columns, and feature columns.
            is_training: When ``True``, validates targets and builds a dict-y;
                when ``False``, runs the inference validator and returns an
                empty dict for ``y``.

        Returns:
            ``(X, y_dict)``. ``X`` is the feature frame (target / member /
            ``USER_ID`` columns removed). ``y_dict`` is keyed by ``target_specs``
            entries — simple targets map to 1-D arrays; multilabel groups map
            to 2-D arrays in declared member order. Empty at inference.
        """
        if users_df is not None or items_df is not None:
            raise ValueError(
                "MixedTypeMultiTargetScorer does not accept users_df or items_df "
                "(wide-format scorer). Pass features as plain columns in "
                "interactions_df."
            )
        if interactions_df is None:
            raise TypeError("interactions_df must be provided.")

        if is_training:
            self._validate_interactions(interactions_df)
            self._init_item_state()
            X, y_dict = self._split_X_y(interactions_df)
            return X, y_dict
        else:
            self._validate_inference_interactions(interactions_df)
            X = self._extract_X_inference(interactions_df)
            return X, {}

    def _init_item_state(self) -> None:
        """Populate ``item_names`` from the declared fanned-out target columns.

        ``item_names`` is the public catalogue exposed via
        :class:`BaseScorer` — for this scorer it's the flat ordered list of
        every column the model predicts (simple targets + multilabel
        members), matching the wide-format output column order.
        """
        self.item_names = np.array(self._fanned_out_target_columns, dtype=np.str_)
        self.items_df = None

    def _split_X_y(self, interactions_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, NDArray]]:
        """Wide-format X/y split. y is dict-keyed by ``target_specs`` entry."""
        # Drop USER_ID + all target columns (simple + multilabel members) to
        # get X.
        drop_cols = [C.USER_ID_NAME] + self._fanned_out_target_columns
        X = interactions_df.drop(columns=drop_cols, errors="ignore")

        y_dict: dict[str, NDArray] = {}
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                y_dict[key] = interactions_df[key].to_numpy()
            else:  # multilabel group — 2-D (n, n_members)
                member_cols = spec["columns"]
                y_dict[key] = interactions_df[list(member_cols)].to_numpy()

        return X, y_dict

    def _extract_X_inference(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Pull X out of an inference frame.

        Drops ``USER_ID``, any present target columns, and any
        ``OBSERVED_*`` columns (those are dispatched separately via
        :meth:`_build_observed_dict_from_interactions` for conditional
        estimators; the underlying estimator's X must remain pure features).
        """
        observed_cols = [c for c in interactions_df.columns if c.startswith(C.OBSERVED_PREFIX)]
        drop_cols = (
            [C.USER_ID_NAME]
            + [c for c in self._fanned_out_target_columns if c in interactions_df.columns]
            + observed_cols
        )
        return interactions_df.drop(columns=drop_cols, errors="ignore")

    # ------------------------------------------------------------------ #
    # Scoring (M4 wiring; M1 ships stubs so the contract surface is testable)
    # ------------------------------------------------------------------ #

    def score_items(
        self,
        interactions: Optional[pd.DataFrame] = None,
        users: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Per-target probabilities/values in wide format.

        OBSERVED-aware dispatch (v3):

        - When the wrapped estimator implements
          :class:`ConditionalMultiTargetEstimator` AND ``interactions``
          carries one or more ``OBSERVED_<suffix>`` columns, predictions
          flow through ``predict_with_observed`` — observed values
          condition per-target predictions; NaN per cell means "not
          observed for this row." Multilabel group members must mask
          together per row (enforced by
          :meth:`_validate_inference_interactions`).
        - When the estimator is vanilla
          (:class:`MultiTargetEstimator` only), any ``OBSERVED_*`` column
          is rejected with a pointer to the conditional estimator classes.

        Args:
            interactions: Inference DataFrame with ``USER_ID``, features,
                optional ``OBSERVED_*`` columns (conditional path), and
                optionally ground-truth target columns (ignored if present).
            users: Must be ``None`` (wide-format scorer).

        Returns:
            Wide DataFrame, one row per input user, with per-target column
            blocks. See the v2 plan's "Output column conventions" table.

        Raises:
            NotImplementedError: If a vanilla estimator receives
                ``OBSERVED_*`` columns. Use a conditional estimator
                (``ConditionalJointMultiTargetMLPEstimator`` or
                ``ConditionalJointMultiTargetTransformerEstimator``).
        """
        if users is not None:
            raise ValueError(
                "MixedTypeMultiTargetScorer does not accept users at inference; pass features in interactions instead."
            )
        if interactions is None:
            raise TypeError("interactions must be provided.")
        self._validate_inference_interactions(interactions)
        X = self._extract_X_inference(interactions)
        proba_dict = self._estimator_predict_proba(X, interactions)
        return self._stitch_score_items(proba_dict, n_rows=X.shape[0])

    def predict_targets(
        self,
        interactions: Optional[pd.DataFrame] = None,
        users: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Per-target point estimates in wide format.

        Args:
            interactions: Inference DataFrame with ``USER_ID`` + features.
            users: Must be ``None``.

        Returns:
            Wide DataFrame with one column per fanned-out target.
        """
        if users is not None:
            raise ValueError(
                "MixedTypeMultiTargetScorer does not accept users at inference; pass features in interactions instead."
            )
        if interactions is None:
            raise TypeError("interactions must be provided.")
        self._validate_inference_interactions(interactions)
        X = self._extract_X_inference(interactions)
        targets_dict = self._estimator_predict_targets(X, interactions)
        return self._stitch_predict_targets(targets_dict, n_rows=X.shape[0])

    def score_fast(self, features: pd.DataFrame) -> pd.DataFrame:
        """Single-row predict_targets.

        Fresh implementation calling ``estimator.predict_targets_dict``
        directly; does NOT route through ``_calculate_scores``.

        Args:
            features: DataFrame with exactly one row. Columns are feature
                names (NOT ``USER_ID``; this is the post-join wide-format
                shape used by :meth:`recommend_online`).

        Returns:
            One-row wide DataFrame with one column per fanned-out target.

        Raises:
            TypeError: If ``features`` is not a DataFrame.
            ValueError: If ``features.shape[0] != 1`` or if the column set
                does not match the training-time feature columns.

        Note: v3 hook — this method will accept ``OBSERVED_*`` columns when
        the estimator is a ``ConditionalMultiTargetEstimator``. v2 rejects
        any ``OBSERVED_*`` upstream.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError(f"score_fast expects a DataFrame; got {type(features).__name__}.")
        if features.shape[0] != 1:
            raise ValueError(f"score_fast requires features.shape[0] == 1; got {features.shape[0]}.")
        # Route through the full inference validator so the same rules fire
        # on this single-row path as on the batch path — orphan ITEM_*,
        # OBSERVED-aware dispatch, multilabel group-mask-together,
        # column-level imbalance. score_fast historically takes a post-join
        # feature frame WITHOUT USER_ID, so synthesise one for the
        # validator and strip it back out afterwards.
        validation_frame = features
        synthesised_user_id = C.USER_ID_NAME not in features.columns
        if synthesised_user_id:
            validation_frame = features.copy()
            validation_frame[C.USER_ID_NAME] = "_score_fast_user"
        self._validate_inference_interactions(validation_frame)
        # USER_ID + OBSERVED_* are not part of the feature row. Strip both
        # (OBSERVED_* was already validated; pulled out separately for
        # conditional dispatch in _estimator_predict_targets).
        observed_cols = [c for c in features.columns if c.startswith(C.OBSERVED_PREFIX)]
        feats = features.drop(columns=[C.USER_ID_NAME] + observed_cols, errors="ignore")
        targets_dict = self._estimator_predict_targets(feats, features)
        return self._stitch_predict_targets(targets_dict, n_rows=1)

    def score_per_target(
        self,
        interactions: pd.DataFrame,
        y_true: pd.DataFrame,
        metric_callables: dict[Union[str, "TargetType"], Callable[[NDArray, NDArray], float]],
    ) -> dict[str, float]:
        """Per-target evaluation escape hatch with user-supplied callables.

        Lookup precedence: target-name key > ``TargetType`` key. A target
        with neither a name override nor a type-keyed default raises
        ``KeyError``.

        Args:
            interactions: Inference frame with ``USER_ID`` + features used to
                compute predictions.
            y_true: Wide DataFrame matching ``predict_targets``'s output
                column set (one column per fanned-out target). Alignment is
                by **column name**, not position.
            metric_callables: Dict keyed by either a target name (override)
                or a ``TargetType`` (default for that type). Callable
                signature per target type — see scorer docstring.

        Returns:
            Dict keyed by fanned-out target name; values are metric floats.

        Raises:
            ValueError: If ``y_true`` columns don't match the fanned-out
                target column set.
            KeyError: If a target has neither a name override nor a
                type-keyed default.
        """
        # Column-set check.
        required = set(self._fanned_out_target_columns)
        provided = set(y_true.columns)
        missing = required - provided
        extra = provided - required
        if missing:
            raise ValueError(f"y_true is missing target column(s): {sorted(missing)}.")
        if extra:
            raise ValueError(f"y_true has unknown column(s) not in target_specs: {sorted(extra)}.")

        # Compute predictions (proba for binary/multiclass/multilabel; values
        # for regression). Drives both probability-style and value-style
        # callables in one pass.
        self._validate_inference_interactions(interactions)
        X = self._extract_X_inference(interactions)
        proba_dict = self._estimator_predict_proba(X, interactions)

        out: dict[str, float] = {}
        for fanned_name in self._fanned_out_target_columns:
            lookup_types = self._metric_lookup_types_for_fanned(fanned_name)
            callable_ = self._lookup_metric_callable(fanned_name, lookup_types, metric_callables)
            y_true_col = y_true[fanned_name].to_numpy()
            preds = proba_dict[fanned_name]
            out[fanned_name] = float(callable_(y_true_col, preds))
        return out

    def preserved_inference_columns(self) -> list[str]:
        """Override of the :class:`BaseScorer` hook (added in v3).

        Returns the ``OBSERVED_<suffix>`` column names matching every
        declared target — **unconditionally**, regardless of whether the
        estimator is vanilla or conditional. Rationale (carried forward
        from the v1 plan):

          - The hook's job is "what columns must survive schema apply so
            the scorer can validate them?" — a scorer property, not an
            estimator property.
          - Returning ``[]`` for vanilla would let ``schema.apply()``
            silently strip OBSERVED_* with a generic unknown-column warning,
            hiding the user's intent ("I want to condition") and leaving
            the scorer unable to raise a clean v2 deferred-to-v3 / v3
            wrong-estimator error.
          - With unconditional preservation, vanilla + OBSERVED_* at
            inference is rejected explicitly by
            :meth:`_validate_inference_interactions` with an actionable
            message naming the conditional estimator classes.

        Suffix mapping: ``ITEM_<suffix>`` (target) ↔ ``OBSERVED_<suffix>``
        (observed input). For multilabel groups, each fanned-out member
        contributes one ``OBSERVED_*`` column (members mask together per
        row at inference).
        """
        out: list[str] = []
        for fanned_name in self._fanned_out_target_columns:
            # Strip ITEM_ prefix; OBSERVED_<remainder>.
            if fanned_name.startswith(C.ITEM_PREFIX):
                suffix = fanned_name[len(C.ITEM_PREFIX) :]
            else:
                # Defensive — fanned-out names are always ITEM_-prefixed by
                # construction (simple targets enforce ITEM_; multilabel
                # members enforce ITEM_). If a future change relaxes that,
                # the suffix is just the name.
                suffix = fanned_name
            out.append(f"{C.OBSERVED_PREFIX}{suffix}")
        return out

    def _is_conditional_estimator(self) -> bool:
        """Strict conditional-estimator check: Protocol + opt-in sentinel.

        ``isinstance(estimator, ConditionalMultiTargetEstimator)`` alone is
        too permissive — ``@runtime_checkable`` Protocol checks are
        structural, so any class with a ``predict_with_observed`` method
        would pass even if it's a sketch implementation that doesn't
        honor the conditional contract. Conditional implementations
        explicitly set the ``is_conditional_multi_target = True``
        sentinel on the class; this method requires BOTH the structural
        match AND the sentinel before routing the scorer through the
        OBSERVED-aware dispatch.
        """
        return isinstance(self.estimator, ConditionalMultiTargetEstimator) and bool(
            getattr(self.estimator, "is_conditional_multi_target", False)
        )

    def preserved_inference_column_prefixes(self) -> list[str]:
        """Preserve every ``OBSERVED_*`` AND ``ITEM_*`` column through
        schema apply, not just the declared ones.

        This lets the scorer's orphan validators (in
        :meth:`_validate_inference_interactions`) see typo'd columns
        like ``OBSERVED_revenuee`` (typo for ``OBSERVED_revenue``) and
        ``ITEM_clicks`` (typo for declared ``ITEM_click``). Without
        prefix-based preservation, schema apply would silently drop both
        typos before the orphan-rejection clauses could fire, and the
        user would see "no OBSERVED columns at inference" or "model
        ignored ITEM_clicks" rather than the actionable orphan errors
        the scorer publishes.
        """
        return [C.OBSERVED_PREFIX, C.ITEM_PREFIX]

    def _calculate_scores(self, joined: pd.DataFrame) -> NDArray:
        """Defunct on this scorer.

        ``score_items`` / ``predict_targets`` / ``score_fast`` are fresh
        implementations that do NOT route through ``_calculate_scores``.
        Raising here means a future change that wires this method back into
        the standard scoring path fails loudly with an actionable message
        rather than silently producing wrong shapes.
        """
        raise NotImplementedError(
            "MixedTypeMultiTargetScorer does not use _calculate_scores. "
            "Use score_items() / predict_targets(). score_fast() is also a "
            "fresh implementation that does NOT route through this method."
        )

    # ------------------------------------------------------------------ #
    # Helpers (used by future milestones; lightweight enough to keep here)
    # ------------------------------------------------------------------ #

    def _output_columns_for_score_items(self) -> list[str]:
        """Wide-format output column order for ``score_items``.

        Per the v2 plan's "Output column conventions" table:
            binary       → ITEM_<col>_0, ITEM_<col>_1
            regression   → ITEM_<col>
            multiclass   → ITEM_<col>_<class_label>  (per class)
            multilabel   → ITEM_<member>_0, ITEM_<member>_1 (per member)
        """
        out: list[str] = []
        multiclass_classes = self._get_multiclass_classes()
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                if spec == TargetType.BINARY:
                    out.extend([f"{key}_0", f"{key}_1"])
                elif spec == TargetType.REGRESSION:
                    out.append(key)
                elif spec == TargetType.MULTICLASS:
                    classes = multiclass_classes.get(key, [])
                    for cls in classes:
                        out.append(f"{key}_{cls}")
            else:  # TargetGroupSpec
                for member in spec["columns"]:
                    out.extend([f"{member}_0", f"{member}_1"])
        return out

    def _output_columns_for_predict_targets(self) -> list[str]:
        """Wide-format output column order for ``predict_targets``.

        One column per fanned-out target (simple targets plus multilabel
        members), in declared order.
        """
        return list(self._fanned_out_target_columns)

    # ------------------------------------------------------------------ #
    # M4: wide-format stitching helpers
    # ------------------------------------------------------------------ #

    def _get_multiclass_classes(self) -> dict[str, list]:
        """Pull each multiclass target's class catalogue off the estimator.

        Both v2 estimator families store ``_multiclass_classes: dict[str,
        list[Any]]`` after fit. Falls back to an empty mapping for
        unfitted/unknown estimators (callers should only invoke after fit).
        """
        return getattr(self.estimator, "_multiclass_classes", {}) or {}

    def _stitch_score_items(self, proba_dict: dict[str, NDArray], n_rows: int) -> pd.DataFrame:
        """Build the wide-format ``score_items`` DataFrame from a per-target dict.

        Column order follows ``_output_columns_for_score_items``.
        """
        multiclass_classes = self._get_multiclass_classes()
        out_cols: list[tuple[str, NDArray]] = []
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                if spec == TargetType.BINARY:
                    arr = proba_dict[key]  # (n, 2)
                    out_cols.append((f"{key}_0", arr[:, 0]))
                    out_cols.append((f"{key}_1", arr[:, 1]))
                elif spec == TargetType.REGRESSION:
                    arr = np.asarray(proba_dict[key]).reshape(-1)
                    out_cols.append((key, arr))
                elif spec == TargetType.MULTICLASS:
                    arr = proba_dict[key]  # (n, K)
                    if key not in multiclass_classes or not multiclass_classes[key]:
                        # Same fail-fast contract as the joint base
                        # estimator's predict_targets_dict — silent
                        # range(K) fallback would produce mislabeled
                        # output columns indistinguishable from the
                        # correct case.
                        raise RuntimeError(
                            f"Multiclass target {key!r} has no class "
                            f"catalogue on the fitted estimator. The "
                            f"estimator must be fit with this target's "
                            f"labels before score_items can map proba "
                            f"columns to class labels."
                        )
                    classes = multiclass_classes[key]
                    for i, cls in enumerate(classes):
                        out_cols.append((f"{key}_{cls}", arr[:, i]))
            else:  # multilabel — proba_dict has fanned-out per-member entries
                for member in spec["columns"]:
                    arr = proba_dict[member]  # (n, 2)
                    out_cols.append((f"{member}_0", arr[:, 0]))
                    out_cols.append((f"{member}_1", arr[:, 1]))
        # Build DataFrame from columns; preserves order.
        df = pd.DataFrame({name: values for name, values in out_cols})
        return df

    def _stitch_predict_targets(self, targets_dict: dict[str, NDArray], n_rows: int) -> pd.DataFrame:
        """Build the wide-format ``predict_targets`` DataFrame.

        Column order follows the fanned-out target order: every simple
        target appears as one column; every multilabel member appears as
        one column.
        """
        out_cols: list[tuple[str, NDArray]] = []
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                out_cols.append((key, np.asarray(targets_dict[key]).reshape(-1)))
            else:
                for member in spec["columns"]:
                    out_cols.append((member, np.asarray(targets_dict[member]).reshape(-1)))
        return pd.DataFrame({name: values for name, values in out_cols})

    def _target_type_for_fanned(self, fanned_name: str) -> "TargetType":
        """Resolve TargetType for a fanned-out target column (simple or member).

        Multilabel members collapse to ``TargetType.BINARY`` here because at
        the fanned-out level each member IS a binary classification head.
        Downstream metric dispatch additionally tries ``TargetType.MULTILABEL``
        as a synonym for the group via
        :meth:`_metric_lookup_types_for_fanned` — that way users who key
        ``metric_callables`` by ``TargetType.MULTILABEL`` get a hit instead
        of an opaque "no metric found" error.
        """
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                if key == fanned_name:
                    return spec
            else:
                if fanned_name in spec["columns"]:
                    return TargetType.BINARY  # multilabel members are binary
        raise KeyError(f"Unknown fanned-out target {fanned_name!r}.")

    def _metric_lookup_types_for_fanned(self, fanned_name: str) -> list["TargetType"]:
        """Ordered list of TargetType keys to try when looking up callables.

        For simple targets, the list has one entry — the declared
        TargetType. For multilabel members, both ``BINARY`` (the
        fanned-out classification semantics) AND ``MULTILABEL`` (the
        group-level semantics declared in target_specs) are returned in
        precedence order: BINARY first (matches the head shape), then
        MULTILABEL (matches what users intuitively type in
        ``metric_callables`` when grouping per the spec). Without the
        MULTILABEL fallback, a caller passing
        ``metric_callables={TargetType.MULTILABEL: my_fn}`` would get an
        opaque "no metric found" error because every fanned-out member
        reports BINARY.
        """
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                if key == fanned_name:
                    return [spec]
            else:
                if fanned_name in spec["columns"]:
                    return [TargetType.BINARY, TargetType.MULTILABEL]
        raise KeyError(f"Unknown fanned-out target {fanned_name!r}.")

    def _estimator_predict_proba(self, X: pd.DataFrame, interactions: pd.DataFrame) -> dict[str, NDArray]:
        """Dispatch ``predict_proba`` against the estimator, threading OBSERVED_* through."""
        if self._is_conditional_estimator():
            observed = self._build_observed_dict_from_interactions(interactions, X.shape[0])
            # Conditional estimators expose predict_with_observed (returns
            # the same dict shape as predict_proba_dict).
            return self.estimator.predict_with_observed(X, observed)
        return self.estimator.predict_proba_dict(X)

    def _estimator_predict_targets(self, X: pd.DataFrame, interactions: pd.DataFrame) -> dict[str, NDArray]:
        """Dispatch ``predict_targets`` against the estimator.

        For conditional estimators we route through ``predict_with_observed``
        and then collapse the proba/value dict into per-target point
        estimates using the same rules as the joint base estimator's
        ``predict_targets_dict``.
        """
        if not self._is_conditional_estimator():
            return self.estimator.predict_targets_dict(X)

        observed = self._build_observed_dict_from_interactions(interactions, X.shape[0])
        proba = self.estimator.predict_with_observed(X, observed)
        # Collapse proba → point estimates per target type.
        out: dict[str, NDArray] = {}
        multiclass_classes = self._get_multiclass_classes()
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                if spec == TargetType.BINARY:
                    out[key] = (proba[key][:, 1] >= 0.5).astype(np.int64)
                elif spec == TargetType.REGRESSION:
                    out[key] = proba[key]
                elif spec == TargetType.MULTICLASS:
                    if key not in multiclass_classes or not multiclass_classes[key]:
                        raise RuntimeError(
                            f"Multiclass target {key!r} has no class "
                            f"catalogue on the fitted estimator. The "
                            f"estimator must be fit with this target's "
                            f"labels before predict_targets can map proba "
                            f"argmax to class labels."
                        )
                    classes = multiclass_classes[key]
                    arg = proba[key].argmax(axis=1)
                    out[key] = np.array([classes[i] for i in arg])
            else:
                for member in spec["columns"]:
                    out[member] = (proba[member][:, 1] >= 0.5).astype(np.int64)
        return out

    def _build_observed_dict_from_interactions(self, interactions: pd.DataFrame, n_rows: int) -> dict[str, NDArray]:
        """Extract per-target observed arrays from OBSERVED_* columns.

        Maps ``OBSERVED_<suffix>`` columns back to their declared targets:
          - Simple target (ITEM_-prefixed): the OBSERVED column populates a
            1-D array; missing column → fully unobserved (all-NaN array).
          - Multilabel group: each member's OBSERVED column becomes one
            column of the group's 2-D array. The "members mask together"
            invariant is enforced by ``_validate_inference_interactions``;
            here we just stitch.
        """
        out: dict[str, NDArray] = {}
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                obs_col = f"{C.OBSERVED_PREFIX}{key[len(C.ITEM_PREFIX) :]}"
                if obs_col in interactions.columns:
                    out[key] = interactions[obs_col].to_numpy()
                else:
                    out[key] = np.full(n_rows, np.nan)
            else:
                member_cols = []
                for m in spec["columns"]:
                    obs_col = f"{C.OBSERVED_PREFIX}{m[len(C.ITEM_PREFIX) :]}"
                    if obs_col in interactions.columns:
                        member_cols.append(interactions[obs_col].to_numpy())
                    else:
                        member_cols.append(np.full(n_rows, np.nan))
                out[key] = np.column_stack(member_cols)
        return out

    @staticmethod
    def _lookup_metric_callable(
        fanned_name: str,
        lookup_types: list["TargetType"],
        metric_callables: dict[Union[str, "TargetType"], Callable[[NDArray, NDArray], float]],
    ) -> Callable[[NDArray, NDArray], float]:
        """Name override beats TargetType default; multiple TargetType keys
        are tried in the order supplied (used by multilabel members so a
        ``MULTILABEL``-keyed default works alongside the implicit
        ``BINARY`` head semantics)."""
        if fanned_name in metric_callables:
            return metric_callables[fanned_name]
        for tt in lookup_types:
            if tt in metric_callables:
                return metric_callables[tt]
        raise KeyError(
            f"No metric callable for target {fanned_name!r} (tried types "
            f"{[t.value for t in lookup_types]}). Provide either a "
            f"target-name override or a TargetType-keyed default."
        )
