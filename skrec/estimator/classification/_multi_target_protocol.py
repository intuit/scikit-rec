# Scorer-facing contract for multi-target estimators.
#
# Why a runtime-checkable Protocol (and not a base class):
#   - The MixedTypeMultiTargetScorer ships with three concrete estimator
#     families that share nothing at the implementation level: two torch-based
#     joint estimators (MLP, Transformer) and one independent estimator that
#     wraps a dict of pre-existing scikit-rec sub-estimators. Forcing them
#     under a common ABC would either bloat the joint estimators with
#     independent-mode bookkeeping or vice-versa. Structural typing via
#     Protocol keeps the scorer's contract explicit while leaving each family
#     free to organize its internals.
#   - runtime_checkable lets the scorer's __init__ raise a clean TypeError on
#     non-conforming estimators at construction time — same UX as
#     MultioutputScorer's isinstance check against BaseClassifier/BaseRegressor,
#     just structurally typed.
#
# Strict attribute set required (no more, no less):
#   - target_specs: dict[str, TargetType | TargetGroupSpec] — the declared
#     per-target schema. Must equal the scorer's target_specs (consistency
#     check enforced at scorer __init__).
#   - fit(X, y, X_valid=None, y_valid=None) — y is dict-shaped (one entry per
#     target spec key), not Series/array. Dict y has no .shape attribute, which
#     is why concrete implementations override _validate_for_fit on the
#     estimator side rather than inheriting BaseEstimator's array-shaped check.
#   - predict_proba_dict(X) — per-target probabilities/values; multilabel
#     groups fan out into per-member entries.
#   - predict_targets_dict(X) — per-target point estimates; multilabel groups
#     fan out into per-member entries.
#
# Note: no predict_with_observed here. That method lives on the v3 subclass
# Protocol ConditionalMultiTargetEstimator so the v2 vanilla contract stays
# narrow. v3 adds the subclass purely additively.
#
# Test #24 in tests/test_mixed_type_multi_target_scorer.py asserts isinstance
# strictness: positive (each of the three v2 families) AND negative (a class
# with target_specs+fit but missing predict_*_dict methods fails the check).
# Without the negative assertion, accidental Protocol widening goes silent.

from enum import Enum
from typing import List, Optional, Protocol, TypedDict, runtime_checkable

import pandas as pd


class TargetType(str, Enum):
    """Declared type of a target column in MixedTypeMultiTargetScorer.

    Co-located with the MultiTargetEstimator Protocol so the contract types
    live next to the contract definition. Re-exported from
    ``skrec.scorer.mixed_type_multi_target`` for the user-facing path —
    that's the canonical import location for end users.
    """

    BINARY = "binary"
    REGRESSION = "regression"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class TargetGroupSpec(TypedDict):
    """Multi-column target grouping (currently used only for multilabel).

    Members:
        type: Must be ``TargetType.MULTILABEL`` in v2.
        columns: Non-empty list of ``ITEM_``-prefixed column names that
            form the group.
    """

    type: TargetType
    columns: List[str]


def sort_multiclass_labels(labels):
    """Sort multiclass label catalogue with the right semantics per dtype.

    ``sorted(labels, key=str)`` (the v2/v3 initial implementation) silently
    breaks integer labels with K≥10 because ``str(10) < str(2)``
    lexicographically, so ``sorted([1, 2, 10], key=str) == [1, 10, 2]``.
    Predictions then round-trip to the wrong class via the wrong-positioned
    argmax index → MULTICLASS_ACCURACY computes against scrambled labels.

    This helper does natural sort when labels are mutually comparable
    (homogeneous ints / floats / strings) and falls back to string sort
    only when natural sort raises ``TypeError`` (mixed types). The fallback
    is purely a safety net — heterogeneous label sets shouldn't reach a
    multiclass target in the first place; the scorer's training validator
    enforces a single value-set per column.
    """
    labels = list(labels)
    try:
        return sorted(labels)
    except TypeError:
        return sorted(labels, key=str)


@runtime_checkable
class MultiTargetEstimator(Protocol):
    """Runtime-checkable contract that scorers use to validate estimators.

    The scorer rejects estimators that don't satisfy this Protocol at
    construction time with a TypeError naming the missing attributes.

    Attributes:
        target_specs: Declared per-target schema. Keys are target names (or
            multilabel group keys). Values are TargetType enums for simple
            targets, or TargetGroupSpec dicts for multilabel groups. Must
            equal the scorer's target_specs (consistency check enforced at
            scorer construction).
    """

    target_specs: dict  # dict[str, TargetType | TargetGroupSpec] — see TYPE_CHECKING note

    def fit(
        self,
        X: pd.DataFrame,
        y: dict,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[dict] = None,
    ) -> "MultiTargetEstimator":
        """Fit the estimator on a wide feature frame and per-target labels.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.
            y: Dict keyed by target_spec key. Values per target type:
                - binary / multiclass: 1-D array of class labels
                - regression: 1-D array of continuous values
                - multilabel group: 2-D array of shape (n_samples, n_members)
                  with members in declared column order
            X_valid: Optional validation features.
            y_valid: Optional validation labels in the same dict shape.

        Returns:
            Self (to support fluent chaining if desired).
        """
        ...

    def predict_proba_dict(self, X: pd.DataFrame) -> dict:
        """Per-target probabilities / values, keyed by **fanned-out** target name.

        For multilabel groups, the group key is replaced by one entry per
        member column, so downstream wide-format stitching is uniform with
        the simple-binary case.

        Returns:
            Dict keyed by target_name (or member column name, for multilabel).
            Values per target type:

            =================  ===================================================
            Target type        Value shape and meaning
            =================  ===================================================
            binary             (n, 2) array of class probabilities, columns [0, 1]
            multiclass         (n, K) array of class probabilities
            multilabel-member  (n, 2) array of class probabilities, columns [0, 1]
            regression         (n,) array of predicted values (de-normalized
                               if the estimator z-score-normalized internally)
            =================  ===================================================
        """
        ...

    def predict_targets_dict(self, X: pd.DataFrame) -> dict:
        """Per-target point estimates, keyed by fanned-out target name.

        Returns:
            Dict keyed the same as ``predict_proba_dict``. Values:

            =================  ===================================================
            Target type        Value shape and meaning
            =================  ===================================================
            binary             (n,) array of predicted labels in {0, 1}
            multiclass         (n,) array of predicted class labels (preserving
                               input label dtype)
            multilabel-member  (n,) array of predicted labels in {0, 1}
            regression         (n,) array of predicted values
            =================  ===================================================
        """
        ...


@runtime_checkable
class ConditionalMultiTargetEstimator(MultiTargetEstimator, Protocol):
    """v3 Protocol subclass: estimators that support real-time-label conditioning.

    A conditional estimator accepts an ``observed`` dict at inference time
    alongside the feature matrix ``X``. For declared targets where the
    caller has observed the ground-truth value in real time, the value is
    used to condition predictions for other targets. For targets where the
    value is NaN (per row), the estimator predicts from features alone, as
    in vanilla mode.

    Joint conditional estimators (v3): ``ConditionalJointMultiTargetMLPEstimator``,
    ``ConditionalJointMultiTargetTransformerEstimator``. Independent
    estimators are NOT conditional in v3 (cross-target observed-as-features
    is structurally different; revisited in v4+ if ever).

    Scorer dispatch: ``MixedTypeMultiTargetScorer._validate_inference_interactions``
    checks ``isinstance(estimator, ConditionalMultiTargetEstimator)`` to
    decide whether ``OBSERVED_*`` columns are allowed at inference. Vanilla
    estimators reject them with a clean error pointing at the conditional
    estimator classes; conditional estimators permit them (with the
    multilabel-group "members must mask together per row" rule enforced
    by the scorer's validator).

    Attributes inherited from :class:`MultiTargetEstimator`. Adds one
    explicit opt-in sentinel attribute (``is_conditional_multi_target``)
    plus one method (``predict_with_observed``).

    The sentinel matters because ``@runtime_checkable`` Protocol
    isinstance checks are STRUCTURAL only: any class that happens to
    declare a ``predict_with_observed`` method (correctly or not) would
    otherwise pass the isinstance check and silently activate the
    conditional dispatch path in the scorer. The sentinel attribute is
    an explicit opt-in — implementations must set
    ``is_conditional_multi_target = True`` as a class attribute. A
    look-alike that forgets the sentinel is rejected, surfacing the
    misconfiguration rather than silently routing through OBSERVED_*
    handling.
    """

    # Class-level sentinel that conditional implementations MUST set to
    # True. Default False at the Protocol declaration so structural
    # look-alikes that omit the attribute don't accidentally satisfy the
    # runtime_checkable isinstance check.
    is_conditional_multi_target: bool = False

    def predict_with_observed(self, X: pd.DataFrame, observed: Optional[dict] = None) -> dict:
        """Per-target probabilities/values conditioned on ``observed``.

        Same output shape contract as :meth:`MultiTargetEstimator.predict_proba_dict`.
        Calling ``predict_with_observed(X, None)`` or
        ``predict_with_observed(X, {})`` is equivalent to
        ``predict_proba_dict(X)`` — the vanilla path falls out as a special
        case where every target is treated as fully unobserved (all
        positions masked). Implementations must accept both ``None`` and
        an empty dict.

        Args:
            X: Feature matrix.
            observed: Optional dict keyed by ``target_specs`` entries;
                per-row NaN means "not observed for this row, predict from
                features." Missing keys (or ``None``/empty dict) → fully
                unobserved for every target. Multilabel groups: members
                must mask together (all observed or all NaN per row); the
                scorer's validator enforces this.

        Returns:
            Dict keyed by fanned-out target name, same shape as
            :meth:`predict_proba_dict`.
        """
        ...
