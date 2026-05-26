# Independent multi-target estimator: per-target sub-estimators.
#
# Sibling to the joint MLP / joint Transformer families but with a different
# learning paradigm. Each target is trained by its own scikit-rec sub-estimator
# (XGB, LightGBM, LogReg, sklearn-universal). No shared representation; no
# joint loss. Predictions are stitched together at the dict level. Multilabel
# groups fan out into per-member binary classifiers.
#
# Why a single concrete class (and not a family like joint MLP / joint
# Transformer):
#   - The "family" dimension for joint estimators is the encoder architecture.
#     Independent estimators have no encoder — the polymorphism is in the
#     dict of per-target sub-estimators. One class is enough.
#
# Two construction paths (both supported):
#
#   1. Direct construction — caller hands in a pre-built dict of sub-estimators:
#
#        estimator = IndependentMultiTargetEstimator(
#            target_specs={
#                "ITEM_clicked": TargetType.BINARY,
#                "ITEM_revenue": TargetType.REGRESSION,
#                "g": TargetGroupSpec(type=TargetType.MULTILABEL,
#                                     columns=["ITEM_a", "ITEM_b"]),
#            },
#            estimators={
#                "ITEM_clicked": XGBClassifierEstimator(...),
#                "ITEM_revenue": LightGBMRegressorEstimator(...),
#                # Multilabel members are keyed by member column, NOT by
#                # the group key. Group key in estimators → clean error.
#                "ITEM_a": XGBClassifierEstimator(...),
#                "ITEM_b": LogRegClassifierEstimator(...),
#            },
#        )
#
#   2. Factory construction — orchestrator (M5) composes sub-estimators from
#      a defaults + per_target spec. Same end state as path 1.
#
# Validation symmetry: type compatibility between sub-estimator class and
# declared TargetType is checked at __init__ for both paths. Catches
# "regressor on binary" / "binary classifier on regression" before any fit
# call runs.
#
# Multilabel group inductive bias: lost in this mode. Each member is its
# own binary classifier. The decision-rule doc calls out the trade-off
# (joint family preserves the group bias; independent mode trades it for
# per-target estimator flexibility).
#
# Partial-fit failure semantics: if any sub-estimator's .fit raises, the
# estimator is left in a state where predict_proba_dict / predict_targets_dict
# raise "not fitted" rather than producing partial output. _fitted flag is
# the gate.
#
# Multiclass class catalogues: captured per target at fit time from
# np.unique(y) in sorted order. Mapped back to original labels at predict
# time via stored catalogue. Mirrors the joint family's catalogue handling.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from skrec.estimator.classification._multi_target_protocol import (
    TargetGroupSpec,
    TargetType,
    sort_multiclass_labels,
)
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.regression.base_regressor import BaseRegressor
from skrec.util.logger import get_logger

logger = get_logger(__name__)


# TargetType → required sub-estimator base class.
_TARGET_TYPE_TO_BASE_CLASS = {
    TargetType.BINARY: BaseClassifier,
    TargetType.REGRESSION: BaseRegressor,
    TargetType.MULTICLASS: BaseClassifier,
    # MULTILABEL is handled per member at the group-spec level; each member
    # is a binary classifier.
}


class IndependentMultiTargetEstimator:
    """Per-target sub-estimators wrapped under one ``MultiTargetEstimator``.

    Implements :class:`MultiTargetEstimator` so the scorer accepts it
    interchangeably with the joint MLP / joint Transformer families.

    Args:
        target_specs: Per-target schema; must match the scorer's.
        estimators: Dict keyed by **fanned-out** target name. For simple
            targets, the key is the target_specs key (ITEM_-prefixed). For
            multilabel groups, the dict has one entry **per member column**,
            keyed by member column name (NOT the group key). Every fanned-out
            target must be covered; the group key itself must NOT appear.

    Raises:
        ValueError: If ``target_specs`` is empty or malformed; if
            ``estimators`` keys don't cover every fanned-out target; if
            ``estimators`` contains a group key (group keys are metadata
            only — fan members into the dict instead); or if a sub-estimator's
            class is incompatible with its declared target type.
    """

    def __init__(
        self,
        target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
        estimators: Dict[str, Any],
    ) -> None:
        self.target_specs = target_specs
        self._validate_target_specs(target_specs)

        # Compute the fanned-out target list (simple targets + per-member
        # columns) — must match exactly what the scorer derives.
        self._fanned_out_targets: List[str] = self._compute_fanned_out_targets(target_specs)
        # Also remember group keys so we can detect "user passed group key
        # in estimators" and explain the convention.
        self._group_keys: set = {key for key, spec in target_specs.items() if isinstance(spec, dict)}

        self._validate_estimators_coverage(estimators)
        self._validate_sub_estimator_types(estimators)

        # Public attribute (estimators keyed by fanned-out target name).
        self.estimators: Dict[str, Any] = dict(estimators)

        # State populated by .fit
        self._feature_names: Optional[List[str]] = None
        self._multiclass_classes: Dict[str, List[Any]] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    # Static helpers (shared with the scorer's validation; kept local so
    # this estimator can be used standalone without scorer coupling).
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_fanned_out_targets(
        target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
    ) -> List[str]:
        out: List[str] = []
        for key, spec in target_specs.items():
            if isinstance(spec, TargetType):
                out.append(key)
            else:
                out.extend(spec["columns"])
        return out

    @staticmethod
    def _validate_target_specs(
        target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
    ) -> None:
        """Minimal duplicate of MixedTypeMultiTargetScorer._validate_target_specs.

        Independent estimators can be constructed without a scorer (e.g., for
        unit-test use), so the validation lives here too. The scorer's
        consistency check ensures the two validators see the same target_specs
        when both are constructed.
        """
        if not target_specs:
            raise ValueError("target_specs must be non-empty.")
        for key, spec in target_specs.items():
            if isinstance(spec, TargetType):
                continue
            if not isinstance(spec, dict):
                raise ValueError(f"target_specs[{key!r}] must be a TargetType or TargetGroupSpec.")
            if "type" not in spec or "columns" not in spec:
                raise ValueError(f"TargetGroupSpec for {key!r} must have 'type' and 'columns'.")
            # Direct/factory parity: the scorer rejects non-MULTILABEL
            # group types with "only MULTILABEL is supported in v2."
            # Without the same check here, direct construction of
            # IndependentMultiTargetEstimator with
            # TargetGroupSpec(type=BINARY, columns=[...]) would
            # silently succeed and fan out members as if it were a
            # multilabel group — the divergence the scorer pins as
            # malformed input would slip through one construction path
            # but not the other. Pin the same rejection here.
            group_type = spec["type"]
            # Accept string form too (matches the scorer's tolerance).
            if isinstance(group_type, str):
                try:
                    group_type = TargetType(group_type)
                except ValueError as exc:
                    raise ValueError(
                        f"TargetGroupSpec for {key!r}: 'type' must be a valid TargetType; got {spec['type']!r}."
                    ) from exc
            if group_type != TargetType.MULTILABEL:
                raise ValueError(
                    f"TargetGroupSpec for key {key!r}: only "
                    f"TargetType.MULTILABEL is supported in v2/v3. Got "
                    f"{group_type}. (Simple typed targets use "
                    f"``target_specs[{key!r}] = TargetType.<TYPE>`` "
                    f"directly — TargetGroupSpec is only for multilabel "
                    f"groups of binary members.)"
                )
            if not isinstance(spec["columns"], (list, tuple)) or len(spec["columns"]) == 0:
                raise ValueError(f"TargetGroupSpec for {key!r}: 'columns' must be a non-empty list.")
            # Duplicate members would silently collapse the fanned-out
            # estimators dict and corrupt the per-member loss / metric path.
            # Reject upfront with the offending duplicates named.
            cols = list(spec["columns"])
            seen: Dict[str, int] = {}
            for c in cols:
                seen[c] = seen.get(c, 0) + 1
            dups = sorted(c for c, n in seen.items() if n > 1)
            if dups:
                raise ValueError(
                    f"TargetGroupSpec for {key!r}: duplicate member column(s) "
                    f"{dups}. Multilabel groups require unique members."
                )

    def _validate_estimators_coverage(self, estimators: Dict[str, Any]) -> None:
        """Every fanned-out target must have an estimator; no extras allowed."""
        provided = set(estimators.keys())
        required = set(self._fanned_out_targets)

        # Detect group-key entries (helpful message — common user mistake).
        offending_group_keys = provided & self._group_keys
        if offending_group_keys:
            raise ValueError(
                f"estimators dict contains multilabel group key(s) "
                f"{sorted(offending_group_keys)}. Multilabel groups must be "
                f"fanned out: provide one estimator per **member column**, not "
                f"per group key. Use the member column name as the dict key."
            )

        missing = required - provided
        if missing:
            raise ValueError(
                f"estimators dict missing entries for target(s): {sorted(missing)}. "
                f"Required: {sorted(required)}. Got: {sorted(provided)}."
            )
        extra = provided - required
        if extra:
            raise ValueError(
                f"estimators dict has unknown key(s): {sorted(extra)}. "
                f"Required: {sorted(required)}. (Did you pass a group key by "
                f"mistake? Use member column names instead.)"
            )

    def _validate_sub_estimator_types(self, estimators: Dict[str, Any]) -> None:
        """Reject regressors on binary targets, classifiers on regression, etc."""
        for target_name in self._fanned_out_targets:
            sub_est = estimators[target_name]
            target_type = self._target_type_for(target_name)
            required_base = _TARGET_TYPE_TO_BASE_CLASS[target_type]
            if not isinstance(sub_est, required_base):
                raise ValueError(
                    f"estimator for target {target_name!r} (declared "
                    f"{target_type.value}) must be a {required_base.__name__}; "
                    f"got {type(sub_est).__name__}."
                )

    def _target_type_for(self, fanned_out_name: str) -> TargetType:
        """Resolve TargetType for a fanned-out target name (simple or member).

        Multilabel members resolve to ``BINARY`` because each fanned-out
        member IS its own binary classifier. ``_validate_target_specs``
        upfront-rejects non-MULTILABEL group types so this method only
        ever sees MULTILABEL groups in the dict branch; a defensive
        assertion below makes the contract explicit for future readers.
        """
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                if key == fanned_out_name:
                    return spec
            else:
                # multilabel group; member columns are all binary.
                # Defensive: _validate_target_specs upfront-rejects any
                # group whose ``type`` isn't MULTILABEL, so spec["type"]
                # must be MULTILABEL here. If a future refactor weakens
                # that validator, this assertion surfaces the regression
                # at predict time rather than silently mis-typing
                # multiclass-group members as binary.
                group_type = spec.get("type")
                if isinstance(group_type, str):
                    group_type = TargetType(group_type)
                assert group_type == TargetType.MULTILABEL, (
                    f"TargetGroupSpec for {key!r} has type {group_type!r}; "
                    f"only MULTILABEL is permitted. _validate_target_specs "
                    f"should have rejected this — investigate."
                )
                if fanned_out_name in spec["columns"]:
                    return TargetType.BINARY
        # Should be unreachable — coverage validator catches first.
        raise KeyError(f"Unknown fanned-out target {fanned_out_name!r}.")

    # ------------------------------------------------------------------ #
    # Public Protocol contract
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: pd.DataFrame,
        y: Dict[str, NDArray],
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[Dict[str, NDArray]] = None,
    ) -> "IndependentMultiTargetEstimator":
        """Train each sub-estimator on its own target.

        On any sub-estimator's failure, ``self._fitted`` is left ``False`` so
        ``predict_*_dict`` raises "not fitted" instead of producing partial
        output.
        """
        # Reset _fitted BEFORE validation. If validation raises (e.g. on a
        # re-fit with malformed y), the prior fitted state must not remain
        # queryable — that contradicts the "no half-fit-state" invariant
        # pinned by test_partial_fit_failure_leaves_estimator_unfitted.
        self._fitted = False
        self._multiclass_classes = {}
        self._validate_for_fit(X, y, X_valid, y_valid)
        self._feature_names = X.columns.tolist()

        # Iterate fanned-out targets in declared order (deterministic).
        for fanned_name in self._fanned_out_targets:
            sub_est = self.estimators[fanned_name]
            y_target = self._extract_y_for_target(y, fanned_name)
            y_valid_target = self._extract_y_for_target(y_valid, fanned_name) if y_valid is not None else None

            # Multiclass: capture class catalogue + integer-encode for the
            # sub-estimator. Sub-estimators like XGBClassifier require
            # integer class labels — encoding here keeps the public
            # predict_targets_dict contract ("preserve input label dtype")
            # while the wire format to the sub-estimator stays numeric.
            if self._target_type_for(fanned_name) == TargetType.MULTICLASS:
                # Natural sort (not lex-of-str) — fixes the K≥10 integer
                # label bug where sorted([1,2,10], key=str) == [1,10,2].
                classes = sort_multiclass_labels(set(np.asarray(y_target).tolist()))
                self._multiclass_classes[fanned_name] = classes
                label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
                y_target = np.array([label_to_idx[v] for v in np.asarray(y_target).tolist()])
                if y_valid_target is not None:
                    y_valid_target = np.array([label_to_idx[v] for v in np.asarray(y_valid_target).tolist()])

            sub_est.fit(
                X,
                y_target,
                X_valid=X_valid if X_valid is not None else None,
                y_valid=y_valid_target,
            )

        self._fitted = True
        return self

    def _safe_sub_estimator_inference(
        self,
        fanned_name: str,
        target_type: TargetType,
        sub_est: Any,
        X: pd.DataFrame,
    ) -> NDArray:
        """Shape-guarded sub-estimator inference call.

        Per-target dispatch:
          - BINARY     → ``sub_est.predict_proba(X)``; returns ``(n, 2)``
                        class-probability stack.
          - REGRESSION → ``sub_est.predict(X)``; returns ``(n,)`` values
                        (reshaped from ``(n, 1)`` if needed). NOT a
                        proba — name change in round 4 reflects this.
          - MULTICLASS → ``sub_est.predict_proba(X)``; returns ``(n, K)``.

        Shared by both ``predict_proba_dict`` and ``predict_targets_dict``
        so the shape guard runs on EVERY inference path — earlier the
        guards only existed in predict_proba_dict, leaving
        predict_targets_dict to crash with a cryptic ``IndexError`` at
        ``proba[:, 1]`` if a sub-estimator returned the wrong shape.
        """
        if target_type == TargetType.BINARY:
            proba = sub_est.predict_proba(X)
            if proba.ndim != 2 or proba.shape[1] != 2:
                raise RuntimeError(
                    f"Binary sub-estimator for target {fanned_name!r} "
                    f"returned predict_proba of shape {proba.shape}; "
                    f"expected (*, 2) — the (1 - p1, p1) class-proba "
                    f"stack. A sub-estimator that emits (n,) or "
                    f"(n, 1) needs to be wrapped to match the "
                    f"BaseClassifier.predict_proba contract."
                )
            return proba
        if target_type == TargetType.REGRESSION:
            # BaseRegressor.predict → (n,). Sub-estimators occasionally
            # hand back (n, 1) (e.g. sklearn MultiOutputRegressor wrapping
            # a single target). Reshape so downstream stitching always
            # sees 1-D, matching the joint family's contract.
            values = np.asarray(sub_est.predict(X))
            if values.ndim == 2 and values.shape[1] == 1:
                values = values.reshape(-1)
            if values.ndim != 1:
                raise RuntimeError(
                    f"Regression sub-estimator for target {fanned_name!r} "
                    f"returned predict() of shape {values.shape}; expected "
                    f"1-D (n,) or 2-D (n, 1)."
                )
            return values
        if target_type == TargetType.MULTICLASS:
            # (n, K_k) — assumed in sorted class label order (sklearn default).
            proba = sub_est.predict_proba(X)
            expected_k = len(self._multiclass_classes[fanned_name])
            if proba.ndim != 2 or proba.shape[1] != expected_k:
                # Catches the known XGBClassifierEstimator+inplace_predict
                # multiclass shape bug (returns (n, 2K) instead of (n, K)).
                raise RuntimeError(
                    f"Multiclass sub-estimator for target {fanned_name!r} returned "
                    f"predict_proba of shape {proba.shape}; expected (*, {expected_k}). "
                    f"This usually means the sub-estimator does not support K>2 "
                    f"classes correctly. For XGBClassifierEstimator on multiclass "
                    f"targets, call set_inplace_predict(False) before fit; "
                    f"alternatively use LightGBMClassifierEstimator or "
                    f"SklearnUniversalClassifierEstimator."
                )
            return proba
        raise NotImplementedError(
            f"Unsupported target_type {target_type!r} for target "
            f"{fanned_name!r}. Update IndependentMultiTargetEstimator to "
            f"handle the new TargetType."
        )

    def predict_proba_dict(self, X: pd.DataFrame) -> Dict[str, NDArray]:
        self._check_fitted()
        X = self._align_X_to_feature_names(X)
        out: Dict[str, NDArray] = {}
        for fanned_name in self._fanned_out_targets:
            sub_est = self.estimators[fanned_name]
            target_type = self._target_type_for(fanned_name)
            out[fanned_name] = self._safe_sub_estimator_inference(fanned_name, target_type, sub_est, X)
        return out

    def predict_targets_dict(self, X: pd.DataFrame) -> Dict[str, NDArray]:
        self._check_fitted()
        X = self._align_X_to_feature_names(X)
        out: Dict[str, NDArray] = {}
        for fanned_name in self._fanned_out_targets:
            sub_est = self.estimators[fanned_name]
            target_type = self._target_type_for(fanned_name)
            # Route through the same shape-guarded helper so the BINARY
            # ``proba[:, 1]`` and MULTICLASS ``proba.argmax(axis=1)``
            # below can't be reached with a malformed sub-estimator output.
            proba_or_values = self._safe_sub_estimator_inference(fanned_name, target_type, sub_est, X)
            if target_type == TargetType.BINARY:
                out[fanned_name] = (proba_or_values[:, 1] >= 0.5).astype(np.int64)
            elif target_type == TargetType.REGRESSION:
                out[fanned_name] = proba_or_values
            elif target_type == TargetType.MULTICLASS:
                class_labels = self._multiclass_classes[fanned_name]
                arg = proba_or_values.argmax(axis=1)
                out[fanned_name] = np.array([class_labels[i] for i in arg])
            else:  # pragma: no cover — _safe_sub_estimator_inference already raises.
                raise NotImplementedError(f"Unsupported target_type {target_type!r} for target {fanned_name!r}.")
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _validate_for_fit(
        self,
        X: pd.DataFrame,
        y: Dict[str, NDArray],
        X_valid: Optional[pd.DataFrame],
        y_valid: Optional[Dict[str, NDArray]],
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame; got {type(X).__name__}.")
        if not isinstance(y, dict):
            raise TypeError(f"y must be a dict keyed by target_specs entries; got {type(y).__name__}.")
        if set(y.keys()) != set(self.target_specs.keys()):
            raise ValueError(
                f"y keys must match target_specs keys exactly. "
                f"y={sorted(y.keys())} target_specs={sorted(self.target_specs.keys())}."
            )
        n = X.shape[0]
        for key, spec in self.target_specs.items():
            arr = np.asarray(y[key])
            if isinstance(spec, TargetType):
                if arr.ndim != 1 or arr.shape[0] != n:
                    raise ValueError(f"y[{key!r}] must be 1-D with {n} samples; got shape {arr.shape}.")
            else:
                expected_members = len(spec["columns"])
                if arr.ndim != 2 or arr.shape != (n, expected_members):
                    raise ValueError(
                        f"y[{key!r}] (multilabel group) must be 2-D with shape "
                        f"({n}, {expected_members}); got {arr.shape}."
                    )

        if X_valid is not None and y_valid is None:
            raise ValueError("X_valid provided but y_valid is None.")
        if y_valid is not None and X_valid is None:
            raise ValueError("y_valid provided but X_valid is None.")

    def _extract_y_for_target(self, y: Optional[Dict[str, NDArray]], fanned_name: str) -> NDArray:
        """Map a fanned-out target name to its 1-D y array.

        For simple targets the dict lookup is direct. For multilabel members
        the y dict is keyed by group key with a 2-D array; we slice the
        member's column out.
        """
        if y is None:
            raise ValueError("Cannot extract y from None.")
        for key, spec in self.target_specs.items():
            if isinstance(spec, TargetType):
                if key == fanned_name:
                    return np.asarray(y[key])
            else:
                if fanned_name in spec["columns"]:
                    member_idx = spec["columns"].index(fanned_name)
                    arr = np.asarray(y[key])
                    return arr[:, member_idx]
        raise KeyError(f"Unknown fanned-out target {fanned_name!r}.")

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "IndependentMultiTargetEstimator is not fitted yet. "
                "Call fit(...) first; if a previous fit raised mid-way, the "
                "estimator is left unfitted so partial predictions cannot leak."
            )

    def _align_X_to_feature_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """Match the joint family's predict-time feature_names contract.

        At fit time we snapshot ``X.columns`` into ``self._feature_names``.
        At predict time, callers may pass a frame with the same columns
        in a different order, missing columns, or with extra columns —
        all three would silently propagate into per-sub-estimator predict
        calls and either misalign features or change the column set the
        sub-estimator sees vs. what it was trained on. Mirror
        :meth:`JointMultiTargetBaseEstimator._align_X` so the two
        estimator families publish the same UX.
        """
        # Both call sites (predict_proba_dict, predict_targets_dict)
        # call ``self._check_fitted()`` before this method, so
        # ``_feature_names is None`` is unreachable in practice. Kept as
        # a defensive assertion for future direct callers.
        assert self._feature_names is not None, "Internal: _check_fitted should run first"
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing training-time feature columns: {missing}")
        extra = [c for c in X.columns if c not in self._feature_names]
        if extra:
            raise ValueError(f"X contains feature columns unseen at training: {extra}")
        return X.loc[:, self._feature_names]
