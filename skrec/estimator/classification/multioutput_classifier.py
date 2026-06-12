from typing import Mapping, Optional, Type

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import ClassifierMixin
from sklearn.multioutput import MultiOutputClassifier

from skrec.estimator._fit_params_mixin import SampleWeightStrategy
from skrec.estimator.classification.sklearn_universal_classifier import (
    SklearnUniversalClassifierEstimator,
    TunedSklearnUniversalClassifierEstimator,
)
from skrec.estimator.datatypes import HPOType


class MultiOutputClassifierEstimator(SklearnUniversalClassifierEstimator):
    """Wrapper for ``sklearn.multioutput.MultiOutputClassifier`` — **binary
    targets only** by contract.

    Pairs with :class:`~skrec.scorer.multioutput.MultioutputScorer` in
    classifier mode. Every target column is expected to have exactly two
    unique values; columns with 3+ classes (multi-class) are rejected at
    fit time with a precise error pointing at migration paths
    (``MulticlassScorer``, one-hot encoding, or the planned mixed-type
    multi-target scorer).

    The contract is enforced both at the scorer level (in
    ``MultioutputScorer._validate_targets``, before this estimator's
    ``fit`` is called) and again here at the wrapper level via a cheap
    pre-flight ``y`` scan. The double check makes the wrapper safe to use
    standalone (i.e. with raw ``y`` outside ``MultioutputScorer``) and
    means the expensive sklearn ``MultiOutputClassifier.fit`` never runs
    on data the eval pipeline can't make sense of.
    """

    def __init__(
        self,
        base_estimator: Type[ClassifierMixin],
        params: Mapping,
        fit_params: Optional[dict] = None,
        sample_weight: SampleWeightStrategy = None,
    ):
        model = base_estimator(**params)
        multioutput_params = {"estimator": model}

        # sklearn's MultiOutputClassifier.fit forwards sample_weight to every
        # per-label sub-estimator, so the generic passthrough works here too.
        super().__init__(MultiOutputClassifier, multioutput_params, fit_params=fit_params, sample_weight=sample_weight)

    @staticmethod
    def _validate_binary_targets(y: object) -> None:
        """Pre-flight check: every target column has exactly 2 unique values
        and contains no nulls.

        Runs before the underlying sklearn ``MultiOutputClassifier.fit`` so
        we fail in O(n_rows × n_targets) instead of after a full fit.

        Rejects three distinct cases with separate, actionable error messages:

        - **NaN / null values** — sklearn ``MultiOutputClassifier.fit`` with
          NaN-in-y produces an opaque internal error; the wrapper catches it
          loudly with a "drop or backfill" hint. Aligned with the scorer's
          ``_validate_interactions`` null check (which fires earlier when the
          wrapper is used through ``MultioutputScorer``).
        - **Multi-class (3+ classes per target)** — never supported in
          binary-only mode. Migration paths point at ``MulticlassScorer``,
          one-hot encoding, or the future mixed-type scorer.
        - **Single-class (1 class per target)** — when used through
          ``MultioutputScorer``, this is handled upstream by
          ``DegenerateTargetPolicy`` and never reaches this wrapper. When
          used standalone (without the scorer), there is no degenerate-policy
          machinery, so we reject explicitly with a pointer to the scorer's
          policy rather than letting sklearn fail inconsistently across
          underlying classifiers.

        When ``y`` is a DataFrame the column names are surfaced in the
        error so the operator can identify which target to fix without
        counting columns. ``pd.isna`` is used for null detection so pandas
        nullable extension dtypes (``Int64``, ``boolean``) are handled the
        same way as numpy float NaN.
        """
        # Capture column names first if available — np.asarray strips them.
        column_names: Optional[list]
        if hasattr(y, "columns"):
            column_names = list(y.columns)
        else:
            column_names = None
        arr = np.asarray(y.values if hasattr(y, "values") else y)
        if arr.ndim != 2:
            raise ValueError(
                f"MultiOutputClassifierEstimator expects 2-D y of shape (n_samples, n_targets), got ndim={arr.ndim}."
            )

        def _label(idx: int) -> str:
            return f"{column_names[idx]!r}" if column_names else f"column index {idx}"

        # First pass: reject NaN / null values explicitly. NaN-in-y produces
        # an opaque sklearn internal error if it reaches MultiOutputClassifier.fit;
        # we catch it here with a clear "drop or backfill" message.
        bad_null: list = []
        for col_idx in range(arr.shape[1]):
            col = arr[:, col_idx]
            null_count = int(pd.isna(col).sum())
            if null_count > 0:
                bad_null.append((col_idx, null_count))
        if bad_null:
            details = "; ".join(f"{_label(idx)} ({count} null value(s))" for idx, count in bad_null)
            raise ValueError(
                f"MultiOutputClassifierEstimator received target column(s) with null/NaN "
                f"values: {details}. Classification targets must be observed for every "
                f"row — drop the affected rows or backfill before fit. NaN as ignore-mask "
                f"isn't supported here (sklearn MultiOutputClassifier doesn't handle it). "
                f"If you have genuinely missing labels for some users, drop those rows "
                f"from the training frame."
            )

        # Second pass: reject any column with values outside the binary
        # numeric set {0, 1}. Mirrors MultioutputScorer._validate_targets
        # so the wrapper's standalone-use contract matches the scorer's
        # contract — without this, a y of `[["yes","no"], ...]` would pass
        # the cardinality check (length 2) and fit silently via sklearn's
        # internal label encoding, even though MultioutputScorer would
        # reject the same input.
        _BINARY = {0, 1, 0.0, 1.0}
        bad_non_numeric: list = []
        for col_idx in range(arr.shape[1]):
            col = arr[:, col_idx]
            unique = np.unique(col)
            extra = set(unique.tolist()) - _BINARY
            if extra:
                bad_non_numeric.append((col_idx, sorted(extra, key=str)))
        if bad_non_numeric:
            details = "; ".join(f"{_label(idx)} (saw values: {vals})" for idx, vals in bad_non_numeric)
            raise ValueError(
                f"MultiOutputClassifierEstimator requires every target column to be "
                f"binary numeric — values strictly in {{0, 1}} (or {{0.0, 1.0}}). "
                f"Non-binary values seen in: {details}. Pre-encode at the caller, "
                f"e.g.: y[col] = (y[col] == 'yes').astype(float). The wrapper-side "
                f"contract intentionally matches MultioutputScorer's so standalone "
                f"and scorer-mediated use behave identically."
            )

        bad_multi: list = []
        bad_single: list = []
        for col_idx in range(arr.shape[1]):
            col = arr[:, col_idx]
            unique = np.unique(col)
            if len(unique) > 2:
                bad_multi.append((col_idx, unique.tolist()))
            elif len(unique) < 2:
                bad_single.append((col_idx, unique.tolist()))
        if bad_multi:
            details = "; ".join(f"{_label(idx)} (classes: {classes})" for idx, classes in bad_multi)
            raise ValueError(
                f"MultiOutputClassifierEstimator requires every target column to be "
                f"binary (exactly 2 classes), but the following column(s) have 3+ "
                f"classes: {details}. This estimator pairs only with binary multi-label "
                f"workflows. For multi-class targets, see migration paths in "
                f"MultioutputScorer's documentation: (1) MulticlassScorer in long format "
                f"for a single multi-class target; (2) one-hot encode multi-class targets "
                f"into binary columns; (3) wait for the planned mixed-type multi-target "
                f"scorer."
            )
        if bad_single:
            details = "; ".join(f"{_label(idx)} (only value: {classes})" for idx, classes in bad_single)
            raise ValueError(
                f"MultiOutputClassifierEstimator received single-class target column(s): "
                f"{details}. Standalone use requires every target to be binary; sklearn's "
                f"MultiOutputClassifier doesn't fit single-class y consistently across "
                f"underlying classifiers. If you're hitting this when training through "
                f"MultioutputScorer, the scorer's DegenerateTargetPolicy normally handles "
                f"this case before the wrapper sees it (DegenerateTargetPolicy.CONSTANT "
                f"to fall back to a constant predictor; DegenerateTargetPolicy.RAISE — "
                f"the default — to fail loudly with column names). Standalone callers "
                f"should drop single-class columns from y before fit."
            )

    def _fit_model(
        self,
        X: DataFrame,
        y: DataFrame,
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[DataFrame] = None,
    ) -> None:
        self._validate_binary_targets(y)
        super()._fit_model(X, y, X_valid, y_valid)


class TunedMultiOutputClassifierEstimator(TunedSklearnUniversalClassifierEstimator):
    def __init__(
        self,
        base_estimator: Type[ClassifierMixin],
        hpo_method: HPOType,
        param_space: dict,
        optimizer_params: dict,
    ):
        model = base_estimator()
        param_space["estimator"] = model
        # Note: in order to parameterize the underlying estimator,
        # we need to use estimator__param_name
        # https://stackoverflow.com/questions/69962287/how-to-use-multioutputclassifier-with-randomizedsearchcv-for-hyperparameter
        updated_param_space = {f"estimator__{k}" if k in model.get_params() else k: v for k, v in param_space.items()}

        super().__init__(MultiOutputClassifier, hpo_method, updated_param_space, optimizer_params)
