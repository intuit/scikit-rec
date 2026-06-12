# Generic fit-time parameter passthrough for sklearn-API estimators.
#
# Why this exists:
#   The sklearn-wrapper estimators (XGBClassifierEstimator, XGBRegressorEstimator,
#   SklearnUniversalClassifier/Regressor, LightGBM*) historically called
#   ``self._model.fit(X, y[, eval_set])`` with no channel for fit-time arguments
#   such as ``sample_weight``, ``feature_weights``, ``base_margin``, a custom
#   objective, or ``callbacks``. Production model migrations (e.g. firmographics
#   industry, which bakes ``compute_sample_weight('balanced')`` into its DMatrix)
#   could not be reproduced through scikit-rec without a dedicated, hardcoded
#   estimator. This mixin adds a single, general mechanism instead of a
#   per-argument special case.
#
# Design:
#   - Instance-stored (constructor / setters), NOT a new ``fit`` positional —
#     the ``fit(X, y, X_valid, y_valid)`` signature is called by every scorer,
#     by BatchTrainingDataset, and by TunedEstimator, so widening it would churn
#     every call site. The estimator carries its fit-time config and resolves it
#     inside ``_fit_model``.
#   - ``sample_weight`` is row-aligned, so it is resolved against ``y`` at fit
#     time (where rows are guaranteed aligned — BaseScorer preserves row order
#     end-to-end). The strategy is pluggable: None (uniform, the default),
#     'balanced' (derive from y via sklearn), a callable fn(y)->weights, or an
#     explicit array. "A constant default, but never an always-constant" — the
#     library ships uniform-by-default and lets the caller plug any scheme.
#   - Static, non-row-aligned kwargs go through ``fit_params`` verbatim.
#
# Composition note: ``BaseEstimator.__init__`` is abstract/no-op and subclasses
# set ``self._model`` without calling ``super().__init__()``. So estimators call
# ``self._init_fit_params(...)`` explicitly in their own ``__init__`` rather than
# relying on cooperative super() chaining.

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

# A sample-weight strategy: None (uniform), the string 'balanced', a callable
# mapping y -> per-row weights, or an explicit per-row array/list.
SampleWeightStrategy = Union[None, str, Callable[[Any], NDArray], NDArray, list]


def _n_rows(y: Any) -> int:
    """Row count of a target container (Series, ndarray, DataFrame, or list)."""
    return y.shape[0] if hasattr(y, "shape") else len(y)


class SklearnFitParamsMixin:
    """Adds instance-stored fit-time kwargs and a pluggable sample-weight
    strategy to a sklearn-API estimator.

    Mix in as the **first** base of an estimator, call
    :meth:`_init_fit_params` in ``__init__``, and splat
    :meth:`_resolve_fit_kwargs` into the wrapped model's ``fit`` inside
    ``_fit_model``.
    """

    def _init_fit_params(
        self,
        fit_params: Optional[Dict[str, Any]] = None,
        sample_weight: SampleWeightStrategy = None,
    ) -> None:
        """Store the fit-time config. Call once from the estimator ``__init__``.

        Args:
            fit_params: Static kwargs forwarded verbatim to the wrapped model's
                ``fit`` (e.g. ``feature_weights``, ``base_margin``, ``callbacks``,
                a custom ``objective``). Not row-aligned.
            sample_weight: Row-weight strategy — ``None`` (uniform; default),
                ``'balanced'`` (``sklearn.utils.class_weight.compute_sample_weight``
                on ``y`` at fit time), a callable ``fn(y) -> weights``, or an
                explicit per-row array.

        Raises:
            ValueError: if ``sample_weight`` is given both as this argument and
                inside ``fit_params`` (ambiguous).
        """
        self._fit_params: Dict[str, Any] = dict(fit_params or {})
        self._sample_weight: SampleWeightStrategy = sample_weight
        if "sample_weight" in self._fit_params and sample_weight is not None:
            raise ValueError(
                "Pass row weights via the `sample_weight` argument OR as "
                "fit_params['sample_weight'], not both. The `sample_weight` argument "
                "additionally supports 'balanced' / a callable; fit_params['sample_weight'] "
                "is treated as a static array."
            )

    def set_fit_params(self, **kwargs: Any) -> None:
        """Update the static fit kwargs after construction (merges, last wins)."""
        # _init_fit_params may not have run if a subclass forgot to call it;
        # be defensive so the setter never AttributeErrors.
        if not hasattr(self, "_fit_params"):
            self._fit_params = {}
        self._fit_params.update(kwargs)

    def set_sample_weight(self, strategy: SampleWeightStrategy) -> None:
        """Replace the sample-weight strategy after construction."""
        self._sample_weight = strategy

    def _resolve_sample_weight(self, y: Any) -> Optional[NDArray]:
        """Resolve the configured strategy into a concrete per-row array (or None).

        Resolution happens against the actual ``y`` passed to ``fit``, so the
        weights are aligned to the training rows.
        """
        sw = getattr(self, "_sample_weight", None)
        if sw is None:
            return None
        if isinstance(sw, str):
            if sw != "balanced":
                raise ValueError(
                    f"Unknown sample_weight strategy {sw!r}. Use 'balanced', a callable "
                    f"fn(y)->weights, an explicit array, or None (uniform)."
                )
            from sklearn.utils.class_weight import compute_sample_weight

            # compute_sample_weight handles 1-D (binary/multiclass) and 2-D
            # (multilabel) y. For 2-D it balances per column and multiplies.
            return compute_sample_weight("balanced", y)
        if callable(sw):
            w = np.asarray(sw(y))
        else:
            w = np.asarray(sw)
        if w.shape[0] != _n_rows(y):
            raise ValueError(f"sample_weight length {w.shape[0]} != n_samples {_n_rows(y)}.")
        return w

    def _resolve_fit_kwargs(
        self,
        X: Any,
        y: Any,
        X_valid: Any = None,
        y_valid: Any = None,
        supports_eval_weight: bool = False,
    ) -> Dict[str, Any]:
        """Build the kwargs dict to splat into the wrapped model's ``fit``.

        Merges the static ``fit_params`` with a resolved ``sample_weight`` (when
        a strategy is configured). When ``supports_eval_weight`` is True and a
        validation set is present, ``sample_weight_eval_set`` is derived **only for
        re-derivable strategies** — the string ``'balanced'`` or a callable
        ``fn(y) -> weights`` — which can be recomputed against ``y_valid``. An
        **explicit array** is train-specific (its length is the train row count) and
        cannot be applied to a differently-sized validation set, so eval weighting is
        skipped for it rather than crash re-validating a train-sized array against the
        eval rows. (Pass eval weights for an explicit-array case via
        ``fit_params['sample_weight_eval_set']`` if you need them.)

        Semantic note: for ``'balanced'``, the eval weights are computed from the
        **validation set's own** class distribution — so the early-stopping objective
        is balanced by the validation split's ratios, not the train ratios. This is a
        reasonable default (each split self-balances) but is a deliberate choice.

        Callers whose backend uses a different eval-weight kwarg (LightGBM's
        ``eval_sample_weight``) should pass ``supports_eval_weight=False`` and handle
        eval weighting themselves.

        If the wrapped model's ``fit`` does not accept ``sample_weight``, the
        resulting error propagates from the backend — surfaced, not swallowed.
        """
        kw = dict(getattr(self, "_fit_params", {}))
        w = self._resolve_sample_weight(y)
        if w is not None:
            kw["sample_weight"] = w
            # Only re-derive eval weights for strategies that map y -> weights per set
            # ('balanced' / callable). An explicit array is train-sized and would raise
            # the length-mismatch guard if re-resolved against y_valid — skip it.
            strategy = getattr(self, "_sample_weight", None)
            is_rederivable = isinstance(strategy, str) or callable(strategy)
            if (
                supports_eval_weight
                and is_rederivable
                and X_valid is not None
                and y_valid is not None
                and "sample_weight_eval_set" not in kw
            ):
                kw["sample_weight_eval_set"] = [self._resolve_sample_weight(y_valid)]
        return kw
