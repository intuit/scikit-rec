from typing import Mapping, Optional, Type

import numpy as np
from pandas import DataFrame
from sklearn.base import RegressorMixin
from sklearn.multioutput import MultiOutputRegressor

from skrec.estimator._fit_params_mixin import SampleWeightStrategy
from skrec.estimator.datatypes import HPOType
from skrec.estimator.regression.sklearn_universal_regressor import (
    SklearnUniversalRegressorEstimator,
    TunedSklearnUniversalRegressorEstimator,
)


class MultiOutputRegressorEstimator(SklearnUniversalRegressorEstimator):
    """Wrapper for ``sklearn.multioutput.MultiOutputRegressor`` — multi-target
    continuous y.

    Pairs with :class:`~skrec.scorer.multioutput.MultioutputScorer` in
    regressor mode. Asymmetric with
    :class:`~skrec.estimator.classification.multioutput_classifier.MultiOutputClassifierEstimator`:
    the classifier has a strict binary ``{0, 1}`` value contract, the
    regressor has none — any continuous values are valid. The pre-flight
    check here only enforces the **shape** (must be 2-D so the underlying
    sklearn ``MultiOutputRegressor`` sees one estimator per column);
    NaN handling is delegated to the underlying regressor (XGBRegressor
    handles NaN, RandomForestRegressor doesn't — caller's responsibility).
    """

    def __init__(
        self,
        base_estimator: Type[RegressorMixin],
        params: Mapping,
        fit_params: Optional[dict] = None,
        sample_weight: SampleWeightStrategy = None,
    ):
        model = base_estimator(**params)
        multioutput_params = {"estimator": model}

        # sklearn's MultiOutputRegressor.fit forwards sample_weight to every
        # per-target sub-estimator, so the generic passthrough works here too.
        super().__init__(MultiOutputRegressor, multioutput_params, fit_params=fit_params, sample_weight=sample_weight)

    @staticmethod
    def _validate_continuous_targets(y: object) -> None:
        """Pre-flight shape check: y is 2-D.

        No value-range or dtype validation — regressor accepts any continuous
        input. Use 2-D rather than 1-D so the underlying sklearn
        ``MultiOutputRegressor`` fits one estimator per column.
        """
        arr = np.asarray(y.values if hasattr(y, "values") else y)
        if arr.ndim != 2:
            raise ValueError(
                f"MultiOutputRegressorEstimator expects 2-D y of shape "
                f"(n_samples, n_targets), got ndim={arr.ndim}. For single-target "
                f"regression use XGBRegressorEstimator (or any single-target "
                f"BaseRegressor) directly."
            )

    def _fit_model(
        self,
        X: DataFrame,
        y: DataFrame,
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[DataFrame] = None,
    ) -> None:
        self._validate_continuous_targets(y)
        super()._fit_model(X, y, X_valid, y_valid)


class TunedMultiOutputRegressorEstimator(TunedSklearnUniversalRegressorEstimator):
    def __init__(
        self,
        base_estimator: Type[RegressorMixin],
        hpo_method: HPOType,
        param_space: dict,
        optimizer_params: dict,
    ):
        model = base_estimator()
        param_space["estimator"] = model
        updated_param_space = {f"estimator__{k}" if k in model.get_params() else k: v for k, v in param_space.items()}

        super().__init__(MultiOutputRegressor, hpo_method, updated_param_space, optimizer_params)
