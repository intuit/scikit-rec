from typing import Optional, Union

from numpy.typing import NDArray
from pandas import DataFrame, Series
from xgboost import XGBRegressor as _XGBRegressor

from skrec.estimator._fit_params_mixin import SampleWeightStrategy, SklearnFitParamsMixin
from skrec.estimator.datatypes import HPOType
from skrec.estimator.regression.base_regressor import BaseRegressor
from skrec.estimator.tuned_estimator import TunedEstimator


class XGBRegressor(_XGBRegressor):
    """XGBRegressor with a fix for the XGBoost 2.x ``get_params()`` serialization bug.

    See ``skrec.estimator.classification.xgb_classifier.XGBClassifier`` for details.
    """

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                try:
                    params[key] = float(value.strip("[]"))
                except ValueError:
                    pass
        return params


class XGBRegressorEstimator(SklearnFitParamsMixin, BaseRegressor):
    def __init__(
        self,
        params: Optional[dict] = None,
        fit_params: Optional[dict] = None,
        sample_weight: SampleWeightStrategy = None,
    ):
        params = params or {}
        self._model = XGBRegressor(**params)
        self._init_fit_params(fit_params, sample_weight)

    def _fit_model(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ):
        fit_kw = self._resolve_fit_kwargs(X, y, X_valid, y_valid, supports_eval_weight=True)
        if X_valid is not None:
            self._model.fit(X, y, eval_set=[(X_valid, y_valid)], **fit_kw)
        else:
            self._model.fit(X, y, **fit_kw)

    def _predict_model(self, X: DataFrame) -> NDArray:
        # Dataframe is very slow. Convert to numpy array
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        return self._model.predict(X)


class TunedXGBRegressorEstimator(TunedEstimator, XGBRegressorEstimator):
    def __init__(
        self,
        hpo_method: HPOType,
        param_space: dict,
        optimizer_params: dict,
        fit_params: Optional[dict] = None,
        sample_weight: SampleWeightStrategy = None,
    ):
        super().__init__(
            XGBRegressor,
            hpo_method,
            param_space,
            optimizer_params,
            fit_params=fit_params,
            sample_weight=sample_weight,
        )
