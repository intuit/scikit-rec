from typing import Optional, Union

from numpy.typing import NDArray
from pandas import DataFrame, Series
from xgboost import XGBRegressor

from skrec.estimator.datatypes import HPOType
from skrec.estimator.regression.base_regressor import BaseRegressor
from skrec.estimator.tuned_estimator import TunedEstimator


class XGBRegressorEstimator(BaseRegressor):
    def __init__(self, params: Optional[dict] = None):
        params = params or {}
        params.setdefault("base_score", 0.5)
        self._model = XGBRegressor(**params)

    def _fit_model(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ):
        if X_valid is not None:
            self._model.fit(X, y, eval_set=[(X_valid, y_valid)])
        else:
            self._model.fit(X, y)

    def _predict_model(self, X: DataFrame) -> NDArray:
        # Dataframe is very slow. Convert to numpy array
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        return self._model.predict(X)


class TunedXGBRegressorEstimator(TunedEstimator, XGBRegressorEstimator):
    def __init__(self, hpo_method: HPOType, param_space: dict, optimizer_params: dict, base_score: float = 0.5):
        super().__init__(XGBRegressor, hpo_method, param_space, optimizer_params, {"base_score": base_score})
