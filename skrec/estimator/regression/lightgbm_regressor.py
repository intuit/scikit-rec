from typing import Dict, Optional

from lightgbm import LGBMRegressor
from pandas import DataFrame

from skrec.estimator.datatypes import HPOType
from skrec.estimator.regression.sklearn_universal_regressor import (
    SklearnUniversalRegressorEstimator,
    TunedSklearnUniversalRegressorEstimator,
)


class LightGBMRegressorEstimator(SklearnUniversalRegressorEstimator):
    def __init__(self, params: Optional[dict] = None, train_params: Optional[dict] = None):
        params = params or {}
        self._model = LGBMRegressor(**params)
        self._train_params: Dict = train_params or {}

    def _fit_model(
        self, X: DataFrame, y: DataFrame, X_valid: Optional[DataFrame] = None, y_valid: Optional[DataFrame] = None
    ):
        if X_valid is not None and y_valid is not None:
            self._model.fit(X, y, eval_set=[(X_valid, y_valid)], **self._train_params)
        else:
            self._model.fit(X, y, **self._train_params)

    def set_training_params(self, train_params: dict):
        self._train_params = train_params


class TunedLightGBMRegressorEstimator(TunedSklearnUniversalRegressorEstimator, LightGBMRegressorEstimator):
    def __init__(self, hpo_method: HPOType, param_space: dict, optimizer_params: dict):
        self._train_params: Dict = {}
        super().__init__(LGBMRegressor, hpo_method, param_space, optimizer_params)
