from typing import Dict, Optional

from lightgbm import LGBMClassifier
from pandas import DataFrame

from skrec.estimator.classification.sklearn_universal_classifier import (
    SklearnUniversalClassifierEstimator,
    TunedSklearnUniversalClassifierEstimator,
)
from skrec.estimator.datatypes import HPOType


class LightGBMClassifierEstimator(SklearnUniversalClassifierEstimator):
    def __init__(self, params: Optional[dict] = None, train_params: Optional[dict] = None):
        params = params or {}
        self._model = LGBMClassifier(**params)
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


class TunedLightGBMClassifierEstimator(TunedSklearnUniversalClassifierEstimator, LightGBMClassifierEstimator):
    def __init__(self, hpo_method: HPOType, param_space: dict, optimizer_params: dict):
        self._train_params: Dict = {}
        super().__init__(LGBMClassifier, hpo_method, param_space, optimizer_params)
