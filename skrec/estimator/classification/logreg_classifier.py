from typing import Optional

from sklearn.linear_model import LogisticRegression

from skrec.estimator.classification.sklearn_universal_classifier import (
    SklearnUniversalClassifierEstimator,
    TunedSklearnUniversalClassifierEstimator,
)
from skrec.estimator.datatypes import HPOType


class LogisticRegressionClassifierEstimator(SklearnUniversalClassifierEstimator):
    def __init__(self, params: Optional[dict] = None):
        super().__init__(LogisticRegression, params or {})


class TunedLogisticRegressionClassifierEstimator(TunedSklearnUniversalClassifierEstimator):
    def __init__(self, hpo_method: HPOType, param_space: dict, optimizer_params: dict):
        super().__init__(LogisticRegression, hpo_method, param_space, optimizer_params)
