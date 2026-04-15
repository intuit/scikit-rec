from typing import Mapping

from sklearn.base import ClassifierMixin
from sklearn.multioutput import MultiOutputClassifier

from skrec.estimator.classification.sklearn_universal_classifier import (
    SklearnUniversalClassifierEstimator,
    TunedSklearnUniversalClassifierEstimator,
)
from skrec.estimator.datatypes import HPOType


class MultiOutputClassifierEstimator(SklearnUniversalClassifierEstimator):
    def __init__(self, base_estimator: ClassifierMixin, params: Mapping):
        model = base_estimator(**params)
        multioutput_params = {"estimator": model}

        super().__init__(MultiOutputClassifier, multioutput_params)


class TunedMultiOutputClassifierEstimator(TunedSklearnUniversalClassifierEstimator):
    def __init__(
        self,
        base_estimator: ClassifierMixin,
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
