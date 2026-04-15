from typing import Optional, Union

from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.base import ClassifierMixin

from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.datatypes import HPOType
from skrec.estimator.tuned_estimator import TunedEstimator


class SklearnUniversalClassifierEstimator(BaseClassifier):
    def __init__(self, model: ClassifierMixin, params: Optional[dict] = None):
        self._model = model(**params)

    def _fit_model(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ):
        if X_valid is not None:
            import warnings

            warnings.warn(
                f"{self.__class__.__name__} does not support early stopping. "
                "Validation data (X_valid, y_valid) will be ignored.",
                stacklevel=2,
            )

        self._model.fit(X, y)

    def _predict_proba_model(self, X: Union[DataFrame, NDArray]) -> NDArray:
        # Dataframe is very slow. Convert to numpy array for sklearn predict_proba if needed
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        return self._model.predict_proba(X)


class TunedSklearnUniversalClassifierEstimator(TunedEstimator, SklearnUniversalClassifierEstimator):
    def __init__(
        self,
        model: ClassifierMixin,
        hpo_method: HPOType,
        param_space: dict,
        optimizer_params: dict,
    ):
        super().__init__(model, hpo_method, param_space, optimizer_params)
