from typing import Optional, Union

from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.base import ClassifierMixin

from skrec.estimator._fit_params_mixin import SampleWeightStrategy, SklearnFitParamsMixin
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.datatypes import HPOType
from skrec.estimator.tuned_estimator import TunedEstimator


class SklearnUniversalClassifierEstimator(SklearnFitParamsMixin, BaseClassifier):
    def __init__(
        self,
        model: ClassifierMixin,
        params: Optional[dict] = None,
        fit_params: Optional[dict] = None,
        sample_weight: SampleWeightStrategy = None,
    ):
        self._model = model(**params)
        self._init_fit_params(fit_params, sample_weight)

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

        # No eval-set for generic sklearn estimators → train-set weighting only.
        self._model.fit(X, y, **self._resolve_fit_kwargs(X, y))

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
        fit_params: Optional[dict] = None,
        sample_weight: SampleWeightStrategy = None,
    ):
        super().__init__(
            model,
            hpo_method,
            param_space,
            optimizer_params,
            fit_params=fit_params,
            sample_weight=sample_weight,
        )
