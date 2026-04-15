from abc import ABC, abstractmethod

from numpy.typing import NDArray
from pandas import DataFrame

from skrec.estimator.base_estimator import BaseEstimator


class BaseClassifier(BaseEstimator, ABC):
    def predict_proba(self, X: DataFrame) -> NDArray:
        """Return class probabilities for each row in ``X``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            NDArray of shape ``(n_samples, n_classes)`` where column ``1``
            contains the positive-class probability.
        """
        X = super()._process_for_predict(X)
        return self._predict_proba_model(X)

    @abstractmethod
    def _predict_proba_model(self, X: DataFrame) -> NDArray:
        # Call stored ML model's predict_proba method here
        pass
