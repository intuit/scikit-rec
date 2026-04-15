from abc import ABC, abstractmethod

from numpy.typing import NDArray
from pandas import DataFrame

from skrec.estimator.base_estimator import BaseEstimator


class BaseRegressor(BaseEstimator, ABC):
    def predict(self, X: DataFrame) -> NDArray:
        """Return predicted values for each row in ``X``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            NDArray of shape ``(n_samples,)`` with continuous predicted values.
        """
        X = super()._process_for_predict(X)
        return self._predict_model(X)

    @abstractmethod
    def _predict_model(self, X: DataFrame) -> NDArray:
        # Call stored ML model's predict method here
        pass
