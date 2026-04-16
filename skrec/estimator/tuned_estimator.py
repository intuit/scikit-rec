from typing import Optional, Union

from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from skrec.estimator.base_estimator import BaseEstimator
from skrec.estimator.datatypes import HPOType


class TunedEstimator(BaseEstimator):
    """Wraps a sklearn ``GridSearchCV`` or ``RandomizedSearchCV`` as a ``BaseEstimator``.

    Intended to be used as a mixin alongside a concrete estimator subclass::

        class TunedXGBClassifierEstimator(TunedEstimator, XGBClassifierEstimator): ...

    In that pattern the concrete class contributes ``_fit_model`` /
    ``_predict_proba_model`` / ``_predict_model`` via MRO, while this class
    contributes ``__init__`` (which builds the CV object).

    ``TunedEstimator`` can also be used standalone when the wrapped CV object
    is sufficient for inference (no estimator-specific predict optimisations).
    """

    def __init__(
        self,
        estimator_class,
        hpo_method: HPOType,
        param_space: dict,
        optimizer_params: dict,
        estimator_kwargs: Optional[dict] = None,
    ):
        if estimator_class.__name__ == "MultiOutputClassifier":
            base_estimator = param_space.pop("estimator")
            estimator = estimator_class(estimator=base_estimator)
        else:
            estimator = estimator_class(**(estimator_kwargs or {}))
        if hpo_method == HPOType.GRID_SEARCH_CV:
            self._model = GridSearchCV(estimator=estimator, param_grid=param_space, **optimizer_params)

        elif hpo_method == HPOType.RANDOMIZED_SEARCH_CV:
            self._model = RandomizedSearchCV(estimator=estimator, param_distributions=param_space, **optimizer_params)

        else:
            raise NotImplementedError(f"Specified HPO method {hpo_method} not implemented")

    def _fit_model(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ) -> None:
        """Fit the wrapped CV object. Validation data is not forwarded because
        ``GridSearchCV`` / ``RandomizedSearchCV`` perform their own cross-validation."""
        self._model.fit(X, y)

    def predict_proba(self, X: DataFrame) -> NDArray:
        """Return class probabilities from the best estimator found by CV.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            NDArray of shape ``(n_samples, n_classes)``.
        """
        X = self._process_for_predict(X)
        return self._model.predict_proba(X)

    def predict(self, X: DataFrame) -> NDArray:
        """Return predictions from the best estimator found by CV.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            NDArray of shape ``(n_samples,)``.
        """
        X = self._process_for_predict(X)
        return self._model.predict(X)
