from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from skrec.util.logger import get_logger

logger = get_logger(__name__)


class BaseEstimator(ABC):
    """Abstract base class for tabular estimators (classifiers and regressors).

    This class defines the contract for estimators that operate on a pre-joined
    feature matrix ``X`` of shape ``(n_samples, n_features)``.

    **Guaranteed interface by subtype:**

    - ``BaseClassifier`` subclasses: ``fit(X, y)`` + ``predict_proba(X)``
    - ``BaseRegressor`` subclasses: ``fit(X, y)`` + ``predict(X)``

    For embedding-based and sequential models (NCF, SASRec, HRNN, etc.) see
    ``BaseEmbeddingEstimator`` and ``SequentialEstimator``, which use a separate
    interface (``fit_embedding_model`` / ``predict_proba_with_embeddings``) that
    operates on factorised user/item/interaction DataFrames rather than a
    pre-joined feature matrix.
    """

    # This contains the smallest ML model, e.g. sklearn estimator, xgboost classifier
    feature_names: Optional[List] = None

    @abstractmethod
    def __init__(self) -> None:
        # Store an ML model as an attribute here
        pass

    def fit(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ) -> None:
        """Validate inputs and fit the underlying ML model.

        Stores ``X.columns`` as ``feature_names`` so that ``predict`` /
        ``predict_proba`` can reorder incoming DataFrames to match the training
        column order.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target labels or values of shape ``(n_samples,)``.
            X_valid: Optional validation feature matrix.  Must have the same
                columns as ``X``.
            y_valid: Optional validation targets aligned with ``X_valid``.
        """
        self._validate_for_fit(X, y, X_valid, y_valid)
        self.feature_names = X.columns.tolist()
        self._fit_model(X, y, X_valid, y_valid)

    @abstractmethod
    def _fit_model(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ):
        # Call stored ML model's fit method here
        pass

    def _validate_for_fit(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("Ill formatted x and y: Rows Mismatch")

        self.X_has_only_1_feature = X.shape[1] == 1

        if X_valid is not None:
            if X_valid.shape[1] != X.shape[1]:
                raise ValueError(
                    "Training and validation data have different number of features: "
                    f"{X.shape[1]} != {X_valid.shape[1]}"
                )
            if y_valid is not None and X_valid.shape[0] != y_valid.shape[0]:
                raise ValueError("Ill formatted validation data: Rows Mismatch")

            if X.columns.tolist() != X_valid.columns.tolist():
                raise ValueError("Training and validation data have different column names")

    def _process_for_predict(self, X: DataFrame) -> DataFrame:
        if isinstance(X, np.ndarray):
            raise TypeError(
                "predict/predict_proba requires a DataFrame, not a numpy array. "
                "Wrap with: pd.DataFrame(X, columns=estimator.feature_names)"
            )

        if self.support_batch_training():
            logger.warning(
                "Predict validation, including column name checking, is not implemented for batch training yet"
            )
            return X

        if self.feature_names is None:
            raise AttributeError("Estimator did not store column names during training")

        return X[self.feature_names]

    def estimator_attributes_are_equal(
        self, estimator_to_compare: "BaseEstimator", attributes_to_compare: List
    ) -> List[bool]:
        """Compare specific model attributes between two estimators.

        Useful for verifying that two estimators trained on the same data
        learned the same parameters (e.g. detecting data leakage or confirming
        a reproducible seed).

        Args:
            estimator_to_compare: Another ``BaseEstimator`` instance to compare
                against.
            attributes_to_compare: List of attribute names to compare on the
                underlying ``_model`` objects (e.g. ``["coef_", "intercept_"]``).

        Returns:
            List of booleans, one per attribute — ``True`` if the values are
            equal, ``False`` otherwise.

        Raises:
            TypeError: If ``estimator_to_compare`` is not a ``BaseEstimator``.
            AttributeError: If either estimator's ``_model`` lacks an attribute.
        """
        if not isinstance(estimator_to_compare, BaseEstimator):
            raise TypeError("The estimator to compare is not a BaseEstimator.")

        attr_equal = []
        for attribute in attributes_to_compare:
            if not hasattr(self._model, attribute) or not hasattr(estimator_to_compare._model, attribute):
                raise AttributeError(f"Estimator does not have the attribute {attribute}")

            attr_base = getattr(self._model, attribute)
            attr_to_compare = getattr(estimator_to_compare._model, attribute)
            result = (
                np.array_equal(attr_base, attr_to_compare)
                if isinstance(attr_base, np.ndarray)
                else attr_base == attr_to_compare
            )
            attr_equal.append(result)
            if result:
                logger.warning(
                    f"Values of the {attribute} are equal for these estimators. "
                    "This might be caused by insufficient amount of data (attributes equal to the initial values), "
                    "models trained on the same dataset "
                    "or the same model retrained on multiple datasets."
                )
            else:
                logger.info(
                    f"Values of the {attribute} are different for these estimators. "
                    "Finished comparing these estimators."
                )
        return attr_equal

    def support_batch_training(self) -> bool:
        """Return ``True`` if this estimator supports batch (partitioned) training.

        Batch training is required when the full dataset does not fit in memory.
        An estimator opts in by implementing the ``_batch_fit_model`` method.
        """
        return hasattr(self, "_batch_fit_model")
