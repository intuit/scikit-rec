# skrec/estimator/embedding/base_embedding_estimator.py
from abc import abstractmethod
from typing import Optional, Union

from numpy.typing import NDArray
from pandas import DataFrame, Series

from skrec.estimator.base_estimator import BaseEstimator


class BaseEmbeddingEstimator(BaseEstimator):
    @abstractmethod
    def fit_embedding_model(
        self,
        users: Optional[DataFrame],
        items: Optional[DataFrame],
        interactions: DataFrame,
        valid_users: Optional[DataFrame] = None,
        valid_interactions: Optional[DataFrame] = None,
    ) -> None:
        """
        Fit the model using factorized user, item, and interaction data.
        This method must be implemented by concrete subclasses.
        """
        pass

    @abstractmethod
    def predict_proba_with_embeddings(
        self,
        interactions: DataFrame,
        users: Optional[DataFrame] = None,
    ) -> NDArray:
        """
        Predicts probabilities for given interactions. Operates in two modes:

        1. Real-time Inference Mode (users DataFrame provided):
           If `users` DataFrame is provided, it MUST contain `USER_ID_NAME` and
           `USER_EMBEDDING_NAME` columns. Pre-computed user embeddings from this
           DataFrame are used. Optionally, if user features are also present in this
           `users` DataFrame and the model was trained with user features, these
           will be used. This mode is for scenarios where user embeddings are
           managed externally (e.g., an embedding store).

        2. Batch Prediction Mode (users is None):
           If `users` is `None`, the model uses its internally learned user embeddings
           and stored user features derived during training. This is the
           typical mode for batch predictions or when not using an external embedding store.

        Args:
            interactions: DataFrame containing interaction data. Must include
                          `USER_ID_NAME`, `ITEM_ID_NAME`, and any context features
                          the model was trained on.
            users: Optional DataFrame.
                   - If provided (Real-time Mode): Must contain `USER_ID_NAME` and
                     `USER_EMBEDDING_NAME` (NumPy arrays). Can also contain user
                     feature columns if the model uses them.
                   - If `None` (Batch Mode): The model uses its internal, learned
                     user embeddings and features.

        Returns:
            NDArray: Predicted probabilities.
        """
        pass

    @abstractmethod
    def get_item_embeddings(self) -> DataFrame:
        """
        Return a DataFrame with columns [ITEM_ID_NAME, ITEM_EMBEDDING_NAME].
        One row per item seen during training (excluding unknown placeholder).
        Raises RuntimeError if called before fit_embedding_model().
        """
        pass

    @abstractmethod
    def get_user_embeddings(self) -> DataFrame:
        """
        Return a DataFrame with columns [USER_ID_NAME, USER_EMBEDDING_NAME].
        One row per user seen during training (excluding unknown placeholder).
        Raises RuntimeError if called before fit_embedding_model().
        """
        pass

    def _fit_model(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ):
        raise NotImplementedError(
            f"{self.__class__.__name__} is a BaseEmbeddingEstimator and requires factorized inputs "
            "via the 'fit_embedding_model' method. It cannot be trained with a single pre-joined X, y matrix."
        )
