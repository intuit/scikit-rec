from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series

from skrec.constants import (
    DEBUG_COLUMNS,
    ITEM_ID_NAME,
    LABEL_NAME,
    TIMESTAMP_COL,
    USER_ID_NAME,
)
from skrec.dataset.batch_training_dataset import BatchTrainingDataset
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.base_estimator import BaseEstimator
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.util.logger import get_logger

if TYPE_CHECKING:
    from skrec.estimator.sequential.base_sequential_estimator import SequentialEstimator

logger = get_logger(__name__)


class BaseScorer(ABC):
    """Abstract base class for all scorers in the recommender pipeline.

    A scorer sits between the recommender (business logic) and the estimator
    (ML model).  It owns data preparation (joining users/items/interactions),
    item catalogue management, and the scoring loop that produces an
    ``(n_users, n_items)`` score matrix from the underlying estimator.
    """

    target_col: Optional[str] = LABEL_NAME
    item_names: Optional[NDArray[np.str_]] = None
    items_df: Optional[pd.DataFrame] = None
    item_subset: Optional[List[str]] = None
    item_subset_df: Optional[pd.DataFrame] = None

    def __init__(self, estimator: Union[BaseEstimator, SequentialEstimator]) -> None:
        self.estimator = estimator

    def set_item_subset(self, item_subset: List[str]) -> None:
        """Restrict scoring and recommendations to a subset of the item catalogue.

        After calling this, ``score_items`` and ``recommend`` will only consider
        items in ``item_subset``.  Call ``clear_item_subset`` to restore
        full-catalogue scoring.

        Args:
            item_subset: List of item IDs to include.  Must be non-empty,
                contain no duplicates, and consist only of items seen during
                training.

        Raises:
            ValueError: If ``item_subset`` is empty, contains duplicates, or
                includes items not present in the training catalogue.
        """
        self.item_subset = self._process_item_subset(item_subset)
        if self.item_subset is None:
            raise RuntimeError("_process_item_subset returned None unexpectedly.")
        if self.items_df is not None:
            self.item_subset_df = self.items_df.set_index(ITEM_ID_NAME).loc[self.item_subset, :]

    def clear_item_subset(self) -> None:
        """Remove any active item subset and restore full-catalogue scoring."""
        self.item_subset = None
        self.item_subset_df = None

    def process_datasets(
        self,
        users_df: Optional[DataFrame] = None,
        items_df: Optional[DataFrame] = None,
        interactions_df: Optional[DataFrame] = None,
        is_training: Optional[bool] = True,
    ) -> Tuple[DataFrame, Series]:  # X, y
        """Validate, join, and encode datasets into a single (X, y) training pair.

        On the first call (``is_training=True``) sets ``self.item_names`` and
        ``self.items_df`` from ``items_df`` (or derives item names from
        ``interactions_df`` when no items dataset is provided).  On subsequent
        calls (``is_training=False``) reuses the item catalogue established
        during training.

        Args:
            users_df: Optional user features DataFrame.  Must include
                ``USER_ID_NAME``.
            items_df: Optional item features DataFrame.  Must include
                ``ITEM_ID_NAME``.
            interactions_df: Required interaction DataFrame.  Must include
                ``USER_ID_NAME``, ``ITEM_ID_NAME``, and the target column.
            is_training: When ``True``, initialises the item catalogue from
                the provided datasets.  Set to ``False`` for validation /
                inference preprocessing.

        Returns:
            A tuple ``(X, y)`` where ``X`` is the joined feature DataFrame
            and ``y`` is the target Series.
        """
        self._validate_interactions(interactions_df)
        if users_df is not None:
            self._validate_users(users_df)
        if items_df is not None:
            self._validate_items(items_df)
        if is_training:
            self.item_names, self.items_df = self._process_items(items_df, interactions_df)

        joined_data = self._join_data_train(users_df, self.items_df, interactions_df)

        X, y = self._process_X_y(joined_data)

        return X, y

    def process_factorized_datasets(
        self,
        users_df: Optional[DataFrame],
        items_df: Optional[DataFrame],
        interactions_df: DataFrame,
        is_training: Optional[bool] = True,
    ) -> Tuple[Optional[DataFrame], Optional[DataFrame], DataFrame]:
        """
        Processes and validates factorized user, item, and interaction datasets.
        This method is intended for estimators that consume distinct user, item,
        and interaction data (e.g., embedding-based models). It ensures necessary
        columns are present, validates inputs, and sets up internal item representations
        (self.item_names, self.items_df) on the first call (typically for training data).
        Args:
            users_df: DataFrame containing user features. Must include USER_ID_NAME.
            items_df: DataFrame containing item features. Must include ITEM_ID_NAME.
            interactions_df: DataFrame containing interaction data. Must include
                             USER_ID_NAME, ITEM_ID_NAME, and self.target_col.
        Returns:
            A tuple containing (users_df, processed_items_df, interactions_df).
            - users_df: The input users_df, validated.
            - processed_items_df: The items_df after processing by _process_items (if it was
              the first call and items were processed) or the existing self.items_df.
            - interactions_df: The input interactions_df, validated.
        """
        self._validate_interactions(interactions_df)
        if users_df is not None:
            self._validate_users(users_df)
        if items_df is not None:
            self._validate_items(items_df)

        # Warning (not error): embedding models keep datasets separate and may handle unseen IDs
        # via default embeddings. Tabular models that join datasets will raise in _join_data_train instead.
        if users_df is not None:
            if not set(interactions_df[USER_ID_NAME]).issubset(set(users_df[USER_ID_NAME])):
                logger.warning(
                    "InteractionsDataset contains Users not present in the UsersDataset. "
                    "This may be acceptable for some embedding models."
                )

        if items_df is not None:
            if not set(interactions_df[ITEM_ID_NAME]).issubset(set(items_df[ITEM_ID_NAME])):
                logger.warning(
                    "InteractionsDataset contains Items not present in the ItemsDataset. "
                    "This may be acceptable for some embedding models."
                )

        if is_training:
            self.item_names, self.items_df = self._process_items(items_df, interactions_df)

        return users_df, self.items_df, interactions_df

    def _validate_interactions_base(self, interactions_df: Optional[DataFrame]) -> None:
        """None check shared by all scorers. Subclasses that fully override
        _validate_interactions should call this first."""
        if interactions_df is None:
            raise TypeError("Interaction Dataset must exist for training.")

    def _validate_interactions(self, interactions_df: DataFrame) -> None:
        self._validate_interactions_base(interactions_df)

        if USER_ID_NAME not in interactions_df.columns:
            raise ValueError(f"'{USER_ID_NAME}' column must exist in interactions_df.")
        if ITEM_ID_NAME not in interactions_df.columns:
            raise ValueError(f"'{ITEM_ID_NAME}' column must exist in interactions_df.")

        for col in (USER_ID_NAME, ITEM_ID_NAME):
            null_count = interactions_df[col].isnull().sum()
            if null_count > 0:
                raise ValueError(
                    f"'{col}' column contains {null_count} null value(s). Remove rows with null IDs before training."
                )

        if TIMESTAMP_COL not in interactions_df.columns:
            dup_count = interactions_df.duplicated(subset=[USER_ID_NAME, ITEM_ID_NAME]).sum()
            if dup_count > 0:
                logger.warning(
                    "interactions_df contains %d duplicate (%s, %s) pair(s). "
                    "This may cause silent wrong results downstream. Deduplicate if unintentional.",
                    dup_count,
                    USER_ID_NAME,
                    ITEM_ID_NAME,
                )

        if self.target_col is None:
            raise ValueError("`target_col` must be set in the Scorer subclass.")

        if self.target_col not in interactions_df.columns:
            raise ValueError(f"{self.target_col} column must exist in Interaction Dataset.")

        null_count = interactions_df[self.target_col].isnull().sum()
        if null_count > 0:
            raise ValueError(
                f"target column '{self.target_col}' contains {null_count} null value(s). "
                "Remove or impute before training."
            )

    def _validate_users(self, users_df: DataFrame) -> None:
        if USER_ID_NAME not in users_df.columns:
            raise ValueError(f"'{USER_ID_NAME}' column must exist in users_df.")
        dup_count = users_df.duplicated(subset=[USER_ID_NAME]).sum()
        if dup_count > 0:
            logger.warning(
                "users_df contains %d duplicate %s value(s). "
                "This may cause row fan-out when joining to interactions. Deduplicate if unintentional.",
                dup_count,
                USER_ID_NAME,
            )

    def _validate_items(self, items_df: DataFrame) -> None:
        if ITEM_ID_NAME not in items_df.columns:
            raise ValueError(f"'{ITEM_ID_NAME}' column must exist in items_df.")
        dup_count = items_df.duplicated(subset=[ITEM_ID_NAME]).sum()
        if dup_count > 0:
            logger.warning(
                "items_df contains %d duplicate %s value(s). "
                "This may cause row fan-out when joining to interactions. Deduplicate if unintentional.",
                dup_count,
                ITEM_ID_NAME,
            )

    def _process_items(
        self, items_df: Optional[DataFrame], interactions_df: Optional[DataFrame], is_partitioned: bool = False
    ) -> Tuple[NDArray[np.str_], Optional[DataFrame]]:
        # Sort item list alphabetically to match sklearn
        if items_df is not None:
            items_df = items_df.sort_values(by=[ITEM_ID_NAME])
            items_df = items_df.reset_index(drop=True)
            item_names = items_df[ITEM_ID_NAME]
            if items_df.shape[1] > 1:  # If there are item-features, we store them
                self.items_array = items_df.drop(columns=[ITEM_ID_NAME]).values
            else:
                self.items_array = None  # type: ignore[assignment]
        else:
            if is_partitioned:
                raise RuntimeError("In batched mode with paratitioned dataset, we assume items dataset is provided")
            if interactions_df is None:
                raise ValueError("Cannot infer item names without interactions_df when items_df is not provided.")
            item_names = sorted(interactions_df[ITEM_ID_NAME].unique())  # type: ignore[assignment]

        return np.array(item_names, dtype=np.str_), items_df

    def _join_data_train(
        self, users_df: Optional[DataFrame], items_df: Optional[DataFrame], interactions_df: Optional[DataFrame]
    ) -> DataFrame:
        if interactions_df is None:
            raise ValueError("interactions_df cannot be None for joining.")
        # User_df and item_df are both optional
        joined_data = interactions_df.copy(deep=True)

        # Error (not warning): tabular models join datasets, so an unseen ID would produce a null row
        # and silently corrupt the training data. process_factorized_datasets warns instead (see comment there).
        if users_df is not None:
            if not set(interactions_df[USER_ID_NAME]).issubset(set(users_df[USER_ID_NAME])):
                raise ValueError("Interactions Dataset contains Users not present in the Users Dataset!")

            joined_data = joined_data.merge(users_df, on=USER_ID_NAME, how="left")

        if items_df is not None:
            if not set(interactions_df[ITEM_ID_NAME]).issubset(set(items_df[ITEM_ID_NAME])):
                raise ValueError("Interactions Dataset contains Items not present in the Items Dataset!")

            joined_data = joined_data.merge(items_df, on=ITEM_ID_NAME, how="left")
        return joined_data

    def _process_X_y(self, joined_data: DataFrame) -> Tuple[DataFrame, Series]:
        X, y = self._generate_X_y(joined_data)
        return X, y

    def _generate_X_y(self, joined_data: DataFrame) -> Tuple[DataFrame, Series]:
        y = joined_data[self.target_col]
        dropped_columns = list(DEBUG_COLUMNS)
        if LABEL_NAME in joined_data.columns:
            dropped_columns.append(LABEL_NAME)
        X = joined_data.drop(columns=dropped_columns)

        return X, y

    def train_model(
        self,
        X: DataFrame,
        y: Union[Series, NDArray],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[Series, NDArray]] = None,
    ) -> None:
        """Fit the scorer's estimator on a pre-processed feature matrix.

        Use ``process_datasets`` to obtain ``X`` and ``y`` from raw dataset
        objects before calling this method.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target labels or values of shape ``(n_samples,)``.
            X_valid: Optional validation features.
            y_valid: Optional validation targets.
        """
        self._fit_estimator(self.estimator, X, y, X_valid, y_valid)

    def _fit_estimator(
        self,
        estimator: BaseEstimator,
        X: DataFrame,
        y: Union[Series, NDArray],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[Series, NDArray]] = None,
    ) -> None:
        estimator.fit(X, y, X_valid, y_valid)

    def train_embedding_model(
        self,
        users: Optional[DataFrame],
        items: Optional[DataFrame],
        interactions: DataFrame,
        valid_users: Optional[DataFrame] = None,
        valid_interactions: Optional[DataFrame] = None,
    ) -> None:
        """
        Trains an embedding-based model using factorized user, item, and interaction data.
        This method checks if the underlying estimator is an instance of
        BaseEmbeddingEstimator and then calls its fit_embedding_model method.
        Args:
            users: DataFrame containing user features for training.
            items: DataFrame containing item features for training.
            interactions: DataFrame containing interaction features and target for training.
            valid_users: Optional DataFrame for user features for validation.
            valid_interactions: Optional DataFrame for interaction features and target for validation.
        Raises:
            TypeError: If the estimator is not an instance of BaseEmbeddingEstimator.
        """
        if not isinstance(self.estimator, BaseEmbeddingEstimator):
            raise TypeError(
                "train_embedding_model can only be called when the scorer's estimator is a BaseEmbeddingEstimator."
            )

        self.estimator.fit_embedding_model(
            users=users,
            items=items,
            interactions=interactions,
            valid_users=valid_users,
            valid_interactions=valid_interactions,
        )

    def batch_train_model(
        self,
        interactions_ds: InteractionsDataset,
        items_ds: Optional[ItemsDataset],
        users_ds: Optional[UsersDataset] = None,
        valid_interactions_ds: Optional[InteractionsDataset] = None,
        valid_users_ds: Optional[UsersDataset] = None,
    ) -> None:
        """Train the scorer's estimator in batch (partitioned) mode.

        Used when the full dataset does not fit in memory.  Wraps the datasets
        in a ``BatchTrainingDataset`` iterator and delegates to the estimator's
        ``_batch_fit_model`` method.

        Args:
            interactions_ds: Required interactions dataset.
            items_ds: Required items dataset (needed for batch mode item
                catalogue initialisation).
            users_ds: Optional user features dataset.
            valid_interactions_ds: Optional validation interactions dataset.
            valid_users_ds: Optional validation user features dataset.

        Raises:
            RuntimeError: If the estimator does not support batch training, or
                if ``valid_interactions_ds`` is provided without
                ``valid_users_ds`` when ``users_ds`` is also provided.
        """
        train = BatchTrainingDataset(
            scorer=self,
            interactions_dataset=interactions_ds,
            users_dataset=users_ds,
            items_dataset=items_ds,  # type: ignore[arg-type]
        )
        if not self.estimator.support_batch_training():
            raise RuntimeError("The estimator does not support batch training.")

        if valid_interactions_ds is not None:
            if users_ds is not None and valid_users_ds is None:
                raise RuntimeError("Validation users dataset not found!")
            valid = BatchTrainingDataset(
                scorer=self,
                interactions_dataset=valid_interactions_ds,
                users_dataset=valid_users_ds,
                items_dataset=items_ds,  # type: ignore[arg-type]
            )
        else:
            valid = None

        self.estimator._batch_fit_model(train, valid)

    def _validate_input_recommend(
        self, interactions: Optional[DataFrame], users: Optional[DataFrame]
    ) -> Tuple[DataFrame, DataFrame]:
        if users is None and interactions is None:
            raise ValueError("Both Users and Interactions are None, specify atleast one of them!")

        if users is not None:
            if USER_ID_NAME not in users.columns:
                raise ValueError(f"{USER_ID_NAME} must exist in Users DataFrame!")

        if interactions is not None:
            if USER_ID_NAME not in interactions.columns:
                raise ValueError(f"{USER_ID_NAME} must exist in Interactions DataFrame!")

        if users is None:
            users_dict = {USER_ID_NAME: list(set(interactions[USER_ID_NAME]))}  # type: ignore[index]
            users = pd.DataFrame(users_dict)

        if interactions is None:
            interactions_dict = {USER_ID_NAME: list(users[USER_ID_NAME])}
            interactions = pd.DataFrame(interactions_dict)

        return interactions, users

    def _get_user_interactions_df(self, interactions: Optional[DataFrame], users: Optional[DataFrame]) -> DataFrame:
        interactions, users = self._validate_input_recommend(interactions, users)
        logger.info("Receiving DataFrames for Interactions and Users")
        logger.info(f"Shape of Interactions DataFrame: {interactions.shape}")
        logger.info(f"Shape of Users DataFrame: {users.shape}")

        logger.info("Merging DataFrames")

        # Mimic training logic during inference too, items won't be available during inference
        user_interactions_df = self._join_data_train(users_df=users, items_df=None, interactions_df=interactions)

        logger.info("Completed Merging User-Interactions DataFrames")
        user_interactions_df.drop(columns=[USER_ID_NAME], inplace=True)
        return user_interactions_df

    # NOTE: Only used by multiclass and multioutput scorers
    def get_item_indices(self) -> Optional[List[int]]:
        """Return the indices of the active item subset within ``item_names``.

        Returns ``None`` when no item subset is active.  Used internally by
        ``MulticlassScorer`` and ``MultioutputScorer`` to slice model output
        columns to match the current subset.

        Returns:
            List of integer indices into ``self.item_names``, or ``None`` if
            no subset is set.
        """
        if self.item_subset is None:
            return None
        if self.item_names is None:
            logger.warning("item_names is None, cannot get item indices.")
            return None

        item_indices = []
        item_names_list = self.item_names.tolist()
        for item in self.item_subset:
            try:
                item_indices.append(item_names_list.index(item))
            except ValueError:
                logger.warning(f"Item {item} not found in item_names!")
        return item_indices

    def _create_df_from_scores(self, scores: NDArray[np.float64]) -> DataFrame:
        calculated_scores_df = pd.DataFrame(scores, dtype=np.float64)

        if self.item_subset:
            calculated_scores_df.columns = self.item_subset  # type: ignore[assignment]
        else:
            if self.item_names is None:
                raise ValueError("item_names is None, cannot assign column names.")
            calculated_scores_df.columns = self.item_names  # type: ignore[assignment]

        return calculated_scores_df

    def score_items(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> DataFrame:
        """Score every item in the catalogue for each user.

        Args:
            interactions: DataFrame with interaction features.  Must include
                ``USER_ID_NAME``.  Pass ``None`` when users are identified
                solely from ``users``.
            users: DataFrame with user features.  Must include ``USER_ID_NAME``.
                Pass ``None`` when all required context is in ``interactions``.

        Returns:
            DataFrame of shape ``(n_users, n_items)`` where columns are item
            IDs and values are predicted scores (higher = more relevant).
            Column order matches the active item subset when one is set, or the
            full training catalogue otherwise.
        """
        user_interactions_df = self._get_user_interactions_df(interactions=interactions, users=users)
        calculated_scores = self._calculate_scores(user_interactions_df)
        return self._create_df_from_scores(calculated_scores)

    def _score_items_np(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> NDArray[np.float64]:
        """Internal: returns raw scores array, skipping DataFrame wrapping."""
        return self.score_items(interactions, users).to_numpy()

    def _score_fast_np(self, features: DataFrame) -> NDArray[np.float64]:
        """Internal: returns raw scores array for a single pre-merged row, skipping DataFrame wrapping."""
        return self.score_fast(features).to_numpy()

    def _process_item_subset(self, item_subset: Optional[List[str]]) -> Optional[List[str]]:
        if item_subset is None:
            return None

        if self.item_names is None:
            raise ValueError("Cannot process item_subset because item_names is None (training likely incomplete).")

        input_length_item_subset = len(item_subset)
        unique_item_subset = sorted(list(set(item_subset)))

        if input_length_item_subset == 0:
            raise ValueError("Length of item_subset cannot be zero")

        if input_length_item_subset != len(unique_item_subset):
            raise ValueError("item_subset contains non-unique values")

        training_items_set = set(self.item_names)
        if not set(unique_item_subset).issubset(training_items_set):
            missing_items = set(unique_item_subset) - training_items_set
            raise ValueError(f"item_subset contains items not used while training: {missing_items}")

        return unique_item_subset

    # Shape of returned scores = (n_rows, n_items)
    @abstractmethod
    def _calculate_scores(self, joined: Union[DataFrame, NDArray]) -> NDArray[np.float64]:
        """
        columns are aligned on `self.item_subset` if set, otherwise on `self.item_names`
        """
        # To be implemented by child classes because we don't know whether to call predict or predict_proba
        pass
