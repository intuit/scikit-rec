from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, get_dummies

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_EMBEDDING_NAME, USER_ID_NAME
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.estimator.regression.base_regressor import BaseRegressor
from skrec.scorer.base_scorer import BaseScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class UniversalScorer(BaseScorer):
    """Score items using a tabular or embedding estimator.

    ``UniversalScorer`` is a factory: calling ``UniversalScorer(estimator)``
    returns a ``TabularUniversalScorer`` for tabular estimators (classifiers,
    regressors) or an ``EmbeddingUniversalScorer`` for embedding estimators
    (NCF, Two-Tower, MatrixFactorization).  Both subclasses share item
    management and preprocessing; they differ in how user-item feature matrices
    are constructed and which estimator method is called for scoring.

    Existing ``isinstance(scorer, UniversalScorer)`` checks continue to work
    because both subclasses inherit from this class.
    """

    target_col = LABEL_NAME
    _items_df_indexed: Optional[DataFrame] = None

    def __new__(cls, estimator, **kwargs):
        if cls is UniversalScorer:
            if isinstance(estimator, BaseEmbeddingEstimator):
                return object.__new__(EmbeddingUniversalScorer)
            return object.__new__(TabularUniversalScorer)
        return object.__new__(cls)

    def __init__(self, estimator: Union[BaseClassifier, BaseEmbeddingEstimator]) -> None:
        super().__init__(estimator)

    def _process_items(
        self, items_df: DataFrame, interactions_df: DataFrame, is_partitioned: bool = False
    ) -> Tuple[NDArray, DataFrame]:
        item_names, returned_items_df = super()._process_items(items_df, interactions_df, is_partitioned=is_partitioned)
        if items_df is None:
            logger.warning("Since item dataset is missing, we create one-hot encodings for items.")
            returned_items_df = get_dummies(item_names, prefix=ITEM_ID_NAME, prefix_sep="=", dtype=int)
            returned_items_df.insert(loc=0, column=ITEM_ID_NAME, value=item_names)
        self.items_array = returned_items_df.drop(columns=[ITEM_ID_NAME]).values
        self._items_df_indexed = returned_items_df.set_index(ITEM_ID_NAME)
        return item_names, returned_items_df

    def _get_relevant_items_df(self):
        logger.info("Adding Item Features for All Items, via Replication")
        if self.item_subset_df is not None:
            return self.item_subset_df
        if self._items_df_indexed is None:
            self._items_df_indexed = self.items_df.set_index(ITEM_ID_NAME)
        return self._items_df_indexed

    def set_new_items(self, new_items_df: DataFrame) -> None:
        """Extend the item catalogue with new items without retraining.

        New items are appended to the existing catalogue and the scorer's
        internal state (``item_names``, ``items_df``, ``_items_df_indexed``) is
        updated in place.  If a new item shares an ID with an existing one,
        the existing entry is replaced with the features from ``new_items_df``
        and a warning is logged.

        .. note::
            Must be called *before* ``set_item_subset``.  The underlying
            estimator is not retrained — scores for new items are produced by
            extrapolating from the trained model's feature space.

        Args:
            new_items_df: DataFrame with the same columns as the training
                ``items_df``.  Must include ``ITEM_ID_NAME``.

        Raises:
            ValueError: If called after ``set_item_subset`` is active, or if
                ``new_items_df`` columns do not match the training items schema.
        """
        if self.item_subset_df is not None:
            raise ValueError("Call set_new_items() before set_item_subset()")

        items_df = self.items_df

        # Check that new_items_df has the same columns as self.items_df
        if new_items_df.columns.to_list() != items_df.columns.to_list():
            raise ValueError("new_items_df must have the same columns as items_df")

        # Check if there is an overlap in ITEM_ID between new_items_df and self.items_df
        overlap_idx = items_df[ITEM_ID_NAME].isin(new_items_df[ITEM_ID_NAME])

        if overlap_idx.any():
            overlapping_items = items_df[overlap_idx][ITEM_ID_NAME].tolist()

            logger.warning(f"Overlap found in ITEM_ID: {overlapping_items}. Using features from new_items_df")

            # Remove overlapping items from the existing items_df
            items_df = items_df[~overlap_idx]

        # Concatenate new_items_df to items_df
        items_df = pd.concat([items_df, new_items_df], axis=0, ignore_index=True)

        # Sort the concatenated DataFrame by ITEM_ID and update item_names
        self.items_df = items_df.sort_values(by=ITEM_ID_NAME).reset_index(drop=True)
        self._items_df_indexed = self.items_df.set_index(ITEM_ID_NAME)

        self.item_names = items_df[ITEM_ID_NAME].values

    def _calculate_scores(self, joined):
        raise NotImplementedError(
            "UniversalScorer dispatches to TabularUniversalScorer or "
            "EmbeddingUniversalScorer. Do not call _calculate_scores on the base class."
        )


class TabularUniversalScorer(UniversalScorer):
    """Universal scorer for tabular estimators (classifiers and regressors).

    Replicates each user's feature row across all candidate items, appends item
    features, and runs a single batched ``predict`` / ``predict_proba`` call.
    """

    def _calculate_scores(self, joined: Union[DataFrame, NDArray]) -> NDArray[np.float64]:
        """
        columns are aligned on `self.item_subset` if set, otherwise on `self.item_names`
        """
        relevant_items = self.item_subset if self.item_subset else self.item_names

        # Check if the estimator is a regressor or a classifier handle accordingly
        if isinstance(self.estimator, BaseRegressor):
            scores = self.estimator.predict(joined)
        else:
            scores = self.estimator.predict_proba(joined)[:, 1]

        scores = np.reshape(scores, (-1, len(relevant_items)))
        return scores

    def score_items(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> DataFrame:
        """Score every item in the catalogue for each user.

        Replicates each user's feature row once per item, appends item features,
        and runs a single batched model call.

        Args:
            interactions: DataFrame with user/interaction context features.
                Must include ``USER_ID_NAME`` when ``users`` is also provided.
            users: Optional DataFrame with user features.

        Returns:
            DataFrame of shape ``(n_users, n_items)`` with item IDs as columns.
        """
        user_interactions_df = super()._get_user_interactions_df(interactions=interactions, users=users)
        user_interactions_items_df = self._replicate_for_items(user_interactions_df)

        logger.info("Calculating Scores for All Items")
        drop_columns = [ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME]
        user_interactions_items_df = user_interactions_items_df.drop(columns=drop_columns, errors="ignore")

        calculated_scores_np = self._calculate_scores(user_interactions_items_df)

        return self._create_df_from_scores(calculated_scores_np)

    def _score_items_np(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> NDArray[np.float64]:
        """Optimised path: returns raw scores array, skipping DataFrame wrapping."""
        user_interactions_df = super()._get_user_interactions_df(interactions=interactions, users=users)
        user_interactions_items_df = self._replicate_for_items(user_interactions_df)
        drop_columns = [ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME]
        user_interactions_items_df = user_interactions_items_df.drop(columns=drop_columns, errors="ignore")
        return self._calculate_scores(user_interactions_items_df)

    def _score_fast_np(self, features: DataFrame) -> NDArray[np.float64]:
        """Internal: returns raw scores array for a single pre-merged row, skipping DataFrame wrapping."""
        if features.shape[0] != 1:
            raise ValueError(
                f"_score_fast_np expects exactly 1 row, got {features.shape[0]}. "
                "Use _score_items_np() for batch scoring."
            )
        drop_cols = [c for c in [USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME] if c in features.columns]
        if drop_cols:
            features = features.drop(columns=drop_cols)
        rep_df = self._replicate_for_items(features)
        return self._calculate_scores(rep_df)

    def score_fast(self, features: DataFrame) -> DataFrame:
        """
        Score all items for a single user without join overhead.

        **Compatibility**: supported on all non-embedding scorers —
        ``UniversalScorer``, ``MulticlassScorer``, ``MultioutputScorer``,
        and ``IndependentScorer``. Not supported for:

        - Embedding-based estimators (NCF, Two-Tower, DeepFM) — use ``score_items()``
          instead; embedding models already score all items in a single forward pass.
        - ``SequentialScorer`` / ``HierarchicalScorer`` — SASRec/HRNN have their own
          efficient forward pass; use ``score_items()`` directly.

        At the recommender level, prefer ``recommender.recommend_online()`` which
        handles schema validation and merging automatically.

        Designed for real-time serving where latency matters. Skips the pandas
        merge/join that ``score_items()`` performs. The caller is responsible for
        passing all user and interaction features in a single pre-merged single-row
        DataFrame. Use ``self.estimator.feature_names`` to verify the expected column
        order after training.

        Args:
            features: Single-row DataFrame with user/interaction features.
                      Must not contain USER_ID, ITEM_ID, or LABEL columns
                      (they are silently dropped if present).

        Returns:
            DataFrame of shape (1, n_items) with item names as columns.

        Raises:
            ValueError: If ``features`` has more than one row.
        """
        return self._create_df_from_scores(self._score_fast_np(features))

    def _replicate_for_items_np(self, user_interactions_array: NDArray) -> NDArray:
        # NO PANDAS OPERATIONS ALLOWED
        if user_interactions_array.shape[0] == 0:
            raise ValueError("No rows input for duplication")

        relevant_items_df = self._get_relevant_items_df()
        relevant_items_array = relevant_items_df.values

        n_items = relevant_items_array.shape[0]
        n_interactions = user_interactions_array.shape[0]

        rep_user_interactions = user_interactions_array.repeat(n_items, axis=0)
        rep_items = np.tile(relevant_items_array, (n_interactions, 1))

        rep_user_interactions_items = np.concatenate([rep_user_interactions, rep_items], axis=1)

        logger.info("Completed Adding Item Features for ALL ITEMS, via Replication")

        return rep_user_interactions_items

    def _replicate_for_items(self, user_interactions_df: DataFrame) -> DataFrame:
        relevant_items_df = self._get_relevant_items_df()

        interactions_cols = user_interactions_df.columns.tolist()
        items_cols = relevant_items_df.columns.tolist()
        all_cols = interactions_cols + items_cols

        rep_user_interactions_items = self._replicate_for_items_np(user_interactions_df.values)

        ml_ready_df = pd.DataFrame(rep_user_interactions_items, columns=all_cols, dtype=np.float64)

        return ml_ready_df


class EmbeddingUniversalScorer(UniversalScorer):
    """Universal scorer for embedding estimators (NCF, Two-Tower, MatrixFactorization).

    Cross-joins user interaction rows with item IDs and calls
    ``predict_proba_with_embeddings`` for a single forward pass.
    """

    def _calculate_scores_with_embeddings(
        self,
        interactions_df: DataFrame,
        users_df: Optional[DataFrame],
    ) -> NDArray[np.float64]:
        """
        Calculates scores using an embedding estimator's predict_proba_with_embeddings method.
        """
        relevant_items_count = len(self.item_subset) if self.item_subset is not None else len(self.item_names)

        scores_flat = self.estimator.predict_proba_with_embeddings(interactions=interactions_df, users=users_df)
        scores = np.reshape(scores_flat, (-1, relevant_items_count))
        return scores

    def score_items(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> DataFrame:
        """Score every item using embedding-based forward pass.

        Cross-joins user interaction rows with item IDs and calls
        ``predict_proba_with_embeddings`` for a single forward pass.

        Args:
            interactions: DataFrame with user/interaction context features.
                For embedding estimators, ``USER_ID_NAME`` is required.
            users: Optional DataFrame with user features.  For embedding
                estimators in real-time mode, must also contain
                ``USER_EMBEDDING_NAME`` (pre-computed user embedding vector).

        Returns:
            DataFrame of shape ``(n_users, n_items)`` with item IDs as columns.
        """
        interactions, users = self._validate_input_recommend(interactions, users)

        # Prepare the base interactions DataFrame (user_id + context features)
        # This will be cross-joined with item IDs.
        relevant_items = self._get_relevant_items_df()
        item_ids = relevant_items.index.to_frame(name=ITEM_ID_NAME, index=False)
        interactions_items_df = self._replicate_for_items_id(interactions, item_ids)

        interactions_items_df = interactions_items_df.drop(columns=[LABEL_NAME], errors="ignore")

        calculated_scores_np = self._calculate_scores_with_embeddings(
            interactions_df=interactions_items_df, users_df=users
        )

        return self._create_df_from_scores(calculated_scores_np)

    def _validate_input_recommend(self, interactions: Optional[DataFrame], users: Optional[DataFrame]):
        if users is not None and USER_EMBEDDING_NAME not in users.columns:
            raise ValueError(f"`users` DataFrame must contain '{USER_EMBEDDING_NAME}' column for embedding estimators.")
        if interactions is None:
            if users is None:
                raise ValueError(
                    "For embedding estimators, at least one of 'interactions' or 'users' must be provided."
                )
            interactions = users[[USER_ID_NAME]]
        return interactions, users

    def _replicate_for_items_id(self, interactions_df: DataFrame, item_ids_df: DataFrame) -> DataFrame:
        """
        like _replicate_for_items but joining ITEM_ID not features
        """
        if interactions_df.empty or item_ids_df.empty:
            expected_cols = interactions_df.columns.tolist()
            if ITEM_ID_NAME not in expected_cols:
                expected_cols.append(ITEM_ID_NAME)
            return pd.DataFrame(columns=expected_cols)

        return pd.merge(interactions_df, item_ids_df, how="cross")

    def score_fast(self, features: DataFrame) -> DataFrame:
        raise NotImplementedError("score_fast is not supported for embedding estimators. Use score_items() instead.")

    def _calculate_scores(self, joined):
        raise TypeError(
            "_calculate_scores is not applicable to EmbeddingUniversalScorer. "
            "Scoring routes through score_items() -> _calculate_scores_with_embeddings()."
        )
