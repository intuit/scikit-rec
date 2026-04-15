from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.scorer.base_scorer import BaseScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class MulticlassScorer(BaseScorer):
    target_col = ITEM_ID_NAME

    def __init__(self, estimator: BaseClassifier) -> None:
        if isinstance(estimator, BaseEmbeddingEstimator):
            raise TypeError(
                "MulticlassScorer does not support BaseEmbeddingEstimator. "
                "Use UniversalScorer for embedding estimators (e.g. MatrixFactorizationEstimator, NCFEstimator)."
            )
        super().__init__(estimator)

    def process_datasets(
        self,
        users_df: Optional[DataFrame] = None,
        items_df: Optional[DataFrame] = None,
        interactions_df: Optional[DataFrame] = None,
        is_training: Optional[bool] = True,
    ) -> Tuple[DataFrame, NDArray]:
        """Validate and prepare interaction data for multiclass classification.

        ``MulticlassScorer`` treats item prediction as a single multiclass
        problem — each interaction row's label is its ``ITEM_ID``, encoded as
        an integer class index.  User and item feature DataFrames are not used
        and must be ``None``.

        Args:
            users_df: Must be ``None``.  Raises ``ValueError`` if provided.
            items_df: Must be ``None``.  Raises ``ValueError`` if provided.
            interactions_df: Interaction DataFrame without an ``OUTCOME``
                column.  Must include ``ITEM_ID_NAME`` as the target.
            is_training: When ``True``, initialises ``item_names`` from the
                unique items observed in ``interactions_df``.

        Returns:
            A tuple ``(X, y)`` where ``y`` contains integer-encoded item
            class indices.

        Raises:
            ValueError: If ``users_df`` or ``items_df`` is provided, or if an
                ``OUTCOME`` column is present in ``interactions_df``.
        """
        if users_df is not None:
            raise ValueError("Users Dataset will not be used in MulticlassScorer.")

        if items_df is not None:
            raise ValueError("Items Dataset will not be used in MulticlassScorer.")

        if "OUTCOME" in interactions_df.columns:
            raise ValueError("OUTCOME field not allowed in Interactions Dataset for MulticlassScorer")

        return super().process_datasets(
            users_df=None,
            items_df=None,
            interactions_df=interactions_df,
            is_training=is_training,
        )

    def _process_X_y(self, joined_data: DataFrame) -> Tuple[NDArray, NDArray]:
        X, y = super()._process_X_y(joined_data)
        y = self._encode_label(y)
        return X, y

    def _encode_label(self, y: NDArray) -> NDArray:
        # encode item names based on its position in self.item_names
        mapping = {item: i for i, item in enumerate(self.item_names)}
        return np.array([mapping[v] for v in y])

    def _process_items(
        self, items_df: DataFrame, interactions_df: DataFrame, is_partitioned=False
    ) -> Tuple[NDArray, DataFrame]:
        items_list, returned_items_df = super()._process_items(items_df, interactions_df, is_partitioned=is_partitioned)
        return items_list, returned_items_df

    def _calculate_scores(self, joined: Union[DataFrame, NDArray]) -> NDArray[np.float64]:
        """
        columns are aligned on `self.item_subset` if set, otherwise on `self.item_names`
        """
        preds = self.estimator.predict_proba(joined)
        if preds.shape[1] == 2 * len(self.item_names):
            logger.info("By using inplace-predict, you have inadvertently stacked 1-pred and pred, unstacking now!")
            preds = preds[:, len(self.item_names) : preds.shape[1]]
        elif preds.shape[1] == len(self.item_names):
            pass
        else:
            raise ValueError("Mismatch in number of expected scores vs items!")
        if self.item_subset is not None:
            relevant_indices = self.get_item_indices()
            preds = preds[:, relevant_indices]
        return preds

    def score_fast(self, features: DataFrame) -> DataFrame:
        """Score all items for a single pre-merged user row without join overhead.

        Args:
            features: Single-row DataFrame with interaction features.
                ``USER_ID_NAME``, ``ITEM_ID_NAME``, and ``LABEL_NAME`` columns
                are silently dropped if present.

        Returns:
            DataFrame of shape ``(1, n_items)`` with item class probabilities.

        Raises:
            ValueError: If ``features`` has more than one row.
        """
        if features.shape[0] != 1:
            raise ValueError(
                f"score_fast() expects exactly 1 row, got {features.shape[0]}. Use score_items() for batch scoring."
            )
        drop_cols = [c for c in [USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME] if c in features.columns]
        if drop_cols:
            features = features.drop(columns=drop_cols)
        scores = self._calculate_scores(features)
        return self._create_df_from_scores(scores)

    def score_items(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,  # Should always be None for this scorer
    ) -> DataFrame:
        """Score all items for each user.

        Args:
            interactions: DataFrame with interaction features.  Must include
                ``USER_ID_NAME``.  Must not contain a ``users`` DataFrame.
            users: Must be ``None``.  Raises ``ValueError`` if provided.

        Returns:
            DataFrame of shape ``(n_users, n_items)`` with item class
            probabilities as values and item IDs as columns.

        Raises:
            ValueError: If ``users`` is not ``None``.
        """
        if users is not None:
            raise ValueError("Multiclass Scorer cannot accept Users Dataframe, set it to None!")
        return super().score_items(interactions, users)
