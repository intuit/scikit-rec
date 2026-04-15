from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from skrec.constants import ITEM_ID_NAME, USER_ID_NAME
from skrec.estimator.sequential.base_sequential_estimator import SequentialEstimator
from skrec.estimator.sequential.sasrec_estimator import ITEM_SEQUENCE_COL
from skrec.scorer.base_scorer import BaseScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class SequentialScorer(BaseScorer):
    """
    Scorer for sequential recommendation models (SASRec, HRNN, etc.).

    Expects pre-built sequences from the recommender layer — it does NOT build
    sequences itself. The interactions DataFrame passed here must already contain
    USER_ID and the appropriate sequence column (ITEM_SEQUENCE or SESSION_SEQUENCES).

    Inherits item management (item_names, items_df, item_subset, set_item_subset,
    clear_item_subset) from BaseScorer. Sequential models score all items in a single
    forward pass — no Cartesian product needed.
    """

    def __init__(self, estimator: SequentialEstimator) -> None:
        # SequentialEstimator intentionally does not inherit BaseEstimator — it uses
        # fit_embedding_model() / predict_proba_with_embeddings() rather than the
        # tabular fit(X, y) / predict(X) interface. BaseScorer stores it as self.estimator
        # and never calls BaseEstimator-specific methods on it directly; all such call
        # sites in SequentialScorer are overridden to route through SequentialEstimator's
        # own interface.
        super().__init__(estimator)

    def train_embedding_model(
        self,
        users: Optional[DataFrame] = None,
        items: Optional[DataFrame] = None,
        interactions: Optional[DataFrame] = None,
        valid_users: Optional[DataFrame] = None,
        valid_interactions: Optional[DataFrame] = None,
    ) -> None:
        """Train the sequential estimator.

        Overrides BaseScorer.train_embedding_model, which guards against
        non-BaseEmbeddingEstimator instances. SequentialEstimator intentionally
        sits outside that hierarchy, so we call fit_embedding_model directly.
        """
        self.estimator.fit_embedding_model(
            users=users,
            items=items,
            interactions=interactions,
            valid_users=valid_users,
            valid_interactions=valid_interactions,
        )

    def process_factorized_datasets(
        self,
        users_df: Optional[DataFrame],
        items_df: Optional[DataFrame],
        interactions_df: DataFrame,
        is_training: Optional[bool] = True,
    ):
        """
        Validate and process sequential data.

        Args:
            users_df: Optional user features (ignored by sequence-only models).
            items_df: DataFrame with ITEM_ID column. Used to set item_names.
            interactions_df: sequences_df with USER_ID and ITEM_SEQUENCE columns.
                             Built by SequentialRecommender, not raw interactions.
            is_training: When True, sets self.item_names and self.items_df from items_df.

        Returns:
            (users_df, items_df, interactions_df)
        """
        if interactions_df is None:
            raise ValueError("interactions_df cannot be None.")
        if USER_ID_NAME not in interactions_df.columns:
            raise ValueError(f"'{USER_ID_NAME}' column must exist in interactions_df.")
        if ITEM_SEQUENCE_COL not in interactions_df.columns:
            raise ValueError(
                f"'{ITEM_SEQUENCE_COL}' column must exist in interactions_df. "
                "Sequences must be built by SequentialRecommender before calling the scorer."
            )

        if is_training:
            if items_df is not None:
                # Normalise ITEM_ID to str so the sort order is always lexicographic and
                # consistent with the estimator's sorted(str(x)...) vocabulary building.
                # Without this, int64 ITEM_IDs (returned by ItemsDataset from CSV) sort
                # numerically [1,2,3,...] while the estimator sorts lexicographically
                # ["1","10","100",...], causing a column-index mismatch in the score matrix.
                items_df = items_df.copy()
                items_df[ITEM_ID_NAME] = items_df[ITEM_ID_NAME].astype(str)
                items_df = items_df.sort_values(by=ITEM_ID_NAME).reset_index(drop=True)
                self.item_names = np.array(items_df[ITEM_ID_NAME].values, dtype=np.str_)
                self.items_df = items_df
            else:
                # Derive item vocabulary from sequences
                all_items = sorted({str(item) for seq in interactions_df[ITEM_SEQUENCE_COL] for item in seq})
                self.item_names = np.array(all_items, dtype=np.str_)
                self.items_df = pd.DataFrame({ITEM_ID_NAME: all_items})

        return users_df, self.items_df, interactions_df

    def score_items(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Score all items for each user using their encoded sequence.

        Args:
            interactions: sequences_df with USER_ID and ITEM_SEQUENCE columns.
                          Built by SequentialRecommender before calling scorer.
            users: Ignored (sequential models derive user representations from sequences).

        Returns:
            DataFrame of shape (num_users, num_items) or (num_users, len(item_subset)).
        """
        if interactions is None:
            raise ValueError(
                "SequentialScorer requires sequences in interactions. "
                "Pass the output of SequentialRecommender._build_sequences()."
            )

        scores = self._calculate_scores_with_embeddings(interactions, users)

        # Apply item_subset filter if set
        if self.item_subset is not None:
            if self.item_names is None:
                raise RuntimeError("item_names must be set before scoring with item_subset.")
            item_to_col = {name: i for i, name in enumerate(self.item_names)}
            subset_indices = [item_to_col[item] for item in self.item_subset]
            scores = scores[:, subset_indices]

        return self._create_df_from_scores(scores)

    def _calculate_scores_with_embeddings(
        self,
        interactions_df: DataFrame,
        users_df: Optional[DataFrame],
    ) -> NDArray[np.float64]:
        """Return (num_users, num_items) scores from the sequential estimator.

        The estimator encodes each user's sequence and dot-products against all
        item embeddings in one forward pass — no Cartesian product replication needed.
        """
        return self.estimator.predict_proba_with_embeddings(  # type: ignore[return-value]
            interactions=interactions_df,
            users=users_df,
        )

    def train_model(self, X, y, X_valid=None, y_valid=None) -> None:
        raise NotImplementedError(
            "SequentialScorer does not support train_model(). Use train_embedding_model() to train sequential models."
        )

    def score_fast(self, features: DataFrame) -> DataFrame:
        raise NotImplementedError(
            "SequentialScorer does not support score_fast(). "
            "Sequential models score all items in a single forward pass via score_items()."
        )

    def set_new_items(self, new_items_df: DataFrame) -> None:
        raise NotImplementedError(
            "SequentialScorer does not support adding new items after training. "
            "Sequential models (SASRec, HRNN) have fixed embedding tables sized at "
            "training time. Retrain with the updated item catalogue instead."
        )

    def _calculate_scores(self, joined):
        # Required by BaseScorer's abstract interface, but never reached in the
        # sequential scoring path — score_items() routes directly through
        # _calculate_scores_with_embeddings() instead.
        raise NotImplementedError(
            "_calculate_scores() is not applicable to SequentialScorer. "
            "Scoring always routes through score_items() → "
            "_calculate_scores_with_embeddings() → predict_proba_with_embeddings()."
        )
