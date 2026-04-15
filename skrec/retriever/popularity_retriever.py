"""Popularity-based candidate retriever using interaction counts."""

from typing import Any, Dict, List, Optional

import pandas as pd

from skrec.constants import ITEM_ID_NAME
from skrec.estimator.base_estimator import BaseEstimator
from skrec.retriever.base_retriever import BaseCandidateRetriever
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class PopularityRetriever(BaseCandidateRetriever):
    """
    Retrieves candidates based on global item popularity (interaction counts).

    Returns the same top-K most interacted items for every user. Simple,
    fast, and works with any estimator type including XGBoost.

    Best for
    --------
    - Getting started quickly with no additional setup.
    - Cold-start scenarios where no user or item features are available.
    - Non-personalized use cases (e.g. homepage recommendations).
    - Any estimator type, including XGBoost and LightGBM.

    Not for
    -------
    - Personalized recommendations — every user receives the same candidate
      set regardless of their history. The ranking stage adds personalization,
      but the retrieval pool is identical for all users.

    Example
    -------
    ::

        from skrec.retriever import PopularityRetriever

        recommender = RankingRecommender(
            scorer=UniversalScorer(estimator=xgb_estimator),
            retriever=PopularityRetriever(top_k=200),
        )
        recommender.train(interactions_ds=interactions_ds)
        recommendations = recommender.recommend(interactions=df, top_k=10)
    """

    def __init__(self, top_k: int = 100):
        """
        Args:
            top_k: Number of popular items to include in the candidate set.
        """
        super().__init__(top_k=top_k)
        self._popular_items: Optional[List[Any]] = None

    def build_index(
        self,
        estimator: Optional[BaseEstimator] = None,
        interactions: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Build popularity index from training interactions.

        Args:
            estimator: Unused.
            interactions: Training interactions DataFrame. Must contain
                ``ITEM_ID_NAME`` column.
            items: Items DataFrame. When provided, the popular-items list is
                filtered to item IDs present in this DataFrame, ensuring the
                scorer always receives candidates it knows about. If ``None``,
                no filtering is applied.

        Raises:
            ValueError: If ``interactions`` is None or missing ``ITEM_ID_NAME``.
        """
        if interactions is None or interactions.empty:
            raise ValueError("PopularityRetriever requires a non-empty interactions DataFrame.")
        if ITEM_ID_NAME not in interactions.columns:
            raise ValueError(f"interactions must contain '{ITEM_ID_NAME}' column.")

        counts = interactions[ITEM_ID_NAME].value_counts()

        if items is not None and not items.empty and ITEM_ID_NAME in items.columns:
            # Filter to items the scorer knows about. Items present in interactions
            # but absent from the items DataFrame (e.g. deleted catalog entries)
            # would be unknown to the scorer and must not be returned as candidates.
            catalog_set = set(items[ITEM_ID_NAME].values)
            self._popular_items = [item_id for item_id in counts.index if item_id in catalog_set]
            if not self._popular_items:
                # All interacted items are outside the current catalog — fall back.
                self._popular_items = items[ITEM_ID_NAME].tolist()
        else:
            self._popular_items = counts.index.tolist()

        logger.info("PopularityRetriever: indexed %d popular items.", len(self._popular_items))

    def retrieve(self, user_ids: List[Any], top_k: int) -> Dict[Any, List[Any]]:
        """
        Return the globally popular items for every user.

        Args:
            user_ids: User IDs (accepted but ignored — all users get the
                same popular candidates).
            top_k: Number of candidates. Capped at the total number of
                unique items seen during training if top_k exceeds it.

        Returns:
            Dict mapping each user_id to the same list of popular item IDs.
        """
        if self._popular_items is None:
            raise RuntimeError("build_index() must be called before retrieve().")

        candidates = self._popular_items[:top_k]
        return {uid: list(candidates) for uid in user_ids}
