"""Abstract base class for candidate retrievers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from skrec.estimator.base_estimator import BaseEstimator


class BaseCandidateRetriever(ABC):
    """
    Abstract base for two-stage retrieval in recommendation pipelines.

    A retriever narrows the full item catalog to a smaller candidate set.
    The ranking model (scorer + estimator) then re-scores only those
    candidates, giving precise personalized recommendations at manageable cost.

    Lifecycle
    ---------
    1. Train the recommender as usual::

           recommender.train(interactions_ds=..., items_ds=...)

       ``RankingRecommender.train()`` automatically calls
       ``retriever.build_index(...)`` at the end of training.

    2. Call ``recommend()`` normally — the retriever runs transparently::

           recommender.recommend(interactions=df, top_k=10)

    Implementing a custom retriever
    --------------------------------
    Subclass ``BaseCandidateRetriever`` and implement both abstract methods.
    Only use the constructor arguments you need — all three of ``estimator``,
    ``interactions``, and ``items`` are optional at the base class level.

    Example (recency-based retriever — returns each user's most recently seen items)::

        from skrec.constants import ITEM_ID_NAME, USER_ID_NAME
        from skrec.retriever.base_retriever import BaseCandidateRetriever

        class RecencyRetriever(BaseCandidateRetriever):
            \"\"\"Retrieves the top_k most recently interacted items per user.
            Falls back to the globally most popular items for unknown users.\"\"\"

            def __init__(self, top_k: int = 100):
                super().__init__(top_k=top_k)
                self._user_recent_items = {}   # user_id -> [item_id, ...] in recency order
                self._popular_fallback = []    # ordered by interaction count

            def build_index(self, estimator=None, interactions=None, items=None):
                if interactions is None or interactions.empty:
                    return
                # Assumes interactions are ordered oldest-first (e.g. sorted by timestamp
                # before passing in). Reverse each group to get most-recent-first.
                for uid, group in interactions.groupby(USER_ID_NAME, sort=False):
                    self._user_recent_items[uid] = group[ITEM_ID_NAME].tolist()[::-1]
                self._popular_fallback = (
                    interactions[ITEM_ID_NAME].value_counts().index.tolist()
                )

            def retrieve(self, user_ids, top_k):
                results = {}
                for uid in user_ids:
                    candidates = self._user_recent_items.get(uid, self._popular_fallback)
                    results[uid] = candidates[:top_k]
                return results
    """

    def __init__(self, top_k: int = 100):
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")
        self.top_k = top_k

    @staticmethod
    def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
        """Return indices of the top-k highest scores in descending order.

        Uses ``np.argpartition`` (O(n)) to select the top-k candidates, then
        sorts only those k elements — avoiding a full O(n log n) sort over the
        entire catalog.

        Args:
            scores: 1-D float array of item scores, length n.
            k: Number of top items to return. Clamped to len(scores).

        Returns:
            1-D int array of length min(k, n), sorted highest-score first.
        """
        n = len(scores)
        if k <= 0:
            # -0 == 0 in Python, so np.argpartition(scores, 0)[-0:] returns the full
            # array rather than an empty slice. Guard explicitly.
            return np.array([], dtype=np.intp)
        if k >= n:
            return np.argsort(scores)[::-1]
        top_idx = np.argpartition(scores, -k)[-k:]
        return top_idx[np.argsort(scores[top_idx])[::-1]]

    @abstractmethod
    def build_index(
        self,
        estimator: Optional[BaseEstimator] = None,
        interactions: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Build the retrieval index after training completes.

        Called automatically by ``RankingRecommender.train()``. Each subclass
        uses only the arguments it needs — unused arguments are ignored.

        Args:
            estimator: The fitted estimator from the scorer. Required by
                ``EmbeddingRetriever``; ignored by the others.
            interactions: Training interactions DataFrame. Required by
                ``PopularityRetriever`` and ``ContentBasedRetriever``.
                Also used by ``EmbeddingRetriever`` when
                ``cold_start_strategy="popular"`` to rank the cold-start
                fallback by interaction count.
            items: Items DataFrame with item features. Required by
                ``ContentBasedRetriever``; ignored by the others.
        """

    @abstractmethod
    def retrieve(self, user_ids: List[Any], top_k: int) -> Dict[Any, List[Any]]:
        """
        Retrieve top-K candidate item IDs for each user.

        Args:
            user_ids: List of user IDs to retrieve candidates for.
            top_k: Number of candidate items to return per user.

        Returns:
            Dict mapping each user_id to a list of candidate item IDs.
            Example: ``{"user_1": ["item_3", "item_7", "item_12"], ...}``
        """
