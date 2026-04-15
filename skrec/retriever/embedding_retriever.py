"""Embedding-based candidate retriever using brute-force dot-product search."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from skrec.constants import (
    ITEM_EMBEDDING_NAME,
    ITEM_ID_NAME,
    USER_EMBEDDING_NAME,
    USER_ID_NAME,
)
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.retriever.base_retriever import BaseCandidateRetriever
from skrec.util.logger import get_logger

logger = get_logger(__name__)

_VALID_COLD_START_STRATEGIES = ("popular", "zero")


class EmbeddingRetriever(BaseCandidateRetriever):
    """
    Retrieves candidates using learned user and item embeddings.

    Performs brute-force dot-product search: ``scores = U @ I.T``, where
    ``U`` is the user factor matrix and ``I`` is the item factor matrix.
    Works with any embedding-based estimator: ``MatrixFactorizationEstimator``,
    ``NCFEstimator``, ``TwoTowerEstimator``.

    Best for
    --------
    - Warm users with interaction history seen during training.
    - Stable item catalogs (new items added after training are invisible).
    - Personalized retrieval based on learned preferences.

    Not for
    -------
    - Items added to the catalog after training (use ``ContentBasedRetriever``).
    - Non-embedding estimators such as XGBoost (use ``PopularityRetriever``
      or ``ContentBasedRetriever`` instead).
    - Very large item catalogs (millions of items): brute-force dot-product
      search is ``O(n_items)`` per user per ``retrieve()`` call. For large
      catalogs implement a custom retriever backed by an ANN index (e.g.
      FAISS) to make item-side search sub-linear.
    - Very large user bases (tens of millions+): all user embeddings are
      cached in memory at ``build_index()`` time —
      ``O(n_users × embedding_dim × 8 bytes)``. At 10 M users and dim=64
      that is ~5 GB. For such scales, fetch user vectors on demand rather
      than caching them all upfront.

    Example
    -------
    ::

        from skrec.retriever import EmbeddingRetriever

        recommender = RankingRecommender(
            scorer=UniversalScorer(estimator=MatrixFactorizationEstimator()),
            retriever=EmbeddingRetriever(top_k=200),
        )
        recommender.train(interactions_ds=interactions_ds)
        recommendations = recommender.recommend(interactions=df, top_k=10)
    """

    def __init__(self, top_k: int = 100, cold_start_strategy: str = "popular"):
        """
        Args:
            top_k: Number of candidate items to retrieve per user.
                   The ranking model re-scores these candidates and returns
                   the final ``top_k`` recommendations passed to ``recommend()``.
            cold_start_strategy: How to handle users not seen during training.
                ``"popular"`` (default) returns the most popular items from training
                interactions, ranked by interaction count and filtered to the current
                embedding catalog. Falls back to catalog order when no interactions
                were provided to ``build_index()``. ``"zero"`` uses a zero embedding
                vector, which produces an arbitrary (implementation-defined) ordering.
        """
        if cold_start_strategy not in _VALID_COLD_START_STRATEGIES:
            raise ValueError(
                f"cold_start_strategy must be one of {_VALID_COLD_START_STRATEGIES}, got {cold_start_strategy!r}."
            )
        super().__init__(top_k=top_k)
        self.cold_start_strategy = cold_start_strategy
        self._item_matrix: Optional[np.ndarray] = None  # (n_items, k)
        self._item_ids: Optional[np.ndarray] = None
        self._user_emb_by_id: Optional[pd.DataFrame] = None  # cached at build_index time
        self._popular_items: Optional[List[Any]] = None  # used when cold_start_strategy="popular"

    def build_index(
        self,
        estimator: Optional[BaseEmbeddingEstimator] = None,
        interactions: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Build the item embedding index from the fitted estimator.

        Args:
            estimator: Must be a fitted ``BaseEmbeddingEstimator``.
            interactions: Optional training interactions DataFrame. Used only
                when ``cold_start_strategy="popular"`` to rank the popularity
                fallback by interaction count. Must contain ``ITEM_ID_NAME`` if
                provided. Ignored when ``cold_start_strategy="zero"``.
            items: Unused.

        Raises:
            TypeError: If ``estimator`` is not a ``BaseEmbeddingEstimator``.
            ValueError: If the estimator does not support precomputed user
                embeddings (e.g. ``ContextualizedTwoTowerEstimator`` with
                ``context_mode='user_tower'`` and context features — user
                representations are context-dependent in that mode).
        """
        if not isinstance(estimator, BaseEmbeddingEstimator):
            raise TypeError(
                f"EmbeddingRetriever requires a BaseEmbeddingEstimator, "
                f"got {type(estimator).__name__}. "
                f"For non-embedding estimators use PopularityRetriever or ContentBasedRetriever."
            )

        item_emb_df = estimator.get_item_embeddings()

        if item_emb_df.empty:
            logger.warning("get_item_embeddings() returned an empty DataFrame. Index will be empty.")
            self._item_matrix = np.empty((0, 0))
            self._item_ids = np.array([])
            self._user_emb_by_id = pd.DataFrame()
            return

        self._item_ids = item_emb_df[ITEM_ID_NAME].values
        self._item_matrix = np.stack(item_emb_df[ITEM_EMBEDDING_NAME].values).astype(np.float64)
        logger.info("EmbeddingRetriever: indexed %d items (dim=%d).", len(self._item_ids), self._item_matrix.shape[1])

        # Cache user embeddings once — they are static after training.
        # get_user_embeddings() raises NotImplementedError for estimators whose user
        # representations are context-dependent (e.g. ContextualizedTwoTowerEstimator
        # with context_mode='user_tower' and context features). Convert to ValueError
        # so the failure is immediate and actionable.
        try:
            user_emb_df = estimator.get_user_embeddings()
        except NotImplementedError as exc:
            raise ValueError(
                "EmbeddingRetriever cannot precompute user embeddings for this estimator. "
                f"Reason: {exc} "
                "Use predict_proba_with_embeddings() at serving time instead of EmbeddingRetriever, "
                "or switch to a context_mode that supports precomputed user embeddings."
            ) from exc
        self._user_emb_by_id = user_emb_df.set_index(USER_ID_NAME) if not user_emb_df.empty else pd.DataFrame()
        logger.info("EmbeddingRetriever: cached %d user embeddings.", len(self._user_emb_by_id))

        # Build popularity fallback for cold-start users.
        # Items are filtered to the embedding catalog so the scorer always
        # receives candidates it knows about.
        if self.cold_start_strategy == "popular":
            if interactions is not None and not interactions.empty and ITEM_ID_NAME in interactions.columns:
                catalog_set = set(self._item_ids.tolist())
                self._popular_items = [
                    item_id for item_id in interactions[ITEM_ID_NAME].value_counts().index if item_id in catalog_set
                ]
                if not self._popular_items:
                    # All interacted items were outside the embedding catalog — fall back.
                    self._popular_items = self._item_ids.tolist()
            else:
                # No interactions provided — use catalog order as fallback.
                self._popular_items = self._item_ids.tolist()

    def retrieve(self, user_ids: List[Any], top_k: int) -> Dict[Any, List[Any]]:
        """
        Retrieve top-K candidate items for each user via dot-product search.

        Complexity
        ----------
        ``O(n_items × embedding_dim)`` per user. This is a brute-force
        dot-product search; see the class-level "Not for" section for
        guidance on when to switch to an ANN index.

        Args:
            user_ids: User IDs to retrieve candidates for.
            top_k: Number of candidates per user.

        Returns:
            Dict mapping user_id to list of candidate item IDs.

        Note:
            Users not seen during training are handled according to
            ``cold_start_strategy``. The default ``"popular"`` returns the
            globally most popular items from training interactions.
            ``"zero"`` falls back to a zero embedding vector, which produces
            an arbitrary (implementation-defined) ordering — use only when
            "anything is better than nothing" applies for cold-start users.
        """
        if self._item_matrix is None or self._user_emb_by_id is None:
            raise RuntimeError("build_index() must be called before retrieve().")

        if len(self._item_ids) == 0:
            return {uid: [] for uid in user_ids}

        effective_k = min(top_k, len(self._item_ids))

        results: Dict[Any, List[Any]] = {}
        unknown_user_vec = None

        for user_id in user_ids:
            try:
                user_vec = np.asarray(self._user_emb_by_id.loc[user_id, USER_EMBEDDING_NAME], dtype=np.float64).ravel()
            except KeyError:
                if self.cold_start_strategy == "popular":
                    logger.warning(
                        "EmbeddingRetriever: user %s was not seen during training — "
                        "returning popular items (cold_start_strategy='popular').",
                        user_id,
                    )
                    results[user_id] = self._popular_items[:effective_k]
                    continue
                else:
                    # "zero" strategy — arbitrary ordering, kept for explicit opt-in.
                    if unknown_user_vec is None:
                        unknown_user_vec = np.zeros(self._item_matrix.shape[1], dtype=np.float64)
                    user_vec = unknown_user_vec
                    logger.warning(
                        "EmbeddingRetriever: user %s was not seen during training — "
                        "returning arbitrary candidates via zero-vector fallback "
                        "(cold_start_strategy='zero').",
                        user_id,
                    )

            scores = self._item_matrix @ user_vec  # (n_items,)
            top_indices = self._topk_indices(scores, effective_k)
            results[user_id] = self._item_ids[top_indices].tolist()

        return results
