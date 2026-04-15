from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.metrics.base_metric import BaseRankingMetric
from skrec.metrics.datatypes import RecommenderMetricType


class RecallMetric(BaseRankingMetric):
    """
    Calculates Recall@k.

    Recall@k = (# relevant items in top-k) / (# total relevant items for user).

    A relevant item is one whose ``modified_rewards`` value is non-NaN and
    positive (> 0).  Users with no relevant items at all are excluded from the
    average (they contribute NaN, handled by ``nanmean``). If every user has
    zero relevant items, ``nanmean`` is undefined and this metric returns
    ``nan`` (not coerced to ``0.0``).
    """

    TYPE = RecommenderMetricType.RECALL_AT_K

    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,
        top_k: Optional[int] = None,
    ) -> float:
        """
        Calculates the Recall@k score.

        Args:
            recommendation_ranks: Rank of each item (0 is best). Shape (N, n_items).
            modified_rewards: Evaluator-specific relevance/reward for each item.
                              NaN entries are treated as non-relevant. Shape (N, n_items).
            recommendation_scores: Raw recommendation scores (unused by this metric).
                                   Shape (N, n_items).
            top_k: The number of top recommendations to consider (k). If None,
                   all items are considered.

        Returns:
            The averaged Recall@k score over all instances, or ``nan`` when
            there is no valid per-user recall (e.g. all users have zero relevant items).
        """
        N, n_items = modified_rewards.shape

        if top_k is None:
            k = n_items
        else:
            k = min(top_k, n_items)
        if k <= 0:
            return 0.0

        # Relevant items: non-NaN and positive
        relevant = (~np.isnan(modified_rewards)) & (modified_rewards > 0)

        # Total relevant items per user
        total_relevant_per_user = relevant.sum(axis=1).astype(float)  # Shape (N,)

        # Relevant items in top-k
        top_k_mask = recommendation_ranks < k
        relevant_in_top_k = (top_k_mask & relevant).sum(axis=1).astype(float)  # Shape (N,)

        # Recall per user: relevant_in_top_k / total_relevant
        # Users with 0 relevant items → NaN (excluded by nanmean)
        with np.errstate(divide="ignore", invalid="ignore"):
            recall_per_user = np.where(
                total_relevant_per_user > 0,
                relevant_in_top_k / total_relevant_per_user,
                np.nan,
            )

        overall_recall = np.nanmean(recall_per_user)
        return float(overall_recall)
