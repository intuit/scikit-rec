from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.metrics.base_metric import BaseRankingMetric
from skrec.metrics.datatypes import RecommenderMetricType


class MRRMetric(BaseRankingMetric):
    """
    Calculates the Mean Reciprocal Rank (MRR).

    MRR is the average of the reciprocal ranks of the *first* relevant item
    found in the top-k recommendations for each instance (user).
    If no relevant item is found within the top-k, the reciprocal rank is 0.
    Assumes binary relevance (0 or 1) from `modified_rewards`.
    """

    TYPE = RecommenderMetricType.MRR_AT_K

    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,
        top_k: Optional[int] = None,
    ) -> float:
        """
        Calculates the Mean Reciprocal Rank (MRR) score efficiently using ranks.

        Args:
            recommendation_ranks: Rank of each item (0 is best). Shape (N, n_items).
            modified_rewards: Evaluator-specific relevance/reward for each item.
                              Assumed to be binary (0 or 1). NaNs are treated as 0.
                              Shape (N, n_items).
            recommendation_scores: Raw recommendation scores (unused by this metric).
                                   Shape (N, n_items).
            top_k: The number of top recommendations to consider (k). If None,
                   all items are considered.

        Returns:
            The averaged MRR@k score over all instances.
        """
        N, n_items = modified_rewards.shape

        # Enforce binary relevance requirement
        valid_mask = ~np.isnan(modified_rewards)
        if valid_mask.any():
            valid_values = modified_rewards[valid_mask]
            if not np.all((valid_values == 0) | (valid_values == 1)):
                raise ValueError(
                    "MRR requires binary rewards (0 or 1). Got non-binary values in modified_rewards. "
                    "Use NDCG_AT_K for graded or continuous rewards."
                )

        # 1. Determine effective k and handle edge cases
        if top_k is None:
            k = n_items
        else:
            k = min(top_k, n_items)
        if k <= 0:
            return 0.0

        # 2. Prepare masks and relevance
        top_k_mask = recommendation_ranks < k  # (N, n_items) - True for items in top k ranks
        is_relevant = np.nan_to_num(modified_rewards, nan=0.0) == 1  # (N, n_items) - True for relevant items
        is_relevant_in_top_k = is_relevant & top_k_mask  # (N, n_items) - True for relevant items in top k

        # 3. Find the rank of the *first* relevant item within top-k for each row
        #    - Set ranks of non-relevant items or items outside top-k to a large value (e.g., k)
        #    - Find the minimum rank for each row. This gives the 0-based rank of the first relevant item.
        ranks_of_relevant_in_top_k = np.where(is_relevant_in_top_k, recommendation_ranks, k)  # Shape (N, n_items)
        first_relevant_rank = np.min(ranks_of_relevant_in_top_k, axis=1)  # Shape (N,)

        # 4. Calculate reciprocal rank (1 / (rank + 1))
        #    If first_relevant_rank is k, it means no relevant item was found in top-k, RR is 0.
        reciprocal_rank = np.where(first_relevant_rank < k, 1.0 / (first_relevant_rank + 1), 0.0)  # Shape (N,)

        # 5. Calculate Mean Reciprocal Rank (MRR)
        mrr_score = np.mean(reciprocal_rank)

        return float(mrr_score)
