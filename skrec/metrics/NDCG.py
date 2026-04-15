from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.metrics.base_metric import BaseRankingMetric
from skrec.metrics.datatypes import RecommenderMetricType


class NDCGMetric(BaseRankingMetric):
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG).

    NDCG measures the quality of the ranking by comparing the Discounted
    Cumulative Gain (DCG) of the recommended list to the Ideal DCG (IDCG),
    which represents the best possible ranking. Relevance scores are taken
    from the `modified_rewards` matrix.
    """

    TYPE = RecommenderMetricType.NDCG_AT_K

    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,  # Not used by NDCG, but part of standard signature
        top_k: Optional[int] = None,
    ) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) score.

        Args:
            recommendation_ranks: Rank of each item (0 is best). Shape (N, n_items).
            modified_rewards: Evaluator-specific relevance/reward for each item.
                              NaNs are treated as 0 relevance. Shape (N, n_items).
            recommendation_scores: Raw recommendation scores (unused by this metric).
                                   Shape (N, n_items).
            top_k: The number of top recommendations to consider (k). If None,
                   all items are considered.

        Returns:
            The averaged NDCG@k score over all instances.
        """
        N, n_items = modified_rewards.shape

        # Determine k
        if top_k is None:
            k = n_items
        else:
            k = min(top_k, n_items)

        if k <= 0:
            return 0.0

        # Create mask for items ranked within top-k
        top_k_mask = recommendation_ranks < k  # Shape (N, n_items)

        # Calculate discounts based on rank (add 1 because ranks are 0-based)
        # Apply discount only to items within top-k
        # Add small epsilon to avoid log2(1) = 0 for rank 0 if needed, but log2(rank+2) avoids this.
        ranks_for_discount = recommendation_ranks + 2  # +2 because discount starts from log2(2) for rank 0
        discounts = np.log2(ranks_for_discount)  # Shape (N, n_items)

        # Calculate DCG@k for each row
        # Get rewards for items in top-k, treat NaNs as 0
        rewards_no_nan = np.nan_to_num(modified_rewards, nan=0.0)
        # Calculate discounted gain only for items in top-k
        discounted_gains = np.where(top_k_mask, rewards_no_nan / discounts, 0)
        dcg = np.sum(discounted_gains, axis=1)  # Shape (N,)

        # Calculate Ideal DCG@k (IDCG@k) for each row
        # Sort *all* modified_rewards per row to find the ideal ordering
        # Treat NaNs as having minimal relevance (-inf) for sorting purposes
        ideal_sorted_rewards = np.sort(np.nan_to_num(modified_rewards, nan=-np.inf), axis=1)[
            :, ::-1
        ]  # Shape (N, n_items)

        # Take the top k rewards from the ideal ranking
        ideal_top_k_rewards = ideal_sorted_rewards[:, :k]  # Shape (N, k)

        # Prepare ideal discounts (log base 2 for positions 1 to k)
        ideal_discounts = np.log2(np.arange(2, k + 2))  # Shape (k,)

        # Calculate IDCG@k
        # NaNs were already handled for sorting, ensure 0 for calculation
        ideal_rewards_no_nan = np.nan_to_num(ideal_top_k_rewards, nan=0.0, neginf=0.0)
        idcg = np.sum(ideal_rewards_no_nan / ideal_discounts, axis=1)  # Shape (N,)

        # Calculate NDCG@k for each row, handling division by zero (IDCG=0 means NDCG=0)
        # If IDCG is 0, it means all top-k ideal rewards were <= 0.
        ndcg_per_row = np.divide(dcg, idcg, out=np.zeros_like(dcg, dtype=float), where=idcg != 0)

        # Return the average NDCG@k over all instances
        # Use nanmean just in case any unexpected NaNs occurred
        return float(np.nanmean(ndcg_per_row))
