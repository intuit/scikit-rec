from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.metrics.base_metric import BaseRankingMetric
from skrec.metrics.datatypes import RecommenderMetricType


class PrecisionMetric(BaseRankingMetric):
    """
    Calculates the Precision@k, also interpretable as Average Reward@k.

    It computes the mean of the `modified_rewards` for the items ranked
    within the top-k recommendations. If `modified_rewards` are binary (0/1),
    this is equivalent to Precision@k. If `modified_rewards` represent
    estimated rewards (e.g., from IPS/DR), this calculates the average
    estimated reward among the top-k items.
    """

    TYPE = RecommenderMetricType.PRECISION_AT_K

    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,
        top_k: Optional[int] = None,
    ) -> float:
        """
        Calculates the Precision@k / Average Reward@k score.

        Args:
            recommendation_ranks: Rank of each item (0 is best). Shape (N, n_items).
            modified_rewards: Evaluator-specific relevance/reward for each item.
                              NaNs are ignored in the mean calculation. Shape (N, n_items).
            recommendation_scores: Raw recommendation scores (unused by this metric).
                                   Shape (N, n_items).
            top_k: The number of top recommendations to consider (k). If None,
                   all items are considered.

        Returns:
            The averaged Precision@k / Average Reward@k score over all instances.
        """
        N, n_items = modified_rewards.shape

        # 1. Determine effective k and handle edge cases
        if top_k is None:
            k = n_items
        else:
            k = min(top_k, n_items)
        if k <= 0:
            return 0.0

        # 2. Create mask for items ranked within top-k
        top_k_mask = recommendation_ranks < k  # Shape (N, n_items)

        # 3. Select rewards for items within top-k, treating others as NaN
        #    so they are ignored by nanmean.
        rewards_in_top_k = np.where(top_k_mask, modified_rewards, np.nan)  # Shape (N, n_items)

        # 4. Calculate the mean reward per instance, ignoring NaNs.
        #    This effectively calculates the mean over the top-k items.
        #    np.nanmean returns NaN for all-NaN slices, which is desired.
        mean_reward_per_row = np.nanmean(rewards_in_top_k, axis=1)  # Shape (N,)

        # 5. Calculate the overall mean across instances.
        #    Use nanmean again to handle cases where some instances might have had no items in top-k
        #    (although with k>0 this shouldn't happen unless n_items=0) or all rewards were NaN.
        overall_mean_reward = np.nanmean(mean_reward_per_row)

        # Return 0.0 if the final result is NaN (e.g., no valid rewards found)
        return float(np.nan_to_num(overall_mean_reward, nan=0.0))
