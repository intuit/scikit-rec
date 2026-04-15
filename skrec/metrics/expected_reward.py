from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.metrics.base_metric import BasePolicyMetric
from skrec.metrics.datatypes import RecommenderMetricType


class ExpectedRewardMetric(BasePolicyMetric):
    """
    Calculates the Expected Reward across all items based on modified rewards.

    This metric computes the simple average of the `modified_rewards` provided
    by the evaluator (e.g., IPS, DR, DirectMethod estimates) across all items
    and all instances. It represents the estimated average reward of the policy
    being evaluated. It ignores the `top_k` parameter.
    """

    TYPE = RecommenderMetricType.EXPECTED_REWARD

    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,
        top_k: Optional[int] = None,
    ) -> float:
        """
        Calculates the mean of the modified rewards, ignoring NaNs.

        Args:
            recommendation_ranks: Rank of each item (unused). Shape (N, n_items).
            modified_rewards: Evaluator-specific reward estimates for each item.
                              NaNs are ignored. Shape (N, n_items).
            recommendation_scores: Raw recommendation scores (unused). Shape (N, n_items).
            top_k: Cutoff parameter (unused).

        Returns:
            The mean modified reward across all items and instances, or 0.0 if all are NaN.
        """
        # Calculate the mean across all items and instances, ignoring NaNs
        mean_reward = np.nanmean(modified_rewards)

        # Return 0.0 if the result is NaN (e.g., all inputs were NaN)
        return float(np.nan_to_num(mean_reward, nan=0.0))
