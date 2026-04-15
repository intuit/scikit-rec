from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.metrics.base_metric import BaseRankingMetric
from skrec.metrics.datatypes import RecommenderMetricType


class MAPMetric(BaseRankingMetric):
    """
    Calculates the Mean Average Precision (MAP).

    MAP is the mean of the Average Precision (AP) scores calculated for each
    instance (user). AP summarizes a precision-recall curve into a single
    value, representing the average precision at different recall levels.
    It's particularly useful when the number of relevant items varies.
    Assumes binary relevance (0 or 1) from `modified_rewards`.
    """

    TYPE = RecommenderMetricType.MAP_AT_K

    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,
        top_k: Optional[int] = None,
    ) -> float:
        """
        Calculates the Mean Average Precision (MAP) score efficiently using ranks.

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
            The averaged MAP@k score over all instances.

        Note:
            In the standard MAP formula, the precision denominator at rank r is r (the
            rank position itself). This implementation uses the count of non-NaN items
            up to rank r instead. When all items have known rewards (no NaNs, e.g. with
            SimpleEvaluator), this is identical to standard MAP. When rewards are partially
            observed (NaN for unlogged items, e.g. with ReplayMatchEvaluator), NaN slots
            are excluded from the denominator — treating unlogged items as unknown rather
            than irrelevant. This is the appropriate behavior for offline bandit evaluation.
        """
        N, n_items = modified_rewards.shape

        # Enforce binary relevance requirement
        valid_mask = ~np.isnan(modified_rewards)
        if valid_mask.any():
            valid_values = modified_rewards[valid_mask]
            if not np.all((valid_values == 0) | (valid_values == 1)):
                raise ValueError(
                    "MAP requires binary rewards (0 or 1). Got non-binary values in modified_rewards. "
                    "Use NDCG_AT_K for graded or continuous rewards."
                )

        # 1. Determine effective k and handle edge cases
        if top_k is None:
            k = n_items
        else:
            k = min(top_k, n_items)
        if k <= 0:
            return 0.0

        # 2. Prepare basic masks and relevance
        top_k_mask = recommendation_ranks < k  # (N, n_items) - True for items in top k ranks
        is_relevant = np.nan_to_num(modified_rewards, nan=0.0) == 1  # (N, n_items) - True for relevant items
        is_relevant_in_top_k = is_relevant & top_k_mask  # (N, n_items) - True for relevant items in top k

        # 3. Calculate denominator for AP: total relevant items in top-k per user
        num_relevant_in_top_k = np.sum(is_relevant_in_top_k, axis=1)  # (N,)

        # Handle cases where there are no relevant items in top-k to avoid division by zero later
        if np.all(num_relevant_in_top_k == 0):
            return 0.0

        # 4. Prepare validity mask and sort by rank
        is_valid = ~np.isnan(modified_rewards)  # (N, n_items) - True for non-NaN items
        valid_by_rank = np.zeros((N, n_items), dtype=float)
        np.put_along_axis(valid_by_rank, recommendation_ranks, is_valid.astype(float), axis=1)
        cumulative_valid_at_rank = np.cumsum(
            valid_by_rank, axis=1
        )  # (N, n_items) - Cumulative count of valid items up to rank i

        # 5. Get the cumulative valid count corresponding to *each item's specific rank*
        count_valid_le_item_rank = np.take_along_axis(
            cumulative_valid_at_rank, recommendation_ranks, axis=1
        )  # (N, n_items)

        # 6. Prepare relevance sorted by rank
        # is_relevant defined in step 2
        relevance_by_rank = np.zeros((N, n_items), dtype=float)
        np.put_along_axis(relevance_by_rank, recommendation_ranks, is_relevant.astype(float), axis=1)
        cumulative_relevance_at_rank = np.cumsum(relevance_by_rank, axis=1)  # (N, n_items)

        # 7. Get the cumulative relevance count corresponding to *each item's specific rank*
        count_relevant_le_item_rank = np.take_along_axis(
            cumulative_relevance_at_rank, recommendation_ranks, axis=1
        )  # (N, n_items)

        # 8. Calculate precision at each item's rank (using valid count as denominator)
        # Denominator is the count of valid (non-NaN) items up to and including the item's rank
        precision_denominator = count_valid_le_item_rank
        precision_at_item_rank = np.divide(
            count_relevant_le_item_rank,
            precision_denominator,
            out=np.zeros_like(count_relevant_le_item_rank, dtype=float),
            where=precision_denominator > 0,
        )  # (N, n_items)

        # 9. Calculate the sum of precisions *only for relevant items within top-k*
        # is_relevant_in_top_k defined in step 2
        precision_sum_terms = precision_at_item_rank * is_relevant_in_top_k
        sum_precision = np.sum(precision_sum_terms, axis=1)  # (N,)

        # 10. Calculate Average Precision (AP) per user
        # Denominator is total relevant items in top-k (calculated in step 3)
        # num_relevant_in_top_k defined in step 3
        ap_per_row = np.divide(
            sum_precision,
            num_relevant_in_top_k,
            out=np.zeros_like(sum_precision, dtype=float),
            where=num_relevant_in_top_k != 0,
        )

        # 11. Calculate Mean Average Precision (MAP)
        map_score = np.mean(ap_per_row)

        return float(map_score)
