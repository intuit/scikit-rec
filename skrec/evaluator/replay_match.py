from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.evaluator.base_evaluator import BaseRecommenderEvaluator
from skrec.evaluator.datatypes import RecommenderEvaluatorType


class ReplayMatchEvaluator(BaseRecommenderEvaluator):
    """
    Evaluator using the "Replay Match" approach.
    Commonly known as "Replay Method" when used with a deterministic reward@1 metric.

    The modified reward for an item is the actual logged reward if that specific
    item was present in the logs for that instance, and `np.nan` otherwise.
    This is calculated across all possible items.

    modified_reward[u,i] = reward[u,i] if logged else NaN
    """

    TYPE = RecommenderEvaluatorType.REPLAY_MATCH
    PRESERVES_LOGGED_REWARD = True

    def _calculate_modified_rewards(
        self,
        logged_items: NDArray,
        logged_rewards: NDArray,
        target_proba: Optional[NDArray] = None,
        expected_rewards: Optional[NDArray] = None,
        logging_proba: Optional[NDArray] = None,
        n_items: Optional[int] = None,
    ) -> NDArray:
        """
        Calculates the modified rewards matrix for all items using the Replay Match approach.

        Args:
            logged_items: Array of shape (N, L_max) with logged item indices
                (0 to n_items-1) or padding (-1).
            logged_rewards: Array of shape (N, L_max) with actual rewards for
                logged items, aligned with `logged_items`.
            target_proba: Array of shape (N, n_items). Used for shape validation.
                          Ignored by ReplayMatchEvaluator logic.
            expected_rewards: Optional array of shape (N, n_items). Ignored by ReplayMatchEvaluator.
            logging_proba: Optional array of shape (N, L_max). Ignored by ReplayMatchEvaluator.

        Returns:
            A NumPy array of shape (N, n_items) containing the modified rewards.
            Values will be np.nan for items not found in logs for a given instance.
        """

        if target_proba is None and n_items is None:
            raise ValueError("target_proba or n_items are required for ReplayMatchEvaluator")

        # 1. Validate shapes and get dimensions
        N, n_items, L_max = self._validate_input_shapes(
            target_proba=target_proba,
            logged_items=logged_items,
            logged_rewards=logged_rewards,
            logging_proba=logging_proba,
            expected_rewards=expected_rewards,
            n_items=n_items,
        )

        # 2. Initialize modified rewards matrix with np.nan
        modified_rewards = np.full((N, n_items), np.nan, dtype=np.float64)

        # 3. Identify valid logged interactions (non-padded)
        valid_log_mask = logged_items != self.ITEM_PAD_VALUE

        # 4. Scatter logged rewards into the modified rewards matrix
        if np.any(valid_log_mask):
            # Get row and column indices for valid logs
            row_indices, _ = np.where(valid_log_mask)
            # Get column indices (item index) for valid logs
            col_indices = logged_items[valid_log_mask].astype(int)
            # Get the corresponding rewards
            rewards_to_scatter = logged_rewards[valid_log_mask]

            # Use advanced indexing to place rewards. If an item is logged multiple
            # times for the same user, the last value overwrites previous ones.
            modified_rewards[row_indices, col_indices] = rewards_to_scatter

        return modified_rewards
