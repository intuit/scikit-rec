from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.evaluator.base_evaluator import BaseRecommenderEvaluator
from skrec.evaluator.datatypes import RecommenderEvaluatorType


class PolicyWeightedEvaluator(BaseRecommenderEvaluator):
    """
    Evaluator using a Policy-Weighted approach, normalized globally.

    This evaluator weights the logged rewards by the target policy's probability
    (`target_proba`) for the logged items. It assumes a uniform logging policy,
    similar to Replay Match, thus ignoring `logging_proba`.

    The normalization is adapted from the SNIPS but assuming uniform sampling:
    the weighted rewards (`reward * target_proba`) are divided by the *global mean*
    of the valid `target_proba` values across all instances and logged items.

    For one user u with logged item i:
    reward[u] = reward[u,i] * target_proba[u,i] / mean(target_proba)

    Therefore we fill modified_reward[u,i] with NaN except the logged entries.
    """

    TYPE = RecommenderEvaluatorType.POLICY_WEIGHTED
    PRESERVES_LOGGED_REWARD = False

    def _calculate_modified_rewards(
        self,
        logged_items: NDArray,
        logged_rewards: NDArray,
        target_proba: NDArray,
        expected_rewards: Optional[NDArray] = None,  # Ignored
        logging_proba: Optional[NDArray] = None,  # Ignored
    ) -> NDArray:
        """
        Calculates the modified rewards matrix using the Policy-Weighted approach.

        Args:
            logged_items: Array of shape (N, L_max) with logged item indices
                (0 to n_items-1) or padding (-1).
            logged_rewards: Array of shape (N, L_max) with actual rewards for
                logged items, aligned with `logged_items`.
            target_proba: Array of shape (N, n_items) with the target policy's
                probability P(action | context) for *all* items. Required.
            expected_rewards: Optional array of shape (N, n_items). Ignored.
            logging_proba: Optional array of shape (N, L_max). Ignored.

        Returns:
            A NumPy array of shape (N, n_items) containing the modified rewards.
            Values will be np.nan for items not logged or if normalization results
            in NaN or infinity.
        """
        # 1. Validate shapes and get dimensions
        # Note: logging_proba is ignored but passed for consistent validation signature
        N, n_items, L_max = self._validate_input_shapes(
            target_proba=target_proba,
            logged_items=logged_items,
            logged_rewards=logged_rewards,
            logging_proba=logging_proba,
            expected_rewards=expected_rewards,
        )

        # 2. Identify valid logged interactions (non-padded)
        valid_log_mask = logged_items != self.ITEM_PAD_VALUE  # Shape (N, L_max)

        if not np.any(valid_log_mask):
            return np.full((N, n_items), np.nan, dtype=np.float64)

        # 3. Calculate Weights (target_proba) & Unnormalized Rewards for valid logs
        row_indices_valid, log_col_indices_valid = np.where(valid_log_mask)
        col_indices_valid = logged_items[row_indices_valid, log_col_indices_valid].astype(int)  # Item indices

        valid_rewards = logged_rewards[row_indices_valid, log_col_indices_valid]
        # Get target probabilities for the *specific logged items*
        target_proba_valid = target_proba[row_indices_valid, col_indices_valid]

        # Calculate unnormalized policy-weighted rewards *only for valid logs*
        unnorm_rewards_valid_logs = valid_rewards * target_proba_valid

        # 4. Calculate Global Normalization Factor (Global Mean Target Proba)
        # Use nanmean as target_proba should be finite (0 to 1)
        global_mean_target_proba = np.nanmean(target_proba_valid)

        # 5. Calculate final normalized rewards for valid logs
        # Standard division handles 0, nan in global_mean_target_proba appropriately
        normalized_rewards_valid_logs = unnorm_rewards_valid_logs / global_mean_target_proba

        # 6. Scatter the final rewards into the (N, n_items) matrix
        modified_rewards = np.full((N, n_items), np.nan, dtype=np.float64)
        modified_rewards[row_indices_valid, col_indices_valid] = normalized_rewards_valid_logs

        return modified_rewards
