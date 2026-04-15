from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.evaluator.base_evaluator import BaseRecommenderEvaluator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.evaluator.utils import calculate_propensity_ratio


class DREvaluator(BaseRecommenderEvaluator):
    """
    Evaluator using the Doubly Robust (DR) approach.

    Combines a direct model (DM) estimate (expected rewards) with an Inverse
    Propensity Score (IPS) correction term applied only to logged items.

    For one user u with logged item i:
    reward[u] = DM[u] + IPS_correction[u,i]
      = sum_j exp_reward[u,j] * target_proba[u,j]
       + (reward[u,i] - exp_reward[u,i]) * target_proba[u,i] / logging_proba[u,i]

    Therefore we fill modified_reward[u,i] with NaN except the logged entries.
    """

    TYPE = RecommenderEvaluatorType.DR
    PRESERVES_LOGGED_REWARD = False

    def _calculate_modified_rewards(
        self,
        logged_items: NDArray,
        logged_rewards: NDArray,
        target_proba: NDArray,
        expected_rewards: Optional[NDArray] = None,
        logging_proba: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Calculates the modified rewards matrix for all items using the Doubly Robust approach.

        Args:
            logged_items: Array of shape (N, L_max) with logged item indices
                (0 to n_items-1) or padding (-1).
            logged_rewards: Array of shape (N, L_max) with actual rewards for
                logged items, aligned with `logged_items`.
            target_proba: Array of shape (N, n_items) with the target policy's
                probability P(action | context) for *all* items. Required.
            expected_rewards: Optional array of shape (N, n_items) with the model's
                expected reward E[Reward | context, action] for *all* items. Required.
            logging_proba: Optional array of shape (N, L_max) with the logging policy's
                probability P(logged_action | context) for each logged item, aligned
                with `logged_items`. Required.

        Returns:
            A NumPy array of shape (N, n_items) containing the modified rewards.
            Values can be np.nan if propensities are zero or result in infinity.

        Raises:
            ValueError: If any of the required arguments (expected_rewards,
                        logging_proba) are None.
        """
        # 1. Check required inputs
        if expected_rewards is None:
            raise ValueError("Expected rewards (expected_rewards) are required for DREvaluator.")
        if logging_proba is None:
            raise ValueError("Logging propensities (logging_proba) are required for DREvaluator.")

        # 2. Validate shapes and get dimensions
        N, n_items, L_max = self._validate_input_shapes(
            target_proba=target_proba,
            logged_items=logged_items,
            logged_rewards=logged_rewards,
            logging_proba=logging_proba,
            expected_rewards=expected_rewards,
        )

        # 3. Initialize modified rewards matrix with np.nan
        modified_rewards = np.full((N, n_items), np.nan, dtype=np.float64)

        # 4. Identify valid logged interactions (non-padded)
        valid_log_mask = logged_items != self.ITEM_PAD_VALUE

        # 5. Calculate and apply the IPS correction term for logged items
        if np.any(valid_log_mask):
            # Get row and column indices for valid logs
            row_indices, log_col_indices = np.where(valid_log_mask)
            items = logged_items[row_indices, log_col_indices].astype(int)  # Get the actual item indices

            # Get the expected reward over all items (Direct Method term)
            direct_method_term = (expected_rewards * target_proba).sum(axis=1)  # (N,)
            modified_rewards[row_indices, items] = direct_method_term[row_indices]

            # Get corresponding data for valid logs using the correct indices
            valid_rewards = logged_rewards[row_indices, log_col_indices]
            valid_logging_proba = logging_proba[row_indices, log_col_indices]
            # Get target probabilities for the *specific logged items*
            valid_target_proba = target_proba[row_indices, items]
            # Get expected rewards for the *specific logged items*
            valid_expected_rewards = expected_rewards[row_indices, items]

            # Calculate reward difference for the correction term
            reward_diff = valid_rewards - valid_expected_rewards

            # Calculate probability ratio (IPS weight) using the utility function
            proba_ratio = calculate_propensity_ratio(valid_target_proba, valid_logging_proba)

            # Calculate the IPS correction term
            correction_term = proba_ratio * reward_diff

            # Handle infinities in correction term -> NaN
            correction_term[np.isinf(correction_term)] = np.nan

            # Add the correction term to the direct estimate for the logged items
            # Note: valid_expected_rewards here refers to the expected reward *for the logged item*,
            # which is already the base value in modified_rewards at this position.
            # We are adding the correction term to it.
            modified_rewards[row_indices, items] += correction_term

        return modified_rewards
