from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.evaluator.base_evaluator import BaseRecommenderEvaluator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.evaluator.utils import calculate_propensity_ratio


class IPSEvaluator(BaseRecommenderEvaluator):
    """
    Evaluator using the Inverse Propensity Score (IPS) approach.

    For one user u with logged item i:
    reward[u] = reward[u,i] * target_proba[u,i] / logging_proba[u,i]

    Therefore we fill modified_reward[u,i] with NaN except the logged entries.
    """

    TYPE = RecommenderEvaluatorType.IPS
    PRESERVES_LOGGED_REWARD = False

    def __init__(self, trim_threshold: Optional[float] = None):
        """
        Initializes the IPS evaluator.

        Args:
            trim_threshold: Optional positive float. If provided, IPS weights
                            (target_proba / logging_proba) exceeding this threshold
                            will result in a modified reward of np.nan for that item.
        """
        if trim_threshold is not None and trim_threshold <= 0:
            raise ValueError("trim_threshold must be positive.")
        self.trim_threshold = trim_threshold

    def _calculate_modified_rewards(
        self,
        logged_items: NDArray,
        logged_rewards: NDArray,
        target_proba: NDArray,
        expected_rewards: Optional[NDArray] = None,  # Ignored by IPS
        logging_proba: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Calculates the modified rewards matrix for all items using the IPS approach.

        Modified reward = actual_reward * target_P(action|x) / logging_P(action|x)
        if the action (item) was logged, np.nan otherwise.

        Args:
            logged_items: Array of shape (N, L_max) with logged item indices
                (0 to n_items-1) or padding (-1).
            logged_rewards: Array of shape (N, L_max) with actual rewards for
                logged items, aligned with `logged_items`.
            target_proba: Array of shape (N, n_items) with the target policy's
                probability P(action | context) for *all* items.
            expected_rewards: Optional array of shape (N, n_items). Ignored by IPS.
            logging_proba: Optional array of shape (N, L_max) with the logging policy's
                probability P(logged_action | context) for each logged item, aligned
                with `logged_items`. Required for IPS.

        Returns:
            A NumPy array of shape (N, n_items) containing the modified rewards.
            Values will be np.nan if trimming is applied, propensities are zero/inf,
            or the item was not logged.

        Raises:
            ValueError: If logging_proba is None.
        """
        if logging_proba is None:
            raise ValueError("Logging propensities (logging_proba) are required for IPSEvaluator.")

        # 1. Validate shapes and get dimensions
        N, n_items, L_max = self._validate_input_shapes(
            target_proba=target_proba,
            logged_items=logged_items,
            logged_rewards=logged_rewards,
            logging_proba=logging_proba,
            expected_rewards=expected_rewards,  # Pass for validation consistency
        )

        # 2. Initialize modified rewards matrix with np.nan
        modified_rewards = np.full((N, n_items), np.nan, dtype=np.float64)

        # 3. Identify valid logged interactions (non-padded)
        valid_log_mask = logged_items != self.ITEM_PAD_VALUE

        # 4. Calculate and scatter IPS rewards for logged items
        if np.any(valid_log_mask):
            # Get row and column indices for valid logs
            row_indices, log_col_indices = np.where(valid_log_mask)
            items = logged_items[row_indices, log_col_indices].astype(int)  # Get the actual item indices

            # Get corresponding data for valid logs using the correct indices
            valid_rewards = logged_rewards[row_indices, log_col_indices]
            valid_logging_proba = logging_proba[row_indices, log_col_indices]
            # Get target probabilities for the *specific logged items*
            valid_target_proba = target_proba[row_indices, items]

            # Calculate IPS weights using the utility function
            ips_weights = calculate_propensity_ratio(valid_target_proba, valid_logging_proba)

            # Calculate raw IPS rewards for logged items: reward * weight
            raw_ips_rewards = valid_rewards * ips_weights

            # Apply trimming if needed
            if self.trim_threshold is not None:
                trim_mask = ips_weights > self.trim_threshold
                raw_ips_rewards = np.where(trim_mask, np.nan, raw_ips_rewards)

            # Replace potential infinities with NaN
            raw_ips_rewards[np.isinf(raw_ips_rewards)] = np.nan

            # Scatter calculated IPS rewards into the main matrix
            modified_rewards[row_indices, items] = raw_ips_rewards

        return modified_rewards
