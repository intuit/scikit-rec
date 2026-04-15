from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.evaluator.base_evaluator import BaseRecommenderEvaluator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.evaluator.utils import calculate_propensity_ratio


class SNIPSEvaluator(BaseRecommenderEvaluator):
    """
    Evaluator using the Self-Normalized Inverse Propensity Score (SNIPS) approach.

    .. warning::
        **Do not use SNIPS with ranking metrics that apply a @k cutoff**
        (e.g. Precision@10, NDCG@5).  SNIPS relies on global normalization
        across all logged interactions.  A @k cutoff restricts the sum to
        top-ranked items, breaking the normalization invariant and producing
        biased estimates.  Use ``SimpleEvaluator`` or ``ReplayMatchEvaluator``
        with ranking metrics instead.

    The standard SNIPS estimator is:
        SNIPS = Σ(reward_i × w_i) / Σ(w_i)
    where w_i = target_proba_i / logging_proba_i for each logged interaction i.

    This implementation stores per-item values `reward_i × w_i / mean(w)` in the
    modified rewards matrix and relies on `ExpectedRewardMetric` (which calls
    `np.nanmean`) to produce the final scalar. The algebra is equivalent:

        nanmean(r_i × w_i / mean(w))
        = (1/L) × Σ(r_i × w_i) / ((1/L) × Σ(w_i))
        = Σ(r_i × w_i) / Σ(w_i)   ← standard SNIPS

    This decomposition is required to fit the (N, n_items) matrix interface shared
    by all evaluators. The result is only meaningful when consumed by
    ExpectedRewardMetric. Using SNIPS with ranking metrics (@k cutoff) breaks the
    global normalization and is not recommended — a UserWarning is issued in that case.

    modified_reward[u,i] = reward[u,i] × w[u,i] / global_mean_weight  if logged
    modified_reward[u,i] = NaN                                          otherwise
    """

    TYPE = RecommenderEvaluatorType.SNIPS
    PRESERVES_LOGGED_REWARD = False

    def _calculate_modified_rewards(
        self,
        logged_items: NDArray,
        logged_rewards: NDArray,
        target_proba: NDArray,
        expected_rewards: Optional[NDArray] = None,  # Ignored by SNIPS
        logging_proba: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Calculates the modified rewards matrix using the SNIPS approach.

        Args:
            logged_items: Array of shape (N, L_max) with logged item indices
                (0 to n_items-1) or padding (-1).
            logged_rewards: Array of shape (N, L_max) with actual rewards for
                logged items, aligned with `logged_items`.
            target_proba: Array of shape (N, n_items) with the target policy's
                probability P(action | context) for *all* items.
            expected_rewards: Optional array of shape (N, n_items). Ignored by SNIPS.
            logging_proba: Optional array of shape (N, L_max) with the logging policy's
                probability P(logged_action | context) for each logged item, aligned
                with `logged_items`. Required for SNIPS.

        Returns:
            A NumPy array of shape (N, n_items) containing the SNIPS modified rewards.
            Values will be np.nan for items not logged or if normalization results
            in NaN or infinity.

        Raises:
            ValueError: If logging_proba is None.
        """
        # 1. Check required inputs
        if logging_proba is None:
            raise ValueError("Logging propensities (logging_proba) are required for SNIPSEvaluator.")

        # 2. Validate shapes and get dimensions
        N, n_items, L_max = self._validate_input_shapes(
            target_proba=target_proba,
            logged_items=logged_items,
            logged_rewards=logged_rewards,
            logging_proba=logging_proba,
            expected_rewards=expected_rewards,  # Pass for validation consistency
        )

        # 3. Identify valid logged interactions (non-padded)
        valid_log_mask = logged_items != self.ITEM_PAD_VALUE  # Shape (N, L_max)

        if not np.any(valid_log_mask):
            return np.full((N, n_items), np.nan, dtype=np.float64)

        # 4. Calculate Weights & Unnormalized Rewards for valid logs
        row_indices_valid, log_col_indices_valid = np.where(valid_log_mask)
        col_indices_valid = logged_items[row_indices_valid, log_col_indices_valid].astype(int)  # Item indices

        valid_rewards = logged_rewards[row_indices_valid, log_col_indices_valid]
        valid_logging_proba = logging_proba[row_indices_valid, log_col_indices_valid]
        valid_target_proba = target_proba[row_indices_valid, col_indices_valid]

        # Calculate IPS weights for valid logs using the utility function
        ips_weights_valid = calculate_propensity_ratio(valid_target_proba, valid_logging_proba)

        # Calculate unnormalized IPS rewards *only for valid logs*
        ips_rewards_valid_logs = valid_rewards * ips_weights_valid

        # 5. Calculate Global Normalization Factor (Global Mean Weight)
        # Use nanmean to handle potential infinities gracefully
        global_mean_weight = np.nanmean(ips_weights_valid)

        # 6. Calculate final SNIPS rewards for valid logs
        snips_rewards_valid_logs = ips_rewards_valid_logs / global_mean_weight

        # 7. Scatter the final SNIPS rewards into the (N, n_items) matrix
        modified_rewards = np.full((N, n_items), np.nan, dtype=np.float64)
        modified_rewards[row_indices_valid, col_indices_valid] = snips_rewards_valid_logs

        return modified_rewards
