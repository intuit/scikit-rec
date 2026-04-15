from typing import Optional

from numpy.typing import NDArray

from skrec.evaluator.base_evaluator import BaseRecommenderEvaluator
from skrec.evaluator.datatypes import RecommenderEvaluatorType


class DirectMethodEvaluator(BaseRecommenderEvaluator):
    """
    Evaluator using the Direct Method approach.

    This method directly uses the predicted expected rewards from a reward model
    as the basis for evaluation, ignoring the actual logged rewards and propensities.
    It essentially evaluates the quality of the reward model itself.

    For one user u:
    reward[u] = sum_i exp_reward[u,i] * target_proba[u,i]

    However we need to return one entry per pair of [u,i], and metrics will average not sum.
    So we return:
    modified_reward[u,i] = n_items * exp_reward[u,i] * target_proba[u,i]
    """

    TYPE = RecommenderEvaluatorType.DIRECT_METHOD
    PRESERVES_LOGGED_REWARD = False

    def _calculate_modified_rewards(
        self,
        logged_items: NDArray,
        logged_rewards: NDArray,
        target_proba: NDArray,
        expected_rewards: Optional[NDArray] = None,
        logging_proba: Optional[NDArray] = None,  # Ignored by DirectMethod
    ) -> NDArray:
        """
        Calculates the modified rewards matrix using the Direct Method.

        Returns the provided `expected_rewards` directly after validation.

        Args:
            logged_items: Array of shape (N, L_max). Used for shape validation only.
            logged_rewards: Array of shape (N, L_max). Used for shape validation only.
            target_proba: Array of shape (N, n_items) with the target policy's
                probability P(action | context) for *all* items.
            expected_rewards: Optional array of shape (N, n_items) with the model's
                predicted E[Reward | context, action] for *all* items. Required.
            logging_proba: Optional array of shape (N, L_max). Ignored by DirectMethod.

        Returns:
            A NumPy array of shape (N, n_items) containing the expected rewards.
        """
        # 1. Check required inputs
        if expected_rewards is None:
            raise ValueError("Expected rewards are required for DirectMethodEvaluator")

        # 2. Validate shapes and get dimensions (ensures consistency)
        self._validate_input_shapes(
            target_proba=target_proba,
            logged_items=logged_items,
            logged_rewards=logged_rewards,
            logging_proba=logging_proba,  # Pass for validation consistency
            expected_rewards=expected_rewards,
        )

        # 3. De-normalize by multiplying with n_items so that `mean` will recover the `sum`
        n_items = expected_rewards.shape[1]
        return n_items * expected_rewards * target_proba
