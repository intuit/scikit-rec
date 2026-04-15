from typing import Tuple

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from skrec.recommender.bandits.datatypes import StrategyFlag
from skrec.recommender.bandits.strategy.base_strategy import BaseStrategy
from skrec.util.logger import get_logger
from skrec.util.numpy_util import index_rows

logger = get_logger(__name__)


class EpsilonGreedy(BaseStrategy):
    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        logger.info(f"Initializing Epsilon Greedy strategy with epsilon = {epsilon} and seed = {seed}")
        super().__init__()
        self.rng = default_rng(seed)
        self.epsilon = epsilon

    def rank(self, scores: NDArray, item_names: NDArray, top_k: int = 1) -> Tuple[NDArray[np.int_], NDArray]:
        scores = self._validate_scores(scores, item_names)

        # Pre-compute random numbers to take advantage of numpy's vectorization
        n_rows, n_items = scores.shape
        rand_nums = self.rng.random(n_rows)
        shuffled_idx = self.rng.permuted(np.tile(np.arange(n_items), [n_rows, 1]), axis=1)

        shuffled_scores = index_rows(scores, shuffled_idx)

        ranked_item_indices = np.empty([n_rows, top_k], dtype=np.int_)
        flags = np.empty(n_rows, dtype=object)

        explore_mask = rand_nums < self.epsilon
        exploit_mask = ~explore_mask

        if explore_mask.any():
            ranked_item_indices[explore_mask, :] = shuffled_idx[explore_mask, :top_k]
            flags[explore_mask] = StrategyFlag.EXPLORE.value

        if exploit_mask.any():
            sorted_idx_within_shuffled = shuffled_scores[exploit_mask, :].argsort(axis=1)[:, ::-1]
            exploited_indices = index_rows(shuffled_idx[exploit_mask, :], sorted_idx_within_shuffled)

            ranked_item_indices[exploit_mask, :] = exploited_indices[:, :top_k]
            flags[exploit_mask] = StrategyFlag.EXPLOIT.value

        return ranked_item_indices, flags

    def get_blended_probabilities(self, base_item_probabilities: NDArray, item_names: NDArray) -> NDArray:
        """
        Blends base item probabilities with a uniform exploration probability.

        For each user/context:
        P_final_user = epsilon * P_explore + (1 - epsilon) * P_base_user

        Args:
            base_item_probabilities: A 2D array (n_users x n_items) of probabilities,
                                     derived from primary scores (e.g. softmax of model scores).
                                     The order of items matches `item_names`.
            item_names: A 1D array of item names, defining the order of items.

        Returns:
            A 2D array (n_users x n_items) where each row is the final blended
            probability distribution over all items for the corresponding user/context,
            aligned with `item_names`.
        """
        n_users, n_items_base = base_item_probabilities.shape
        n_items = len(item_names)

        if n_items_base != n_items:
            raise ValueError(
                f"Mismatch in number of items. Base probabilities have {n_items_base} items, "
                f"while item_names parameter has {n_items} items."
            )

        # Calculate Exploration Probabilities (P_explore)
        # Uniform distribution over all n_items
        uniform_item_probabilities = np.full((1, n_items), 1.0 / n_items)

        # Calculate Blended Final Probabilities (P_final)
        final_item_probabilities = (
            self.epsilon * uniform_item_probabilities + (1 - self.epsilon) * base_item_probabilities
        )
        return final_item_probabilities
