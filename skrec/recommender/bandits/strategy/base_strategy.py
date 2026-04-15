from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class BaseStrategy(ABC):
    """Abstract base class for contextual bandit exploration strategies.

    A strategy takes raw model scores and item names, and returns the top-k
    recommended item indices — optionally injecting exploration (e.g.
    epsilon-greedy random selection, static forced actions).
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def rank(self, scores: NDArray, item_names: NDArray, top_k: int = 1) -> Tuple[NDArray[np.int_], NDArray]:
        # Takes a list of scores and returns the top_k recommended action indices
        # relative to the item_names array.
        # Also returns a StrategyFlag to indicate what path the strategy took (e.g. explore vs exploit)
        pass

    def get_blended_probabilities(self, base_item_probabilities: NDArray, item_names: NDArray) -> NDArray:
        """Blend base item probabilities with strategy-specific exploration logic.

        Strategies that support probabilistic blending (e.g. EpsilonGreedy) override
        this method. Strategies that do not (e.g. StaticAction) leave this default in
        place — ContextualBanditsRecommender guards against calling it for such strategies
        before this method is ever reached.

        Args:
            base_item_probabilities: A 2D array (n_users x n_items) of probabilities,
                                     typically derived from model scores via softmax.
                                     The order of items in this array corresponds to `item_names`.
            item_names: A 1D array of item names, defining the order of items
                        in `base_item_probabilities` and the output.

        Returns:
            A 2D array (n_users x n_items) representing the final blended
            probability distribution over all items, aligned with `item_names`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support get_blended_probabilities(). "
            "Use a probabilistic strategy such as EpsilonGreedy instead."
        )

    def _validate_scores(self, scores: NDArray, item_names: NDArray) -> NDArray:
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        if scores.shape[1] != len(item_names):
            raise ValueError(f"Shape of scores {scores.shape} does not match length of item_names {len(item_names)}")

        return scores
