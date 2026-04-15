from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from skrec.recommender.bandits.datatypes import StrategyFlag
from skrec.recommender.bandits.strategy.base_strategy import BaseStrategy
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class StaticAction(BaseStrategy):
    def __init__(self, ranked_item_names: NDArray):
        # The ranked_item_names should be passed in the order they will be recommended (e.g. best first)
        super().__init__()
        self.ranked_item_names = ranked_item_names
        if sorted(list(self.ranked_item_names)) == list(self.ranked_item_names):
            logger.warning(
                "For static action strategy, ranked_item_names should be passed in best-to-worst order,"
                " but they are in alphabetical order. This could be a coincidence, but it is worth checking."
            )

    def rank(self, scores: NDArray, item_names: NDArray, top_k: int = 1) -> Tuple[NDArray[np.int_], NDArray]:
        scores = self._validate_scores(scores, item_names)

        n_rows = scores.shape[0]

        # Get the top_k item names from the pre-defined ranked list
        top_k_ranked_names = self.ranked_item_names[
            :top_k
        ]  # if top_k > len(self.ranked_item_names), this will return the entire ranked_item_names

        # Convert these names to indices based on the provided `item_names` (scorer's list)
        # Create a mapping from item name to its index in the scorer's list
        scorer_item_to_idx_map = {name: i for i, name in enumerate(item_names)}

        ranked_item_indices_for_top_k = []
        for name in top_k_ranked_names:
            if name not in scorer_item_to_idx_map:
                raise ValueError(f"Item '{name}' from static ranking not found in scorer's item list")
            ranked_item_indices_for_top_k.append(scorer_item_to_idx_map[name])

        # Convert to NDArray and tile for each row
        ranked_item_indices_1d = np.array(ranked_item_indices_for_top_k, dtype=np.int_)
        final_ranked_item_indices = np.tile(ranked_item_indices_1d, [n_rows, 1])

        flags = np.array([StrategyFlag.EXPLOIT.value] * n_rows)
        return final_ranked_item_indices, flags
