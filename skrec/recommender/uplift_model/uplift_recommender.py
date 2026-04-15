from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.recommender.base_recommender import BaseRecommender
from skrec.recommender.uplift_model.uplift_scorer_adapter import (
    UpliftMode,
    UpliftScorerAdapter,
)
from skrec.scorer.base_scorer import BaseScorer
from skrec.scorer.independent import IndependentScorer
from skrec.scorer.universal import UniversalScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class UpliftRecommender(BaseRecommender):
    def __init__(self, scorer: BaseScorer, control_item_id: str, mode: Optional[str] = None):
        """Uplift recommender supporting T-Learner, S-Learner, and X-Learner.

        Args:
            scorer: An IndependentScorer (for T/X-Learner) or UniversalScorer (for S-Learner).
            control_item_id: The item ID representing the control (no-treatment) group.
            mode: Explicit uplift mode. One of ``"t_learner"``, ``"s_learner"``,
                or ``"x_learner"``. When ``None`` (default), auto-detected from
                the scorer type: IndependentScorer → T-Learner,
                UniversalScorer → S-Learner. Must be set explicitly for X-Learner.
        """
        if mode == "x_learner":
            if not isinstance(scorer, IndependentScorer):
                raise TypeError("X-Learner requires IndependentScorer.")
            uplift_mode = UpliftMode.X_LEARNER
        elif mode is not None and mode in ("t_learner", "s_learner"):
            uplift_mode = UpliftMode(mode)
        elif mode is None:
            if isinstance(scorer, UniversalScorer):
                uplift_mode = UpliftMode.S_LEARNER
            elif isinstance(scorer, IndependentScorer):
                uplift_mode = UpliftMode.T_LEARNER
            else:
                raise TypeError(
                    f"Scorer of type {type(scorer).__name__} is not supported for UpliftRecommender. "
                    "Please use UniversalScorer (for S-Learner) or IndependentScorer (for T-Learner)."
                )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 't_learner', 's_learner', or 'x_learner'.")

        factory = None
        est_dict = None
        if isinstance(scorer.estimator, dict):
            est_dict = scorer.estimator
        else:

            def factory_func():
                return scorer.estimator

            factory = factory_func

        adapter = UpliftScorerAdapter(
            scorer_class=type(scorer),
            mode=uplift_mode,
            control_item_id=control_item_id,
            estimator_factory=factory,
            estimator_dict=est_dict,
        )
        super().__init__(adapter)
        self.control_item_id = control_item_id

    def _get_item_names(self) -> NDArray[np.str_]:
        """
        Returns item names in the order [treatments, control].
        This matches the order of columns in the augmented scores returned by score_items + control.
        """
        adapter = self.scorer
        if adapter.item_subset is not None:
            # When item_subset is set, use only those treatments plus control
            treatments = [t for t in adapter.treatment_item_ids if t in adapter.item_subset]
        else:
            treatments = adapter.treatment_item_ids

        # Return treatments followed by control
        return np.array(treatments + [self.control_item_id], dtype=np.str_)

    def _recommend_from_scores(self, scores: NDArray[np.float64], top_k: int = 1) -> NDArray[np.int_]:
        # scores is a numpy array with shape (n_users, n_treatments)
        # Add control scores (zeros) as an additional column
        num_users = scores.shape[0]
        control_scores = np.zeros((num_users, 1))

        # Augment scores with control (control is appended as the last column)
        augmented_scores = np.hstack([scores, control_scores])

        # Sort indices in descending order of scores
        sorted_idx = augmented_scores.argsort(axis=1)[:, ::-1][:, :top_k]

        # Check for non-positive recommendations and log info if needed
        recommended_scores = np.take_along_axis(augmented_scores, sorted_idx, axis=1)
        positive_counts_per_user = np.sum(recommended_scores > 0, axis=1)

        if np.any(positive_counts_per_user < top_k):
            num_affected_users = np.sum(positive_counts_per_user < top_k)
            logger.info(
                f"For {num_affected_users} user(s), the number of recommendations "
                f"with positive uplift was less than top_k={top_k}."
            )

        return sorted_idx
