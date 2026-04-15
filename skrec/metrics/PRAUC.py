from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score

from skrec.metrics.base_metric import BaseClassificationMetric
from skrec.metrics.datatypes import RecommenderMetricType


class PRAUCMetric(BaseClassificationMetric):
    """
    Calculates the Area Under the Precision-Recall Curve (PR AUC), also known
    as Average Precision (AP).

    This metric summarizes the Precision-Recall curve and is particularly useful
    for imbalanced datasets where the number of relevant items (positives) is small.
    It expects `modified_rewards` to represent binary ground truth labels
    (values > 0.5 are treated as 1, others as 0), typically from a
    ReplayMatchEvaluator or SimpleEvaluator. `recommendation_scores` are used
    as the prediction scores. NaNs in `modified_rewards` are ignored.
    """

    TYPE = RecommenderMetricType.PR_AUC

    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,
        top_k: Optional[int] = None,
    ) -> float:
        """
        Calculates the PR AUC / Average Precision score after binarizing modified_rewards.

        Args:
            recommendation_ranks: Rank of each item (unused). Shape (N, n_items).
            modified_rewards: Ground truth labels (values > 0.5 -> 1, else 0). NaNs ignored.
                              Shape (N, n_items).
            recommendation_scores: Prediction scores (e.g., logits). Shape (N, n_items).
            top_k: Cutoff parameter (unused).

        Returns:
            The PR AUC / Average Precision score, or 0.0 if calculation is not possible.
        """
        # Flatten arrays and create mask for valid (non-NaN) ground truth labels
        y_true_flat = modified_rewards.ravel()
        y_score_flat = recommendation_scores.ravel()
        valid_mask = ~np.isnan(y_true_flat)

        if not np.any(valid_mask):
            return 0.0  # No valid labels

        y_true_valid = y_true_flat[valid_mask]
        y_score_valid = y_score_flat[valid_mask]

        # Enforce that rewards are in [0, 1] — counterfactual evaluators (IPS, DR, SNIPS)
        # produce values outside this range and are not compatible with classification metrics.
        if np.any((y_true_valid < 0) | (y_true_valid > 1)):
            raise ValueError(
                "PR AUC requires modified_rewards in [0, 1]. Got values outside this range. "
                "Counterfactual evaluators (IPS, DR, SNIPS) are not compatible with classification metrics. "
                "Use SimpleEvaluator or ReplayMatchEvaluator."
            )

        # Binarize the ground truth labels
        y_true_binarized = (y_true_valid > self.THRESHOLD).astype(int)

        # average_precision_score handles cases with only one class gracefully (returns NaN or 0 depending on class)
        # We'll convert potential NaN result to 0.0
        ap_score = average_precision_score(y_true=y_true_binarized, y_score=y_score_valid)
        return float(np.nan_to_num(ap_score, nan=0.0))
