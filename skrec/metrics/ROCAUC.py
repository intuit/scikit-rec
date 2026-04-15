from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from skrec.metrics.base_metric import BaseClassificationMetric
from skrec.metrics.datatypes import RecommenderMetricType


class ROCAUCMetric(BaseClassificationMetric):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    This metric treats the recommendation problem as a binary classification task,
    evaluating the model's ability to rank relevant items higher than non-relevant ones
    across all possible thresholds. It expects `modified_rewards` to represent
    binary ground truth labels (values > 0.5 are treated as 1, others as 0),
    typically from a ReplayMatchEvaluator or SimpleEvaluator. `recommendation_scores`
    are used as the prediction scores. NaNs in `modified_rewards` are ignored.
    """

    TYPE = RecommenderMetricType.ROC_AUC

    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,
        top_k: Optional[int] = None,
    ) -> float:
        """
        Calculates the ROC AUC score after binarizing modified_rewards.

        Args:
            recommendation_ranks: Rank of each item (unused). Shape (N, n_items).
            modified_rewards: Ground truth labels (values > 0.5 -> 1, else 0). NaNs ignored.
                              Shape (N, n_items).
            recommendation_scores: Prediction scores (e.g., logits). Shape (N, n_items).
            top_k: Cutoff parameter (unused).

        Returns:
            The ROC AUC score, or 0.0 if calculation is not possible (e.g., only one class present after binarization).
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
                "ROC AUC requires modified_rewards in [0, 1]. Got values outside this range. "
                "Counterfactual evaluators (IPS, DR, SNIPS) are not compatible with classification metrics. "
                "Use SimpleEvaluator or ReplayMatchEvaluator."
            )

        # Binarize the ground truth labels
        y_true_binarized = (y_true_valid > self.THRESHOLD).astype(int)

        if y_true_binarized.min() == y_true_binarized.max():
            # ROC AUC is not defined if only one class is present
            return 0.0

        # Use the binarized labels for AUC calculation
        auc_score = roc_auc_score(y_true=y_true_binarized, y_score=y_score_valid)
        return float(auc_score)
