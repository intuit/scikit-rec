from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.metrics.base_metric import BaseRegressionMetric
from skrec.metrics.datatypes import RecommenderMetricType


class RMSEMetric(BaseRegressionMetric):
    """Root Mean Squared Error.

    Treats ``recommendation_scores`` as predicted continuous values and
    ``modified_rewards`` as the ground truth. NaNs in ``modified_rewards``
    are ignored. Lower is better; ``0.0`` is a perfect predictor.

    Used by :class:`~skrec.scorer.multioutput.MultioutputScorer` in
    regressor mode for per-target prediction error.
    """

    TYPE = RecommenderMetricType.RMSE

    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,
        top_k: Optional[int] = None,
    ) -> float:
        y_true = modified_rewards.ravel()
        y_pred = recommendation_scores.ravel()
        valid = ~np.isnan(y_true)
        if not np.any(valid):
            return float("nan")
        diff = y_pred[valid] - y_true[valid]
        return float(np.sqrt(np.mean(diff * diff)))
