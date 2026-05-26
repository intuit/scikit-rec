# Multiclass top-1 accuracy metric.
#
# v2 introduces a single multiclass metric for the MixedTypeMultiTargetScorer
# evaluation contract. Log-loss, macro-F1, and other multiclass metrics are
# reachable via score_per_target's user-supplied callable path; only the
# canonical top-1 lands as a named RecommenderMetricType in v2.

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skrec.metrics.base_metric import BaseMulticlassMetric
from skrec.metrics.datatypes import RecommenderMetricType


class MulticlassAccuracy(BaseMulticlassMetric):
    """Top-1 multiclass accuracy.

    Computes ``mean(argmax(scores, axis=1) == labels)`` on rows where the
    label is non-NaN. Returns ``NaN`` if every row has a NaN label.

    Contract (inherited from :class:`BaseMulticlassMetric`):
      - ``recommendation_scores``: ``(n, K)`` class probabilities (or any
        per-class score). Column order is the training-time catalogue.
      - ``modified_rewards``: ``(n,)`` class indices. NaN rows are masked.
      - ``recommendation_ranks`` / ``top_k`` are unused.
    """

    TYPE = RecommenderMetricType.MULTICLASS_ACCURACY

    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,
        top_k: Optional[int] = None,
    ) -> float:
        labels = np.asarray(modified_rewards).reshape(-1)
        scores = np.asarray(recommendation_scores)
        if scores.ndim != 2:
            raise ValueError(
                f"MulticlassAccuracy requires recommendation_scores of shape (n, K); got shape {scores.shape}."
            )
        if scores.shape[0] != labels.shape[0]:
            raise ValueError(
                f"recommendation_scores has {scores.shape[0]} rows but modified_rewards has {labels.shape[0]} rows."
            )
        # Mask NaN labels.
        try:
            valid = ~np.isnan(labels.astype(np.float64))
        except (TypeError, ValueError):
            # Non-numeric labels (strings, etc.) — assume no NaN.
            valid = np.ones_like(labels, dtype=bool)
        if not np.any(valid):
            return float("nan")
        labels_valid = labels[valid]
        scores_valid = scores[valid]
        preds = scores_valid.argmax(axis=1)
        # If labels are class indices (int-like), direct compare; if strings,
        # caller is responsible for pre-mapping. v2 keeps the metric scalar
        # and dtype-agnostic — equality semantics handle both.
        return float(np.mean(preds == labels_valid))
