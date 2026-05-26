from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from numpy.typing import NDArray

from skrec.metrics.datatypes import RecommenderMetricType


class BaseRecommenderMetric(ABC):
    """Base class for all recommender evaluation metrics."""

    TYPE: ClassVar[RecommenderMetricType]

    @abstractmethod
    def calculate(
        self,
        recommendation_ranks: NDArray,
        modified_rewards: NDArray,
        recommendation_scores: NDArray,
        top_k: Optional[int] = None,
    ) -> float:
        """
        Calculates the metric score.

        Args:
            recommendation_ranks: Rank of each item (0 is best) based on recommendation_scores.
                                  Shape (N, n_items).
            modified_rewards: Evaluator-specific values for each item. Interpretation depends
                              on the evaluator (e.g., 0/1 relevance for Simple/DM, estimated
                              reward for IPS/DR, predicted reward for DirectMethod).
                              Implementations should handle NaNs appropriately (e.g., ignore).
                              Shape (N, n_items).
            recommendation_scores: Raw recommendation scores. Primarily used by classification
                                   metrics as prediction scores. Shape (N, n_items).
            top_k: Cutoff for ranking metrics. If None, consider all items where applicable.
                   Should typically be ignored by Classification and policy-level Reward metrics.

        Returns:
            The calculated metric score (float).
        """
        pass


class BaseRankingMetric(BaseRecommenderMetric):
    """
    Base class for metrics evaluating recommendation ranking quality.

    These metrics typically use `recommendation_ranks` and `modified_rewards`
    (interpreted as relevance) up to a specified `top_k`. They generally
    do not use `recommendation_scores` directly. `top_k` is usually required.
    """


class BaseClassificationMetric(BaseRecommenderMetric):
    """
    Base class for metrics evaluating recommendation as a classification task.

    These metrics typically use `recommendation_scores` (as predictions) and
    `modified_rewards` (interpreted as ground truth labels, usually from
    Simple/RM evaluators). They generally do not use `recommendation_ranks`
    and ignore the `top_k` parameter.
    """

    THRESHOLD = 0.5


class BaseRegressionMetric(BaseRecommenderMetric):
    """Base class for metrics evaluating recommendations as a regression task.

    These metrics treat ``recommendation_scores`` as predicted continuous
    values and ``modified_rewards`` as the corresponding ground-truth
    targets, computing point-wise error statistics (RMSE, MAE, ...). They
    do not use ``recommendation_ranks`` and ignore ``top_k``.

    Used primarily by :class:`~skrec.scorer.multioutput.MultioutputScorer`
    in regressor mode, where each ``ITEM_<name>`` target is continuous and
    per-target prediction error is the natural quality signal.
    """


class BaseMulticlassMetric(BaseRecommenderMetric):
    """Base class for metrics evaluating multiclass classification per target.

    Parallel to :class:`BaseClassificationMetric` but for K>2 classes. The
    contract:
      - ``recommendation_scores``: ``(n, K)`` class probabilities (rows sum
        to 1; column order matches the training-time class catalogue).
      - ``modified_rewards``: ``(n,)`` class indices into the same catalogue.
        NaN-tolerant: rows with NaN labels are masked out.
      - ``recommendation_ranks`` and ``top_k`` are unused.

    Used by :class:`~skrec.scorer.mixed_type_multi_target.MixedTypeMultiTargetScorer`
    for multiclass-typed targets.
    """


class BasePolicyMetric(BaseRecommenderMetric):
    """
    Base class for metrics evaluating the overall policy-level performance.

    These metrics typically use only `modified_rewards` (interpreted as
    estimated rewards from counterfactual evaluators like IPS, DR, DM).
    They generally do not use `recommendation_ranks`, `recommendation_scores`,
    and ignore the `top_k` parameter.
    """
