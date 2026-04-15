from typing import Dict, Type

from skrec.metrics.base_metric import BaseRecommenderMetric
from skrec.metrics.datatypes import RecommenderMetricType
from skrec.metrics.expected_reward import ExpectedRewardMetric

# Import concrete metric classes
from skrec.metrics.MAP import MAPMetric
from skrec.metrics.MRR import MRRMetric
from skrec.metrics.NDCG import NDCGMetric
from skrec.metrics.PRAUC import PRAUCMetric
from skrec.metrics.recall import RecallMetric
from skrec.metrics.recommender_precision import PrecisionMetric
from skrec.metrics.ROCAUC import ROCAUCMetric


class RecommenderMetricFactory:
    _METRIC_MAP: Dict[RecommenderMetricType, Type[BaseRecommenderMetric]] = {
        PrecisionMetric.TYPE: PrecisionMetric,
        RecallMetric.TYPE: RecallMetric,
        MAPMetric.TYPE: MAPMetric,
        MRRMetric.TYPE: MRRMetric,
        NDCGMetric.TYPE: NDCGMetric,
        ExpectedRewardMetric.TYPE: ExpectedRewardMetric,
        ROCAUCMetric.TYPE: ROCAUCMetric,
        PRAUCMetric.TYPE: PRAUCMetric,
        # Add alias for AVERAGE_REWARD_AT_K -> PrecisionMetric
        RecommenderMetricType.AVERAGE_REWARD_AT_K: PrecisionMetric,
    }

    @classmethod
    def create(cls, flavor: RecommenderMetricType) -> BaseRecommenderMetric:
        try:
            metric_class = cls._METRIC_MAP[flavor]
            return metric_class()
        except KeyError:
            raise KeyError(f"Cannot find metric with type '{flavor}'")
