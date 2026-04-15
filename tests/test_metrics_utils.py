import pytest

from skrec.metrics.datatypes import RecommenderMetricType
from skrec.metrics.utils import parse_metric_name


@pytest.mark.parametrize(
    "metric_name_full, expected_type, expected_k",
    [
        ("NDCG@10", RecommenderMetricType.NDCG_AT_K, 10),
        ("Precision@5", RecommenderMetricType.PRECISION_AT_K, 5),
        ("MAP@20", RecommenderMetricType.MAP_AT_K, 20),
        ("MRR@100", RecommenderMetricType.MRR_AT_K, 100),
        ("Average_Reward@3", RecommenderMetricType.AVERAGE_REWARD_AT_K, 3),
        # Test case sensitivity (should be handled by .upper())
        ("ndcg@7", RecommenderMetricType.NDCG_AT_K, 7),
        ("average_reward@1", RecommenderMetricType.AVERAGE_REWARD_AT_K, 1),
        # Test metrics without explicit K (should default to k=1, type needs _AT_K)
        # Assuming the function correctly maps base names to _AT_K enums where appropriate
        # Adjust these based on actual enum definitions if base names don't map directly
        ("NDCG", RecommenderMetricType.NDCG_AT_K, 1),
        ("PRECISION", RecommenderMetricType.PRECISION_AT_K, 1),
        ("MAP", RecommenderMetricType.MAP_AT_K, 1),
        ("MRR", RecommenderMetricType.MRR_AT_K, 1),
        ("AVERAGE_REWARD", RecommenderMetricType.AVERAGE_REWARD_AT_K, 1),
        ("Recall@10", RecommenderMetricType.RECALL_AT_K, 10),
        ("recall@5", RecommenderMetricType.RECALL_AT_K, 5),
        ("RECALL", RecommenderMetricType.RECALL_AT_K, 1),
    ],
)
def test_parse_metric_name_success(metric_name_full, expected_type, expected_k):
    """Tests successful parsing of valid metric names."""
    metric_type, k = parse_metric_name(metric_name_full)
    assert metric_type == expected_type
    assert k == expected_k


@pytest.mark.parametrize(
    "invalid_metric_name",
    [
        "InvalidMetric@10",
        "NDCG@",
        "Precision@-5",
        "MAP@k",
        "FooBar@10",  # Not a defined metric
    ],
)
def test_parse_metric_name_failure(invalid_metric_name):
    """Tests that parsing invalid metric names raises ValueError."""
    with pytest.raises(ValueError):
        parse_metric_name(invalid_metric_name)
