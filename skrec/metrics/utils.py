import re
from typing import Tuple

from skrec.metrics.datatypes import RecommenderMetricType


def _get_base_name_from_enum(enum_member: RecommenderMetricType) -> str:
    """Derives the base name (e.g., 'NDCG') from an enum member (e.g., NDCG_AT_K)."""
    name = enum_member.name
    if name.endswith("_AT_K"):
        base = name.replace("_AT_K", "")
        # Handle special case: PRECISION_AT_K -> PRECISION
        return "PRECISION" if base == "RECOMMENDER_PRECISION" else base
    else:
        return name


def parse_metric_name(metric_name_full: str) -> Tuple[RecommenderMetricType, int]:
    """
    Parses a metric name string (e.g., "NDCG@10", "Precision@5", "MAP", "roc_auc")
    into its corresponding RecommenderMetricType enum and integer k.

    Args:
        metric_name_full: The full metric name string (case-insensitive for base name).

    Returns:
        A tuple containing the RecommenderMetricType enum and the integer k.
        k defaults to 1 if not specified in the name.

    Raises:
        ValueError: If the base metric name does not match any known metric type.
    """
    metric_name_base_input = metric_name_full
    k = 1  # Default k value

    match = re.search(r"@(\d+)$", metric_name_full)
    if match:
        k = int(match.group(1))
        metric_name_base_input = metric_name_full[: match.start()]

    # Normalize input base name for comparison (case-insensitive)
    normalized_input_base = metric_name_base_input.upper()

    # Find the matching enum member based on the base name
    for enum_member in RecommenderMetricType:
        enum_base_name = _get_base_name_from_enum(enum_member)
        if enum_base_name == normalized_input_base:
            return enum_member, k
    raise ValueError(f"Invalid metric base name: '{metric_name_base_input}'")
