from enum import Enum


class RecommenderMetricType(str, Enum):
    # Ranking Metrics (@k)
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"
    MAP_AT_K = "MAP_at_k"
    MRR_AT_K = "MRR_at_k"
    NDCG_AT_K = "NDCG_at_k"
    AVERAGE_REWARD_AT_K = "average_reward_at_k"

    # Classification Metrics
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"

    # Policy Metric
    EXPECTED_REWARD = "expected_reward"
