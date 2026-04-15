# Recommender Metrics

This module provides various metrics for evaluating recommender systems, designed to work in conjunction with the `recommender.evaluator` module, particularly in off-policy evaluation scenarios.

## Metric Categories

Metrics are broadly categorized based on what aspect of the recommendation performance they evaluate:

1.  **Ranking Metrics:**
    *   **Purpose:** Evaluate the quality of the item ordering produced by the recommender. They measure how well relevant items are ranked higher than non-relevant ones.
    *   **Examples:** `NDCG`, `MAP`, `MRR`, `RecommenderPrecision` (Precision@k / Average Reward@k).
    *   **Inputs:** Primarily use `recommendation_ranks` (0=best rank) to determine item order and `modified_rewards` as the relevance score for each item. Respect the `top_k` parameter.
    *   **Evaluator Interaction:** The interpretation of `modified_rewards` as relevance is crucial. These metrics are most straightforwardly interpreted when used with evaluators like `SimpleEvaluator` or `ReplayMatchEvaluator`, where `modified_rewards` directly reflect logged interactions (e.g., 0/1 relevance). Using them with counterfactual estimators (IPS, DR) is possible, but the `modified_rewards` then represent estimated relevance or reward, which might require careful interpretation.

2.  **Policy/Reward Metrics:**
    *   **Purpose:** Estimate the overall value or average reward of the recommendation policy across all items, as estimated by a counterfactual evaluator.
    *   **Examples:** `ExpectedRewardMetric`.
    *   **Inputs:** Primarily use the `modified_rewards` matrix, calculating an aggregate statistic (like the mean) over all items. They typically ignore `recommendation_ranks`, `recommendation_scores`, and `top_k`.
    *   **Evaluator Interaction:** Designed to be used with counterfactual estimators like `IPSEvaluator`, `DREvaluator`, or `DirectMethodEvaluator`. The `modified_rewards` from these evaluators represent the estimated reward for each item under the target policy.

3.  **Classification Metrics:**
    *   **Purpose:** Evaluate the recommender's ability to discriminate between relevant and non-relevant items, framing the problem as a binary classification task.
    *   **Examples:** `ROCAUCMetric`, `PRAUCMetric`.
    *   **Inputs:** Use `recommendation_scores` as the prediction scores and `modified_rewards` as the ground truth labels. They ignore `recommendation_ranks` and `top_k`.
    *   **Evaluator Interaction:** These metrics fundamentally require ground truth labels. Therefore, they should only be used with `SimpleEvaluator` or `ReplayMatchEvaluator`. In this context, `modified_rewards` represent the actual observed outcome (e.g., 1 for a click/purchase, 0 otherwise, NaN if not observed). Using classification metrics with other evaluators where `modified_rewards` are estimates will likely produce misleading results (warnings should be issued by the evaluator).

## Summary Table

| Metric Category    | Examples       | Primary Inputs Used                         | `top_k` | Typical Evaluators                    | `modified_rewards` Interpretation |
| :----------------- | :------------- | :------------------------------------------ | :------ | :------------------------------------ | :--- |
| **Ranking**        | NDCG, MAP, MRR | `recommendation_ranks`, `modified_rewards`  |    ✓    | Simple, (ReplayMatch)                 | Relevance |
| **Policy/Reward**  | ExpectedReward | `modified_rewards`                          |    ✕    | IPS, DR, DirectMethod, PolicyWeighted | Weighted Reward |
| **Classification** | ROCAUC, PRAUC  | `recommendation_scores`, `modified_rewards` |    ✕    | ReplayMatch, (Simple)                 | Ground Truth Label |

**Note:** Using metrics with evaluators outside their "Typical" pairing is possible but requires careful consideration of how `modified_rewards` are interpreted. The `BaseRecommenderEvaluator` includes warnings for common mismatches.
