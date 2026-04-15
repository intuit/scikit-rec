# Recommender Evaluators (Off-Policy Evaluation)

This directory contains different strategies for **off-policy evaluation** (also known as **counterfactual evaluation**) of recommender systems. These methods estimate how a new target recommendation policy would perform using only historical interaction data collected under a different (logging) policy.

## Evaluation Strategies

*   **`SimpleEvaluator`:** A naive off-policy method. It uses the actual reward from the logs if an item was logged for a user, and **0** otherwise. It makes the strong assumption that the logged action was the only one possible or relevant, ignoring potential selection bias. The zero-filling implies unobserved items yield zero reward.
*   **`ReplayMatchEvaluator` (RM):** Another naive off-policy method, often called the "Replay" method. Similar to `SimpleEvaluator`, it uses the actual reward from the logs if an item was logged, but fills with **`np.nan`** otherwise. The NaN-filling explicitly marks unobserved items as having unknown reward based on the logs.
*   **`DirectMethodEvaluator` (DM):** Evaluates a direct reward model (`expected_rewards`) itself, ignoring logged rewards and propensities. This model predicts the expected reward for *any* item, given the context. This evaluator assesses the quality of this reward model directly. The reward model is often a component of more advanced counterfactual estimators like DR.
*   **`IPSEvaluator` (IPS):** A classic counterfactual estimator using Inverse Propensity Scoring (importance sampling). It weights the logged reward by the ratio of the target policy's probability to the logging policy's probability (`target_proba / logging_proba`). This aims to correct for the distribution shift (selection bias) between the logging and target policies but can suffer from high variance if propensities are small. Unobserved items implicitly have zero contribution after weighting.
*   **`SNIPSEvaluator` (SNIPS):** A variant of the IPS counterfactual estimator using Self-Normalized Importance Sampling. It normalizes the importance weights (`target_proba / logging_proba`) per user/instance by their mean. This aims to reduce the variance compared to standard IPS, potentially at the cost of introducing some bias.
*   **`DREvaluator` (DR):** A Doubly Robust counterfactual estimator. It combines a direct estimate (using the `DirectMethodEvaluator`'s underlying `expected_rewards` model) with an IPS-based correction term applied only to logged items. It aims to provide an unbiased estimate with lower variance than IPS if *either* the reward model *or* the propensity model is correctly specified.
*   **`PolicyWeightedEvaluator`:** A counterfactual estimator that weights the logged rewards by the `target_proba` (similar to SNIPS) but assumes a uniform logging probability (similar to RM/Simple), effectively ignoring `logging_proba`. It normalizes these weights (`target_proba`) per user/instance by their mean. This can be seen as a simplified SNIPS under strong logging assumptions.

## Comparison

| Evaluator                 | Type             | Uses `logged_rewards` | Uses `logging_proba` | Uses `target_proba` | Uses `expected_rewards` | Normalization   | Fill Value (Unlogged) | Key Goal / Assumption                                      |
| :------------------------ | :--------------- | :--------------------: | :------------------: | :-----------------: | :---------------------: | :-------------: | :--------------------: | :--------------------------------------------------------- |
| `SimpleEvaluator`         | Naive Off-Policy |           ✓            |          ✕           |          ✕          |            ✕            |        ✕        |           0            | Logged action only possibility; Unobserved = 0 reward      |
| `ReplayMatchEvaluator`    | Naive Off-Policy |           ✓            |          ✕           |          ✕          |            ✕            |        ✕        |         `NaN`          | Logged action only possibility; Unobserved = Unknown reward |
| `DirectMethodEvaluator`   | Direct Model     |           ✕            |          ✕           |          ✕          |            ✓            |        ✕        |  `expected_reward`    | Evaluate reward model quality directly                     |
| `IPSEvaluator`            | Counterfactual   |           ✓            |          ✓           |          ✓          |            ✕            |        ✕        |         `NaN`          | Correct selection bias via propensities (Unbiased, High Var) |
| `SNIPSEvaluator`          | Counterfactual   |           ✓            |          ✓           |          ✓          |            ✕            | IPS Weights     |         `NaN`          | Reduce IPS variance via normalization (Biased, Lower Var)  |
| `DREvaluator`             | Counterfactual   |           ✓            |          ✓           |          ✓          |            ✓            |        ✕        |   `expected_reward`    | Robust to one model misspecification (Unbiased, Lower Var) |
| `PolicyWeightedEvaluator` | Counterfactual   |           ✓            |          ✕           |          ✓          |            ✕            | Target Proba    |         `NaN`          | Simplified SNIPS assuming uniform logging (Biased)         |


| Evaluator                 | Typical Metric     | logged_rewards | logging_proba | target_proba | expected_rewards | Notes |
| :------------------------ | :----------------- | :------------: | :-----------: | :----------: | :--------------: | :---- |
| `SimpleEvaluator`         | Ranking@k          |       ✓        |       ✕       |       ✕      |        ✕         | Unobserved = 0  |
| `ReplayMatchEvaluator`    | Classif, Ranking@1 |       ✓        |       ✕       |       ✕      |        ✕         | Unobserved = Skip |
| `DirectMethodEvaluator`   | Policy             |       ✕        |       ✕       |       ✕      |        ✓         | Use reward model |
| `IPSEvaluator`            | Policy             |       ✓        |       ✓       |       ✓      |        ✕         | Correct selection bias via propensities |
| `SNIPSEvaluator`          | Policy             |       ✓        |       ✓       |       ✓      |        ✕         | Reduce IPS variance via normalization |
| `DREvaluator`             | Policy             |       ✓        |       ✓       |       ✓      |        ✓         | DirectMethod+IPS |
| `PolicyWeightedEvaluator` | Policy             |       ✓        |       ✕       |       ✓      |        ✕         | Simplified SNIPS assuming uniform logging |
