import numpy as np
import pandas as pd

from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_items,
    sample_binary_reward_users,
)
from skrec.metrics.datatypes import RecommenderMetricType
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.universal import UniversalScorer

# Estimator params
xgb_params = {"learning_rate": 0.1, "n_estimators": 10}

# Instantiate Estimate, Scorer, and Recommender
estimator = XGBClassifierEstimator(xgb_params)
scorer = UniversalScorer(estimator)
recommender = RankingRecommender(scorer)

# Evaluation data
interactions_df = pd.DataFrame({"USER_ID": ["user_1", "user_2"]})
users_df = pd.DataFrame({"USER_ID": ["user_1", "user_2"], "feat1": [2000, 0], "feat2": [100, 0.1]})
rec_top_k = 3
eval_top_k = 3
eval_data = pd.DataFrame(
    data={
        "logged_rewards": [np.array([0]), np.array([1])],
        "logged_items": [np.array(["ITEM_3"]), np.array(["ITEM_2"])],
    }
)


if __name__ == "__main__":
    # Training
    recommender.train(
        interactions_ds=sample_binary_reward_interactions,
        users_ds=sample_binary_reward_users,
        items_ds=sample_binary_reward_items,
    )

    # Get Recommendations

    recommendations = recommender.recommend(interactions=interactions_df, users=users_df, top_k=3)
    scores = recommender.scorer.score_items(interactions=interactions_df, users=users_df)
    print(f"Recommendations = {recommendations}")
    print(f"Scores = {scores}")

    # Evaluate

    # Evaluation - Calculation
    simple_eval_value_p_at_k = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.PRECISION_AT_K,
        eval_top_k=eval_top_k,
        eval_kwargs=eval_data,
        recommend_kwargs={"top_k": rec_top_k, "interactions": interactions_df, "users": users_df},
    )
    print(f"Simple Evaluation Metric - Precision at k  = {simple_eval_value_p_at_k}")

    # We can change the metric type using the same evaluator, and skip eval_kwargs and recommend_kwargs
    # This will skip the "recommendation computation, and use the previously calculated values"

    simple_eval_value_reward_at_k = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.AVERAGE_REWARD_AT_K,
        eval_top_k=eval_top_k,
    )
    print(f"Simple Evaluation Metric - Reward at k  = {simple_eval_value_reward_at_k}")

    # But we cannot change the evaluator type and skip eval_kwargs and recommend_kwargs
    # We will get the old result with a warning. This is incorrect usage!

    incorrect_rm_eval_p_at_k = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.REPLAY_MATCH,
        metric_type=RecommenderMetricType.PRECISION_AT_K,
        eval_top_k=eval_top_k,
    )

    print(f"Incorrect Replay Match Evaluation - Precision at k  = {incorrect_rm_eval_p_at_k}")

    correct_rm_eval_p_at_k = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.REPLAY_MATCH,
        metric_type=RecommenderMetricType.PRECISION_AT_K,
        eval_top_k=eval_top_k,
        eval_kwargs=eval_data,
        recommend_kwargs={"top_k": rec_top_k, "interactions": interactions_df, "users": users_df},
    )

    print(f"Correct Replay Match Evaluation - Precision at k  = {correct_rm_eval_p_at_k}")
