import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from xgboost import XGBClassifier

from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.classification.multioutput_classifier import (
    MultiOutputClassifierEstimator,
)
from skrec.estimator.classification.xgb_classifier import (
    BatchXGBClassifierEstimator,
    XGBClassifierEstimator,
)
from skrec.estimator.embedding.neural_factorization_estimator import (
    NeuralFactorizationEstimator,
)
from skrec.estimator.regression.xgb_regressor import XGBRegressorEstimator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.evaluator.utils import create_padded_matrix
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_items,
    sample_binary_reward_users,
    sample_continuous_reward_interactions,
    sample_continuous_reward_items,
    sample_continuous_reward_users,
    sample_multi_class_interactions,
    sample_multi_output_interactions,
    sample_multi_output_multi_class_interactions,
)
from skrec.metrics.datatypes import RecommenderMetricType
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.independent import IndependentScorer
from skrec.scorer.multiclass import MulticlassScorer
from skrec.scorer.multioutput import MultioutputScorer
from skrec.scorer.universal import UniversalScorer
from skrec.util.config_loader import load_config
from skrec.util.logger import get_logger
from tests.utils import parse_config

logger = get_logger(__name__)


@pytest.fixture
def setup_fixture(setup_small_datasets):
    estimator_config_dir = Path.cwd() / "skrec/examples/estimators/"
    estimator_config = load_config(estimator_config_dir / "estimator_hyperparameters.yaml")
    _, _, _, setup_small_datasets["xgb_params"] = parse_config(estimator_config, "XGBoostClassifier")
    _, _, _, setup_small_datasets["xgb_early_stopping_params"] = parse_config(
        estimator_config, "XGBoostClassifierEarlyStopping"
    )
    _, _, _, setup_small_datasets["xgb_multioutput_params"] = parse_config(
        estimator_config, "MultiOutputXGBoostClassifier"
    )

    return setup_small_datasets


def test_universal_recommender(setup_fixture):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)

    try:
        universal_recommender.train(
            sample_binary_reward_users,
            sample_binary_reward_items,
            sample_binary_reward_interactions,
        )
        logger.info("Finished Training of a Universal Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Universal Ranking Recommender Failed.")

    assert_array_equal(universal_recommender.users_schema.columns, ["USER_ID", "feat1", "feat2"])
    assert_array_equal(
        universal_recommender.items_schema.columns,
        ["ITEM_ID", "item_feat1", "item_feat2", "item_feat3", "item_feat4", "item_feat5", "item_feat6", "item_feat7"],
    )
    assert_array_equal(universal_recommender.interactions_schema.columns, ["USER_ID", "ITEM_ID", "OUTCOME"])
    assert_array_equal(
        universal_recommender.scorer.estimator.feature_names,
        [
            "feat1",
            "feat2",
            "item_feat1",
            "item_feat2",
            "item_feat3",
            "item_feat4",
            "item_feat5",
            "item_feat6",
            "item_feat7",
        ],
    )

    # recommending
    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [18, 65], "feat2": [0, 1]})
    top_k = 3

    recommended_items = universal_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    print("output of recommend", recommended_items)
    assert recommended_items.shape == (2, 3)
    # Do not copy paste expected answers as part of unittests and insert assert statements

    recommended_items = universal_recommender.recommend(interactions, users, top_k, sampling_temperature=0.9)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    print("output of recommend", recommended_items)
    assert recommended_items.shape == (2, 3)

    recommended_items = universal_recommender.recommend(interactions, users, top_k, sampling_temperature=1)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    print("output of recommend", recommended_items)
    assert recommended_items.shape == (2, 3)

    items_scores = universal_recommender.score_items(interactions, users)
    logger.info("Finished Scoring of a Universal Ranking Recommender.")
    assert items_scores.shape == (2, 3)

    # Even changing the order should work
    users = users[["USER_ID", "feat2", "feat1"]]
    recommended_items = universal_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    assert recommended_items.shape == (2, 3)

    # But missing a feature should not work
    users = users[["USER_ID", "feat2"]]
    expected_error_msg = "Column 'feat1' not found in dataset"
    with pytest.raises(RuntimeError, match=expected_error_msg):
        recommended_items = universal_recommender.recommend(interactions, users, top_k)

    # Again, missing a feature should not work
    users = users[["USER_ID", "feat2"]]
    expected_error_msg = "Column 'feat1' not found in dataset"
    with pytest.raises(RuntimeError, match=expected_error_msg):
        recommended_items = universal_recommender.recommend(interactions, users, top_k)


def test_multiclass_recommender(setup_fixture):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = MulticlassScorer(estimator)
    multiclass_recommender = RankingRecommender(scorer)

    expected_msg = "Users Dataset will not be used in MulticlassScorer."
    with pytest.raises(ValueError, match=expected_msg):
        multiclass_recommender.train(
            sample_binary_reward_users, sample_binary_reward_items, sample_multi_class_interactions
        )

    # training
    try:
        multiclass_recommender.train(interactions_ds=sample_multi_class_interactions)
        logger.info("Finished Training of a Multiclass Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Multiclass Ranking Recommender Failed.")

    # recommending
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [2, 3]})
    interactions = pd.DataFrame(
        {
            "USER_ID": ["Emma", "Jasper"],
            "feat1": [18, 65],
            "feat2": [0, 1],
            "feat3": [1, 1],
            "feat4": [5, 6],
            "feat5": [11, 9],
        }
    )
    top_k = 3

    expected_msg = "For this scorer, users should be set to None!"
    with pytest.raises(ValueError, match=expected_msg):
        multiclass_recommender.recommend(interactions, users, top_k)

    recommended_items = multiclass_recommender.recommend(interactions=interactions, users=None, top_k=top_k)
    logger.info("Finished Recommending of a Multiclass Ranking Recommender.")
    assert recommended_items.shape == (2, 3)

    expected_msg = "This scorer cannot accept Users, set it to None!"
    with pytest.raises(ValueError, match=expected_msg):
        multiclass_recommender.score_items(interactions, users)
    items_scores = multiclass_recommender.score_items(interactions)
    logger.info("Finished Scoring of a Multiclass Ranking Recommender.")
    assert items_scores.shape == (2, 4)


def test_multioutput_recommender(setup_fixture):
    estimator = MultiOutputClassifierEstimator(XGBClassifier, setup_fixture["xgb_multioutput_params"])
    scorer = MultioutputScorer(estimator)
    multioutput_recommender = RankingRecommender(scorer)
    # training
    try:
        multioutput_recommender.train(interactions_ds=sample_multi_output_interactions)
        logger.info("Finished Training of a Multioutput Ranking Recommender.")
    except Exception as e:
        pytest.fail("Training of a Multioutput Ranking Recommender Failed, error: " + str(e))

    # recommending
    list_of_users = ["Emma", "Jasper", "John", "Doe", "Smith", "Anna"]
    list_of_age = [28, 30, 41, 33, 100, 101]
    list_of_income = [10000, 20000, 30000, 40000, 50000, 60000]
    interactions = pd.DataFrame({"USER_ID": list_of_users, "age": list_of_age, "income": list_of_income})
    scores = multioutput_recommender.scorer.predict_classes(interactions=interactions)
    logger.info("Finished Scoring of a Multioutput Ranking Recommender.")
    # predict_classes returns one predicted-class column per item; top_k is ignored
    assert scores.shape == (6, 11)
    recommendations = multioutput_recommender.recommend(interactions=interactions, top_k=3)
    assert recommendations.shape == (6, 11)


def test_multioutput_recommender_with_subset(setup_fixture):
    estimator = MultiOutputClassifierEstimator(XGBClassifier, setup_fixture["xgb_multioutput_params"])
    scorer = MultioutputScorer(estimator)
    multioutput_recommender = RankingRecommender(scorer)
    # training
    try:
        multioutput_recommender.train(interactions_ds=sample_multi_output_interactions)
        logger.info("Finished Training of a Multioutput Ranking Recommender.")
    except Exception as e:
        pytest.fail("Training of a Multioutput Ranking Recommender Failed, error: " + str(e))

    # recommending
    items = ["ITEM_600", "ITEM_965", "ITEM_970"]
    multioutput_recommender.set_item_subset(item_subset=items)
    list_of_users = ["Emma", "Jasper", "John", "Doe", "Smith", "Anna"]
    list_of_age = [28, 30, 41, 33, 100, 101]
    list_of_income = [10000, 20000, 30000, 40000, 50000, 60000]
    interactions = pd.DataFrame({"USER_ID": list_of_users, "age": list_of_age, "income": list_of_income})
    scores = multioutput_recommender.scorer.predict_classes(interactions=interactions)
    logger.info("Finished Scoring of a Multioutput Ranking Recommender.")
    # 3 items, one predicted-class column each
    assert scores.shape == (6, 3)

    # Try fitting with users dataset, this should throw an error
    expected_error_msg = "Item Dataset and User Dataset will not be used in MultioutputScorer."
    with pytest.raises(ValueError, match=expected_error_msg):
        random_ds = sample_multi_output_interactions
        multioutput_recommender.train(interactions_ds=random_ds, users_ds=random_ds)


def test_multi_output_multi_class_recommender(setup_fixture):
    estimator = MultiOutputClassifierEstimator(XGBClassifier, setup_fixture["xgb_multioutput_params"])
    scorer = MultioutputScorer(estimator)
    multioutput_recommender = RankingRecommender(scorer)
    # training
    try:
        multioutput_recommender.train(interactions_ds=sample_multi_output_multi_class_interactions)
        logger.info("Finished Training of a Multioutput Ranking Recommender.")
    except Exception as e:
        pytest.fail("Training of a Multioutput Ranking Recommender Failed, error: " + str(e))

    # recommending
    list_of_users = ["Emma", "Jasper", "John", "Doe", "Smith", "Anna"]
    feat1 = [28, 30, 41, 33, 100, 101]
    feat2 = [10000, 20000, 30000, 40000, 50000, 60000]
    feat3 = [1, 2, 3, 4, 5, 6]

    interactions = pd.DataFrame({"USER_ID": list_of_users, "feat1": feat1, "feat2": feat2, "feat3": feat3})
    item_scores = multioutput_recommender.scorer.predict_classes(interactions=interactions)
    # predict_classes returns one predicted-class column per item: 3 items → (6, 3)
    assert item_scores.shape == (6, 3)

    item_scores = multioutput_recommender.scorer.predict_classes(interactions=interactions)
    assert item_scores.shape == (6, 3)

    # recommend() works for MultioutputScorer — top_k is ignored, all item predictions returned
    interactions_copy = pd.DataFrame({"USER_ID": list_of_users, "feat1": feat1, "feat2": feat2, "feat3": feat3})
    recommendations = multioutput_recommender.recommend(interactions=interactions_copy)
    assert recommendations.shape == (6, 3)

    features = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3}).head(1)
    item_scores = multioutput_recommender.scorer.score_fast(features)
    # score_fast returns one predicted-class column per item
    assert item_scores.shape == (1, 3)
    interactions = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3})

    recommendations = multioutput_recommender.recommend_online(interactions=interactions.head(1))
    assert recommendations.shape == (1, 3)

    # recommend() fails without USER_ID when schema is active
    expected_message = "Column 'USER_ID' not found in dataset"
    with pytest.raises(RuntimeError) as excinfo:
        multioutput_recommender.recommend(interactions=interactions.head(1))
    assert expected_message in str(excinfo.value)


def test_independent_models_recommender(setup_fixture):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = IndependentScorer(estimator)
    independent_models_recommender = RankingRecommender(scorer)
    # training
    try:
        independent_models_recommender.train(
            sample_binary_reward_users, sample_binary_reward_items, sample_binary_reward_interactions
        )
        logger.info("Finished Training of a Item Models Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Item Models Ranking Recommender Failed.")

    # recommending
    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [18, 65], "feat2": [0, 1]})
    top_k = 3

    recommended_items = independent_models_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Item Models Ranking Recommender.")
    assert recommended_items.shape == (2, 3)

    # verify the copied xgboost models are trained separately
    item_names = independent_models_recommender.scorer.item_names
    estimators = independent_models_recommender.scorer.estimator

    assert_array_equal(item_names, np.array(["ITEM_1", "ITEM_2", "ITEM_3"]))
    assert_array_equal(np.array(list(estimators.keys())), item_names)
    for estimator in list(estimators.values()):
        isinstance(estimator, XGBClassifierEstimator) is True


def test_independent_models_recommender_continuous_rewards(setup_fixture):
    estimator = XGBRegressorEstimator(setup_fixture["xgb_params"])
    scorer = IndependentScorer(estimator)
    independent_models_recommender = RankingRecommender(scorer)
    # training
    try:
        independent_models_recommender.train(
            sample_continuous_reward_users,
            sample_continuous_reward_items,
            sample_continuous_reward_interactions,
        )

        logger.info("Finished Training of a Item Models Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Item Models Ranking Recommender Failed.")

    # recommending
    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame(
        {
            "USER_ID": ["Emma", "Jasper"],
            "feat1": [18, 65],
            "feat2": [0, 1],
            "feat3": [5, 6],
            "onehot1": [0, 1],
            "onehot2": [1, 1],
            "pca_0": [0.1, 0.3],
            "pca_1": [0.2, 0.7],
            "pca_2": [0.4, 0.8],
        }
    )
    top_k = 3

    recommended_items = independent_models_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Item Models Ranking Recommender.")
    assert recommended_items.shape == (2, 3)


@patch("skrec.scorer.universal.TabularUniversalScorer.score_items")
def test_evaluate_recommender(mock_score_items, setup_fixture, caplog):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)
    # training
    universal_recommender.train(
        sample_binary_reward_users, sample_binary_reward_items, sample_binary_reward_interactions
    )
    # Assert item names after training
    assert_array_equal(universal_recommender.scorer.item_names, ["ITEM_1", "ITEM_2", "ITEM_3"])  # Added assertion

    # Configure mock scores to ensure deterministic ranking
    # Target Ranks (0-based for ITEM_1, ITEM_2, ITEM_3): [[2, 1, 0], [0, 2, 1]]
    mock_scores = pd.DataFrame(
        [
            # Use log for simpler softmax
            [0.0, np.log(2), np.log(3)],
            [np.log(3), 0.0, np.log(2)],
        ],
        columns=["ITEM_1", "ITEM_2", "ITEM_3"],
    )
    mock_score_items.return_value = mock_scores

    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [2000, 0], "feat2": [100, 0.1]})
    metric_type = RecommenderMetricType.PRECISION_AT_K
    eval_top_k = 3

    # --- Prepare eval_kwargs for Simple/ReplayMatch (first part) ---
    eval_kwargs_1 = {
        "logged_rewards": np.array([[0.0], [1.0]], dtype=float),
        "logged_items": np.array([["ITEM_3"], ["ITEM_2"]], dtype=object),
    }
    # ---

    # # ----------simple evaluator--------------
    # test evaluate when eval_kwargs is not given for the first call
    expected_error_msg = "No cached recommendation scores available"
    with pytest.raises(ValueError, match=expected_error_msg):
        universal_recommender.evaluate(
            RecommenderEvaluatorType.SIMPLE,
            metric_type,
            eval_top_k,
        )

    # Mocking UniversalScorer.score_items ensures deterministic ranks based on mock_scores.
    # This makes the test independent of the actual model's scoring behavior,
    # focusing instead on the evaluator and metric logic given a fixed ranking.
    # Expected Ranks (0-based for ITEM_1, ITEM_2, ITEM_3) based on mock_scores:
    # User 1 ('Emma'): [2, 1, 0] -> ITEM_3 > ITEM_2 > ITEM_1
    # User 2 ('Jasper'): [0, 2, 1] -> ITEM_1 > ITEM_3 > ITEM_2

    # test evaluate
    simple_eval = universal_recommender.evaluate(
        RecommenderEvaluatorType.SIMPLE,
        metric_type,
        eval_top_k,
        eval_kwargs=eval_kwargs_1,
        score_items_kwargs={"interactions": interactions, "users": users},
    )
    assert (1 / 3 + 0) / 2 == simple_eval

    # test evaluate with same data eval_kwargs and recommend_kwargs, and different metric
    simple_eval = universal_recommender.evaluate(
        RecommenderEvaluatorType.SIMPLE, RecommenderMetricType.AVERAGE_REWARD_AT_K, eval_top_k=3
    )
    assert (0 + 1 / 3) / 2 == simple_eval

    # Switching eval_type without providing eval_kwargs raises ValueError because
    # modified_rewards must be recomputed for the new evaluator.
    with pytest.raises(ValueError, match="eval_kwargs is required to compute modified rewards"):
        universal_recommender.evaluate(RecommenderEvaluatorType.REPLAY_MATCH, metric_type, eval_top_k)

    # # ----------replay_match evaluator-------------
    # # recommend top 1 item
    replay_match_eval = universal_recommender.evaluate(
        RecommenderEvaluatorType.REPLAY_MATCH,
        metric_type,
        eval_top_k=1,
        eval_kwargs=eval_kwargs_1,
        score_items_kwargs={"interactions": interactions, "users": users},
    )
    assert 0 == replay_match_eval

    # --- Prepare eval_kwargs for ReplayMatch (second part), IPS, DR ---
    # Data with variable lengths, requires padding
    logged_items_list = [["ITEM_3", "ITEM_2", "ITEM_1"], ["ITEM_2"]]
    logged_rewards_list = [[0.0, 1.0, 1.0], [1.0]]
    logging_proba_list = [[0.5, 0.9, 0.8], [0.4]]
    # expected_rewards for DR evaluator: (N_users, N_items)
    # User 1 (Emma) for (ITEM_1, ITEM_2, ITEM_3): [0.2, 0.4, 0.8]
    # User 2 (Jasper) for (ITEM_1, ITEM_2, ITEM_3): [0.7, 0.1, 0.3]
    expected_rewards_arr = np.array([[0.2, 0.4, 0.8], [0.7, 0.1, 0.3]])

    # Define padding values
    pad_value_item = ""  # Use empty string for object type padding

    # Dictionary for ReplayMatch (second part)
    eval_kwargs_replay_2 = {
        "logged_items": create_padded_matrix(logged_items_list, pad_value=pad_value_item, dtype=object),
        "logged_rewards": create_padded_matrix(logged_rewards_list, dtype=float),
    }

    # Dictionary for IPS (requires propensities)
    eval_kwargs_ips = {
        "logged_items": eval_kwargs_replay_2["logged_items"],
        "logged_rewards": eval_kwargs_replay_2["logged_rewards"],
        "logging_proba": create_padded_matrix(logging_proba_list, dtype=float),
    }

    # Dictionary for DR (requires propensities and expected rewards)
    eval_kwargs_dr = {
        "logged_items": eval_kwargs_replay_2["logged_items"],
        "logged_rewards": eval_kwargs_replay_2["logged_rewards"],
        "logging_proba": eval_kwargs_ips["logging_proba"],
        "expected_rewards": expected_rewards_arr,
    }
    # --- End preparation ---

    replay_match_eval = universal_recommender.evaluate(
        RecommenderEvaluatorType.REPLAY_MATCH,
        metric_type,  # Precision@k
        eval_top_k,  # 3
        eval_kwargs=eval_kwargs_replay_2,
        score_items_kwargs={"interactions": interactions, "users": users},
    )
    assert (2 / 3 + 1) / 2 == replay_match_eval

    # test evaluate with same data eval_kwargs and recommend_kwargs, and different metric
    assert (2 / 3 + 1) / 2 == universal_recommender.evaluate(
        RecommenderEvaluatorType.REPLAY_MATCH, RecommenderMetricType.AVERAGE_REWARD_AT_K, eval_top_k
    )
    assert ((1 / 2 + 2 / 3) / 2 + 1) / 2 == universal_recommender.evaluate(
        RecommenderEvaluatorType.REPLAY_MATCH, RecommenderMetricType.MAP_AT_K, eval_top_k
    )
    assert (1 / 2 + 1 / 3) / 2 == universal_recommender.evaluate(
        RecommenderEvaluatorType.REPLAY_MATCH, RecommenderMetricType.MRR_AT_K, eval_top_k
    )
    assert (
        (0 + 1 / np.log2(2 + 1) + 1 / np.log2(2 + 2)) / (1 / np.log2(2 + 0) + 1 / np.log2(2 + 1))
        + (1 / np.log2(2 + 2)) / (1 / np.log2(2 + 0))
    ) / 2 == universal_recommender.evaluate(
        RecommenderEvaluatorType.REPLAY_MATCH, RecommenderMetricType.NDCG_AT_K, eval_top_k
    )

    # # ---------IPS Evaluator-----------------------------------
    # We pass the full eval_kwargs_ips dictionary prepared earlier.
    IPS_eval = universal_recommender.evaluate(
        RecommenderEvaluatorType.IPS,
        metric_type,
        eval_top_k,  # Use eval_top_k=3
        eval_kwargs=eval_kwargs_ips,
        # score_items_kwargs are not needed if evaluator state is reused, but let's keep it consistent
        score_items_kwargs={"interactions": interactions, "users": users},
    )
    expected_metric = (
        # IPS: (target_proba) / logging_proba * reward
        (
            ((3 / 6) / 0.5 * 0.0)  # User 1, ITEM_3
            + ((2 / 6) / 0.9 * 1.0)  # User 1, ITEM_2
            + ((1 / 6) / 0.8 * 1.0)  # User 1, ITEM_1
        )
        / 3  # Average for User 1
        + ((1 / 6) / 0.4 * 1.0) / 1  # User 2, ITEM_2
    ) / 2  # Average over users
    assert IPS_eval == pytest.approx(expected_metric)

    # # ---------DR Evaluator------------------------------------
    # We pass the full eval_kwargs_dr dictionary prepared earlier.
    DR_eval = universal_recommender.evaluate(
        RecommenderEvaluatorType.DR,
        RecommenderMetricType.EXPECTED_REWARD,
        eval_top_k,  # unused
        eval_kwargs=eval_kwargs_dr,
        # score_items_kwargs are not needed if evaluator state is reused, but let's keep it consistent
        score_items_kwargs={"interactions": interactions, "users": users},
    )
    # Metric is EXPECTED_REWARD: nanmean over users of (nanmean over items of DR-modified_rewards)
    # DR-modified_rewards are: DM_u + (logged_reward_ui - ER_ui) * TP_ui / LP_ui for logged items i; NaN otherwise.

    # User 1 (Emma): logged=[I3,I2,I1], TP_softmax=[1/6,2/6,3/6] for I1,I2,I3. ER_model=[0.2,0.4,0.8] for I1,I2,I3
    # Logged Rewards: I3=0.0, I2=1.0, I1=1.0. Logging Probas: I3=0.5, I2=0.9, I1=0.8
    dm_1 = (0.2 * (1 / 6)) + (0.4 * (2 / 6)) + (0.8 * (3 / 6))  # DM for User 1

    # ITEM_3 (idx 2) for User 1:
    dr_1_item3 = dm_1 + (0.0 - 0.8) * (3 / 6) / 0.5
    # ITEM_2 (idx 1) for User 1:
    dr_1_item2 = dm_1 + (1.0 - 0.4) * (2 / 6) / 0.9
    # ITEM_1 (idx 0) for User 1:
    dr_1_item1 = dm_1 + (1.0 - 0.2) * (1 / 6) / 0.8

    # User 2 (Jasper): logged=[I2], TP_softmax=[3/6,1/6,2/6] for I1,I2,I3. ER_model=[0.7,0.1,0.3] for I1,I2,I3
    # Logged Rewards: I2=1.0. Logging Probas: I2=0.4
    dm_2 = (0.7 * (3 / 6)) + (0.1 * (1 / 6)) + (0.3 * (2 / 6))  # DM for User 2

    # Only ITEM_2 is logged for User 2
    dr_2_item2 = dm_2 + (1.0 - 0.1) * (1 / 6) / 0.4

    expected_metric = np.mean([dr_1_item3, dr_1_item2, dr_1_item1, dr_2_item2])
    assert DR_eval == pytest.approx(expected_metric)


@patch("skrec.scorer.universal.TabularUniversalScorer.score_items")
def test_clear_evaluation_cache_invalidates_scores(mock_score_items, setup_fixture):
    """``clear_evaluation_cache`` clears the evaluate session; a later ``evaluate`` needs ``score_items_kwargs``."""
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)
    universal_recommender.train(
        sample_binary_reward_users, sample_binary_reward_items, sample_binary_reward_interactions
    )

    mock_scores_df = pd.DataFrame(
        np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]]),
        columns=["ITEM_1", "ITEM_2", "ITEM_3"],
    )
    mock_score_items.return_value = mock_scores_df

    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [2000, 0], "feat2": [100, 0.1]})
    eval_kwargs_1 = {
        "logged_rewards": np.array([[0.0], [1.0]], dtype=float),
        "logged_items": np.array([["ITEM_3"], ["ITEM_2"]], dtype=object),
    }
    universal_recommender.evaluate(
        RecommenderEvaluatorType.SIMPLE,
        RecommenderMetricType.PRECISION_AT_K,
        3,
        eval_kwargs=eval_kwargs_1,
        score_items_kwargs={"interactions": interactions, "users": users},
    )
    assert universal_recommender.evaluation_session is universal_recommender._eval_session
    universal_recommender.clear_evaluation_cache()
    assert universal_recommender.evaluator is None
    with pytest.raises(ValueError, match="No cached recommendation scores"):
        universal_recommender.evaluate(
            RecommenderEvaluatorType.SIMPLE,
            RecommenderMetricType.PRECISION_AT_K,
            3,
            eval_kwargs=eval_kwargs_1,
        )


def test_training_with_validation_sets_recommender(setup_fixture):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_early_stopping_params"])
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)

    # training
    try:
        universal_recommender.train(
            sample_binary_reward_users,
            sample_binary_reward_items,
            sample_binary_reward_interactions,
            valid_users_ds=sample_binary_reward_users,
            valid_interactions_ds=sample_binary_reward_interactions,
        )
        logger.info("Finished Training of a Universal Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Universal Ranking Recommender Failed.")

    # recommending
    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [18, 65], "feat2": [0, 1]})
    top_k = 3

    recommended_items = universal_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    assert recommended_items.shape == (2, 3)

    with tempfile.TemporaryDirectory() as tmpdirname:
        valid_users_path = os.path.join(tmpdirname, "valid_users.parquet")
        users_df = sample_binary_reward_users.fetch_data().copy(deep=True)
        # create extra feature to raise error
        users_df["extra_feature"] = range(len(users_df))
        users_df.to_parquet(valid_users_path)
        valid_users_ds = UsersDataset(
            data_location=valid_users_path,
        )

        expected_msg = "Training and validation data have different number of features: 9 != 10"
        with pytest.raises(ValueError, match=expected_msg):
            universal_recommender.train(
                sample_binary_reward_users,
                sample_binary_reward_items,
                sample_binary_reward_interactions,
                valid_users_ds=valid_users_ds,
                valid_interactions_ds=sample_binary_reward_interactions,
            )


def test_batch_universal_recommender(setup_fixture):
    estimator_params = {"num_boost_round": 10, "tree_method": "hist"}
    estimator = BatchXGBClassifierEstimator(estimator_params)
    assert estimator.support_batch_training()
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)

    # training
    try:
        universal_recommender.train(
            sample_binary_reward_users,
            sample_binary_reward_items,
            sample_binary_reward_interactions,
        )
        logger.info("Finished Training of a Universal Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Universal Ranking Recommender Failed.")

    # recommending
    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [18, 65], "feat2": [0, 1]})
    top_k = 3

    recommended_items = universal_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    assert recommended_items.shape == (2, 3)
    catalog = set(universal_recommender.scorer.item_names)
    for i in range(len(interactions)):
        assert set(recommended_items[i]) == catalog, f"User {i}: items outside catalog"
        assert len(recommended_items[i]) == len(set(recommended_items[i])), f"User {i}: duplicate items"

    universal_recommender.train(
        sample_binary_reward_users,
        sample_binary_reward_items,
        sample_binary_reward_interactions,
        valid_users_ds=sample_binary_reward_users,
        valid_interactions_ds=sample_binary_reward_interactions,
    )

    recommended_items = universal_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    assert recommended_items.shape == (2, 3)
    catalog = set(universal_recommender.scorer.item_names)
    for i in range(len(interactions)):
        assert set(recommended_items[i]) == catalog, f"User {i}: items outside catalog after retrain"
        assert len(recommended_items[i]) == len(set(recommended_items[i])), f"User {i}: duplicate items after retrain"


def test_mismatch_training_inference(setup_fixture):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)

    # training
    try:
        universal_recommender.train(
            sample_binary_reward_users,
            sample_binary_reward_items,
            sample_binary_reward_interactions,
        )
        logger.info("Finished Training of a Universal Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Universal Ranking Recommender Failed.")

    # Provide extra user feature
    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [18, 65], "feat2": [0, 1], "Salary": [100, 500]})
    top_k = 3
    recs = universal_recommender.recommend(interactions, users, top_k)
    assert recs.shape == (2, 3)

    # Provide one less user feature
    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [18, 65]})
    top_k = 3

    expected_msg = "Column 'feat2' not found in dataset"
    with pytest.raises(RuntimeError, match=expected_msg):
        universal_recommender.recommend(interactions, users, top_k)

    # Provide one extra interaction feature
    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "cf": [5, 6]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [18, 65], "feat2": [0, 1]})
    top_k = 3
    recs = universal_recommender.recommend(interactions, users, top_k)
    assert recs.shape == (2, 3)


def test_1row_recommender_multiclass(setup_fixture):
    # If we test multiclass, the same is code is used for multioutput-scorer and independent-scorer models
    # We need to test binary separately in another test though
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = MulticlassScorer(estimator)
    multiclass_recommender = RankingRecommender(scorer)

    # training
    try:
        multiclass_recommender.train(
            interactions_ds=sample_multi_class_interactions,
        )
        logger.info("Finished Training of a Universal Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Universal Ranking Recommender Failed.")

    # recommending
    interactions = pd.DataFrame(
        {"USER_ID": ["Emma"], "feat1": [18], "feat2": [0], "feat3": [4], "feat4": [6], "feat5": [2]}
    )

    score_items = multiclass_recommender.score_items(interactions)
    features = interactions.drop(columns=["USER_ID"])
    score_items_fast = multiclass_recommender.scorer.score_fast(features)
    assert_array_equal(score_items, score_items_fast)


def test_1_row_recommender_with_nones(setup_fixture):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)

    # training
    try:
        universal_recommender.train(
            sample_binary_reward_users, sample_binary_reward_items, sample_binary_reward_interactions
        )
        logger.info("Finished Training of a Universal Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Universal Ranking Recommender Failed.")

    # recommending
    interactions = pd.DataFrame({"USER_ID": ["Emma"]})
    users = pd.DataFrame({"USER_ID": ["Emma"], "feat1": [18], "feat2": [0]})
    users_no_id = users.drop(columns="USER_ID")

    # testing with users = None - should break because users-features are expected
    expected_error_msg = r"Expecting User Columns: \['USER_ID', 'feat1', 'feat2'\]"
    with pytest.raises(ValueError, match=expected_error_msg):
        universal_recommender.score_items(interactions=interactions, users=None)

    # testing with Interactions = None, should work because interactions-features are not present in this problem
    recommender_score_items_fast = universal_recommender.scorer.score_fast(users_no_id.copy(deep=True))

    # Verify scorer can also handle it directly
    scorer_items_fast = universal_recommender.scorer.score_fast(users_no_id.copy(deep=True))

    score_items_through_join = universal_recommender.score_items(interactions=None, users=users)
    assert_array_equal(scorer_items_fast, score_items_through_join)
    assert_array_equal(recommender_score_items_fast, score_items_through_join)


def test_1row_recommender_universal(setup_fixture):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)

    # training
    try:
        universal_recommender.train(
            sample_binary_reward_users, sample_binary_reward_items, sample_binary_reward_interactions
        )
        logger.info("Finished Training of a Universal Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Universal Ranking Recommender Failed.")

    # recommending
    interactions = pd.DataFrame({"USER_ID": ["Emma"]})
    users = pd.DataFrame({"USER_ID": ["Emma"], "feat1": [18], "feat2": [0]})

    score_items = universal_recommender.score_items(interactions, users)

    features = users.drop(columns=["USER_ID"])
    score_items_fast = universal_recommender.scorer.score_fast(features)
    assert_array_equal(score_items, score_items_fast)

    # recommending with extra interactions features
    interactions = pd.DataFrame({"USER_ID": ["Emma"], "extra_feat": [1]})
    users = pd.DataFrame({"USER_ID": ["Emma"], "feat1": [18], "feat2": [0]})
    features = pd.concat([interactions.drop(columns=["USER_ID"]), users.drop(columns=["USER_ID"])], axis=1)
    universal_recommender.scorer.score_fast(features)

    # recommending with extra user features
    interactions = pd.DataFrame({"USER_ID": ["Emma"]})
    users = pd.DataFrame({"USER_ID": ["Emma"], "feat1": [18], "feat2": [0], "extra_feat": [1]})
    features = users.drop(columns=["USER_ID"])
    universal_recommender.scorer.score_fast(features)


def test_score_items_np_parity_universal(setup_fixture):
    """_score_items_np must return the same values as score_items().to_numpy() for UniversalScorer."""
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    recommender = RankingRecommender(scorer)
    recommender.train(sample_binary_reward_users, sample_binary_reward_items, sample_binary_reward_interactions)

    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [18, 30], "feat2": [0, 1]})

    expected = scorer.score_items(interactions, users).to_numpy()
    actual = scorer._score_items_np(interactions, users)

    assert_array_equal(expected, actual)


def test_score_fast_np_parity_universal(setup_fixture):
    """_score_fast_np must return the same values as score_fast().to_numpy() for UniversalScorer."""
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    recommender = RankingRecommender(scorer)
    recommender.train(sample_binary_reward_users, sample_binary_reward_items, sample_binary_reward_interactions)

    users = pd.DataFrame({"USER_ID": ["Emma"], "feat1": [18], "feat2": [0]})
    features = users.drop(columns=["USER_ID"])

    expected = scorer.score_fast(features).to_numpy()
    actual = scorer._score_fast_np(features)

    assert_array_equal(expected, actual)

    # also verify extra columns are handled identically
    features_extra = pd.concat([pd.DataFrame({"extra_feat": [1]}), features], axis=1)
    expected_extra = scorer.score_fast(features_extra).to_numpy()
    actual_extra = scorer._score_fast_np(features_extra)
    assert_array_equal(expected_extra, actual_extra)


def test_recommender_with_subset(setup_fixture):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)

    # training
    try:
        universal_recommender.train(
            sample_binary_reward_users, sample_binary_reward_items, sample_binary_reward_interactions
        )
        logger.info("Finished Training of a Universal Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Universal Ranking Recommender Failed.")

    new_items = pd.DataFrame(
        {
            "ITEM_ID": ["ITEM_6", "ITEM_7"],
            "item_feat1": [2, 0],
            "item_feat2": [0, 0],
            "item_feat3": [0, 2],
            "item_feat4": [0, 0],
            "item_feat5": [1, 0],
            "item_feat6": [0, 0],
            "item_feat7": [0, 1],
        }
    )
    item_subset = ["ITEM_2", "ITEM_3", "ITEM_6"]
    top_k = 5

    interactions = pd.DataFrame({"USER_ID": ["Emma"]})
    users = pd.DataFrame({"USER_ID": ["Emma"], "feat1": [18], "feat2": [0]})

    universal_recommender.set_new_items(new_items)
    universal_recommender.set_item_subset(item_subset)
    universal_recommender.score_items(interactions, users)

    recommended_items = universal_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    logger.info("Recommended Items: %s", recommended_items)

    assert recommended_items.shape == (1, len(item_subset))
    assert set(recommended_items[0]) == set(item_subset)

    # testing without item_subset
    recommended_items = universal_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    assert recommended_items.shape == (1, len(item_subset))

    # testing with item_subset
    item_subset = ["ITEM_2", "ITEM_3"]
    universal_recommender.set_item_subset(item_subset)
    recommended_items = universal_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    logger.info("Recommended Items: %s", recommended_items)
    assert recommended_items.shape == (1, len(item_subset))
    assert set(recommended_items[0]) == set(item_subset)

    # clearing item subset
    universal_recommender.clear_item_subset()
    top_k = len(universal_recommender.scorer.item_names)
    recommended_items = universal_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    logger.info("Recommended Items: %s", recommended_items)

    assert recommended_items.shape == (1, top_k)
    assert set(recommended_items[0]) == set(universal_recommender.scorer.item_names)

    # testing with item_subset with only one item
    item_subset = ["ITEM_2"]
    universal_recommender.set_item_subset(item_subset)
    recommended_items = universal_recommender.recommend(interactions, users, top_k)
    logger.info("Finished Recommending of a Universal Ranking Recommender.")
    assert recommended_items.shape == (1, 1)
    assert set(recommended_items[0]) == set(item_subset)

    # testing with item_subset with no items
    item_subset = []
    # expecting error since it is not possible to recommend 0 items
    with pytest.raises(ValueError):
        universal_recommender.set_item_subset(item_subset)
        recommended_items = universal_recommender.recommend(interactions, users, top_k)


def test_recommender_with_random_data(setup_fixture):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)

    n_users = 10
    n_items = 10
    n_interactions = 100
    n_features = 5
    rng = np.random.default_rng(100)

    users_values = rng.random(size=(n_users, n_features))
    users_df = pd.DataFrame(users_values, columns=[f"user_feat_{i}" for i in range(n_features)])
    user_ids = [f"user_{i}" for i in range(n_users)]
    users_df["USER_ID"] = user_ids

    items_values = rng.random(size=(n_items, n_features))
    items_df = pd.DataFrame(items_values, columns=[f"item_feat_{i}" for i in range(n_features)])
    item_ids = [f"item_{i}" for i in range(n_items)]
    items_df["ITEM_ID"] = item_ids

    interactions_values = rng.random(size=(n_interactions, n_features))
    interactions_df = pd.DataFrame(interactions_values, columns=[f"interaction_feat_{i}" for i in range(n_features)])
    interactions_df["USER_ID"] = rng.choice(user_ids, n_interactions)
    interactions_df["ITEM_ID"] = rng.choice(item_ids, n_interactions)
    interactions_df["OUTCOME"] = rng.choice([0, 1], n_interactions)

    X, y = universal_recommender.scorer.process_datasets(users_df, items_df, interactions_df)
    universal_recommender.scorer.train_model(X, y)

    n_samples = 5
    rec_interactions_df = interactions_df.head(n_samples).copy()
    rec_interactions_df = rec_interactions_df.drop(columns=["OUTCOME", "ITEM_ID"])
    rec_interactions_df["USER_ID"] = rng.choice(user_ids, n_samples)

    universal_recommender.interactions_schema = None
    universal_recommender.users_schema = None
    top_k = 3
    all_items = set(item_ids)

    # Full-catalog recommend
    recommended_items = universal_recommender.recommend(rec_interactions_df, users_df, top_k)
    assert recommended_items.shape == (n_samples, top_k)
    for row in recommended_items:
        assert len(set(row)) == top_k, "Recommendations should be unique per user"
        assert set(row).issubset(all_items), "All recommendations should be valid items"

    full_recs = recommended_items.copy()

    # testing with item_subset
    item_subset = ["item_2", "item_3", "item_4"]
    universal_recommender.set_item_subset(item_subset)
    recommended_items = universal_recommender.recommend(rec_interactions_df, users_df, top_k)
    assert recommended_items.shape == (n_samples, top_k)
    for row in recommended_items:
        assert set(row) == set(item_subset), "Recommendations should only contain subset items"
    subset_recs = recommended_items.copy()

    # now change the order — results should be the same regardless of subset order
    item_subset_rev = item_subset[::-1]
    universal_recommender.set_item_subset(item_subset_rev)
    recommended_items = universal_recommender.recommend(rec_interactions_df, users_df, top_k)
    assert_array_equal(recommended_items, subset_recs)

    # try recommend_online (single-user, no join overhead)
    rec_interactions_df = interactions_df.head(1).copy()
    rec_interactions_df = rec_interactions_df.drop(columns=["OUTCOME", "ITEM_ID", "USER_ID"])
    rec_users_df = users_df.head(1).copy().drop(columns="USER_ID")
    universal_recommender.clear_item_subset()
    recommended_items = universal_recommender.recommend_online(
        interactions=rec_interactions_df, users=rec_users_df, top_k=top_k
    )
    assert len(recommended_items) == top_k
    assert set(recommended_items).issubset(all_items)
    # Online and batch should return the same items for the same user
    assert set(recommended_items) == set(full_recs[0])

    # testing with item_subset
    universal_recommender.set_item_subset(item_subset)
    recommended_items = universal_recommender.recommend_online(
        interactions=rec_interactions_df, users=rec_users_df, top_k=top_k
    )
    assert set(recommended_items) == set(item_subset)

    # now change the order
    universal_recommender.set_item_subset(item_subset_rev)
    recommended_items_rev = universal_recommender.recommend_online(
        interactions=rec_interactions_df, users=rec_users_df, top_k=top_k
    )
    assert_array_equal(recommended_items_rev, recommended_items)


def test_multiclass_recommender_with_subset(setup_fixture, caplog):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = MulticlassScorer(estimator)
    multiclass_recommender = RankingRecommender(scorer)

    expected_msg = "Users Dataset will not be used in MulticlassScorer."
    with pytest.raises(ValueError, match=expected_msg):
        multiclass_recommender.train(
            sample_binary_reward_users, sample_binary_reward_items, sample_multi_class_interactions
        )

    # training
    try:
        multiclass_recommender.train(interactions_ds=sample_multi_class_interactions)
        logger.info("Finished Training of a Multiclass Ranking Recommender.")
    except Exception:
        pytest.fail("Training of a Multiclass Ranking Recommender Failed.")

    # recommending
    interactions = pd.DataFrame(
        {
            "USER_ID": ["Emma", "Jasper"],
            "feat1": [18, 65],
            "feat2": [0, 1],
            "feat3": [1, 1],
            "feat4": [5, 6],
            "feat5": [11, 9],
        }
    )
    top_k = 3
    item_subset = ["item_2", "item_3"]
    multiclass_recommender.set_item_subset(item_subset)
    recommended_items = multiclass_recommender.recommend(interactions=interactions, users=None, top_k=top_k)
    logger.info("Finished Recommending of a Multiclass Ranking Recommender.")
    assert recommended_items.shape == (2, 2)
    assert set(recommended_items[0]) == set(item_subset)

    item_subset = ["ITEM_ALL_CAP"]
    expected_msg = "item_subset contains items not used while training"
    with pytest.raises(ValueError, match=expected_msg):
        multiclass_recommender.set_item_subset(item_subset)
        recommended_items = multiclass_recommender.recommend(interactions=interactions, users=None, top_k=top_k)

    # testing with item_subset that has duplicate items
    item_subset = ["ITEM_2", "ITEM_2"]
    expected_msg = "item_subset contains non-unique values"
    with pytest.raises(ValueError, match=expected_msg):
        multiclass_recommender.set_item_subset(item_subset)
        recommended_items = multiclass_recommender.recommend(interactions=interactions, users=None, top_k=top_k)

    # testing with item_subset with no items
    item_subset = []
    expected_msg = "Length of item_subset cannot be zero"
    with pytest.raises(ValueError, match=expected_msg):
        multiclass_recommender.set_item_subset(item_subset)
        recommended_items = multiclass_recommender.recommend(interactions=interactions, users=None, top_k=top_k)


def test_universal_recommender_sampling(setup_fixture):
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    recommender = RankingRecommender(scorer)

    recommender.train(
        sample_binary_reward_users,
        sample_binary_reward_items,
        sample_binary_reward_interactions,
    )

    interactions_df = pd.DataFrame({"USER_ID": ["Emma", "Jasper", "John"]})
    users_df = pd.DataFrame(
        {
            "USER_ID": ["Emma", "Jasper", "John"],
            "feat1": [18, 65, 30],
            "feat2": [0, 1, 0],
        }
    )
    top_k = 2
    n_users = len(users_df)

    # Test sampling with replacement
    recommended_items_replace = recommender.recommend(
        interactions=interactions_df,
        users=users_df,
        top_k=top_k,
        sampling_temperature=1.0,
        replace=True,
    )
    assert recommended_items_replace.shape == (n_users, top_k)

    # Test sampling without replacement
    recommended_items_no_replace = recommender.recommend(
        interactions=interactions_df,
        users=users_df,
        top_k=top_k,
        sampling_temperature=1.0,
        replace=False,
    )
    assert recommended_items_no_replace.shape == (n_users, top_k)
    for user_recs in recommended_items_no_replace:
        assert len(set(user_recs)) == len(user_recs)

    # Test edge case: top_k > n_items when sampling without replacement
    # Scorer has 3 items: ITEM_1, ITEM_2, ITEM_3
    n_available_items = len(recommender.scorer.item_names)

    recommended_items_k_gt_items = recommender.recommend(
        interactions=interactions_df,
        users=users_df,
        top_k=n_available_items + 1,
        sampling_temperature=1.0,
        replace=False,
    )
    # Should return all available items (n_available_items)
    assert recommended_items_k_gt_items.shape == (n_users, n_available_items)
    for user_recs in recommended_items_k_gt_items:
        assert len(set(user_recs)) == n_available_items


def test_base_recommender_set_item_specific_features(setup_fixture):
    # Test with non IndependentScorer (should raise ValueError)
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = UniversalScorer(estimator)
    universal_recommender = RankingRecommender(scorer)

    # FAIL - Because we don't have an IndependentScorer
    item_specific_features_users = {"ITEM_2": ["feat1"], "ITEM_3": ["feat1", "feat2"]}
    with pytest.raises(ValueError, match="Item specific features can only be set for IndependentScorer"):
        universal_recommender.set_item_specific_features(
            item_specific_features_users=item_specific_features_users, item_specific_features_interactions=None
        )

    # Test with IndependentScorer (should work)
    estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    scorer = IndependentScorer(estimator)
    independent_models_recommender = RankingRecommender(scorer)
    # training

    # FAIL - Because we are using USER_ID in item_specific_features_users
    item_specific_features_users = {"ITEM_2": ["feat1", "USER_ID"], "ITEM_3": ["feat1", "feat2"]}
    with pytest.raises(ValueError, match="USER_ID must not be in item_specific_features_users for item ITEM_2"):
        independent_models_recommender.set_item_specific_features(
            item_specific_features_users=item_specific_features_users, item_specific_features_interactions=None
        )

    # FAIL - Because we are using USER_ID in item_specific_features_interactions
    item_specific_features_users = {"ITEM_2": ["feat1"], "ITEM_3": ["feat1", "feat2"]}
    item_specific_features_interactions = {"ITEM_2": ["feat1"], "ITEM_3": ["feat1", "feat2", "USER_ID"]}
    with pytest.raises(ValueError, match="USER_ID must not be in item_specific_features_interactions for item ITEM_3"):
        independent_models_recommender.set_item_specific_features(
            item_specific_features_users=item_specific_features_users,
            item_specific_features_interactions=item_specific_features_interactions,
        )

    # Test with IndependentScorer, without USER_ID in item specific features (should work)
    item_specific_features_users = {
        "ITEM_2": [
            "feat1",
        ],
        "ITEM_3": ["feat1", "feat2"],
    }

    independent_models_recommender.set_item_specific_features(
        item_specific_features_users=item_specific_features_users,
        item_specific_features_interactions=None,
    )

    independent_models_recommender.train(
        sample_binary_reward_users, sample_binary_reward_items, sample_binary_reward_interactions
    )

    # recommending
    interactions = pd.DataFrame({"USER_ID": ["Emma", "Jasper"]})
    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [18, 65], "feat2": [0, 1]})
    top_k = 3

    # Test for normal recommending
    recommended_items = independent_models_recommender.recommend(
        interactions,
        users,
        top_k,
    )
    logger.info("Finished Recommending of a Item Models Ranking Recommender.")
    assert recommended_items.shape == (2, 3)

    feats = [["feat1", "feat2"], ["feat1"], ["feat1", "feat2"]]

    for idx, item in enumerate(independent_models_recommender.scorer.item_names):
        assert independent_models_recommender.scorer.estimator[item].feature_names == feats[idx]

    # Test for one row recommending
    interactions = pd.DataFrame({"USER_ID": ["Emma"]})
    users = pd.DataFrame({"USER_ID": ["Emma"], "feat1": [18], "feat2": [0]})
    recommended_items = independent_models_recommender.recommend_online(
        interactions=interactions, users=users, top_k=top_k
    )
    logger.info("Finished Recommending of a Item Models Ranking Recommender.")
    assert recommended_items.shape == (3,)

    feats = [["feat1", "feat2"], ["feat1"], ["feat1", "feat2"]]

    for idx, item in enumerate(independent_models_recommender.scorer.item_names):
        assert independent_models_recommender.scorer.estimator[item].feature_names == feats[idx]


def test_ranking_recommender_with_embedding_estimator_realtime(setup_fixture):
    """
    Tests RankingRecommender with UniversalScorer and an EmbeddingEstimator
    in a real-time inference scenario (providing pre-computed user embeddings)
    by calling train, recommend, score_items, and evaluate.
    """
    embedding_estimator_params = {"embedding_dim": 8, "epochs": 2, "verbose": 0, "device": "cpu"}
    embedding_estimator = NeuralFactorizationEstimator(**embedding_estimator_params)
    scorer = UniversalScorer(estimator=embedding_estimator)
    recommender = RankingRecommender(scorer)

    recommender.train(
        users_ds=sample_binary_reward_users,
        items_ds=sample_binary_reward_items,
        interactions_ds=sample_binary_reward_interactions,
    )
    logger.info("Finished Training of RankingRecommender with EmbeddingEstimator.")

    # Call evaluate to test the batch prediction path (users=None internally for predict_proba_with_embeddings)
    eval_kwargs_simple_batch = {
        "logged_rewards": np.array([[0.0], [1.0]], dtype=float),
        "logged_items": np.array([["ITEM_3"], ["ITEM_2"]], dtype=object),
    }
    score_items_kwargs_batch = {
        "interactions": pd.DataFrame({"USER_ID": ["Emma", "Jasper"]}),
        # users=None
    }

    _ = recommender.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.PRECISION_AT_K,
        eval_top_k=3,
        score_items_kwargs=score_items_kwargs_batch,
        eval_kwargs=eval_kwargs_simple_batch,
    )
