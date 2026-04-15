from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
from skrec.recommender.bandits.contextual_bandits import ContextualBanditsRecommender
from skrec.recommender.bandits.datatypes import StrategyType
from skrec.scorer.universal import UniversalScorer
from skrec.util.config_loader import load_config
from tests.utils import parse_config


@pytest.fixture
def setup_fixture(setup_small_datasets):
    setup_small_datasets["binary_interactions_dataset"] = setup_small_datasets["interactions_dataset"]
    estimator_config_dir = Path.cwd() / "skrec/examples/estimators"
    estimator_config = load_config(estimator_config_dir / "estimator_hyperparameters.yaml")
    _, _, _, setup_small_datasets["xgb_params"] = parse_config(estimator_config, "XGBoostClassifier")
    estimator = XGBClassifierEstimator(setup_small_datasets["xgb_params"])
    scorer = UniversalScorer(estimator)
    recommender = ContextualBanditsRecommender(
        scorer,
        StrategyType.STATIC_ACTION,
        {"ranked_item_names": np.array(["Item3", "Item1", "Item2"])},
    )
    setup_small_datasets["scorer"] = scorer
    setup_small_datasets["recommender"] = recommender
    return setup_small_datasets


def test_recommend_isolated_strategy(setup_fixture):
    recommender_isolated_strategy = ContextualBanditsRecommender(setup_fixture["scorer"])
    recommender_isolated_strategy.train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["binary_interactions_dataset"],
    )

    setup_fixture["users"] = setup_fixture["users_dataset"].fetch_data()

    setup_fixture["interactions"] = (
        setup_fixture["binary_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)
    )

    recommender_isolated_strategy.set_strategy(
        StrategyType.STATIC_ACTION, {"ranked_item_names": np.array(["Item3", "Item1", "Item2"])}
    )

    ranked_items = recommender_isolated_strategy.recommend(
        setup_fixture["interactions"], setup_fixture["users"], top_k=2
    )

    expected = np.array(
        [
            ["Item3", "Item1"],
            ["Item3", "Item1"],
            ["Item3", "Item1"],
            ["Item3", "Item1"],
        ]
    )
    assert_array_equal(ranked_items, expected)


def test_recommend_raises_when_strategy_not_set(setup_fixture):
    recommender = ContextualBanditsRecommender(setup_fixture["scorer"])
    recommender.train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["binary_interactions_dataset"],
    )
    interactions = setup_fixture["binary_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)
    users = setup_fixture["users_dataset"].fetch_data()
    with pytest.raises(RuntimeError, match="Strategy not set"):
        recommender.recommend(interactions, users, top_k=1)


def test_recommend(setup_fixture):
    setup_fixture["recommender"].train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["binary_interactions_dataset"],
    )

    setup_fixture["users"] = setup_fixture["users_dataset"].fetch_data()

    setup_fixture["interactions"] = (
        setup_fixture["binary_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)
    )
    ranked_items = setup_fixture["recommender"].recommend(
        setup_fixture["interactions"], setup_fixture["users"], top_k=2
    )

    expected = np.array(
        [
            ["Item3", "Item1"],
            ["Item3", "Item1"],
            ["Item3", "Item1"],
            ["Item3", "Item1"],
        ]
    )
    assert_array_equal(ranked_items, expected)


def test_get_latest_strategy_flags(setup_fixture):
    with pytest.raises(ValueError) as cm:
        flags = setup_fixture["recommender"].get_latest_strategy_flags()
    assert str(cm.value) == "Strategy flags not found; recommend must be called before this method"

    test_recommend(setup_fixture)

    flags = setup_fixture["recommender"].get_latest_strategy_flags()
    expected = np.array(["exploit"] * 4)
    assert_array_equal(flags, expected)


def test_evaluate(setup_fixture):
    test_recommend(setup_fixture)

    metric_type = RecommenderMetricType.AVERAGE_REWARD_AT_K

    eval_kwargs = {
        "logged_rewards": np.array([[0.0], [1.0], [0.0], [1.0]], dtype=float),
        "logged_items": np.array([["Item3"], ["Item1"], ["Item3"], ["Item1"]], dtype=object),
    }

    simple_eval = setup_fixture["recommender"].evaluate(
        RecommenderEvaluatorType.REPLAY_MATCH,
        metric_type,
        eval_top_k=1,
        eval_kwargs=eval_kwargs,
        score_items_kwargs={
            "interactions": setup_fixture["interactions"],
            "users": setup_fixture["users"],
        },
    )
    assert simple_eval == 0


def test_recommend_static_action_temperature(setup_fixture):
    """Test that static action strategy works with deterministic ranking (temperature=0)
    but does not support probabilistic sampling (temperature>0)."""
    # Create a recommender with static action strategy
    recommender = ContextualBanditsRecommender(
        setup_fixture["scorer"],
        StrategyType.STATIC_ACTION,
        {"ranked_item_names": np.array(["Item3", "Item1", "Item2"])},
    )

    # Train the recommender
    recommender.train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["binary_interactions_dataset"],
    )

    # Prepare test data
    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["binary_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)

    # Test with temperature = None (should use deterministic ranking)
    ranked_items_no_temp = recommender.recommend(interactions, users, top_k=2)
    expected = np.array(
        [
            ["Item3", "Item1"],
            ["Item3", "Item1"],
            ["Item3", "Item1"],
            ["Item3", "Item1"],
        ]
    )
    assert_array_equal(ranked_items_no_temp, expected)

    # Test with temperature = 0 (should use deterministic ranking)
    ranked_items_temp0 = recommender.recommend(interactions, users, top_k=2, sampling_temperature=0)
    assert_array_equal(ranked_items_temp0, expected)

    # Test with positive temperature (should raise error since probabilistic sampling is not supported)
    with pytest.raises(NotImplementedError) as cm:
        recommender.recommend(interactions, users, top_k=2, sampling_temperature=1.0)
    assert str(cm.value) == "Static action strategy does not support probabilistic approach"


def test_recommend_epsilon_greedy_temperature(setup_fixture):
    """Test that epsilon greedy strategy behaves appropriately based on temperature."""
    # Create a recommender with epsilon greedy strategy
    recommender1 = ContextualBanditsRecommender(
        setup_fixture["scorer"], StrategyType.EPSILON_GREEDY, {"epsilon": 0.5, "seed": 42}
    )
    recommender2 = ContextualBanditsRecommender(
        setup_fixture["scorer"], StrategyType.EPSILON_GREEDY, {"epsilon": 0.5, "seed": 42}
    )

    # Train the recommender
    recommender1.train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["binary_interactions_dataset"],
    )
    recommender2.train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["binary_interactions_dataset"],
    )

    # Prepare test data
    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["binary_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)

    # Test with temperature = 0 (should be deterministic for same seed)
    ranked_items_temp0_1 = recommender1.recommend(interactions, users, top_k=2, sampling_temperature=0)
    ranked_items_temp0_2 = recommender2.recommend(interactions, users, top_k=2, sampling_temperature=0)
    assert_array_equal(ranked_items_temp0_1, ranked_items_temp0_2)

    # Test with temperature > 0 (should be probabilistic)
    ranked_items_temp1_1 = recommender1.recommend(interactions, users, top_k=2, sampling_temperature=1.0)
    ranked_items_temp1_2 = recommender2.recommend(interactions, users, top_k=2, sampling_temperature=1.0)

    # With temperature > 0, recommendations should be probabilistic
    # Due to epsilon=0.5 and probabilistic sampling, outputs should differ
    assert not np.array_equal(ranked_items_temp1_1, ranked_items_temp1_2)
