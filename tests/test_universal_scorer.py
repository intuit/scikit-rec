import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pandas.testing import assert_frame_equal

from skrec.constants import ITEM_ID_NAME
from skrec.estimator.regression.xgb_regressor import (
    TunedXGBRegressorEstimator,
    XGBRegressorEstimator,
)
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_items,
    sample_binary_reward_users,
    sample_continuous_reward_interactions,
    sample_continuous_reward_items,
    sample_continuous_reward_users,
)
from skrec.scorer.universal import UniversalScorer
from skrec.util.config_loader import load_config
from tests.utils import MockClassifier, parse_config


@pytest.fixture
def setup_fixture(setup_small_datasets):
    scorer = UniversalScorer(estimator=MockClassifier())
    scorer.estimator.feature_names = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "ItemFeature1",
        "ItemFeature2",
    ]
    scorer.items_df = setup_small_datasets["items_dataset"].fetch_data()
    scorer.items_array = scorer.items_df.drop(columns=[ITEM_ID_NAME]).values
    scorer.item_names = np.array(["Item1", "Item2", "Item3"])

    setup_objects = {}
    setup_objects["universal_scorer"] = scorer
    setup_objects["interactions"] = pd.DataFrame({"USER_ID": [11, 12], "a": [1, 4], "b": [2, 5], "c": [3.3, 6.6]})
    setup_objects["users"] = pd.DataFrame({"USER_ID": [11, 12], "d": [7, 9], "e": [8, 10]})
    setup_objects["users_dataset"] = setup_small_datasets["users_dataset"]
    setup_objects["items_dataset"] = setup_small_datasets["items_dataset"]
    setup_objects["interactions_dataset"] = setup_small_datasets["interactions_dataset"]

    return setup_objects


def test_process_datasets(setup_fixture, caplog):
    # test with the existence of item dataset
    correct_item_names = np.array(["Item1", "Item2", "Item3"])
    correct_item_df = pd.DataFrame(
        [["Item1", 1, 2], ["Item2", 0, 1], ["Item3", 2, 4]], columns=["ITEM_ID", "ItemFeature1", "ItemFeature2"]
    )
    correct_item_df[["ItemFeature1", "ItemFeature2"]] = correct_item_df[["ItemFeature1", "ItemFeature2"]].astype(
        "float32"
    )

    X_correct = pd.DataFrame(
        {
            "Context1": [1.0, 2.0, 3.0, 4.0],
            "Context2": [0.1, 0.2, 0.3, 0.4],
            "Age": [30, 28, 49, 28],
            "Gender": [1, 1, 0, 1],
            "ItemFeature1": [0.0, 1.0, 1.0, 0.0],
            "ItemFeature2": [1.0, 2.0, 2.0, 1.0],
        }
    )
    X_correct[["Context1"]] = X_correct[["Context1"]].astype("float32")
    X_correct[["Context2"]] = X_correct[["Context2"]].astype("float32")
    X_correct[["ItemFeature1"]] = X_correct[["ItemFeature1"]].astype("float32")
    X_correct[["ItemFeature2"]] = X_correct[["ItemFeature2"]].astype("float32")

    y_correct = np.array([0, 1, 0, 1])

    X, y = setup_fixture["universal_scorer"].process_datasets(
        setup_fixture["users_dataset"].fetch_data(),
        setup_fixture["items_dataset"].fetch_data(),
        setup_fixture["interactions_dataset"].fetch_data(),
    )

    assert_array_equal(setup_fixture["universal_scorer"].item_names, correct_item_names)
    assert_frame_equal(setup_fixture["universal_scorer"].items_df, correct_item_df)

    assert_frame_equal(X, X_correct)
    assert_array_equal(y, y_correct)

    # test with item dataset missing
    correct_item_names = np.array(["Item1", "Item2"])
    correct_item_df = pd.DataFrame(
        [["Item1", 1, 0], ["Item2", 0, 1]], columns=["ITEM_ID", "ITEM_ID=Item1", "ITEM_ID=Item2"]
    )

    X_correct = pd.DataFrame(
        {
            "Context1": [1.0, 2.0, 3.0, 4.0],
            "Context2": [0.1, 0.2, 0.3, 0.4],
            "Age": [30, 28, 49, 28],
            "Gender": [1, 1, 0, 1],
            "ITEM_ID=Item1": [0, 1, 1, 0],
            "ITEM_ID=Item2": [1, 0, 0, 1],
        }
    )
    X_correct[["Context1"]] = X_correct[["Context1"]].astype("float32")
    X_correct[["Context2"]] = X_correct[["Context2"]].astype("float32")

    expected_msg = "Since item dataset is missing, we create one-hot encodings for items.\n"
    caplog.set_level(logging.WARNING, logger="skrec.scorer.universal")
    with caplog.at_level(logging.WARNING, logger="skrec.scorer.universal"):
        X, y = setup_fixture["universal_scorer"].process_datasets(
            users_df=setup_fixture["users_dataset"].fetch_data(),
            items_df=None,
            interactions_df=setup_fixture["interactions_dataset"].fetch_data(),
        )

    assert_array_equal(setup_fixture["universal_scorer"].item_names, correct_item_names)
    assert_frame_equal(setup_fixture["universal_scorer"].items_df, correct_item_df)
    assert expected_msg in caplog.text
    assert_frame_equal(X, X_correct)
    assert_array_equal(y, y_correct)


def test_replicate_for_items(setup_fixture):
    user_interactions = pd.DataFrame([(1, 2, 3.3, 7, 8), (5, 10, 1, 1.5, 6.2)])
    user_interactions.columns = ["col1", "col2", "col3", "col4", "col5"]

    corrected_X_data = [
        (1, 2, 3.3, 7, 8, 1, 2),
        (1, 2, 3.3, 7, 8, 0, 1),
        (1, 2, 3.3, 7, 8, 2, 4),
        (5, 10, 1, 1.5, 6.2, 1, 2),
        (5, 10, 1, 1.5, 6.2, 0, 1),
        (5, 10, 1, 1.5, 6.2, 2, 4),
    ]
    corrected_X_data = pd.DataFrame(corrected_X_data, dtype=np.float64)
    corrected_X_data.columns = ["col1", "col2", "col3", "col4", "col5", "ItemFeature1", "ItemFeature2"]

    X_data = setup_fixture["universal_scorer"]._replicate_for_items(user_interactions)
    assert_frame_equal(X_data, corrected_X_data)

    user_interactions = pd.DataFrame([])
    expected_msg = "No rows input for duplication"
    with pytest.raises(ValueError, match=expected_msg):
        setup_fixture["universal_scorer"]._replicate_for_items(user_interactions)

    interactions = setup_fixture["interactions"]
    users = setup_fixture["users"]

    expected_X_data = pd.DataFrame(
        {
            "a": np.array([1, 1, 1, 4, 4, 4], dtype=np.float64),
            "b": np.array([2, 2, 2, 5, 5, 5], dtype=np.float64),
            "c": np.array([3.3, 3.3, 3.3, 6.6, 6.6, 6.6], dtype=np.float64),
            "d": np.array([7, 7, 7, 9, 9, 9], dtype=np.float64),
            "e": np.array([8, 8, 8, 10, 10, 10], dtype=np.float64),
            "ItemFeature1": [np.float64(x) for x in [1.0, 0.0, 2.0, 1.0, 0.0, 2.0]],
            "ItemFeature2": [np.float64(x) for x in [2.0, 1.0, 4.0, 2.0, 1.0, 4.0]],
        }
    )

    user_interactions_data = setup_fixture["universal_scorer"]._get_user_interactions_df(
        interactions=interactions, users=users
    )
    X_data = setup_fixture["universal_scorer"]._replicate_for_items(user_interactions_data)

    assert_frame_equal(X_data, expected_X_data)


def test_score_items(setup_fixture):
    interactions = setup_fixture["interactions"]
    users = setup_fixture["users"]

    result = setup_fixture["universal_scorer"].score_items(interactions, users)
    expected = np.array([[3.47142857, 3.18571429, 3.9], [5.37142857, 5.08571429, 5.8]])
    assert_array_almost_equal(result, expected)


def test_score_items_unknown_users(setup_fixture):
    interactions = setup_fixture["interactions"].copy()
    interactions["USER_ID"] = [11, 1300]
    users = setup_fixture["users"]

    expected_msg = "Interactions Dataset contains Users not present in the Users Dataset!"
    with pytest.raises(ValueError, match=expected_msg):
        setup_fixture["universal_scorer"].score_items(interactions, users)


def test_scorer(setup_fixture):
    X, y = setup_fixture["universal_scorer"].process_datasets(
        users_df=sample_binary_reward_users.fetch_data(),
        items_df=sample_binary_reward_items.fetch_data(),
        interactions_df=sample_binary_reward_interactions.fetch_data(),
    )

    assert X.shape == (5000, 9)
    assert y.shape == (5000,)


def test_scorer_continuous_reward(setup_fixture):
    estimator_config_dir = Path.cwd() / "skrec/examples/estimators/"
    estimator_config = load_config(estimator_config_dir / "estimator_hyperparameters.yaml")
    _, _, _, setup_fixture["xgb_params"] = parse_config(estimator_config, "XGBoostClassifier")
    univeral_propensity_continuous = UniversalScorer(estimator=XGBRegressorEstimator(setup_fixture["xgb_params"]))
    X, y = univeral_propensity_continuous.process_datasets(
        users_df=sample_continuous_reward_users.fetch_data(),
        items_df=sample_continuous_reward_items.fetch_data(),
        interactions_df=sample_continuous_reward_interactions.fetch_data(),
    )

    assert X.shape == (5780, 8)
    assert y.shape == (5780,)


def test_score_items_with_item_subset(setup_fixture):
    interactions = setup_fixture["interactions"]
    users = setup_fixture["users"]
    scorer = setup_fixture["universal_scorer"]

    item_subset = ["Item1", "Item3"]
    scorer.set_item_subset(item_subset)
    result = scorer.score_items(interactions, users)
    # same as test_score_items but only for Item1 and Item3
    expected = np.array([[3.47142857, 3.9], [5.37142857, 5.8]])
    assert_array_almost_equal(result, expected)


def test_set_new_items(setup_fixture, caplog):
    scorer = setup_fixture["universal_scorer"]

    # Verify if they are sorted and unique
    new_items = pd.DataFrame({"ITEM_ID": ["Item5", "Item6"], "ItemFeature1": [0, 0], "ItemFeature2": [0, 0]})

    items_df_final = pd.concat([scorer.items_df, new_items], ignore_index=True)

    scorer.set_new_items(new_items)
    assert_frame_equal(scorer.items_df, items_df_final)
    assert_array_equal(scorer.item_names, np.array(["Item1", "Item2", "Item3", "Item5", "Item6"]))

    # Column names different: Verify if the column names are different
    new_items = pd.DataFrame({"ITEM_ID": ["ITEM_7", "ITEM_8"], "Foo": [1, 1], "Feature4": [1, 1]})
    expected_msg = "new_items_df must have the same columns as items_df"
    with pytest.raises(ValueError, match=expected_msg):
        scorer.set_new_items(new_items)

    # Overlap: Verify what happens when there is an overlap
    new_items = pd.DataFrame(
        {"ITEM_ID": ["Item2", "Item6", "Item1"], "ItemFeature1": [0, 0, 0], "ItemFeature2": [0, 0, 0]}
    )
    with caplog.at_level(logging.WARNING):
        scorer.set_new_items(new_items)
    assert "Overlap found in ITEM_ID: ['Item1', 'Item2', 'Item6']. Using features from new_items_df" in caplog.text

    # Verify behavior with an empty DataFrame
    items_df_before = scorer.items_df.copy()
    new_items = pd.DataFrame(columns=["ITEM_ID", "ItemFeature1", "ItemFeature2"])
    scorer.set_new_items(new_items)
    assert_frame_equal(scorer.items_df, items_df_before)

    new_items = pd.DataFrame(columns=[])
    expected_msg = "new_items_df must have the same columns as items_df"
    with pytest.raises(ValueError, match=expected_msg):
        scorer.set_new_items(new_items)


def test_score_with_set_new_items(setup_fixture):
    interactions = setup_fixture["interactions"]
    users = setup_fixture["users"]
    scorer = setup_fixture["universal_scorer"]

    set_new_items = pd.DataFrame({"ITEM_ID": ["ITEM_5", "ITEM_6"], "ItemFeature1": [0, 0], "ItemFeature2": [0, 0]})
    item_subset = ["Item1", "Item3"]

    scorer.set_new_items(set_new_items)
    scorer.set_item_subset(item_subset)
    items_scores = scorer.score_items(interactions, users)

    assert items_scores is not None

    expected_msg = r"Call set_new_items\(\) before set_item_subset\(\)"

    with pytest.raises(ValueError, match=expected_msg):
        scorer.set_item_subset(item_subset)
        scorer.set_new_items(set_new_items)
        items_scores = scorer.score_items(interactions, users)


def test_tuned_xgb_regressor_with_universal_scorer(setup_fixture):
    """Test that both XGBRegressorEstimator and TunedXGBRegressorEstimator work with UniversalScorer."""
    from skrec.estimator.datatypes import HPOType

    # Test 1: Regular XGBRegressorEstimator
    xgb_params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    regular_regressor = XGBRegressorEstimator(xgb_params)

    regular_scorer = UniversalScorer(estimator=regular_regressor)
    regular_scorer.items_df = setup_fixture["universal_scorer"].items_df
    regular_scorer.items_array = setup_fixture["universal_scorer"].items_array
    regular_scorer.item_names = setup_fixture["universal_scorer"].item_names

    # This should not fail with isinstance check now that BaseRegressor is imported
    X, y = regular_scorer.process_datasets(
        users_df=sample_continuous_reward_users.fetch_data(),
        items_df=sample_continuous_reward_items.fetch_data(),
        interactions_df=sample_continuous_reward_interactions.fetch_data(),
    )

    # Verify the datasets were processed without errors for regular XGB
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 0

    # Test 2: TunedXGBRegressorEstimator
    param_space = {"n_estimators": [10, 20], "max_depth": [3, 5]}
    optimizer_params = {"cv": 2, "scoring": "neg_mean_squared_error"}

    tuned_regressor = TunedXGBRegressorEstimator(
        hpo_method=HPOType.GRID_SEARCH_CV, param_space=param_space, optimizer_params=optimizer_params
    )

    # Create scorer with the tuned regressor
    tuned_scorer = UniversalScorer(estimator=tuned_regressor)
    tuned_scorer.items_df = setup_fixture["universal_scorer"].items_df
    tuned_scorer.items_array = setup_fixture["universal_scorer"].items_array
    tuned_scorer.item_names = setup_fixture["universal_scorer"].item_names

    # This should not fail with isinstance check now that BaseRegressor is imported
    # We're mainly testing that the isinstance(self.estimator[item], BaseRegressor) check works
    X, y = tuned_scorer.process_datasets(
        users_df=sample_continuous_reward_users.fetch_data(),
        items_df=sample_continuous_reward_items.fetch_data(),
        interactions_df=sample_continuous_reward_interactions.fetch_data(),
    )

    # Verify the datasets were processed without errors for tuned XGB
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 0
