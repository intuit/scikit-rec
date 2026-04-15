import logging
from copy import copy

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_items,
    sample_binary_reward_users,
)
from skrec.scorer.independent import IndependentScorer
from tests.utils import MockClassifier, MockClassifier_v2


@pytest.fixture
def setup_fixture(setup_small_datasets):
    setup_small_datasets["items_propensity"] = IndependentScorer(estimator=MockClassifier())
    setup_small_datasets["items_propensity_dict"] = IndependentScorer(
        estimator={"Item1": MockClassifier(), "Item2": MockClassifier_v2(), "Item3": MockClassifier()}
    )
    setup_small_datasets["items_propensity_missing_item"] = IndependentScorer(
        estimator={"Item1": MockClassifier(), "Item2": MockClassifier()}
    )
    setup_small_datasets["items_propensity_unmatched_item"] = IndependentScorer(
        estimator={"Item1": MockClassifier(), "Item2": MockClassifier(), "ABC": MockClassifier()}
    )
    return setup_small_datasets


def test_filter_items_by_interactions(setup_fixture):
    # We will create a new use case where interactions has more items than items_df
    interactions_df = setup_fixture["interactions_dataset"].fetch_data()
    original_items_array = np.array(["Item1", "Item2", "Item3"])
    setup_fixture["items_propensity"].items_df = setup_fixture["items_dataset"].fetch_data()

    setup_fixture["items_propensity"].item_names = original_items_array
    assert_array_equal(setup_fixture["items_propensity"].items_df["ITEM_ID"].unique(), original_items_array)

    setup_fixture["items_propensity"]._filter_items_by_interactions(interactions_df)
    assert_array_equal(setup_fixture["items_propensity"].item_names, np.array(["Item1", "Item2"]))
    final_items_array = np.array(["Item1", "Item2"])
    assert_array_equal(setup_fixture["items_propensity"].items_df["ITEM_ID"].unique(), final_items_array)


def test_process_datasets(setup_fixture, caplog):
    correct_item_names = np.array(["Item1", "Item2"])

    df_item1 = pd.DataFrame({"Context1": [2.0, 3.0], "Context2": [0.2, 0.3], "Age": [28, 49], "Gender": [1, 0]})
    df_item2 = pd.DataFrame({"Context1": [1.0, 4.0], "Context2": [0.1, 0.4], "Age": [30, 28], "Gender": [1, 1]})
    df_item3 = pd.DataFrame({"Context1": [], "Context2": [], "Age": [], "Gender": []})

    float_features = ["Context1", "Context2"]
    df_item1[float_features] = df_item1[float_features].astype("float32")
    df_item2[float_features] = df_item2[float_features].astype("float32")

    X_correct = {"Item1": df_item1, "Item2": df_item2, "Item3": df_item3}
    y_correct = {"Item1": np.array([1, 0]), "Item2": np.array([0, 1]), "Item3": np.array([])}

    caplog.set_level(logging.WARNING, logger="skrec.scorer.independent")
    with caplog.at_level(logging.WARNING, logger="skrec.scorer.independent"):
        X, y = setup_fixture["items_propensity"].process_datasets(
            users_df=setup_fixture["users_dataset"].fetch_data(),
            items_df=setup_fixture["items_dataset"].fetch_data(),
            interactions_df=setup_fixture["interactions_dataset"].fetch_data(),
        )
    assert_array_equal(setup_fixture["items_propensity"].item_names, correct_item_names)
    assert setup_fixture["items_propensity"].items_df is None
    for item in setup_fixture["items_propensity"].item_names:
        assert_array_equal(X[item], X_correct[item])
        assert_array_equal(y[item], y_correct[item])

    expected_msg = "Item Dataset will not be used in IndependentScorer"
    assert expected_msg in caplog.text


def test_validate_input_train(setup_fixture):
    # interactions data with both reward 1 and 0 for each item
    interactions_df = setup_fixture["interactions_dataset"].fetch_data()
    setup_fixture["items_propensity"]._validate_interactions(interactions_df)

    # interactions data with reward = 0, 1, 2 for Item1 and reward = 1 for Item3.
    interactions_df = pd.DataFrame(
        {
            "USER_ID": ["John", "John", "Amy", "Bill", "Amy", "Doe"],
            "ITEM_ID": ["Item1", "Item2", "Item1", "Item2", "Item1", "Item3"],
            "OUTCOME": [0, 1, 1, 0, 2, 1],
        }
    )
    pytest.raises(ValueError, setup_fixture["items_propensity"]._validate_interactions, interactions_df)


def test_process_estimators_per_item(setup_fixture):
    item_names = np.array(["Item1", "Item2", "Item3"])
    # the initial estimator is one classifier
    estimator = copy(setup_fixture["items_propensity"].estimator)
    setup_fixture["items_propensity"]._process_estimators_per_item(item_names)

    assert isinstance(setup_fixture["items_propensity"].estimator, dict)

    for item in item_names:
        assert isinstance(setup_fixture["items_propensity"].estimator[item], type(estimator))

    # the initial estimator is a dictionary of (item_name, classifier)
    correct_estimator = {
        "Item1": setup_fixture["items_propensity_dict"].estimator["Item1"],
        "Item2": setup_fixture["items_propensity_dict"].estimator["Item2"],
        "Item3": setup_fixture["items_propensity_dict"].estimator["Item3"],
    }
    setup_fixture["items_propensity_dict"]._process_estimators_per_item(item_names)
    assert setup_fixture["items_propensity_dict"].estimator == correct_estimator

    # the initial estimator is a dictionary of (item_name, classifier), but the estimator for Item3 is missing
    expected_error_msg = (
        "When multiple binary classification models are used,"
        "the key names in the dictionary of estimators and the item names must be the same!!"
    )

    with pytest.raises(ValueError, match=expected_error_msg):
        setup_fixture["items_propensity_missing_item"]._process_estimators_per_item(item_names)

    # the initial estimator is a dictionary of (item_name, classifier), but the keys do not match with item_names

    with pytest.raises(ValueError, match=expected_error_msg):
        setup_fixture["items_propensity_unmatched_item"]._process_estimators_per_item(item_names)


def test_process_X_y(setup_fixture):
    users_df = setup_fixture["users_dataset"].fetch_data()
    setup_fixture["items_propensity"].items_df = None
    setup_fixture["items_propensity"].item_names = np.array(["Item1", "Item2", "Item3"])
    interactions_df = setup_fixture["interactions_dataset"].fetch_data()
    joined_data = setup_fixture["items_propensity"]._join_data_train(
        users_df, setup_fixture["items_propensity"].items_df, interactions_df
    )
    X, y = setup_fixture["items_propensity"]._process_X_y_join_and_filter(joined_data)

    df_item1 = pd.DataFrame({"Context1": [2.0, 3.0], "Context2": [0.2, 0.3], "Age": [28, 49], "Gender": [1, 0]})
    df_item2 = pd.DataFrame({"Context1": [1.0, 4.0], "Context2": [0.1, 0.4], "Age": [30, 28], "Gender": [1, 1]})
    df_item3 = pd.DataFrame({"Context1": [], "Context2": [], "Age": [], "Gender": []})

    float_features = ["Context1", "Context2"]
    df_item1[float_features] = df_item1[float_features].astype("float32")
    df_item2[float_features] = df_item2[float_features].astype("float32")

    correct_X = {"Item1": df_item1, "Item2": df_item2, "Item3": df_item3}
    correct_y = {"Item1": np.array([1, 0]), "Item2": np.array([0, 1]), "Item3": np.array([])}

    for item in setup_fixture["items_propensity"].item_names:
        assert_array_equal(X[item], correct_X[item])
        assert_array_equal(y[item], correct_y[item])


def test_score_items(setup_fixture):
    interactions = pd.DataFrame({"USER_ID": [11, 12], "a": [1, 4], "b": [2, 5], "c": [3.3, 6.6]})
    users = pd.DataFrame({"USER_ID": [11, 12], "d": [7, 9], "e": [8, 10]})

    # test _score_items when the estimator is one classifier
    setup_fixture["items_propensity"].estimator.feature_names = ["a", "b", "c", "d", "e"]
    setup_fixture["items_propensity"].items_df = setup_fixture["items_dataset"].fetch_data()
    setup_fixture["items_propensity"].item_names = np.array(["Item1", "Item2", "Item3"])
    setup_fixture["items_propensity"]._process_estimators_per_item(setup_fixture["items_propensity"].item_names)

    scores = setup_fixture["items_propensity"].score_items(interactions, users)
    expected = np.array([[4.26, 4.26, 4.26], [6.92, 6.92, 6.92]])
    assert_array_almost_equal(scores, expected)

    # test _score_items when the initial estimator is a dictionary of (item_name, classifier)
    #     and the classifier for Item 2 is different from those for Item1 and Item3.
    for _, estimator in setup_fixture["items_propensity_dict"].estimator.items():
        estimator.feature_names = ["a", "b", "c", "d", "e"]
    setup_fixture["items_propensity_dict"].items_df = setup_fixture["items_dataset"].fetch_data()
    setup_fixture["items_propensity_dict"].item_names = np.array(["Item1", "Item2", "Item3"])

    scores = setup_fixture["items_propensity_dict"].score_items(interactions, users)
    expected = np.array([[4.26, 21.3, 4.26], [6.92, 34.6, 6.92]])
    assert_array_almost_equal(scores, expected)

    features = pd.concat(
        [interactions.head(1).drop(columns=["USER_ID"]), users.head(1).drop(columns=["USER_ID"])], axis=1
    )
    scores = setup_fixture["items_propensity_dict"].score_fast(features)
    expected = np.array([[4.26, 21.3, 4.26]])
    assert_array_almost_equal(scores, expected)


def test_score_fast_parity(setup_fixture):
    """score_fast and score_items must return identical results for the same single row."""
    interactions = pd.DataFrame({"USER_ID": [11, 12], "a": [1, 4], "b": [2, 5], "c": [3.3, 6.6]})
    users = pd.DataFrame({"USER_ID": [11, 12], "d": [7, 9], "e": [8, 10]})

    # single-estimator scorer
    scorer = setup_fixture["items_propensity"]
    scorer.estimator.feature_names = ["a", "b", "c", "d", "e"]
    scorer.items_df = setup_fixture["items_dataset"].fetch_data()
    scorer.item_names = np.array(["Item1", "Item2", "Item3"])
    scorer._process_estimators_per_item(scorer.item_names)

    scores_normal = scorer.score_items(interactions.head(1), users.head(1))
    features = pd.concat(
        [interactions.head(1).drop(columns=["USER_ID"]), users.head(1).drop(columns=["USER_ID"])], axis=1
    )
    scores_fast = scorer.score_fast(features)
    assert_array_almost_equal(scores_fast, scores_normal)

    # dict-of-estimators scorer
    scorer_dict = setup_fixture["items_propensity_dict"]
    for estimator_instance in scorer_dict.estimator.values():
        estimator_instance.feature_names = ["a", "b", "c", "d", "e"]
    scorer_dict.items_df = setup_fixture["items_dataset"].fetch_data()
    scorer_dict.item_names = np.array(["Item1", "Item2", "Item3"])

    scores_normal_dict = scorer_dict.score_items(interactions.head(1), users.head(1))
    scores_fast_dict = scorer_dict.score_fast(features)
    assert_array_almost_equal(scores_fast_dict, scores_normal_dict)


def test_scorer(setup_fixture):
    X, y = setup_fixture["items_propensity"].process_datasets(
        users_df=sample_binary_reward_users.fetch_data(),
        items_df=sample_binary_reward_items.fetch_data(),
        interactions_df=sample_binary_reward_interactions.fetch_data(),
    )

    assert len(X) == 3
    assert len(y) == 3


def test_score_items_with_item_subset(setup_fixture):
    interactions = pd.DataFrame({"USER_ID": [11, 12], "a": [1, 4], "b": [2, 5], "c": [3.3, 6.6]})
    users = pd.DataFrame({"USER_ID": [11, 12], "d": [7, 9], "e": [8, 10]})

    # test _score_items when the estimator is one classifier
    scorer = setup_fixture["items_propensity"]
    scorer.estimator.feature_names = ["a", "b", "c", "d", "e"]
    scorer.items_df = setup_fixture["items_dataset"].fetch_data()
    scorer.item_names = np.array(["Item1", "Item2", "Item3"])
    scorer._process_estimators_per_item(scorer.item_names)

    item_subset = ["Item2", "Item3"]
    scorer.set_item_subset(item_subset)
    scores = scorer.score_items(interactions, users)
    expected = np.array([[4.26, 4.26], [6.92, 6.92]])
    assert_array_almost_equal(scores, expected)


def test_score_items_parallelization(setup_fixture):
    """Test that score_items produces the same result in parallel mode as in sequential mode."""
    # Setup the scorer as in test_score_items

    interactions = pd.DataFrame({"USER_ID": [11, 12], "a": [1, 4], "b": [2, 5], "c": [3.3, 6.6]})
    users = pd.DataFrame({"USER_ID": [11, 12], "d": [7, 9], "e": [8, 10]})

    # test _score_items when the estimator is one classifier
    setup_fixture["items_propensity"].estimator.feature_names = ["a", "b", "c", "d", "e"]
    setup_fixture["items_propensity"].items_df = setup_fixture["items_dataset"].fetch_data()
    setup_fixture["items_propensity"].item_names = np.array(["Item1", "Item2", "Item3"])
    setup_fixture["items_propensity"]._process_estimators_per_item(setup_fixture["items_propensity"].item_names)

    # Calculate scores in sequential mode
    setup_fixture["items_propensity"].set_parallel_inference(False)
    scores_sequential = setup_fixture["items_propensity"].score_items(interactions, users)

    # Calculate scores in parallel mode
    setup_fixture["items_propensity"].set_parallel_inference(True)
    scores_parallel = setup_fixture["items_propensity"].score_items(interactions, users)

    # Assert that results are almost equal (due to potential float precision differences in parallelization)
    assert_array_almost_equal(scores_sequential, scores_parallel)
    assert scores_sequential.shape == (2, 3)
    assert scores_parallel.shape == (2, 3)


def test_score_fast_parallel(setup_fixture):
    """Test that score_fast works correctly in parallel mode and matches sequential."""
    # Setup the scorer
    scorer = setup_fixture["items_propensity"]
    scorer.items_df = setup_fixture["items_dataset"].fetch_data()
    scorer.item_names = np.array(["Item1", "Item2", "Item3"])
    scorer._process_estimators_per_item(scorer.item_names)

    # Set feature_names on each estimator in the dictionary after processing
    for estimator_instance in scorer.estimator.values():
        estimator_instance.feature_names = ["a", "b", "c", "d", "e"]

    # Prepare merged single-row features DataFrame (user+interaction, no USER_ID)
    features = pd.DataFrame({"a": [1], "b": [2], "c": [3.3], "d": [7], "e": [8]})

    # Get sequential baseline for comparison
    scorer.set_parallel_inference(False)
    scores_sequential = scorer.score_fast(features)

    # Calculate scores in parallel mode for 1-row
    scorer.set_parallel_inference(parallel_inference_status=True, num_cores=2)
    scores_parallel = scorer.score_fast(features)
    assert scorer.num_cores == 2

    # Assert that results are almost equal and shape is correct
    assert_array_almost_equal(scores_parallel, scores_sequential)
    assert scores_parallel.shape == (1, 3)
