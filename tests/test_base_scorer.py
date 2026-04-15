import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from skrec.constants import ITEM_ID_NAME, USER_ID_NAME
from tests.utils import MockEstimator, MockScorer, MockScorerWithoutCalculateScores


@pytest.fixture
def setup_fixture(setup_small_datasets):
    setup_small_datasets["propensity_scorer"] = MockScorer(estimator=None)
    setup_small_datasets["users_df"] = setup_small_datasets["users_dataset"].fetch_data()
    setup_small_datasets["items_df"] = setup_small_datasets["items_dataset"].fetch_data()
    setup_small_datasets["interactions_df"] = setup_small_datasets["interactions_dataset"].fetch_data()
    return setup_small_datasets


def test_validate_interactions(setup_fixture):
    scorer = setup_fixture["propensity_scorer"]
    scorer.target_col = ITEM_ID_NAME

    # valid input
    scorer._validate_interactions(setup_fixture["interactions_df"])

    # missing target column — use OUTCOME as target so USER_ID/ITEM_ID are still present
    scorer.target_col = "OUTCOME"
    interactions_no_target = setup_fixture["interactions_df"].drop(columns=["OUTCOME"])
    with pytest.raises(ValueError, match="OUTCOME column must exist in Interaction Dataset."):
        scorer._validate_interactions(interactions_no_target)
    scorer.target_col = ITEM_ID_NAME  # restore

    # None interactions
    with pytest.raises(TypeError, match="Interaction Dataset must exist"):
        scorer._validate_interactions(None)

    # null values in target column
    interactions_with_nulls = setup_fixture["interactions_df"].copy()
    interactions_with_nulls.loc[0, ITEM_ID_NAME] = None
    with pytest.raises(ValueError, match="null value"):
        scorer._validate_interactions(interactions_with_nulls)

    # missing USER_ID
    interactions_no_user = setup_fixture["interactions_df"].drop(columns=[USER_ID_NAME])
    with pytest.raises(ValueError, match=f"'{USER_ID_NAME}' column must exist"):
        scorer._validate_interactions(interactions_no_user)

    # missing ITEM_ID (use OUTCOME as target_col to avoid dropping target)
    scorer.target_col = "OUTCOME"
    interactions_no_item = setup_fixture["interactions_df"].drop(columns=[ITEM_ID_NAME])
    with pytest.raises(ValueError, match=f"'{ITEM_ID_NAME}' column must exist"):
        scorer._validate_interactions(interactions_no_item)
    scorer.target_col = ITEM_ID_NAME  # restore


def test_validate_users(setup_fixture):
    scorer = setup_fixture["propensity_scorer"]

    # valid users_df
    scorer._validate_users(setup_fixture["users_df"])

    # missing USER_ID column
    users_no_id = setup_fixture["users_df"].drop(columns=[USER_ID_NAME])
    with pytest.raises(ValueError, match=f"'{USER_ID_NAME}' column must exist in users_df"):
        scorer._validate_users(users_no_id)

    # duplicate USER_IDs — warns but does not raise
    users_with_dups = pd.concat([setup_fixture["users_df"], setup_fixture["users_df"].head(1)], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scorer._validate_users(users_with_dups)  # should not raise


def test_validate_items(setup_fixture):
    scorer = setup_fixture["propensity_scorer"]

    # valid items_df
    scorer._validate_items(setup_fixture["items_df"])

    # missing ITEM_ID column
    items_no_id = setup_fixture["items_df"].drop(columns=[ITEM_ID_NAME])
    with pytest.raises(ValueError, match=f"'{ITEM_ID_NAME}' column must exist in items_df"):
        scorer._validate_items(items_no_id)

    # duplicate ITEM_IDs — warns but does not raise
    items_with_dups = pd.concat([setup_fixture["items_df"], setup_fixture["items_df"].head(1)], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scorer._validate_items(items_with_dups)  # should not raise


def test_process_items_order(setup_fixture):
    correct_item_names = np.array(["Item1", "Item2", "Item3"])
    correct_item_df = pd.DataFrame(
        [["Item1", 1, 2], ["Item2", 0, 1], ["Item3", 2, 4]], columns=["ITEM_ID", "ItemFeature1", "ItemFeature2"]
    )
    correct_item_df[["ItemFeature1", "ItemFeature2"]] = correct_item_df[["ItemFeature1", "ItemFeature2"]].astype(
        "float32"
    )

    setup_fixture["items_df"].iloc[0], setup_fixture["items_df"].iloc[1] = (
        setup_fixture["items_df"].iloc[1].copy(),
        setup_fixture["items_df"].iloc[0].copy(),
    )
    setup_fixture["items_df"] = setup_fixture["items_df"].reset_index(drop=True)
    item_names, returned_items_df = setup_fixture["propensity_scorer"]._process_items(
        setup_fixture["items_df"], setup_fixture["interactions_df"]
    )
    assert_array_equal(item_names, correct_item_names)
    assert_frame_equal(returned_items_df, correct_item_df)


def test_process_items(setup_fixture):
    # test with the existence of item dataset
    correct_item_names = np.array(["Item1", "Item2", "Item3"])
    correct_item_df = pd.DataFrame(
        [["Item1", 1, 2], ["Item2", 0, 1], ["Item3", 2, 4]], columns=["ITEM_ID", "ItemFeature1", "ItemFeature2"]
    )
    correct_item_df[["ItemFeature1", "ItemFeature2"]] = correct_item_df[["ItemFeature1", "ItemFeature2"]].astype(
        "float32"
    )
    item_names, returned_items_df = setup_fixture["propensity_scorer"]._process_items(
        setup_fixture["items_df"], setup_fixture["interactions_df"]
    )
    assert_array_equal(item_names, correct_item_names)
    assert_frame_equal(returned_items_df, correct_item_df)

    # test with item dataset missing
    correct_item_names = np.array(["Item1", "Item2"])
    items_df = None
    item_names, returned_items_df = setup_fixture["propensity_scorer"]._process_items(
        items_df, setup_fixture["interactions_df"]
    )
    assert_array_equal(item_names, correct_item_names)
    assert returned_items_df is None


def test_process_items_without_item_features(setup_fixture):
    correct_item_names = np.array(["Item1", "Item2", "Item3"])
    correct_item_df = pd.DataFrame([["Item1"], ["Item2"], ["Item3"]], columns=["ITEM_ID"])
    items_df_no_features = setup_fixture["items_df"][["ITEM_ID"]]
    item_names, returned_items_df = setup_fixture["propensity_scorer"]._process_items(
        items_df_no_features, setup_fixture["interactions_df"]
    )
    assert_array_equal(item_names, correct_item_names)
    assert_frame_equal(returned_items_df, correct_item_df)
    assert setup_fixture["propensity_scorer"].items_array is None


def test_join_data_train_incorrect_items(setup_fixture):
    interactions_extra_row = setup_fixture["interactions_df"]
    last = len(interactions_extra_row)
    interactions_extra_row.loc[last] = ["John", "UnknownItem", "1.0", 5, 6]
    expected_error_msg = "Interactions Dataset contains Items not present in the Items Dataset!"
    with pytest.raises(ValueError, match=expected_error_msg):
        setup_fixture["propensity_scorer"]._join_data_train(
            setup_fixture["users_df"],
            setup_fixture["items_df"],
            interactions_extra_row,
        )


def test_join_data_train_incorrect_users(setup_fixture):
    interactions_extra_item = setup_fixture["interactions_df"].copy(deep=True)
    last = len(interactions_extra_item)
    interactions_extra_item.loc[last] = ["UnknownUser", "Item1", "1.0", 5, 6]
    expected_error_msg = "Interactions Dataset contains Users not present in the Users Dataset!"
    with pytest.raises(ValueError, match=expected_error_msg):
        setup_fixture["propensity_scorer"]._join_data_train(
            setup_fixture["users_df"],
            setup_fixture["items_df"],
            interactions_extra_item,
        )


def test_join_data_train(setup_fixture):
    # test when users, items and interactions datasets all exist
    correct_joined_data = pd.DataFrame(
        [
            ["John", "Item2", 0, 1, 0.1, 30, 1, 0.0, 1.0],
            ["Amy", "Item1", 1, 2, 0.2, 28, 1, 1.0, 2.0],
            ["Bill", "Item1", 0, 3, 0.3, 49, 0, 1.0, 2.0],
            ["Amy", "Item2", 1, 4, 0.4, 28, 1, 0.0, 1.0],
        ],
        columns=[
            "USER_ID",
            "ITEM_ID",
            "OUTCOME",
            "Context1",
            "Context2",
            "Age",
            "Gender",
            "ItemFeature1",
            "ItemFeature2",
        ],
    )

    float_features = ["OUTCOME", "Context1", "Context2", "ItemFeature1", "ItemFeature2"]
    correct_joined_data[float_features] = correct_joined_data[float_features].astype("float32")

    joined_data = setup_fixture["propensity_scorer"]._join_data_train(
        setup_fixture["users_df"],
        setup_fixture["items_df"],
        setup_fixture["interactions_df"],
    )
    assert_frame_equal(joined_data, correct_joined_data)

    # test when only interactions dataset exists
    correct_joined_data = pd.DataFrame(
        [
            ["John", "Item2", 0, 1, 0.1],
            ["Amy", "Item1", 1, 2, 0.2],
            ["Bill", "Item1", 0, 3, 0.3],
            ["Amy", "Item2", 1, 4, 0.4],
        ],
        columns=["USER_ID", "ITEM_ID", "OUTCOME", "Context1", "Context2"],
    )
    joined_data = setup_fixture["propensity_scorer"]._join_data_train(
        users_df=None, items_df=None, interactions_df=setup_fixture["interactions_df"]
    )

    float_features = ["OUTCOME", "Context1", "Context2"]
    correct_joined_data[float_features] = correct_joined_data[float_features].astype("float32")
    assert_frame_equal(joined_data, correct_joined_data)


def test_generate_X_y(setup_fixture):
    # test target_col = ITEM_ID_NAME and LABEL_NAME exists in joined_data
    joined_data = pd.DataFrame(
        [
            ["John", "Item2", 1, 30, 0],
            ["Doe", "Item1", 1, 35, 1],
            ["Bill", "Item1", 1, 49, 0],
            ["Amy", "Item2", 1, 28, 1],
        ],
        columns=["USER_ID", "ITEM_ID", "OUTCOME", "Age", "Gender"],
    )

    setup_fixture["propensity_scorer"].target_col = ITEM_ID_NAME
    X, y = setup_fixture["propensity_scorer"]._generate_X_y(joined_data)

    X_correct = np.array([[30, 0], [35, 1], [49, 0], [28, 1]])
    y_correct = np.array(["Item2", "Item1", "Item1", "Item2"])
    assert_array_equal(X, X_correct)
    assert_array_equal(y, y_correct)

    # test target_col = ITEM_ID_NAME and LABEL_NAME is missing in joined_data
    joined_data = pd.DataFrame(
        [
            ["John", "Item2", 30, 0],
            ["Doe", "Item1", 35, 1],
            ["Bill", "Item1", 49, 0],
            ["Amy", "Item2", 28, 1],
        ],
        columns=["USER_ID", "ITEM_ID", "Age", "Gender"],
    )
    X, y = setup_fixture["propensity_scorer"]._generate_X_y(joined_data)

    assert_array_equal(X, X_correct)
    assert_array_equal(y, y_correct)


def test_validate_input_recommend():
    scorer = MockScorer(estimator=None)

    interactions = pd.DataFrame({"USER_ID": [1, 4], "b": [2, 5], "c": [3.3, 6.6]})
    users = pd.DataFrame({"USER_ID": [10, 40], "b_col": [21, 51], "c_col": [33, 67]})

    # Testing default case of both users and interactions present
    df1, df2 = scorer._validate_input_recommend(interactions, users)
    assert_frame_equal(interactions, df1)
    assert_frame_equal(users, df2)

    # If both users and interactions are absent, throw error
    expected_error_msg = "Both Users and Interactions are None, specify atleast one of them!"
    with pytest.raises(ValueError, match=expected_error_msg):
        df1, df2 = scorer._validate_input_recommend(interactions=None, users=None)

    # If only users is absent, construct users_df from interactions
    df1, df2 = scorer._validate_input_recommend(interactions=interactions, users=None)
    assert_frame_equal(interactions, df1)
    assert_frame_equal(pd.DataFrame({USER_ID_NAME: [1, 4]}), df2)

    # If only interactions is absent, construct interactions_df from users
    df1, df2 = scorer._validate_input_recommend(interactions=None, users=users)
    assert_frame_equal(pd.DataFrame({USER_ID_NAME: [10, 40]}), df1)
    assert_frame_equal(users, df2)

    # If user_id is absent in interactions, throw error
    interactions.drop(columns="USER_ID", inplace=True)
    expected_error_msg = f"{USER_ID_NAME} must exist in Interactions DataFrame!"
    with pytest.raises(ValueError, match=expected_error_msg):
        df1, df2 = scorer._validate_input_recommend(interactions, users)

    # If user_id is absent in users, throw error
    users.drop(columns="USER_ID", inplace=True)
    expected_error_msg = f"{USER_ID_NAME} must exist in Users DataFrame!"
    with pytest.raises(ValueError, match=expected_error_msg):
        df1, df2 = scorer._validate_input_recommend(interactions, users)


def test_score_items():
    scorer = MockScorer(estimator=MockEstimator())

    interactions = pd.DataFrame({"USER_ID": [1, 4], "b": [2, 5], "c": [3.3, 6.6]})
    users = pd.DataFrame({"USER_ID": [1, 4], "b_col": [21, 51], "c_col": [33, 67]})

    result = scorer.score_items(interactions, users)

    # We know this because this is implemented in the MockEstimator
    expected = pd.DataFrame(np.array([59.3, 129.6]).reshape(1, -1))
    expected.columns = ["col_0", "col_1"]
    assert_frame_equal(result, expected)


def test_scorer_without_calculate_scores():
    with pytest.raises(TypeError) as context:
        MockScorerWithoutCalculateScores(estimator=MockEstimator())

    expected_error_msg1 = "Can't instantiate abstract class"
    assert expected_error_msg1 in str(context.value)


def test_get_user_interactions_df():
    scorer = MockScorer(estimator=MockEstimator())

    interactions = pd.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3.3, 6.6]})
    users = pd.DataFrame({"d": [7, 9], "e": [8, 10]})

    users_original = users.copy(deep=True)
    interactions_original = interactions.copy(deep=True)
    user_interactions_original = pd.concat([interactions, users], axis=1)

    # Test for empty Interactions, users with USER_ID
    users[USER_ID_NAME] = ["a", "b"]
    estimator_ready_df = scorer._get_user_interactions_df(interactions=None, users=users)
    result_scores = scorer._calculate_scores(estimator_ready_df)

    expected_scores = np.array([[15, 19]])
    assert_frame_equal(estimator_ready_df, users_original)
    assert_array_equal(result_scores, expected_scores)

    # Test for empty Users, Interactions with USER_ID
    interactions[USER_ID_NAME] = ["a", "b"]
    estimator_ready_df = scorer._get_user_interactions_df(interactions=interactions, users=None)
    result_scores = scorer._calculate_scores(estimator_ready_df)

    expected_scores = np.array([[6.3, 15.6]])
    assert_frame_equal(estimator_ready_df, interactions_original)
    assert_array_equal(result_scores, expected_scores)

    # Test for both (Interactions, Users) Non-Empty, both with USER_ID
    estimator_ready_df = scorer._get_user_interactions_df(interactions=interactions, users=users)
    result_scores = scorer._calculate_scores(estimator_ready_df)

    expected_scores = np.array([[21.3, 34.6]])
    assert_frame_equal(estimator_ready_df, user_interactions_original)
    assert_array_equal(result_scores, expected_scores)
