import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from skrec.constants import USER_ID_NAME
from skrec.dataset.interactions_dataset import InteractionMultiOutputDataset
from skrec.estimator.classification.multioutput_classifier import (
    MultiOutputClassifierEstimator,
)
from skrec.scorer.multioutput import MultioutputScorer


@pytest.fixture
def setup_fixture(setup_small_datasets):
    setup_small_datasets["multioutput_interactions_dataset"] = InteractionMultiOutputDataset(
        data_location=setup_small_datasets["dst"] / setup_small_datasets["multioutput_interactions_data"]
    )

    setup_small_datasets["multioutput_propensity_xgboost"] = MultioutputScorer(
        estimator=MultiOutputClassifierEstimator(
            base_estimator=XGBClassifier,
            params={
                "objective": "binary:logistic",
            },
        )
    )

    setup_small_datasets["multioutput_propensity_dummy"] = MultioutputScorer(
        estimator=MultiOutputClassifierEstimator(
            base_estimator=DummyClassifier,
            params={
                "strategy": "stratified",
            },
        )
    )
    return setup_small_datasets


def test_error_handling(setup_fixture):
    expected_msg = "Item Dataset and User Dataset will not be used in MultioutputScorer"
    with pytest.raises(ValueError, match=expected_msg):
        X, y = setup_fixture["multioutput_propensity_xgboost"].process_datasets(
            users_df=setup_fixture["users_dataset"].fetch_data(),
            items_df=setup_fixture["items_dataset"].fetch_data(),
            interactions_df=setup_fixture["multioutput_interactions_dataset"].fetch_data(),
        )


def test_process_datasets(setup_fixture):
    # test with items_dataset
    # Because multi-output scorer has its own custom generate_X_y
    # The number of rows will not be the same as the number of interaction-rows
    # There will be one row per user, amongst all users who have >0 interaction
    X_correct = pd.DataFrame({"age": [25, 30, 40, 21], "income": [50000, 60000, 80000, 40000]})
    X_correct = X_correct.values

    y_correct = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]
    )
    correct_item_names = np.array(["ITEM_1", "ITEM_2", "ITEM_3"])
    X, y = setup_fixture["multioutput_propensity_xgboost"].process_datasets(
        interactions_df=setup_fixture["multioutput_interactions_dataset"].fetch_data(),
    )

    assert_array_equal(X, X_correct)
    assert_array_equal(y, y_correct)
    assert_array_equal(setup_fixture["multioutput_propensity_xgboost"].item_names, correct_item_names)
    assert setup_fixture["multioutput_propensity_xgboost"].items_df is None

    # test without items_dataset
    X, y = setup_fixture["multioutput_propensity_xgboost"].process_datasets(
        interactions_df=setup_fixture["multioutput_interactions_dataset"].fetch_data(),
    )
    assert_array_equal(X, X_correct)
    assert_array_equal(y, y_correct)
    assert_array_equal(setup_fixture["multioutput_propensity_xgboost"].item_names, correct_item_names)
    assert setup_fixture["multioutput_propensity_xgboost"].items_df is None


def test_score_items_with_users(setup_fixture):
    X, y = setup_fixture["multioutput_propensity_xgboost"].process_datasets(
        interactions_df=setup_fixture["multioutput_interactions_dataset"].fetch_data(),
    )

    setup_fixture["multioutput_propensity_xgboost"].train_model(X, y)

    X_test = pd.DataFrame(
        {USER_ID_NAME: [11, 12, 13, 14, 15, 16], "age": [28, 30, 41, 33, 100, 101], "Income": [0, 1, 1, 0, 1, 0]}
    )

    expected_msg = "Multioutput Scorer cannot accept Users Dataframe, set it to None!"
    with pytest.raises(ValueError, match=expected_msg):
        setup_fixture["multioutput_propensity_xgboost"].score_items(interactions=X_test, users=X_test)


def test_score_items_xgboost(setup_fixture):
    X, y = setup_fixture["multioutput_propensity_xgboost"].process_datasets(
        interactions_df=setup_fixture["multioutput_interactions_dataset"].fetch_data(),
    )

    setup_fixture["multioutput_propensity_xgboost"].train_model(X, y)

    X_test = pd.DataFrame(
        {USER_ID_NAME: [11, 12, 13, 14, 15, 16], "age": [28, 30, 41, 33, 100, 101], "Income": [0, 1, 1, 0, 1, 0]}
    )

    prediction = setup_fixture["multioutput_propensity_xgboost"].predict_classes(X_test)

    # make sure we have a prediction for each user
    assert len(prediction) == len(X_test)

    # predict_classes returns one predicted-class column per item
    assert list(prediction.columns) == ["ITEM_1", "ITEM_2", "ITEM_3"]


def test_score_items_xgboost_with_item_subset(setup_fixture):
    scorer = setup_fixture["multioutput_propensity_xgboost"]

    X, y = scorer.process_datasets(
        interactions_df=setup_fixture["multioutput_interactions_dataset"].fetch_data(),
    )

    scorer.train_model(X, y)

    X_test = pd.DataFrame(
        {USER_ID_NAME: [11, 12, 13, 14, 15, 16], "age": [28, 30, 41, 33, 100, 101], "Income": [0, 1, 1, 0, 1, 0]}
    )
    item_subset = ["ITEM_1", "ITEM_2"]

    scorer.set_item_subset(item_subset)
    prediction = scorer.predict_classes(X_test)

    # make sure we have a prediction for each user
    assert len(prediction) == len(X_test)

    # predict_classes returns one predicted-class column per item
    assert prediction.columns.to_list() == ["ITEM_1", "ITEM_2"]


def test_score_items(setup_fixture):
    """score_items returns per-class probability columns, not predicted classes."""
    scorer = setup_fixture["multioutput_propensity_xgboost"]
    X, y = scorer.process_datasets(
        interactions_df=setup_fixture["multioutput_interactions_dataset"].fetch_data(),
    )
    scorer.train_model(X, y)

    X_test = pd.DataFrame(
        {USER_ID_NAME: [11, 12, 13, 14, 15, 16], "age": [28, 30, 41, 33, 100, 101], "Income": [0, 1, 1, 0, 1, 0]}
    )
    proba = scorer.score_items(X_test)

    assert len(proba) == len(X_test)
    # binary items: each item has 2 class-probability columns → 3 items × 2 = 6 columns
    assert proba.columns.to_list() == ["ITEM_1_0", "ITEM_1_1", "ITEM_2_0", "ITEM_2_1", "ITEM_3_0", "ITEM_3_1"]
    # probabilities must sum to 1 across classes for each item
    assert np.allclose(proba[["ITEM_1_0", "ITEM_1_1"]].sum(axis=1), 1.0)
    assert np.allclose(proba[["ITEM_2_0", "ITEM_2_1"]].sum(axis=1), 1.0)
    assert np.allclose(proba[["ITEM_3_0", "ITEM_3_1"]].sum(axis=1), 1.0)

    # users=None should be enforced
    with pytest.raises(ValueError, match="Multioutput Scorer cannot accept Users Dataframe"):
        scorer.score_items(interactions=X_test, users=X_test)


def test_score_fast_parity(setup_fixture):
    """score_fast and predict_classes must return identical results for the same single row."""
    scorer = setup_fixture["multioutput_propensity_xgboost"]
    X, y = scorer.process_datasets(
        interactions_df=setup_fixture["multioutput_interactions_dataset"].fetch_data(),
    )
    scorer.train_model(X, y)

    X_test = pd.DataFrame(
        {USER_ID_NAME: [11, 12, 13, 14, 15, 16], "age": [28, 30, 41, 33, 100, 101], "Income": [0, 1, 1, 0, 1, 0]}
    )
    scores_normal = scorer.predict_classes(X_test.head(1))
    features = X_test.head(1).drop(columns=[USER_ID_NAME])
    scores_fast = scorer.score_fast(features)
    assert_array_equal(scores_normal, scores_fast)


def test_score_items_dummy(setup_fixture):
    X, y = setup_fixture["multioutput_propensity_dummy"].process_datasets(
        interactions_df=setup_fixture["multioutput_interactions_dataset"].fetch_data(),
    )
    setup_fixture["multioutput_propensity_dummy"].train_model(X, y)
    X_test = pd.DataFrame(
        {USER_ID_NAME: [11, 12, 13, 14, 15, 16], "age": [28, 30, 41, 33, 100, 101], "Income": [0, 1, 1, 0, 1, 0]}
    )
    prediction = setup_fixture["multioutput_propensity_dummy"].predict_classes(X_test)
    # make sure we have a prediction for each user
    assert len(prediction) == len(X_test)
    # predict_classes returns one predicted-class column per item
    assert prediction.columns.to_list() == ["ITEM_1", "ITEM_2", "ITEM_3"]


# @pytest.mark.parametrize("propensity_datasets", ["LargeDataset"], indirect=True)
# def test_scorer(setup_fixture):
#     X, y = setup_fixture["multioutput_propensity_xgboost"].process_datasets(
#         interactions_df=setup_fixture["multioutput_interactions_dataset"].fetch_data(),
#     )

#     assert X.shape == (790, 2)
#     assert y.shape == (790, 11)
