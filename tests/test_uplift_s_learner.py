import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from skrec.estimator.classification.logreg_classifier import (
    LogisticRegressionClassifierEstimator,
)
from skrec.examples.datasets import sample_uplift_interactions, sample_uplift_users
from skrec.recommender.uplift_model.uplift_recommender import UpliftRecommender
from skrec.scorer.universal import UniversalScorer


@pytest.fixture(scope="module")
def trained_s_learner():
    """Provides a trained S-Learner recommender for tests."""
    scorer = UniversalScorer(estimator=LogisticRegressionClassifierEstimator({}))
    recommender = UpliftRecommender(scorer=scorer, control_item_id="control")
    recommender.train(
        interactions_ds=sample_uplift_interactions,
        users_ds=sample_uplift_users,
        items_ds=None,
    )
    return recommender


@pytest.mark.uplift
def test_s_learner_scoring_and_recommendation(trained_s_learner):
    """Tests the basic scoring and recommendation flow."""
    users_df = sample_uplift_users.fetch_data()

    uplift_scores = trained_s_learner.score_items(users=users_df)
    assert uplift_scores.shape[0] == users_df.shape[0]
    assert set(uplift_scores.columns) == {"item_a", "item_b"}

    top_k = trained_s_learner.recommend(users=users_df, top_k=1)
    assert top_k.shape == (users_df.shape[0], 1)


@pytest.mark.uplift
def test_s_learner_recommend_more_than_available(trained_s_learner):
    """
    Tests that the recommender returns the max available items when top_k is too high.
    """
    users_df = sample_uplift_users.fetch_data()

    # There are 3 options (item_a, item_b, control), so ask for 4
    recommendations = trained_s_learner.recommend(users=users_df, top_k=4)

    # The recommender should return the max available items, which is 3
    assert recommendations.shape[1] == 3


@pytest.mark.uplift
def test_s_learner_with_unknown_feature(trained_s_learner):
    """
    Tests that the recommender raises an error if the input dataframe has wrong columns.
    """
    users_df_wrong_feature = pd.DataFrame({"USER_ID": ["user_1", "user_2"], "wrong_feature_name": [15, 25]})

    with pytest.raises(Exception):
        trained_s_learner.recommend(users=users_df_wrong_feature, top_k=1)


@pytest.mark.uplift
def test_s_learner_with_item_subset(trained_s_learner):
    """Tests the item_subset functionality for the S-Learner."""
    users_df = sample_uplift_users.fetch_data()

    trained_s_learner.set_item_subset(["item_a"])

    scores = trained_s_learner.score_items(users=users_df)
    assert list(scores.columns) == ["item_a"]

    recommendations = trained_s_learner.recommend(users=users_df, top_k=1)
    assert recommendations[0, 0] in ["item_a", "control"]

    trained_s_learner.clear_item_subset()
    scores_after_clear = trained_s_learner.score_items(users=users_df)
    assert set(scores_after_clear.columns) == {"item_a", "item_b"}


@pytest.mark.uplift
def test_s_learner_with_validation_data():
    """Tests that training with validation data doesn't destroy the shared scorer."""
    from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator

    scorer = UniversalScorer(estimator=XGBClassifierEstimator({"max_depth": 2, "verbosity": 0}))
    recommender = UpliftRecommender(scorer=scorer, control_item_id="control")
    recommender.train(
        interactions_ds=sample_uplift_interactions,
        users_ds=sample_uplift_users,
        items_ds=None,
        valid_interactions_ds=sample_uplift_interactions,
        valid_users_ds=sample_uplift_users,
    )
    users_df = sample_uplift_users.fetch_data()

    uplift_scores = recommender.score_items(users=users_df)
    assert uplift_scores.shape[0] == users_df.shape[0]
    assert set(uplift_scores.columns) == {"item_a", "item_b"}

    recommendations = recommender.recommend(users=users_df, top_k=1)
    assert recommendations.shape == (users_df.shape[0], 1)


@pytest.mark.uplift
def test_s_learner_recommends_control_on_negative_uplift(trained_s_learner):
    """
    Tests that 'control' is recommended when all treatment uplifts are negative.
    """
    users_df = sample_uplift_users.fetch_data()

    negative_scores = pd.DataFrame({"item_a": [-0.5, -0.2], "item_b": [-0.1, -0.8]}, index=users_df.index[:2])
    trained_s_learner._score_items_np = MagicMock(return_value=negative_scores.to_numpy())

    recommendations = trained_s_learner.recommend(users=users_df.head(2), top_k=1)

    assert recommendations.shape == (2, 1)
    assert recommendations[0, 0] == "control"
    assert recommendations[1, 0] == "control"


@pytest.mark.uplift
def test_s_learner_logs_warning_for_negative_uplift(trained_s_learner, caplog):
    """Tests that a log message is printed when returning non-positive recommendations."""
    users_df = sample_uplift_users.fetch_data()

    # Mock scores where one user gets a positive score, the other gets negative
    mixed_scores = pd.DataFrame({"item_a": [0.5, -0.2], "item_b": [0.1, -0.8]}, index=users_df.index[:2])

    trained_s_learner._score_items_np = MagicMock(return_value=mixed_scores.to_numpy())

    with caplog.at_level(logging.INFO):
        trained_s_learner.recommend(users=users_df.head(2), top_k=2)

    # The user with negative scores will get 'control' (score=0), triggering the warning
    assert "positive uplift was less than top_k" in caplog.text
