import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from skrec.estimator.classification.logreg_classifier import (
    LogisticRegressionClassifierEstimator,
)
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.examples.datasets import sample_uplift_interactions, sample_uplift_users
from skrec.recommender.uplift_model.uplift_recommender import UpliftRecommender
from skrec.scorer.independent import IndependentScorer


@pytest.fixture
def t_learner_factory():
    """A factory fixture that creates and trains a T-Learner on demand."""

    def _create_and_train(estimator):
        scorer = IndependentScorer(estimator=estimator)
        recommender = UpliftRecommender(scorer=scorer, control_item_id="control")
        recommender.train(
            interactions_ds=sample_uplift_interactions,
            users_ds=sample_uplift_users,
            items_ds=None,
        )
        return recommender

    return _create_and_train


@pytest.mark.uplift
def test_t_learner_with_factory(t_learner_factory):
    """Tests the full flow using a single estimator."""
    recommender = t_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df = sample_uplift_users.fetch_data()

    uplift_scores = recommender.score_items(users=users_df)
    assert uplift_scores.shape[0] == users_df.shape[0]
    assert set(uplift_scores.columns) == {"item_a", "item_b"}


@pytest.mark.uplift
def test_t_learner_with_estimator_dict(t_learner_factory):
    """Tests the full flow using an estimator_dict."""
    estimator_dict = {
        "control": XGBClassifierEstimator({"max_depth": 3, "verbosity": 0}),
        "item_a": XGBClassifierEstimator({"max_depth": 5, "verbosity": 0}),
        "item_b": XGBClassifierEstimator({"max_depth": 2, "verbosity": 0}),
    }
    recommender = t_learner_factory(estimator=estimator_dict)
    users_df = sample_uplift_users.fetch_data()

    uplift_scores = recommender.score_items(users=users_df)
    assert uplift_scores.shape[0] == users_df.shape[0]
    assert set(uplift_scores.columns) == {"item_a", "item_b"}


@pytest.mark.uplift
def test_t_learner_recommend_more_than_available(t_learner_factory):
    """
    Tests that the recommender returns the max available items when top_k is too high.
    """
    recommender = t_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df = sample_uplift_users.fetch_data()

    # There are 3 options (item_a, item_b, control), so ask for 4
    recommendations = recommender.recommend(users=users_df, top_k=4)

    # The recommender should return the max available items, which is 3
    assert recommendations.shape[1] == 3


@pytest.mark.uplift
def test_t_learner_with_unknown_feature(t_learner_factory):
    """
    Tests that the recommender raises an error if the input dataframe has wrong columns.
    """
    recommender = t_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df_wrong_feature = pd.DataFrame({"USER_ID": ["user_1", "user_2"], "wrong_feature_name": [15, 25]})

    with pytest.raises(Exception):
        recommender.recommend(users=users_df_wrong_feature, top_k=1)


@pytest.mark.uplift
def test_t_learner_with_item_subset(t_learner_factory):
    """Tests the item_subset functionality for the T-Learner."""
    recommender = t_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df = sample_uplift_users.fetch_data()

    recommender.set_item_subset(["item_a"])

    scores = recommender.score_items(users=users_df)
    assert list(scores.columns) == ["item_a"]

    recommendations = recommender.recommend(users=users_df, top_k=1)
    assert recommendations[0, 0] in ["item_a", "control"]

    recommender.clear_item_subset()
    scores_after_clear = recommender.score_items(users=users_df)
    assert set(scores_after_clear.columns) == {"item_a", "item_b"}


@pytest.mark.uplift
def test_t_learner_with_validation_data():
    """Tests that training with validation data doesn't destroy scorers."""
    scorer = IndependentScorer(estimator=XGBClassifierEstimator({"max_depth": 2, "verbosity": 0}))
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
def test_t_learner_recommends_control_on_negative_uplift(t_learner_factory):
    """
    Tests that 'control' is recommended when all treatment uplifts are negative.
    """
    recommender = t_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df = sample_uplift_users.fetch_data()

    negative_scores = pd.DataFrame({"item_a": [-0.5, -0.2], "item_b": [-0.1, -0.8]}, index=users_df.index[:2])

    recommender._score_items_np = MagicMock(return_value=negative_scores.to_numpy())

    recommendations = recommender.recommend(users=users_df.head(2), top_k=1)

    assert recommendations.shape == (2, 1)
    assert recommendations[0, 0] == "control"
    assert recommendations[1, 0] == "control"


@pytest.mark.uplift
def test_t_learner_logs_warning_for_negative_uplift(t_learner_factory, caplog):
    """Tests that a log message is printed when returning non-positive recommendations."""
    recommender = t_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df = sample_uplift_users.fetch_data()

    mixed_scores = pd.DataFrame({"item_a": [0.5, -0.2], "item_b": [0.1, -0.8]}, index=users_df.index[:2])

    recommender._score_items_np = MagicMock(return_value=mixed_scores.to_numpy())

    with caplog.at_level(logging.INFO):
        recommender.recommend(users=users_df.head(2), top_k=2)

    assert "positive uplift was less than top_k" in caplog.text


@pytest.mark.uplift
def test_recommend_online_raises_not_implemented(t_learner_factory):
    """recommend_online() must raise NotImplementedError for uplift recommenders."""
    recommender = t_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df = sample_uplift_users.fetch_data()
    with pytest.raises(NotImplementedError):
        recommender.recommend_online(users=users_df)
