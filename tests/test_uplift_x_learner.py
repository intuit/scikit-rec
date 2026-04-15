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
from skrec.scorer.universal import UniversalScorer


@pytest.fixture
def x_learner_factory():
    """A factory fixture that creates and trains an X-Learner on demand."""

    def _create_and_train(estimator):
        scorer = IndependentScorer(estimator=estimator)
        recommender = UpliftRecommender(scorer=scorer, control_item_id="control", mode="x_learner")
        recommender.train(
            interactions_ds=sample_uplift_interactions,
            users_ds=sample_uplift_users,
            items_ds=None,
        )
        return recommender

    return _create_and_train


@pytest.mark.uplift
def test_x_learner_scoring_and_recommendation(x_learner_factory):
    """Tests the basic X-Learner scoring and recommendation flow."""
    recommender = x_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df = sample_uplift_users.fetch_data()

    uplift_scores = recommender.score_items(users=users_df)
    assert uplift_scores.shape[0] == users_df.shape[0]
    assert set(uplift_scores.columns) == {"item_a", "item_b"}

    recommendations = recommender.recommend(users=users_df, top_k=1)
    assert recommendations.shape == (users_df.shape[0], 1)


@pytest.mark.uplift
def test_x_learner_recommend_more_than_available(x_learner_factory):
    """Tests that the recommender returns the max available items when top_k is too high."""
    recommender = x_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df = sample_uplift_users.fetch_data()

    recommendations = recommender.recommend(users=users_df, top_k=4)
    assert recommendations.shape[1] == 3


@pytest.mark.uplift
def test_x_learner_with_item_subset(x_learner_factory):
    """Tests the item_subset functionality for the X-Learner."""
    recommender = x_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df = sample_uplift_users.fetch_data()

    recommender.set_item_subset(["item_a"])

    scores = recommender.score_items(users=users_df)
    assert list(scores.columns) == ["item_a"]

    recommender.clear_item_subset()
    scores_after_clear = recommender.score_items(users=users_df)
    assert set(scores_after_clear.columns) == {"item_a", "item_b"}


@pytest.mark.uplift
def test_x_learner_recommends_control_on_negative_uplift(x_learner_factory):
    """Tests that 'control' is recommended when all treatment uplifts are negative."""
    recommender = x_learner_factory(estimator=LogisticRegressionClassifierEstimator({}))
    users_df = sample_uplift_users.fetch_data()

    negative_scores = pd.DataFrame({"item_a": [-0.5, -0.2], "item_b": [-0.1, -0.8]}, index=users_df.index[:2])
    recommender._score_items_np = MagicMock(return_value=negative_scores.to_numpy())

    recommendations = recommender.recommend(users=users_df.head(2), top_k=1)

    assert recommendations.shape == (2, 1)
    assert recommendations[0, 0] == "control"
    assert recommendations[1, 0] == "control"


@pytest.mark.uplift
def test_x_learner_with_validation_data():
    """Tests that training with validation data works for X-Learner."""
    scorer = IndependentScorer(estimator=XGBClassifierEstimator({"max_depth": 2, "verbosity": 0}))
    recommender = UpliftRecommender(scorer=scorer, control_item_id="control", mode="x_learner")
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


@pytest.mark.uplift
def test_x_learner_requires_independent_scorer():
    """X-Learner must raise TypeError if given a UniversalScorer."""
    scorer = UniversalScorer(estimator=LogisticRegressionClassifierEstimator({}))
    with pytest.raises(TypeError, match="X-Learner requires IndependentScorer"):
        UpliftRecommender(scorer=scorer, control_item_id="control", mode="x_learner")


@pytest.mark.uplift
def test_x_learner_invalid_mode():
    """Unknown mode string must raise ValueError."""
    scorer = IndependentScorer(estimator=LogisticRegressionClassifierEstimator({}))
    with pytest.raises(ValueError, match="Unknown mode"):
        UpliftRecommender(scorer=scorer, control_item_id="control", mode="invalid")
