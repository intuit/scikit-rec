import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
from skrec.recommender.gcsl.gcsl_recommender import GcslRecommender
from skrec.recommender.gcsl.inference.base_inference import NotFittedError
from skrec.recommender.gcsl.inference.mean_scalarization import MeanScalarization
from skrec.recommender.gcsl.inference.percentile_value import PercentileValue
from skrec.recommender.gcsl.inference.predefined_value import PredefinedValue
from skrec.retriever.popularity_retriever import PopularityRetriever
from skrec.scorer.universal import UniversalScorer
from skrec.util.config_loader import load_config
from tests.utils import parse_config


@pytest.fixture
def setup_fixture(setup_small_datasets):
    interactions_data_loc = setup_small_datasets["dst"] / setup_small_datasets["multioutcome_interactions_data"]
    setup_small_datasets["multioutcome_interactions_dataset"] = InteractionsDataset(interactions_data_loc)
    estimator_config_dir = Path.cwd() / "skrec/examples/estimators/"
    estimator_config = load_config(estimator_config_dir / "estimator_hyperparameters.yaml")
    _, _, _, setup_small_datasets["xgb_params"] = parse_config(estimator_config, "XGBoostClassifier")
    # Training data has OUTCOME_1 = OUTCOME_2 = 1.5 (constant), so training range is [1.5, 1.5].
    # A scalar of 1.0 gives goal = 1.5 exactly (in-range); any other scalar triggers OOD warning.
    inference_method = MeanScalarization({"OUTCOME_1": 1.0, "OUTCOME_2": 1.0})

    estimator = XGBClassifierEstimator(setup_small_datasets["xgb_params"])
    scorer = UniversalScorer(estimator)
    recommender = GcslRecommender(scorer, inference_method)

    setup_small_datasets["scorer"] = scorer
    setup_small_datasets["recommender"] = recommender
    return setup_small_datasets


# ---------------------------------------------------------------------------
# Existing tests — updated for new API
# ---------------------------------------------------------------------------


def test_recommend_isolated_strategy(setup_fixture):
    recommender = GcslRecommender(setup_fixture["scorer"])
    recommender.train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["multioutcome_interactions_dataset"],
    )

    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["multioutcome_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)

    # set_inference_method auto-fits using the stored training data
    recommender.set_inference_method(PredefinedValue({"OUTCOME_1": 1.5, "OUTCOME_2": 1.5}))
    ranked_items = recommender.recommend(interactions, users, top_k=2)
    assert_array_equal(ranked_items.shape, (4, 2))

    # Missing scalar key raises at fit time; previous inference method is restored
    good_method = recommender.inference_method
    with pytest.raises(ValueError, match="No scalar provided for outcome column 'OUTCOME_1'"):
        recommender.set_inference_method(MeanScalarization({"OUTCOME_2": 1.0}))
    assert recommender.inference_method is good_method  # rollback verified

    # Calling transform() before fit() raises NotFittedError
    unfitted = MeanScalarization({"OUTCOME_1": 1.0, "OUTCOME_2": 1.0})
    with pytest.raises(NotFittedError):
        unfitted.transform(interactions)


def test_empty_outcome_cols_raises(setup_fixture):
    """_prepare_interactions raises when outcome_cols is empty."""
    setup_fixture["recommender"].train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["multioutcome_interactions_dataset"],
    )
    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["multioutcome_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)
    setup_fixture["recommender"].outcome_cols = []
    with pytest.raises(ValueError, match="Outcome column names are not set"):
        setup_fixture["recommender"].recommend(interactions, users, top_k=2)


def test_recommend_predefined_value(setup_fixture):
    inference_method = PredefinedValue({"OUTCOME_1": 1.5, "OUTCOME_2": 1.5})
    recommender_predefined_value = GcslRecommender(
        setup_fixture["scorer"],
        inference_method,
    )
    recommender_predefined_value.train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["multioutcome_interactions_dataset"],
    )

    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["multioutcome_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)

    ranked_items = recommender_predefined_value.recommend(interactions, users, top_k=2)
    assert_array_equal(ranked_items.shape, (4, 2))


def test_recommend(setup_fixture):
    setup_fixture["recommender"].train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["multioutcome_interactions_dataset"],
    )

    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["multioutcome_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)

    ranked_items = setup_fixture["recommender"].recommend(interactions, users, top_k=2)
    assert_array_equal(ranked_items.shape, (4, 2))

    wrong_users = users.drop(columns=["Age"])
    with pytest.raises(RuntimeError, match="Column 'Age' not found in dataset"):
        setup_fixture["recommender"].recommend(interactions, wrong_users, top_k=2)


def test_evaluate(setup_fixture):
    setup_fixture["recommender"].train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["multioutcome_interactions_dataset"],
    )

    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["multioutcome_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)

    metric_type = RecommenderMetricType.AVERAGE_REWARD_AT_K
    eval_kwargs = {
        "logged_rewards": np.array([[0.0], [1.0], [0.0], [1.0]], dtype=float),
        "logged_items": np.array([["Item3"], ["Item1"], ["Item3"], ["Item1"]], dtype=object),
    }

    simple_eval = setup_fixture["recommender"].evaluate(
        RecommenderEvaluatorType.REPLAY_MATCH,
        metric_type,
        eval_top_k=3,
        eval_kwargs=eval_kwargs,
        score_items_kwargs={"interactions": interactions, "users": users},
    )
    # recommendations scores are all 0.5 so ranking is arbitrary
    assert simple_eval == 0.5


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------


def test_recommend_online_raises(setup_fixture):
    """recommend_online() is not supported — must raise NotImplementedError."""
    setup_fixture["recommender"].train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["multioutcome_interactions_dataset"],
    )
    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["multioutcome_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)

    with pytest.raises(NotImplementedError, match="recommend_online\\(\\) is not supported"):
        setup_fixture["recommender"].recommend_online(interactions, users)


def test_goal_values_injected_into_interactions(setup_fixture):
    """transform() overwrites outcome columns with goal values and does not mutate the input."""
    # Input has the original training values (1.5). Goals are deliberately different (0.0)
    # so we can confirm the write actually happened — not just that values happened to match.
    interactions = pd.DataFrame(
        {
            "USER_ID": ["John"],
            "Context1": [1],
            "Context2": [0.1],
            "OUTCOME_1": [1.5],
            "OUTCOME_2": [1.5],
        }
    )
    outcome_cols = ["OUTCOME_1", "OUTCOME_2"]
    training_df = setup_fixture["multioutcome_interactions_dataset"].fetch_data()

    method = PredefinedValue({"OUTCOME_1": 0.0, "OUTCOME_2": 0.0})
    method.fit(training_df, outcome_cols)

    # Goals (0.0) are below training min (1.5) → OOD warning expected
    with pytest.warns(UserWarning, match="outside the training range"):
        result = method.transform(interactions)

    assert (result["OUTCOME_1"] == 0.0).all()
    assert (result["OUTCOME_2"] == 0.0).all()
    # Original DataFrame must not be mutated
    assert interactions["OUTCOME_1"].iloc[0] == 1.5
    assert interactions["OUTCOME_2"].iloc[0] == 1.5


def test_ood_warning_emitted(setup_fixture):
    """A UserWarning is emitted when a goal is outside the observed training range."""
    # Training range is [1.5, 1.5] — any other value triggers the warning.
    inference_method = PredefinedValue({"OUTCOME_1": 9999.0, "OUTCOME_2": 9999.0})
    recommender = GcslRecommender(setup_fixture["scorer"], inference_method)
    recommender.train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["multioutcome_interactions_dataset"],
    )
    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["multioutcome_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)

    with pytest.warns(UserWarning, match="outside the training range"):
        recommender.recommend(interactions, users, top_k=2)


def test_not_fitted_error():
    """transform() raises NotFittedError when called before fit() for all inference methods."""
    interactions = pd.DataFrame({"OUTCOME_1": [1.5], "OUTCOME_2": [1.5]})

    for method in [
        MeanScalarization({"OUTCOME_1": 1.0, "OUTCOME_2": 1.0}),
        PredefinedValue({"OUTCOME_1": 1.5, "OUTCOME_2": 1.5}),
        PercentileValue({"OUTCOME_1": 80, "OUTCOME_2": 90}),
    ]:
        with pytest.raises(NotFittedError):
            method.transform(interactions)


def test_percentile_value_end_to_end(setup_fixture):
    """PercentileValue produces valid recommendations without OOD warnings."""
    inference_method = PercentileValue({"OUTCOME_1": 50, "OUTCOME_2": 75})
    recommender = GcslRecommender(setup_fixture["scorer"], inference_method)
    recommender.train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["multioutcome_interactions_dataset"],
    )
    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["multioutcome_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)

    # Percentile goals are in-range by construction — no warning should be emitted
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ranked_items = recommender.recommend(interactions, users, top_k=2)

    assert_array_equal(ranked_items.shape, (4, 2))
    # Computed goals are real values derived from the training data
    assert "OUTCOME_1" in inference_method.goal_values_
    assert "OUTCOME_2" in inference_method.goal_values_


def test_percentile_value_invalid_range():
    """PercentileValue rejects percentiles outside [0, 100] at construction time."""
    with pytest.raises(ValueError, match="between 0 and 100"):
        PercentileValue({"OUTCOME_1": 150})

    with pytest.raises(ValueError, match="between 0 and 100"):
        PercentileValue({"OUTCOME_1": -5})


def test_missing_outcome_raises_at_fit(setup_fixture):
    """Missing outcome column in inference params raises at fit time, not at recommend time."""
    training_df = setup_fixture["multioutcome_interactions_dataset"].fetch_data()
    outcome_cols = ["OUTCOME_1", "OUTCOME_2"]

    with pytest.raises(ValueError, match="No scalar provided for outcome column 'OUTCOME_1'"):
        MeanScalarization({"OUTCOME_2": 1.0}).fit(training_df, outcome_cols)

    with pytest.raises(ValueError, match="No goal value provided for outcome column 'OUTCOME_1'"):
        PredefinedValue({"OUTCOME_2": 1.5}).fit(training_df, outcome_cols)

    with pytest.raises(ValueError, match="No percentile provided for outcome column 'OUTCOME_1'"):
        PercentileValue({"OUTCOME_2": 80}).fit(training_df, outcome_cols)


def test_recommend_with_retriever(setup_fixture):
    """recommend() with a retriever injects goal values before the per-user loop."""
    inference_method = PredefinedValue({"OUTCOME_1": 1.5, "OUTCOME_2": 1.5})
    retriever = PopularityRetriever(top_k=2)
    recommender = GcslRecommender(
        setup_fixture["scorer"],
        inference_method=inference_method,
        retriever=retriever,
    )
    recommender.train(
        setup_fixture["users_dataset"],
        setup_fixture["items_dataset"],
        setup_fixture["multioutcome_interactions_dataset"],
    )

    users = setup_fixture["users_dataset"].fetch_data()
    interactions = setup_fixture["multioutcome_interactions_dataset"].fetch_data().drop(["ITEM_ID", "OUTCOME"], axis=1)

    ranked_items = recommender.recommend(interactions, users, top_k=2)
    # Retriever path returns one row per unique user (3), not per interaction row (4).
    assert ranked_items.shape == (3, 2)
