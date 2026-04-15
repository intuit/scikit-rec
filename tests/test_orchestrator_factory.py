import logging
from unittest.mock import MagicMock, patch

import pytest

from skrec.estimator.base_estimator import BaseEstimator
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.classification.multioutput_classifier import (
    MultiOutputClassifierEstimator,
)
from skrec.estimator.classification.xgb_classifier import (
    WeightedXGBClassifierEstimator,
    XGBClassifierEstimator,
)
from skrec.estimator.regression.xgb_regressor import XGBRegressorEstimator
from skrec.orchestrator.factory import (
    create_estimator,
    create_recommender,
    create_recommender_pipeline,
    create_scorer,
)
from skrec.recommender.bandits.contextual_bandits import ContextualBanditsRecommender
from skrec.recommender.base_recommender import BaseRecommender
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.base_scorer import BaseScorer
from skrec.scorer.independent import IndependentScorer
from skrec.scorer.multiclass import MulticlassScorer
from skrec.scorer.multioutput import MultioutputScorer
from skrec.scorer.universal import UniversalScorer

# --- Tests for create_estimator ---


@pytest.mark.parametrize(
    "estimator_config, scorer_type, expected_type, expected_checks",
    [
        # Simple Classification (Default)
        ({}, None, XGBClassifierEstimator, {"check_xgb_params": {}}),
        # Simple Classification (Explicit)
        ({"ml_task": "classification"}, None, XGBClassifierEstimator, {"check_xgb_params": {}}),
        # Simple Classification with XGB params
        (
            {"ml_task": "classification", "xgboost": {"n_estimators": 150, "learning_rate": 0.05}},
            None,
            XGBClassifierEstimator,
            {"check_xgb_params": {"n_estimators": 150, "learning_rate": 0.05}},
        ),
        # Weighted Classification (action_weight only)
        (
            {"ml_task": "classification", "weights": {"action_weight": 0.8}, "xgboost": {"colsample_bynode": 0.9}},
            None,
            WeightedXGBClassifierEstimator,
            {"action_weight": 0.8, "item_sample_weights": None, "check_xgb_params": {"colsample_bynode": 0.9}},
        ),
        # Weighted Classification (item_sample_weights only) - action_weight defaults to 1
        (
            {"ml_task": "classification", "weights": {"item_sample_weights": {"itemA": 1.2}}},
            None,
            WeightedXGBClassifierEstimator,
            {"action_weight": 1, "item_sample_weights": {"itemA": 1.2}, "check_xgb_params": {}},
        ),
        # Weighted Classification (both weights)
        (
            {
                "ml_task": "classification",
                "weights": {"action_weight": 0.8, "item_sample_weights": {"itemA": 1.2}},
                "xgboost": {"n_estimators": 50, "colsample_bynode": 0.7},
            },
            None,
            WeightedXGBClassifierEstimator,
            {
                "action_weight": 0.8,
                "item_sample_weights": {"itemA": 1.2},
                "check_xgb_params": {"n_estimators": 50, "colsample_bynode": 0.7},
            },
        ),
        # MultiOutput Classification (scorer_type provided)
        (
            {"ml_task": "classification", "xgboost": {"n_estimators": 120}},
            "multioutput",  # scorer_type needed to trigger MultiOutput
            MultiOutputClassifierEstimator,
            {},  # Keep checks minimal for factory test - just verify type
        ),
        # Simple Regression
        (
            {"ml_task": "regression", "xgboost": {"n_estimators": 100}},
            None,
            XGBRegressorEstimator,
            {"check_xgb_params": {"n_estimators": 100}},
        ),
    ],
)
def test_create_estimator_success_cases(estimator_config, scorer_type, expected_type, expected_checks):
    """Test successful creation of various non-tuned estimator types."""
    estimator = create_estimator(estimator_config, scorer_type=scorer_type)
    assert isinstance(estimator, expected_type)

    # Check attributes based on expected_checks dictionary structure
    for key, expected_value in expected_checks.items():
        if key == "check_xgb_params":
            # Check params passed to the underlying XGBoost model
            assert hasattr(estimator, "_model"), f"Estimator {type(estimator).__name__} missing '_model' attribute"
            model_params = estimator._model.get_params()
            for k, v in expected_value.items():
                assert k in model_params, f"Model {type(estimator._model).__name__} missing param '{k}' in get_params()"
                assert model_params[k] == v, f"Model param '{k}' mismatch: expected {v}, got {model_params[k]}"
        else:
            # Check attributes directly on the estimator wrapper (e.g., action_weight)
            assert hasattr(estimator, key), f"Estimator missing attribute '{key}'"
            actual_value = getattr(estimator, key)
            assert actual_value == expected_value, (
                f"Attribute '{key}' mismatch: expected {expected_value}, got {actual_value}"
            )


# --- Tests for create_scorer ---


@pytest.fixture
def mock_estimator():
    """Fixture for a mock BaseClassifier."""
    # Mock BaseClassifier so isinstance checks pass in create_scorer
    mock = MagicMock(spec=BaseClassifier)
    # Make the mock appear as an instance of BaseClassifier
    mock.__class__ = BaseClassifier
    return mock


@pytest.mark.parametrize(
    "config, expected_type",
    [
        ({"scorer_type": "multioutput"}, MultioutputScorer),
        ({"scorer_type": "multiclass"}, MulticlassScorer),
        ({"scorer_type": "independent"}, IndependentScorer),
        ({"scorer_type": "universal"}, UniversalScorer),
    ],
)
def test_create_scorer_success_cases(mock_estimator, config, expected_type):
    """Test successful creation of various scorer types."""
    scorer = create_scorer(mock_estimator, config)
    assert isinstance(scorer, expected_type)
    assert scorer.estimator == mock_estimator


@pytest.mark.parametrize(
    "recommender_config, expected_error, match_pattern",
    [
        (
            {"scorer_type": "unknown_scorer"},
            NotImplementedError,
            r"Scorer type 'unknown_scorer' not supported\.",
        ),
        (
            {},  # Missing scorer_type in full config
            ValueError,
            r"'scorer_type' must be specified in the configuration\.",
        ),
        (
            {"scorer_type": None},  # scorer_type is None in full config
            ValueError,
            r"'scorer_type' must be specified in the configuration\.",
        ),
    ],
)
def test_create_scorer_error_cases(mock_estimator, recommender_config, expected_error, match_pattern):
    """Test error conditions during scorer creation."""
    with pytest.raises(expected_error, match=match_pattern):
        create_scorer(mock_estimator, recommender_config)


# --- Tests for create_recommender ---


@pytest.fixture
def mock_scorer():
    """Fixture for a mock BaseScorer."""
    return MagicMock(spec=BaseScorer)


@pytest.mark.parametrize(
    "config, expected_type",
    [
        ({"recommender_type": "bandits"}, ContextualBanditsRecommender),
        ({"recommender_type": "ranking"}, RankingRecommender),
        ({}, RankingRecommender),  # Test default
        ({"recommender_type": None}, RankingRecommender),  # Test default when None
        ({"recommender_type": "unsupported_type"}, RankingRecommender),  # Test fallback to default
    ],
)
def test_create_recommender_success_cases(mock_scorer, config, expected_type):
    """Test successful creation of various recommender types and defaults."""
    recommender = create_recommender(mock_scorer, config)
    assert isinstance(recommender, expected_type)
    assert recommender.scorer == mock_scorer


def test_create_recommender_warning_on_unsupported(mock_scorer, caplog):
    """Test that a warning is logged for unsupported recommender types."""
    config = {"recommender_type": "fancy_new_recommender"}
    with caplog.at_level(logging.WARNING):  # Use logging.WARNING
        recommender = create_recommender(mock_scorer, config)
    assert isinstance(recommender, RankingRecommender)  # Falls back to default
    assert "Unsupported recommender_type 'fancy_new_recommender'. Defaulting to RankingRecommender." in caplog.text


# --- Tests for create_recommender_pipeline ---


@patch("skrec.orchestrator.factory.create_estimator")
@patch("skrec.orchestrator.factory.create_scorer")
@patch("skrec.orchestrator.factory.create_recommender")
def test_create_recommender_pipeline_success(mock_create_recommender, mock_create_scorer, mock_create_estimator):
    """Test successful creation of the full pipeline."""
    mock_est = MagicMock(spec=BaseEstimator)
    mock_sco = MagicMock(spec=BaseScorer)
    mock_rec = MagicMock(spec=BaseRecommender)

    mock_create_estimator.return_value = mock_est
    mock_create_scorer.return_value = mock_sco
    mock_create_recommender.return_value = mock_rec

    # Config structure matching factory expectations
    config = {
        "estimator_config": {"ml_task": "classification", "xgboost": {}, "weights": {}, "hpo": {}},
        "scorer_type": "independent",
        "recommender_type": "ranking",
        # Other top-level keys if needed by scorer/recommender
    }
    pipeline = create_recommender_pipeline(config)

    # Estimator receives estimator_config and scorer_type hint
    mock_create_estimator.assert_called_once_with(config["estimator_config"], scorer_type=config["scorer_type"])
    # Scorer and Recommender receive the full config
    mock_create_scorer.assert_called_once_with(mock_est, config)
    mock_create_recommender.assert_called_once_with(mock_sco, config)
    assert pipeline == mock_rec


@patch("skrec.orchestrator.factory.create_estimator")
def test_create_recommender_pipeline_estimator_error(mock_create_estimator):
    """Test that errors from create_estimator propagate."""
    mock_create_estimator.side_effect = ValueError("Estimator config error")
    # Config structure matching factory expectations
    config = {
        "estimator_config": {"ml_task": "invalid"},
        "scorer_type": "independent",  # Need scorer_type for the call signature
    }
    with pytest.raises(ValueError, match="Estimator config error"):
        create_recommender_pipeline(config)
    # Check that create_estimator was called with estimator_config and scorer_type
    mock_create_estimator.assert_called_once_with(config["estimator_config"], scorer_type=config["scorer_type"])


@patch("skrec.orchestrator.factory.create_estimator")
@patch("skrec.orchestrator.factory.create_scorer")
def test_create_recommender_pipeline_scorer_error(mock_create_scorer, mock_create_estimator):
    """Test that errors from create_scorer propagate."""
    mock_est = MagicMock(spec=BaseEstimator)
    mock_create_estimator.return_value = mock_est
    mock_create_scorer.side_effect = NotImplementedError("Scorer type error")

    # Config structure matching factory expectations
    config = {
        "estimator_config": {"ml_task": "classification", "xgboost": {}, "weights": {}, "hpo": {}},
        "scorer_type": "invalid",  # Scorer type at top level
        "recommender_type": "ranking",  # Need recommender_type as well
    }
    with pytest.raises(NotImplementedError, match="Scorer type error"):
        create_recommender_pipeline(config)
    # Estimator receives estimator_config and scorer_type hint
    mock_create_estimator.assert_called_once_with(config["estimator_config"], scorer_type=config["scorer_type"])
    # Scorer receives the full config
    mock_create_scorer.assert_called_once_with(mock_est, config)
