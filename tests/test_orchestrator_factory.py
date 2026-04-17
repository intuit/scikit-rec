import importlib
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
from skrec.estimator.embedding.contextualized_two_tower_estimator import ContextualizedTwoTowerEstimator
from skrec.estimator.embedding.deep_cross_network_estimator import DeepCrossNetworkEstimator
from skrec.estimator.embedding.matrix_factorization_estimator import MatrixFactorizationEstimator
from skrec.estimator.embedding.ncf_estimator import NCFEstimator
from skrec.estimator.embedding.neural_factorization_estimator import NeuralFactorizationEstimator
from skrec.estimator.regression.xgb_regressor import XGBRegressorEstimator
from skrec.estimator.sequential.base_sequential_estimator import SequentialEstimator
from skrec.estimator.sequential.hrnn_estimator import HRNNClassifierEstimator, HRNNRegressorEstimator
from skrec.estimator.sequential.sasrec_estimator import SASRecClassifierEstimator, SASRecRegressorEstimator
from skrec.orchestrator.factory import (
    _create_inference_method,
    _create_retriever,
    create_estimator,
    create_recommender,
    create_recommender_pipeline,
    create_scorer,
)
from skrec.recommender.bandits.contextual_bandits import ContextualBanditsRecommender
from skrec.recommender.base_recommender import BaseRecommender
from skrec.recommender.gcsl.gcsl_recommender import GcslRecommender
from skrec.recommender.gcsl.inference.mean_scalarization import MeanScalarization
from skrec.recommender.gcsl.inference.percentile_value import PercentileValue
from skrec.recommender.gcsl.inference.predefined_value import PredefinedValue
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.recommender.sequential.hierarchical_recommender import HierarchicalSequentialRecommender
from skrec.recommender.sequential.sequential_recommender import SequentialRecommender
from skrec.recommender.uplift_model.uplift_recommender import UpliftRecommender
from skrec.retriever.content_based_retriever import ContentBasedRetriever
from skrec.retriever.embedding_retriever import EmbeddingRetriever
from skrec.retriever.popularity_retriever import PopularityRetriever
from skrec.scorer.base_scorer import BaseScorer
from skrec.scorer.hierarchical import HierarchicalScorer
from skrec.scorer.independent import IndependentScorer
from skrec.scorer.multiclass import MulticlassScorer
from skrec.scorer.multioutput import MultioutputScorer
from skrec.scorer.sequential import SequentialScorer
from skrec.scorer.universal import UniversalScorer

_torch_available = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")

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
    ],
)
def test_create_recommender_success_cases(mock_scorer, config, expected_type):
    """Test successful creation of various recommender types."""
    recommender = create_recommender(mock_scorer, config)
    assert isinstance(recommender, expected_type)
    assert recommender.scorer == mock_scorer


@pytest.mark.parametrize(
    "config, expected_error, match_pattern",
    [
        (
            {"recommender_type": "fancy_new_recommender"},
            NotImplementedError,
            r"Recommender type 'fancy_new_recommender' not supported",
        ),
        (
            {},
            ValueError,
            r"'recommender_type' must be specified in the configuration\.",
        ),
        (
            {"recommender_type": None},
            ValueError,
            r"'recommender_type' must be specified in the configuration\.",
        ),
    ],
)
def test_create_recommender_error_cases(mock_scorer, config, expected_error, match_pattern):
    """Test error conditions during recommender creation."""
    with pytest.raises(expected_error, match=match_pattern):
        create_recommender(mock_scorer, config)


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
        "scorer_type": "independent",
        "recommender_type": "ranking",
    }
    with pytest.raises(ValueError, match="Estimator config error"):
        create_recommender_pipeline(config)
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


# --- Tests for embedding estimators ---


def test_create_embedding_estimator_matrix_factorization():
    """Test MF embedding estimator (pure NumPy, no torch required)."""
    config = {
        "estimator_type": "embedding",
        "embedding": {"model_type": "matrix_factorization", "params": {}},
    }
    estimator = create_estimator(config)
    assert isinstance(estimator, MatrixFactorizationEstimator)


@requires_torch
@pytest.mark.parametrize(
    "model_type, expected_cls",
    [
        ("ncf", NCFEstimator),
        ("two_tower", ContextualizedTwoTowerEstimator),
        ("deep_cross_network", DeepCrossNetworkEstimator),
        ("neural_factorization", NeuralFactorizationEstimator),
    ],
)
def test_create_embedding_estimator_torch(model_type, expected_cls):
    """Test PyTorch-based embedding estimators (requires torch)."""
    config = {
        "estimator_type": "embedding",
        "embedding": {"model_type": model_type, "params": {}},
    }
    estimator = create_estimator(config)
    assert isinstance(estimator, expected_cls)


def test_create_embedding_estimator_with_params():
    """Test embedding estimator with custom params."""
    config = {
        "estimator_type": "embedding",
        "embedding": {
            "model_type": "matrix_factorization",
            "params": {"n_factors": 64, "epochs": 50},
        },
    }
    estimator = create_estimator(config)
    assert isinstance(estimator, MatrixFactorizationEstimator)
    assert estimator.n_factors == 64
    assert estimator.epochs == 50


def test_create_embedding_estimator_missing_config():
    """Test that missing embedding key raises ValueError."""
    with pytest.raises(ValueError, match="'embedding' key is required"):
        create_estimator({"estimator_type": "embedding"})


def test_create_embedding_estimator_invalid_model_type():
    """Test that invalid embedding model_type raises NotImplementedError."""
    config = {
        "estimator_type": "embedding",
        "embedding": {"model_type": "unknown_model"},
    }
    with pytest.raises(NotImplementedError, match="Embedding model type 'unknown_model' not supported"):
        create_estimator(config)


def test_create_embedding_estimator_empty_section():
    """Test that embedding section present but empty raises ValueError with specific message."""
    config = {
        "estimator_type": "embedding",
        "embedding": {},
    }
    with pytest.raises(ValueError, match="'embedding' config is empty"):
        create_estimator(config)


def test_create_embedding_estimator_missing_model_type():
    """Test that embedding section with params but no model_type raises ValueError."""
    config = {
        "estimator_type": "embedding",
        "embedding": {"params": {"n_factors": 32}},
    }
    with pytest.raises(ValueError, match="'model_type' is required in embedding config"):
        create_estimator(config)


# --- Tests for sequential estimators ---


@requires_torch
@pytest.mark.parametrize(
    "model_type, expected_cls",
    [
        ("sasrec_classifier", SASRecClassifierEstimator),
        ("sasrec_regressor", SASRecRegressorEstimator),
        ("hrnn_classifier", HRNNClassifierEstimator),
        ("hrnn_regressor", HRNNRegressorEstimator),
    ],
)
def test_create_sequential_estimator(model_type, expected_cls):
    """Test that each sequential model_type creates the correct class."""
    config = {
        "estimator_type": "sequential",
        "sequential": {"model_type": model_type, "params": {}},
    }
    estimator = create_estimator(config)
    assert isinstance(estimator, expected_cls)


@requires_torch
def test_create_sequential_estimator_with_params():
    """Test sequential estimator with custom params."""
    config = {
        "estimator_type": "sequential",
        "sequential": {
            "model_type": "sasrec_classifier",
            "params": {"hidden_units": 128, "num_blocks": 4, "max_len": 100},
        },
    }
    estimator = create_estimator(config)
    assert isinstance(estimator, SASRecClassifierEstimator)
    assert estimator.hidden_units == 128
    assert estimator.num_blocks == 4
    assert estimator.max_len == 100


def test_create_sequential_estimator_missing_config():
    """Test that missing sequential key raises ValueError."""
    with pytest.raises(ValueError, match="'sequential' key is required"):
        create_estimator({"estimator_type": "sequential"})


def test_create_sequential_estimator_invalid_model_type():
    """Test that invalid sequential model_type raises NotImplementedError."""
    config = {
        "estimator_type": "sequential",
        "sequential": {"model_type": "unknown_model"},
    }
    with pytest.raises(NotImplementedError, match="Sequential model type 'unknown_model' not supported"):
        create_estimator(config)


def test_create_sequential_estimator_empty_section():
    """Test that sequential section present but empty raises ValueError with specific message."""
    config = {
        "estimator_type": "sequential",
        "sequential": {},
    }
    with pytest.raises(ValueError, match="'sequential' config is empty"):
        create_estimator(config)


def test_create_sequential_estimator_missing_model_type():
    """Test that sequential section with params but no model_type raises ValueError."""
    config = {
        "estimator_type": "sequential",
        "sequential": {"params": {"hidden_units": 64}},
    }
    with pytest.raises(ValueError, match="'model_type' is required in sequential config"):
        create_estimator(config)


def test_create_estimator_invalid_estimator_type():
    """Test that invalid estimator_type raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Estimator type 'fancy' not supported"):
        create_estimator({"estimator_type": "fancy"})


def test_create_tabular_estimator_warns_on_unused_embedding_key(caplog):
    """Test that tabular estimator warns when embedding/sequential keys are present."""
    config = {
        "estimator_type": "tabular",
        "embedding": {"model_type": "ncf"},
    }
    with caplog.at_level(logging.WARNING):
        estimator = create_estimator(config)
    assert isinstance(estimator, XGBClassifierEstimator)
    assert "will be ignored" in caplog.text
    assert "embedding" in caplog.text


def test_create_tabular_estimator_warns_on_unused_sequential_key(caplog):
    """Test that tabular estimator warns when sequential key is present."""
    config = {
        "sequential": {"model_type": "sasrec_classifier"},
    }
    with caplog.at_level(logging.WARNING):
        estimator = create_estimator(config)
    assert isinstance(estimator, XGBClassifierEstimator)
    assert "will be ignored" in caplog.text
    assert "sequential" in caplog.text


# --- Tests for new scorer types ---


@pytest.fixture
def mock_sequential_estimator():
    """Fixture for a mock SequentialEstimator."""
    mock = MagicMock(spec=SequentialEstimator)
    mock.__class__ = SequentialEstimator
    return mock


def test_create_sequential_scorer(mock_sequential_estimator):
    """Test creation of SequentialScorer."""
    scorer = create_scorer(mock_sequential_estimator, {"scorer_type": "sequential"})
    assert isinstance(scorer, SequentialScorer)


def test_create_hierarchical_scorer(mock_sequential_estimator):
    """Test creation of HierarchicalScorer."""
    scorer = create_scorer(mock_sequential_estimator, {"scorer_type": "hierarchical"})
    assert isinstance(scorer, HierarchicalScorer)


def test_create_sequential_scorer_wrong_estimator_type(mock_estimator):
    """Test that sequential scorer with non-sequential estimator raises TypeError."""
    with pytest.raises(TypeError, match="Sequential scorer requires a SequentialEstimator"):
        create_scorer(mock_estimator, {"scorer_type": "sequential"})


def test_create_hierarchical_scorer_wrong_estimator_type(mock_estimator):
    """Test that hierarchical scorer with non-sequential estimator raises TypeError."""
    with pytest.raises(TypeError, match="Hierarchical scorer requires a SequentialEstimator"):
        create_scorer(mock_estimator, {"scorer_type": "hierarchical"})


@pytest.mark.parametrize("scorer_type", ["multioutput", "multiclass", "independent", "universal"])
def test_create_tabular_scorer_rejects_sequential_estimator(mock_sequential_estimator, scorer_type):
    """Test that tabular scorers reject SequentialEstimator with clear error."""
    with pytest.raises(TypeError, match=f"Scorer type '{scorer_type}' requires a BaseEstimator"):
        create_scorer(mock_sequential_estimator, {"scorer_type": scorer_type})


@pytest.mark.parametrize("scorer_type", ["multioutput", "multiclass", "independent"])
def test_create_scorer_rejects_embedding_estimator(scorer_type):
    """Test that multioutput/multiclass/independent scorers reject embedding estimators."""
    # Use a real embedding estimator (MatrixFactorizationEstimator) to exercise
    # the isinstance(estimator, BaseEmbeddingEstimator) guard end-to-end.
    embedding_estimator = MatrixFactorizationEstimator()
    with pytest.raises(TypeError, match=f"Scorer type '{scorer_type}' does not support embedding"):
        create_scorer(embedding_estimator, {"scorer_type": scorer_type})


def test_create_universal_scorer_accepts_embedding_estimator():
    """Test that UniversalScorer accepts embedding estimators (the one valid combo)."""
    embedding_estimator = MatrixFactorizationEstimator()
    scorer = create_scorer(embedding_estimator, {"scorer_type": "universal"})
    assert isinstance(scorer, UniversalScorer)


# --- Tests for new recommender types ---


@pytest.fixture
def mock_sequential_scorer():
    """Fixture for a mock SequentialScorer."""
    return MagicMock(spec=SequentialScorer)


@pytest.fixture
def mock_hierarchical_scorer():
    """Fixture for a mock HierarchicalScorer."""
    return MagicMock(spec=HierarchicalScorer)


def test_create_sequential_recommender(mock_sequential_scorer):
    """Test creation of SequentialRecommender."""
    config = {"recommender_type": "sequential", "recommender_params": {"max_len": 75}}
    recommender = create_recommender(mock_sequential_scorer, config)
    assert isinstance(recommender, SequentialRecommender)
    assert recommender.max_len == 75


def test_create_sequential_recommender_defaults(mock_sequential_scorer):
    """Test SequentialRecommender uses default max_len."""
    config = {"recommender_type": "sequential"}
    recommender = create_recommender(mock_sequential_scorer, config)
    assert isinstance(recommender, SequentialRecommender)
    assert recommender.max_len == 50


def test_create_hierarchical_recommender(mock_hierarchical_scorer):
    """Test creation of HierarchicalSequentialRecommender."""
    config = {
        "recommender_type": "hierarchical_sequential",
        "recommender_params": {
            "max_sessions": 15,
            "max_session_len": 25,
            "session_timeout_minutes": 60.0,
        },
    }
    recommender = create_recommender(mock_hierarchical_scorer, config)
    assert isinstance(recommender, HierarchicalSequentialRecommender)
    assert recommender.max_sessions == 15
    assert recommender.max_session_len == 25
    assert recommender.session_timeout_minutes == 60.0


def test_create_uplift_recommender():
    """Test creation of UpliftRecommender."""
    # UpliftRecommender accesses scorer.estimator to auto-detect mode,
    # so we need a scorer with an estimator attribute.
    scorer = MagicMock(spec=IndependentScorer)
    scorer.estimator = MagicMock(spec=BaseEstimator)
    config = {
        "recommender_type": "uplift",
        "recommender_params": {"control_item_id": "ctrl_arm", "mode": "t_learner"},
    }
    recommender = create_recommender(scorer, config)
    assert isinstance(recommender, UpliftRecommender)


def test_create_uplift_recommender_missing_control_id(mock_scorer):
    """Test that uplift recommender without control_item_id raises ValueError."""
    config = {"recommender_type": "uplift", "recommender_params": {}}
    with pytest.raises(ValueError, match="'control_item_id' is required"):
        create_recommender(mock_scorer, config)


def test_create_gcsl_recommender(mock_scorer):
    """Test creation of GcslRecommender without inference method."""
    config = {"recommender_type": "gcsl"}
    recommender = create_recommender(mock_scorer, config)
    assert isinstance(recommender, GcslRecommender)


def test_create_gcsl_recommender_with_inference_method(mock_scorer):
    """Test creation of GcslRecommender with inference method."""
    config = {
        "recommender_type": "gcsl",
        "recommender_params": {
            "inference_method": {
                "type": "percentile_value",
                "params": {"percentiles": {"OUTCOME_revenue": 80}},
            },
        },
    }
    recommender = create_recommender(mock_scorer, config)
    assert isinstance(recommender, GcslRecommender)


def test_create_ranking_recommender_with_retriever(mock_scorer):
    """Test creation of RankingRecommender with retriever."""
    config = {
        "recommender_type": "ranking",
        "recommender_params": {
            "retriever": {"type": "popularity", "params": {"top_k": 50}},
        },
    }
    recommender = create_recommender(mock_scorer, config)
    assert isinstance(recommender, RankingRecommender)
    assert recommender.retriever is not None
    assert isinstance(recommender.retriever, PopularityRetriever)


# --- Tests for _create_inference_method ---


@pytest.mark.parametrize(
    "type_str, expected_cls",
    [
        ("mean_scalarization", MeanScalarization),
        ("percentile_value", PercentileValue),
        ("predefined_value", PredefinedValue),
    ],
)
def test_create_inference_method(type_str, expected_cls):
    """Test each inference method type creates the correct class."""
    # Each inference method constructor requires specific params
    if type_str == "mean_scalarization":
        params = {"scalars": {"OUTCOME_a": 0.5}}
    elif type_str == "percentile_value":
        params = {"percentiles": {"OUTCOME_a": 80}}
    else:  # predefined_value
        params = {"goal_values": {"OUTCOME_a": 1.0}}
    config = {"type": type_str, "params": params}
    method = _create_inference_method(config)
    assert isinstance(method, expected_cls)


def test_create_inference_method_invalid_type():
    """Test that invalid inference method type raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Inference method type 'unknown' not supported"):
        _create_inference_method({"type": "unknown"})


def test_create_inference_method_missing_type():
    """Test that missing type raises ValueError."""
    with pytest.raises(ValueError, match="'type' is required"):
        _create_inference_method({})


# --- Tests for _create_retriever ---


@pytest.mark.parametrize(
    "type_str, expected_cls",
    [
        ("popularity", PopularityRetriever),
        ("content_based", ContentBasedRetriever),
        ("embedding", EmbeddingRetriever),
    ],
)
def test_create_retriever(type_str, expected_cls):
    """Test each retriever type creates the correct class."""
    config = {"type": type_str, "params": {"top_k": 100}}
    retriever = _create_retriever(config)
    assert isinstance(retriever, expected_cls)


def test_create_retriever_invalid_type():
    """Test that invalid retriever type raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Retriever type 'unknown' not supported"):
        _create_retriever({"type": "unknown"})


def test_create_retriever_missing_type():
    """Test that missing type raises ValueError."""
    with pytest.raises(ValueError, match="'type' is required"):
        _create_retriever({})


# --- Tests for pipeline-level validation ---


def test_pipeline_sequential_requires_sequential_estimator():
    """Test that sequential recommender with non-sequential estimator raises ValueError."""
    config = {
        "recommender_type": "sequential",
        "scorer_type": "sequential",
        "estimator_config": {"estimator_type": "tabular", "ml_task": "classification"},
    }
    with pytest.raises(ValueError, match="requires estimator_type 'sequential'"):
        create_recommender_pipeline(config)


def test_pipeline_sequential_requires_sequential_scorer():
    """Test that sequential recommender with non-sequential scorer raises ValueError."""
    config = {
        "recommender_type": "sequential",
        "scorer_type": "universal",
        "estimator_config": {
            "estimator_type": "sequential",
            "sequential": {"model_type": "sasrec_classifier"},
        },
    }
    with pytest.raises(ValueError, match="requires scorer_type 'sequential'"):
        create_recommender_pipeline(config)


def test_pipeline_hierarchical_requires_hierarchical_scorer():
    """Test that hierarchical recommender with wrong scorer raises ValueError."""
    config = {
        "recommender_type": "hierarchical_sequential",
        "scorer_type": "sequential",
        "estimator_config": {
            "estimator_type": "sequential",
            "sequential": {"model_type": "hrnn_classifier"},
        },
    }
    with pytest.raises(ValueError, match="requires scorer_type 'hierarchical'"):
        create_recommender_pipeline(config)


def test_pipeline_sequential_scorer_requires_sequential_estimator():
    """Test that sequential scorer with tabular estimator raises ValueError."""
    config = {
        "recommender_type": "ranking",
        "scorer_type": "sequential",
        "estimator_config": {"estimator_type": "tabular"},
    }
    with pytest.raises(ValueError, match="scorer_type 'sequential' requires estimator_type 'sequential'"):
        create_recommender_pipeline(config)


@pytest.mark.parametrize("scorer_type", ["multioutput", "multiclass", "independent"])
def test_pipeline_embedding_rejects_incompatible_scorers(scorer_type):
    """Test that embedding estimator with multioutput/multiclass/independent scorer raises early."""
    config = {
        "recommender_type": "ranking",
        "scorer_type": scorer_type,
        "estimator_config": {
            "estimator_type": "embedding",
            "embedding": {"model_type": "ncf"},
        },
    }
    with pytest.raises(ValueError, match=f"scorer_type '{scorer_type}' does not support embedding"):
        create_recommender_pipeline(config)


# --- Full pipeline integration tests (mocked sub-factories) ---


@patch("skrec.orchestrator.factory.create_estimator")
@patch("skrec.orchestrator.factory.create_scorer")
@patch("skrec.orchestrator.factory.create_recommender")
def test_pipeline_sequential_config_flow(mock_create_recommender, mock_create_scorer, mock_create_estimator):
    """Test that sequential pipeline config flows correctly through the chain."""
    mock_est = MagicMock(spec=SequentialEstimator)
    mock_sco = MagicMock(spec=SequentialScorer)
    mock_rec = MagicMock(spec=SequentialRecommender)

    mock_create_estimator.return_value = mock_est
    mock_create_scorer.return_value = mock_sco
    mock_create_recommender.return_value = mock_rec

    config = {
        "recommender_type": "sequential",
        "scorer_type": "sequential",
        "estimator_config": {
            "estimator_type": "sequential",
            "sequential": {"model_type": "sasrec_classifier", "params": {"hidden_units": 64}},
        },
        "recommender_params": {"max_len": 50},
    }
    pipeline = create_recommender_pipeline(config)

    mock_create_estimator.assert_called_once_with(config["estimator_config"], scorer_type="sequential")
    mock_create_scorer.assert_called_once_with(mock_est, config)
    mock_create_recommender.assert_called_once_with(mock_sco, config)
    assert pipeline == mock_rec


@patch("skrec.orchestrator.factory.create_estimator")
@patch("skrec.orchestrator.factory.create_scorer")
@patch("skrec.orchestrator.factory.create_recommender")
def test_pipeline_uplift_config_flow(mock_create_recommender, mock_create_scorer, mock_create_estimator):
    """Test that uplift pipeline config flows correctly through the chain."""
    mock_est = MagicMock(spec=BaseEstimator)
    mock_sco = MagicMock(spec=BaseScorer)
    mock_rec = MagicMock(spec=UpliftRecommender)

    mock_create_estimator.return_value = mock_est
    mock_create_scorer.return_value = mock_sco
    mock_create_recommender.return_value = mock_rec

    config = {
        "recommender_type": "uplift",
        "scorer_type": "independent",
        "estimator_config": {"ml_task": "classification"},
        "recommender_params": {"control_item_id": "ctrl", "mode": "t_learner"},
    }
    pipeline = create_recommender_pipeline(config)

    mock_create_estimator.assert_called_once_with(config["estimator_config"], scorer_type="independent")
    mock_create_scorer.assert_called_once_with(mock_est, config)
    mock_create_recommender.assert_called_once_with(mock_sco, config)
    assert pipeline == mock_rec


@patch("skrec.orchestrator.factory.create_estimator")
@patch("skrec.orchestrator.factory.create_scorer")
@patch("skrec.orchestrator.factory.create_recommender")
def test_pipeline_gcsl_config_flow(mock_create_recommender, mock_create_scorer, mock_create_estimator):
    """Test that GCSL pipeline config flows correctly through the chain."""
    mock_est = MagicMock(spec=BaseEstimator)
    mock_sco = MagicMock(spec=BaseScorer)
    mock_rec = MagicMock(spec=GcslRecommender)

    mock_create_estimator.return_value = mock_est
    mock_create_scorer.return_value = mock_sco
    mock_create_recommender.return_value = mock_rec

    config = {
        "recommender_type": "gcsl",
        "scorer_type": "universal",
        "estimator_config": {"ml_task": "classification"},
        "recommender_params": {
            "inference_method": {
                "type": "mean_scalarization",
                "params": {"scalars": {"OUTCOME_a": 0.5}},
            },
        },
    }
    pipeline = create_recommender_pipeline(config)

    mock_create_estimator.assert_called_once_with(config["estimator_config"], scorer_type="universal")
    mock_create_scorer.assert_called_once_with(mock_est, config)
    mock_create_recommender.assert_called_once_with(mock_sco, config)
    assert pipeline == mock_rec


@patch("skrec.orchestrator.factory.create_estimator")
@patch("skrec.orchestrator.factory.create_scorer")
@patch("skrec.orchestrator.factory.create_recommender")
def test_pipeline_embedding_ranking_config_flow(mock_create_recommender, mock_create_scorer, mock_create_estimator):
    """Test that embedding + ranking pipeline config flows correctly."""
    mock_est = MagicMock(spec=BaseEstimator)
    mock_sco = MagicMock(spec=BaseScorer)
    mock_rec = MagicMock(spec=RankingRecommender)

    mock_create_estimator.return_value = mock_est
    mock_create_scorer.return_value = mock_sco
    mock_create_recommender.return_value = mock_rec

    config = {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {
            "estimator_type": "embedding",
            "embedding": {"model_type": "ncf", "params": {"embedding_dim": 32}},
        },
    }
    pipeline = create_recommender_pipeline(config)

    mock_create_estimator.assert_called_once_with(config["estimator_config"], scorer_type="universal")
    mock_create_scorer.assert_called_once_with(mock_est, config)
    mock_create_recommender.assert_called_once_with(mock_sco, config)
    assert pipeline == mock_rec


# --- Pipeline validation: uplift scorer compatibility ---


@pytest.mark.parametrize("scorer_type", ["multioutput", "multiclass"])
def test_pipeline_uplift_rejects_incompatible_scorers(scorer_type):
    """Test that uplift recommender with incompatible scorer raises early.

    Note: sequential/hierarchical scorer_types are already caught by the
    'scorer requires estimator_type=sequential' validation, which fires first.
    """
    config = {
        "recommender_type": "uplift",
        "scorer_type": scorer_type,
        "estimator_config": {"ml_task": "classification"},
        "recommender_params": {"control_item_id": "ctrl"},
    }
    with pytest.raises(ValueError, match="recommender_type 'uplift' requires scorer_type"):
        create_recommender_pipeline(config)


# --- Runtime type checks in create_recommender ---


def test_create_sequential_recommender_rejects_wrong_scorer(mock_scorer):
    """Test that SequentialRecommender rejects non-SequentialScorer."""
    config = {"recommender_type": "sequential", "recommender_params": {"max_len": 50}}
    with pytest.raises(TypeError, match="SequentialRecommender requires a SequentialScorer"):
        create_recommender(mock_scorer, config)


def test_create_hierarchical_recommender_rejects_wrong_scorer(mock_scorer):
    """Test that HierarchicalSequentialRecommender rejects non-HierarchicalScorer."""
    config = {"recommender_type": "hierarchical_sequential"}
    with pytest.raises(TypeError, match="HierarchicalSequentialRecommender requires a HierarchicalScorer"):
        create_recommender(mock_scorer, config)


# --- Tuned mode edge case ---


def test_create_tuned_regressor_ignores_multioutput_scorer_type():
    """Test that ml_task=regression with scorer_type=multioutput creates a plain regressor, not multioutput."""
    from skrec.estimator.datatypes import HPOType
    from skrec.estimator.regression.xgb_regressor import TunedXGBRegressorEstimator

    config = {
        "ml_task": "regression",
        "hpo": {
            "hpo_method": HPOType.RANDOMIZED_SEARCH_CV,
            "param_space": {"n_estimators": [50, 100]},
            "optimizer_params": {"n_iter": 1, "cv": 2},
        },
    }
    # scorer_type="multioutput" is only meaningful for classification;
    # for regression, the factory creates a TunedXGBRegressorEstimator regardless.
    estimator = create_estimator(config, scorer_type="multioutput")
    assert isinstance(estimator, TunedXGBRegressorEstimator)


# --- Unmocked end-to-end integration tests ---


def test_e2e_tabular_ranking_pipeline():
    """End-to-end: tabular XGB + universal scorer + ranking recommender."""
    config = {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 5, "max_depth": 2},
        },
    }
    pipeline = create_recommender_pipeline(config)
    assert isinstance(pipeline, RankingRecommender)
    assert isinstance(pipeline.scorer, UniversalScorer)
    assert isinstance(pipeline.scorer.estimator, XGBClassifierEstimator)


def test_e2e_embedding_ranking_pipeline():
    """End-to-end: MF embedding + universal scorer + ranking recommender."""
    config = {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {
            "estimator_type": "embedding",
            "embedding": {"model_type": "matrix_factorization", "params": {"n_factors": 16}},
        },
    }
    pipeline = create_recommender_pipeline(config)
    assert isinstance(pipeline, RankingRecommender)
    assert isinstance(pipeline.scorer, UniversalScorer)
    assert isinstance(pipeline.scorer.estimator, MatrixFactorizationEstimator)
    assert pipeline.scorer.estimator.n_factors == 16


@requires_torch
def test_e2e_sequential_pipeline():
    """End-to-end: SASRec + sequential scorer + sequential recommender."""
    config = {
        "recommender_type": "sequential",
        "scorer_type": "sequential",
        "estimator_config": {
            "estimator_type": "sequential",
            "sequential": {
                "model_type": "sasrec_classifier",
                "params": {"hidden_units": 32, "max_len": 25},
            },
        },
        "recommender_params": {"max_len": 25},
    }
    pipeline = create_recommender_pipeline(config)
    assert isinstance(pipeline, SequentialRecommender)
    assert isinstance(pipeline.scorer, SequentialScorer)
    assert isinstance(pipeline.scorer.estimator, SASRecClassifierEstimator)
    assert pipeline.max_len == 25


def test_e2e_uplift_pipeline():
    """End-to-end: tabular XGB + independent scorer + uplift recommender."""
    from skrec.recommender.uplift_model.uplift_scorer_adapter import UpliftScorerAdapter

    config = {
        "recommender_type": "uplift",
        "scorer_type": "independent",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 5},
        },
        "recommender_params": {"control_item_id": "control", "mode": "t_learner"},
    }
    pipeline = create_recommender_pipeline(config)
    assert isinstance(pipeline, UpliftRecommender)
    # UpliftRecommender wraps the original scorer in UpliftScorerAdapter
    assert isinstance(pipeline.scorer, UpliftScorerAdapter)


def test_e2e_gcsl_pipeline():
    """End-to-end: tabular XGB + universal scorer + GCSL recommender with inference method."""
    config = {
        "recommender_type": "gcsl",
        "scorer_type": "universal",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 5},
        },
        "recommender_params": {
            "inference_method": {
                "type": "predefined_value",
                "params": {"goal_values": {"OUTCOME_revenue": 1.0}},
            },
        },
    }
    pipeline = create_recommender_pipeline(config)
    assert isinstance(pipeline, GcslRecommender)
    assert isinstance(pipeline.scorer, UniversalScorer)


def test_e2e_bandits_pipeline():
    """End-to-end: tabular XGB + universal scorer + bandits recommender."""
    config = {
        "recommender_type": "bandits",
        "scorer_type": "universal",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 5},
        },
    }
    pipeline = create_recommender_pipeline(config)
    assert isinstance(pipeline, ContextualBanditsRecommender)
    assert isinstance(pipeline.scorer, UniversalScorer)
