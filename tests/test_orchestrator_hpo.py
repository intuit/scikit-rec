from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.orchestrator.hpo import HyperparameterOptimizer, deep_update

# --- Synthetic Data Fixtures ---


@pytest.fixture
def synthetic_interaction_data():
    """Creates a simple DataFrame for interactions with a deterministic label."""
    n_samples = 30  # Increased sample size slightly
    users = [f"u{i}" for i in range(n_samples)]
    user_is_odd = np.arange(n_samples) % 2
    user_is_even = np.arange(1, 1 + n_samples) % 2
    items = [f"i{i % 5}" for i in range(n_samples)]  # 5 distinct items
    # Label=1 if user number parity == item number parity
    labels = [(int(u[1:]) % 2) == (int(i[1:]) % 2) for u, i in zip(users, items)]
    labels = [int(label) for label in labels]  # Convert bool to int
    return pd.DataFrame(
        {
            "USER_ID": users,
            "ITEM_ID": items,
            "OUTCOME": labels,
            "is_odd": user_is_odd,
            "is_even": user_is_even,
        }
    )


@pytest.fixture
def train_interactions_ds(synthetic_interaction_data):
    """Creates a real InteractionsDataset for training."""
    train_data = synthetic_interaction_data.iloc[:20].copy()  # Use 20 for training
    ds = InteractionsDataset(data_location=None)
    ds.fetch_data = MagicMock(return_value=train_data)
    return ds


@pytest.fixture
def validation_interactions_ds(synthetic_interaction_data):
    """Creates a real InteractionsDataset for validation."""
    val_data = synthetic_interaction_data.iloc[20:].copy()  # Use remaining 10 for validation
    ds = InteractionsDataset(data_location=None)
    ds.fetch_data = MagicMock(return_value=val_data)
    return ds


# --- Config Fixtures ---


@pytest.fixture
def base_config():
    """Base config using XGBoost and including dataset schema."""
    return {
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {
                # Keep base xgboost params minimal, HPO will override
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "use_label_encoder": False,
                "n_jobs": 1,  # Ensure single-threaded for predictability in tests
            },
        },
        "scorer_type": "independent",
        "recommender_type": "ranking",
    }


@pytest.fixture
def search_space():
    """Search space relevant to XGBoost using optuna-style dicts."""
    return {
        "estimator_config.xgboost.learning_rate": {"type": "float", "low": 0.05, "high": 0.2, "log": True},
        "estimator_config.xgboost.n_estimators": {"type": "int", "low": 10, "high": 30},
        "estimator_config.xgboost.max_depth": {"type": "int", "low": 3, "high": 5},
    }


@pytest.fixture
def metric_definitions():
    """Metric definitions as strings."""
    return ["NDCG@3", "Precision@3"]


# --- Tests for deep_update ---


def test_deep_update():
    source = {"a": 1, "b": {"c": 2, "d": 3}}
    overrides = {"b": {"c": 4, "e": 5}, "f": 6}
    expected = {"a": 1, "b": {"c": 4, "d": 3, "e": 5}, "f": 6}
    result = deep_update(source, overrides)
    assert result == expected
    # Check if source was modified in place
    assert source == expected


# --- Tests for HyperparameterOptimizer Initialization and Persistence ---


@patch("pandas.read_parquet")
def test_hpo_init_no_persistence(
    mock_read_parquet, base_config, search_space, metric_definitions, train_interactions_ds, validation_interactions_ds
):
    """Test initialization without a persistence path."""
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        search_space=search_space,
        metric_definitions=metric_definitions,
        training_interactions_ds=train_interactions_ds,
        validation_interactions_ds=validation_interactions_ds,
        persistence_path=None,
    )
    assert optimizer.persistence_path is None
    assert optimizer.results_df.empty
    mock_read_parquet.assert_not_called()


@patch("pandas.read_parquet")
def test_hpo_init_persistence_file_not_found(
    mock_read_parquet, base_config, search_space, metric_definitions, train_interactions_ds, validation_interactions_ds
):
    """Test initialization when persistence file doesn't exist."""
    mock_read_parquet.side_effect = FileNotFoundError
    persistence_path = "/fake/path/results.parquet"

    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        search_space=search_space,
        metric_definitions=metric_definitions,
        training_interactions_ds=train_interactions_ds,
        validation_interactions_ds=validation_interactions_ds,
        persistence_path=persistence_path,
    )
    assert optimizer.persistence_path == persistence_path
    assert optimizer.results_df.empty
    mock_read_parquet.assert_called_once_with(persistence_path)


@patch("pandas.read_parquet")
def test_hpo_init_persistence_load_error(
    mock_read_parquet, base_config, search_space, metric_definitions, train_interactions_ds, validation_interactions_ds
):
    """Test initialization when loading persistence file raises an error."""
    mock_read_parquet.side_effect = Exception("Corrupted file")
    persistence_path = "/fake/path/results.parquet"

    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        search_space=search_space,
        metric_definitions=metric_definitions,
        training_interactions_ds=train_interactions_ds,
        validation_interactions_ds=validation_interactions_ds,
        persistence_path=persistence_path,
    )
    assert optimizer.persistence_path == persistence_path
    assert optimizer.results_df.empty  # Should reset on error
    mock_read_parquet.assert_called_once_with(persistence_path)


@patch("pandas.read_parquet")
def test_hpo_init_persistence_success(
    mock_read_parquet, base_config, search_space, metric_definitions, train_interactions_ds, validation_interactions_ds
):
    """Test successful initialization with loading previous results."""
    # Adjust previous results to match the new XGBoost search space and metrics
    previous_results = pd.DataFrame(
        {
            "estimator_config.xgboost.learning_rate": [0.1],
            "estimator_config.xgboost.n_estimators": [20],
            "estimator_config.xgboost.max_depth": [4],
            "NDCG@3": [0.5],  # Match updated metric names
            "Precision@3": [0.2],  # Match updated metric names
            "trial_duration": [10.0],
        }
    )
    mock_read_parquet.return_value = previous_results
    persistence_path = "/fake/path/results.parquet"

    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        search_space=search_space,
        metric_definitions=metric_definitions,
        training_interactions_ds=train_interactions_ds,
        validation_interactions_ds=validation_interactions_ds,
        persistence_path=persistence_path,
    )
    assert optimizer.persistence_path == persistence_path
    pd.testing.assert_frame_equal(optimizer.results_df, previous_results)
    mock_read_parquet.assert_called_once_with(persistence_path)


# --- Objective Function Tests (End-to-End) ---


# We only mock file system operations (saving results)
@patch("pandas.DataFrame.to_parquet")
@patch("os.makedirs")
def test_run_trial_end_to_end(
    mock_makedirs,
    mock_to_parquet,
    base_config,
    search_space,
    metric_definitions,
    train_interactions_ds,
    validation_interactions_ds,
    tmp_path,
):
    """
    Test _run_trial runs end-to-end with a real pipeline,
    synthetic data, and real metrics, checking if results are reasonable.
    """
    persistence_path = str(tmp_path / "results.parquet")

    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        search_space=search_space,
        metric_definitions=metric_definitions,
        training_interactions_ds=train_interactions_ds,
        validation_interactions_ds=validation_interactions_ds,
        persistence_path=persistence_path,
    )

    # Define sample parameters matching the search space
    params_dict = {
        "estimator_config.xgboost.learning_rate": 0.1,
        "estimator_config.xgboost.n_estimators": 20,
        "estimator_config.xgboost.max_depth": 4,
    }

    # --- Execute the trial ---
    metric_scores = optimizer._run_trial(params_dict)

    # --- Assertions ---

    # 1. Check results DataFrame structure and content
    assert len(optimizer.results_df) == 1, "Should have one row after one trial"
    result_row = optimizer.results_df.iloc[0].to_dict()

    # Check hyperparameters are recorded correctly
    for param_name, param_value in params_dict.items():
        assert result_row[param_name] == param_value, f"Param '{param_name}' not recorded correctly"

    # Check metric columns exist and contain plausible float values (better than random)
    primary_metric_name = metric_definitions[0]  # e.g., "NDCG@3"

    for metric_name in metric_definitions:
        assert metric_name in result_row, f"Metric '{metric_name}' missing from results"
        metric_value = result_row[metric_name]
        assert isinstance(metric_value, float), f"Metric '{metric_name}' should be a float"
        # Assert metric is better than a baseline random guess (e.g., > 0.05)
        assert metric_value > 0.05, f"Metric '{metric_name}' ({metric_value:.4f}) is too low, expected > 0.05"

    # Check trial duration exists and is positive
    assert "trial_duration" in result_row, "Trial duration missing"
    assert isinstance(result_row["trial_duration"], float), "Trial duration should be float"
    assert result_row["trial_duration"] > 0, "Trial duration should be positive"

    # 2. Check the returned metric scores dict
    assert primary_metric_name in metric_scores
    assert np.isclose(metric_scores[primary_metric_name], result_row[primary_metric_name])

    # 3. Check saving results (mocks are still used for filesystem interaction)
    mock_makedirs.assert_called_once_with(str(tmp_path), exist_ok=True)
    mock_to_parquet.assert_called_once_with(persistence_path, index=False)
