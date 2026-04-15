import pickle
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import pytest

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_EMBEDDING_NAME, USER_ID_NAME
from skrec.estimator.embedding.ncf_estimator import NCFEstimator


@pytest.fixture(scope="module")
def ncf_test_data():
    """Provides sample data for NCF estimator tests."""
    users_data = {
        USER_ID_NAME: ["u1", "u2", "u3", "u4"],
        "user_feat1": [0.1, 0.2, 0.3, 0.4],
        "user_feat2": [1, 2, 3, 4],
    }
    items_data = {
        ITEM_ID_NAME: ["i1", "i2", "i3", "i4", "i5"],
        "item_feat1": [10.0, 20.0, 30.0, 40.0, 50.0],
    }
    interactions_data = {
        USER_ID_NAME: ["u1", "u1", "u2", "u3", "u2", "u4", "u5", "u1"],  # u5 is unknown user
        ITEM_ID_NAME: ["i1", "i2", "i1", "i3", "i4", "i5", "i1", "i6"],  # i6 is unknown item
        LABEL_NAME: [1, 0, 1, 0, 1, 0, 1, 0],
        "inter_feat1": [0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4],
    }
    return {
        "users_df": pd.DataFrame(users_data),
        "items_df": pd.DataFrame(items_data),
        "interactions_df": pd.DataFrame(interactions_data),
    }


NCFParams = Dict[str, Union[int, float, str, list]]


@pytest.fixture(params=["gmf", "mlp", "neumf"])
def ncf_variant(request) -> Tuple[str, NCFParams]:
    """
    Parameterized fixture providing each NCF variant type and its parameters.
    """
    ncf_type = request.param
    common_params = {"epochs": 1, "random_state": 42, "verbose": 0}

    if ncf_type == "gmf":
        specific_params = {
            "ncf_type": "gmf",
            "gmf_embedding_dim": 8,
        }
    elif ncf_type == "mlp":
        specific_params = {
            "ncf_type": "mlp",
            "mlp_embedding_dim": 8,
            "mlp_layers": [16, 8],
        }
    else:  # neumf
        specific_params = {
            "ncf_type": "neumf",
            "gmf_embedding_dim": 8,
            "mlp_embedding_dim": 8,
            "mlp_layers": [16, 8],
        }

    all_params = {**common_params, **specific_params}
    return ncf_type, all_params


def test_ncf_variants_fit_predict(ncf_variant: Tuple[str, NCFParams], ncf_test_data: dict):
    """Tests the basic fit and predict flow for each NCF variant (GMF, MLP, NeuMF)."""
    ncf_type, params = ncf_variant
    estimator = NCFEstimator(**params)

    # Verify ncf_type is set correctly
    assert estimator.ncf_type == ncf_type.lower()

    estimator.fit_embedding_model(
        users=ncf_test_data["users_df"],
        items=ncf_test_data["items_df"],
        interactions=ncf_test_data["interactions_df"],
    )
    assert estimator.model is not None, f"Model should be trained for {ncf_type}."

    # Test batch prediction mode (using learned embeddings)
    predictions = estimator.predict_proba_with_embeddings(
        interactions=ncf_test_data["interactions_df"],
    )

    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array."
    assert predictions.shape == (len(ncf_test_data["interactions_df"]), 1), "Predictions shape mismatch."
    assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions should be probabilities (0-1)."


def test_ncf_get_config():
    """Tests if get_config returns the correct hyperparameters for NCF."""
    params = {
        "ncf_type": "neumf",
        "gmf_embedding_dim": 16,
        "mlp_embedding_dim": 24,
        "mlp_layers": [64, 32, 16],
        "dropout": 0.2,
        "epochs": 5,
        "learning_rate": 0.01,
        "random_state": 123,
    }

    estimator = NCFEstimator(**params)
    config = estimator.get_config()

    assert config["ncf_type"] == "neumf"
    assert config["gmf_embedding_dim"] == 16
    assert config["mlp_embedding_dim"] == 24
    assert config["mlp_layers"] == [64, 32, 16]
    assert config["dropout"] == 0.2
    assert config["epochs"] == 5
    assert config["learning_rate"] == 0.01
    assert config["random_state"] == 123


def test_ncf_predict_without_fit_raises_error(ncf_test_data: dict):
    """Tests that predict_proba_with_embeddings raises RuntimeError if called before fit."""
    estimator = NCFEstimator(ncf_type="gmf", epochs=1, random_state=42)

    with pytest.raises(RuntimeError, match="Model has not been trained"):
        estimator.predict_proba_with_embeddings(ncf_test_data["interactions_df"])


def test_ncf_handles_missing_features(ncf_variant: Tuple[str, NCFParams], ncf_test_data: dict):
    """Tests NCF robustness to missing or empty feature DataFrames."""
    ncf_type, params = ncf_variant
    base_interactions = ncf_test_data["interactions_df"]

    scenarios = {
        "no_user_features": (
            pd.DataFrame({USER_ID_NAME: ncf_test_data["users_df"][USER_ID_NAME]}),
            ncf_test_data["items_df"],
            base_interactions,
        ),
        "empty_user_df": (
            pd.DataFrame(columns=[USER_ID_NAME, "user_feat1"]),
            ncf_test_data["items_df"],
            base_interactions,
        ),
        "no_item_features": (
            ncf_test_data["users_df"],
            pd.DataFrame({ITEM_ID_NAME: ncf_test_data["items_df"][ITEM_ID_NAME]}),
            base_interactions,
        ),
        "empty_item_df": (
            ncf_test_data["users_df"],
            pd.DataFrame(columns=[ITEM_ID_NAME, "item_feat1"]),
            base_interactions,
        ),
        "no_interaction_features": (
            ncf_test_data["users_df"],
            ncf_test_data["items_df"],
            base_interactions[[USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME]],
        ),
    }

    for scenario_name, (users_df, items_df, interactions_df) in scenarios.items():
        estimator = NCFEstimator(**params)

        estimator.fit_embedding_model(users=users_df, items=items_df, interactions=interactions_df)
        assert estimator.model is not None, f"Model not trained for {ncf_type} in scenario: {scenario_name}"

        if scenario_name == "no_user_features" or scenario_name == "empty_user_df":
            assert estimator.user_features_dim == 0, f"User features dim should be 0 for {scenario_name}"
        if scenario_name == "no_item_features" or scenario_name == "empty_item_df":
            assert estimator.item_features_dim == 0, f"Item features dim should be 0 for {scenario_name}"
        if scenario_name == "no_interaction_features":
            assert estimator.interaction_features_dim == 0, f"Interaction features dim should be 0 for {scenario_name}"

        predictions = estimator.predict_proba_with_embeddings(interactions_df)
        assert predictions.shape == (len(interactions_df), 1), f"Predictions shape mismatch for {scenario_name}"


def test_ncf_handles_unknown_ids(ncf_variant: Tuple[str, NCFParams], ncf_test_data: dict):
    """Tests prediction with user/item IDs not seen during training."""
    ncf_type, params = ncf_variant
    estimator = NCFEstimator(**params)

    estimator.fit_embedding_model(
        users=ncf_test_data["users_df"],
        items=ncf_test_data["items_df"],
        interactions=ncf_test_data["interactions_df"],
    )

    predict_interactions_data = {
        USER_ID_NAME: ["u1", "unknown_user_1", "u2", "unknown_user_2"],
        ITEM_ID_NAME: ["i1", "i2", "unknown_item_1", "i3"],
        LABEL_NAME: [1, 0, 1, 0],
        "inter_feat1": [0.1, 0.2, 0.3, 0.4],
    }
    predict_interactions_df = pd.DataFrame(predict_interactions_data)

    predictions = estimator.predict_proba_with_embeddings(predict_interactions_df)

    assert predictions.shape == (len(predict_interactions_df), 1), "Predictions shape mismatch for unknown IDs."
    assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions for unknown IDs should be probabilities."


def test_ncf_with_empty_features(ncf_variant: Tuple[str, NCFParams], ncf_test_data: dict):
    """Test case where users_df and items_df contain only IDs, no feature columns."""
    ncf_type, params = ncf_variant
    users_df_no_feats = ncf_test_data["users_df"][[USER_ID_NAME]].copy()
    items_df_no_feats = ncf_test_data["items_df"][[ITEM_ID_NAME]].copy()
    interactions_df_no_inter_feats = ncf_test_data["interactions_df"][[USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME]].copy()

    estimator = NCFEstimator(**params)
    estimator.fit_embedding_model(
        users=users_df_no_feats, items=items_df_no_feats, interactions=interactions_df_no_inter_feats
    )
    assert estimator.user_features_dim == 0
    assert estimator.item_features_dim == 0
    assert estimator.interaction_features_dim == 0

    predictions = estimator.predict_proba_with_embeddings(interactions_df_no_inter_feats)
    assert predictions.shape == (len(interactions_df_no_inter_feats), 1)


def test_ncf_with_no_features(ncf_variant: Tuple[str, NCFParams], ncf_test_data: dict):
    """Test case where users_df and items_df are None during fit."""
    ncf_type, params = ncf_variant
    interactions_df_no_inter_feats = ncf_test_data["interactions_df"][[USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME]].copy()

    estimator = NCFEstimator(**params)
    estimator.fit_embedding_model(users=None, items=None, interactions=interactions_df_no_inter_feats)
    assert estimator.user_features_dim == 0
    assert estimator.item_features_dim == 0
    assert estimator.interaction_features_dim == 0
    assert estimator.user_features_tensor is None
    assert estimator.item_features_tensor is None

    predictions = estimator.predict_proba_with_embeddings(interactions_df_no_inter_feats)
    assert predictions.shape == (len(interactions_df_no_inter_feats), 1)


def test_ncf_embedding_extraction_truncation_pickle(ncf_test_data: dict):
    """
    Tests the full production workflow for NCF:
    1. Train model.
    2. Extract user embeddings (for Feature Management Platform).
    3. Truncate user data (for model size reduction).
    4. Pickle and unpickle the truncated model.
    5. Predict using the unpickled model with provided embeddings.
    """
    # Use NeuMF for this comprehensive test
    estimator = NCFEstimator(
        ncf_type="neumf",
        gmf_embedding_dim=8,
        mlp_embedding_dim=8,
        mlp_layers=[16, 8],
        epochs=1,
        random_state=42,
        verbose=0,
    )

    # 1. Train model
    estimator.fit_embedding_model(
        users=ncf_test_data["users_df"],
        items=ncf_test_data["items_df"],
        interactions=ncf_test_data["interactions_df"],
    )
    n_users = len(estimator.user_id_index)

    # 2. Get user embeddings
    user_embeddings_df = estimator.get_user_embeddings()
    assert isinstance(user_embeddings_df, pd.DataFrame)
    assert USER_ID_NAME in user_embeddings_df.columns
    assert USER_EMBEDDING_NAME in user_embeddings_df.columns
    assert len(user_embeddings_df) == n_users

    # 3. Truncate user data
    estimator.truncate_user_data()
    assert estimator._user_data_truncated is True
    assert estimator.num_users == 1  # Only placeholder user concept remains
    assert estimator.model.user_id_embedding.num_embeddings == 1

    # 4. Pickle and unpickle
    pickled_estimator = pickle.dumps(estimator)
    unpickled_estimator = pickle.loads(pickled_estimator)

    assert unpickled_estimator is not None
    assert unpickled_estimator._user_data_truncated is True
    assert unpickled_estimator.num_users == 1
    assert unpickled_estimator.unknown_user_idx == 0
    assert unpickled_estimator.model.user_id_embedding.num_embeddings == 1

    # 5. Predict using the unpickled model with provided embeddings (real-time mode)
    users_for_prediction_df = user_embeddings_df.head(2).copy()
    items_for_prediction = ncf_test_data["items_df"][[ITEM_ID_NAME]].head(2)

    interactions_for_prediction_list = []
    for _, user_row in users_for_prediction_df.iterrows():
        user_id = user_row[USER_ID_NAME]
        for item_id in items_for_prediction[ITEM_ID_NAME]:
            interaction_entry = {
                USER_ID_NAME: user_id,
                ITEM_ID_NAME: item_id,
                "inter_feat1": 0.5,
            }
            interactions_for_prediction_list.append(interaction_entry)

    interactions_predict_df = pd.DataFrame(interactions_for_prediction_list)
    users_for_prediction_df = pd.merge(users_for_prediction_df, ncf_test_data["users_df"], on=USER_ID_NAME, how="left")

    predictions = unpickled_estimator.predict_proba_with_embeddings(
        interactions=interactions_predict_df, users=users_for_prediction_df
    )

    assert isinstance(predictions, np.ndarray)
    expected_predictions_count = len(interactions_for_prediction_list)
    assert predictions.shape == (
        expected_predictions_count,
        1,
    ), f"Predictions shape mismatch. Expected ({expected_predictions_count}, 1), got {predictions.shape}"

    # Test with an unknown user
    unknown_user_id_for_test = "unknown_predict_user_realtime"
    interactions_unknown_user_data = {
        USER_ID_NAME: [unknown_user_id_for_test, unknown_user_id_for_test],
        ITEM_ID_NAME: items_for_prediction[ITEM_ID_NAME].tolist(),
        "inter_feat1": [0.5, 0.5],
    }
    interactions_unknown_user_df = pd.DataFrame(interactions_unknown_user_data)

    predictions_unknown = unpickled_estimator.predict_proba_with_embeddings(
        interactions=interactions_unknown_user_df, users=users_for_prediction_df
    )
    assert predictions_unknown.shape == (len(interactions_unknown_user_df), 1)
    assert np.all((predictions_unknown >= 0) & (predictions_unknown <= 1))


def test_ncf_custom_mlp_layers():
    """Tests NCF with custom MLP layer configurations."""
    # Test with different MLP layer sizes
    configs = [
        {"ncf_type": "mlp", "mlp_layers": [32, 16], "mlp_embedding_dim": 8},
        {"ncf_type": "mlp", "mlp_layers": [128, 64, 32, 16, 8], "mlp_embedding_dim": 16},
        {"ncf_type": "neumf", "mlp_layers": [64, 32], "gmf_embedding_dim": 8, "mlp_embedding_dim": 8},
    ]

    users_data = {USER_ID_NAME: ["u1", "u2"], "feat": [1.0, 2.0]}
    items_data = {ITEM_ID_NAME: ["i1", "i2"], "feat": [3.0, 4.0]}
    interactions_data = {
        USER_ID_NAME: ["u1", "u1", "u2", "u2"],
        ITEM_ID_NAME: ["i1", "i2", "i1", "i2"],
        LABEL_NAME: [1, 0, 0, 1],
    }

    users_df = pd.DataFrame(users_data)
    items_df = pd.DataFrame(items_data)
    interactions_df = pd.DataFrame(interactions_data)

    for config in configs:
        estimator = NCFEstimator(**config, epochs=1, random_state=42, verbose=0)

        estimator.fit_embedding_model(users=users_df, items=items_df, interactions=interactions_df)
        assert estimator.model is not None

        # Verify MLP layers were set correctly
        assert estimator.mlp_layers == config["mlp_layers"]

        predictions = estimator.predict_proba_with_embeddings(interactions_df)
        assert predictions.shape == (len(interactions_df), 1)


def test_ncf_with_dropout():
    """Tests NCF with dropout for regularization."""
    estimator = NCFEstimator(
        ncf_type="mlp",
        mlp_embedding_dim=16,
        mlp_layers=[32, 16],
        dropout=0.3,
        epochs=1,
        random_state=42,
        verbose=0,
    )

    users_data = {USER_ID_NAME: ["u1", "u2"]}
    items_data = {ITEM_ID_NAME: ["i1", "i2"]}
    interactions_data = {
        USER_ID_NAME: ["u1", "u1", "u2", "u2"],
        ITEM_ID_NAME: ["i1", "i2", "i1", "i2"],
        LABEL_NAME: [1, 0, 0, 1],
    }

    estimator.fit_embedding_model(
        users=pd.DataFrame(users_data),
        items=pd.DataFrame(items_data),
        interactions=pd.DataFrame(interactions_data),
    )

    assert estimator.dropout == 0.3
    assert estimator.model is not None

    predictions = estimator.predict_proba_with_embeddings(pd.DataFrame(interactions_data))
    assert predictions.shape == (4, 1)


def test_ncf_invalid_type_raises_error():
    """Tests that invalid ncf_type raises ValueError."""
    with pytest.raises(ValueError, match="ncf_type must be"):
        NCFEstimator(ncf_type="invalid_type")


def test_ncf_with_mse_loss(ncf_test_data: dict):
    """Tests NCF with MSE loss for explicit rating prediction."""
    estimator = NCFEstimator(
        ncf_type="gmf",
        gmf_embedding_dim=8,
        loss_fn_name="mse",
        epochs=1,
        random_state=42,
        verbose=0,
    )

    estimator.fit_embedding_model(
        users=ncf_test_data["users_df"],
        items=ncf_test_data["items_df"],
        interactions=ncf_test_data["interactions_df"],
    )

    predictions = estimator.predict_proba_with_embeddings(ncf_test_data["interactions_df"])
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(ncf_test_data["interactions_df"]), 1)
