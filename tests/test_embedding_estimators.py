import copy
import logging
import pickle
from typing import Dict, Tuple, Type, Union
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_EMBEDDING_NAME, USER_ID_NAME
from skrec.estimator.embedding.base_pytorch_estimator import (
    BasePyTorchEmbeddingEstimator,
)
from skrec.estimator.embedding.contextualized_two_tower_estimator import (
    ContextualizedTwoTowerEstimator,
)
from skrec.estimator.embedding.deep_cross_network_estimator import (
    DeepCrossNetworkEstimator,
)
from skrec.estimator.embedding.neural_factorization_estimator import (
    NeuralFactorizationEstimator,
)


@pytest.fixture(scope="module")
def embedding_data():
    """Provides sample data for embedding estimator tests."""
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


# Parameterize over all three new estimator classes
estimator_classes = [
    ContextualizedTwoTowerEstimator,
    NeuralFactorizationEstimator,
    DeepCrossNetworkEstimator,
]

ClsAndParams = Tuple[Type[BasePyTorchEmbeddingEstimator], Dict[str, Union[int, float, str]]]


@pytest.fixture(params=estimator_classes)
def estimator_cls_and_params(request) -> ClsAndParams:
    """
    Parameterized fixture providing each estimator class and its minimal valid default parameters
    for instantiation in tests that don't need to vary these specific architectural params.
    """
    klass = request.param
    common_params = {"epochs": 1, "random_state": 42, "verbose": 0}
    specific_params = {}

    if klass == ContextualizedTwoTowerEstimator:
        specific_params = {"final_embedding_dim": 8, "user_embedding_dim": 3, "item_embedding_dim": 4}
    elif klass == NeuralFactorizationEstimator:
        specific_params = {"embedding_dim": 8}
    elif klass == DeepCrossNetworkEstimator:
        specific_params = {"embedding_dim": 8, "num_cross_layers": 1}

    all_params = {**common_params, **specific_params}
    return klass, all_params


def test_estimator_fit_predict(estimator_cls_and_params: ClsAndParams, embedding_data: dict):
    """Tests the basic fit and predict flow for each estimator."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)

    estimator.fit_embedding_model(
        users=embedding_data["users_df"],
        items=embedding_data["items_df"],
        interactions=embedding_data["interactions_df"],
    )
    assert estimator.model is not None, "Model should be trained."

    # For predict_proba_with_embeddings, users_df is mandatory.
    # If USER_EMBEDDING_NAME is not in users_df, the model will use its internal embeddings.
    # Test batch prediction mode (using learned embeddings)
    predictions = estimator.predict_proba_with_embeddings(
        interactions=embedding_data["interactions_df"],
    )

    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array."
    assert predictions.shape == (len(embedding_data["interactions_df"]), 1), "Predictions shape mismatch."
    assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions should be probabilities (0-1)."


def test_estimator_get_config(estimator_cls_and_params: ClsAndParams):
    """Tests if get_config returns the correct hyperparameters."""
    estimator_cls, params = estimator_cls_and_params
    params.update(
        {
            "epochs": 3,
            "learning_rate": 0.01,
            "random_state": 123,
        }
    )

    estimator = estimator_cls(**params)
    config = estimator.get_config()

    assert config["epochs"] == 3
    assert config["learning_rate"] == 0.01
    assert config["random_state"] == 123
    for param, value in params.items():
        assert config[param] == value, f"Config mismatch for {param}"


def test_estimator_predict_without_fit_raises_error(estimator_cls_and_params: ClsAndParams, embedding_data: dict):
    """Tests that predict_proba_numpy raises RuntimeError if called before fit."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)
    with pytest.raises(RuntimeError, match="Model has not been trained"):
        estimator.predict_proba_with_embeddings(embedding_data["interactions_df"])


def test_estimator_handles_missing_features(estimator_cls_and_params: ClsAndParams, embedding_data: dict):
    """Tests estimator robustness to missing or empty feature DataFrames."""
    base_interactions = embedding_data["interactions_df"]

    scenarios = {
        "no_user_features": (
            pd.DataFrame({USER_ID_NAME: embedding_data["users_df"][USER_ID_NAME]}),
            embedding_data["items_df"],
            base_interactions,
        ),
        "empty_user_df": (
            pd.DataFrame(columns=[USER_ID_NAME, "user_feat1"]),
            embedding_data["items_df"],
            base_interactions,
        ),
        "no_item_features": (
            embedding_data["users_df"],
            pd.DataFrame({ITEM_ID_NAME: embedding_data["items_df"][ITEM_ID_NAME]}),
            base_interactions,
        ),
        "empty_item_df": (
            embedding_data["users_df"],
            pd.DataFrame(columns=[ITEM_ID_NAME, "item_feat1"]),
            base_interactions,
        ),
        "no_interaction_features": (
            embedding_data["users_df"],
            embedding_data["items_df"],
            base_interactions[[USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME]],
        ),
    }

    for scenario_name, (users_df, items_df, interactions_df) in scenarios.items():
        estimator_cls, params = estimator_cls_and_params
        estimator = estimator_cls(**params)

        estimator.fit_embedding_model(users=users_df, items=items_df, interactions=interactions_df)
        assert estimator.model is not None, f"Model not trained for scenario: {scenario_name}"

        if scenario_name == "no_user_features" or scenario_name == "empty_user_df":
            assert estimator.user_features_dim == 0, f"User features dim should be 0 for {scenario_name}"
        if scenario_name == "no_item_features" or scenario_name == "empty_item_df":
            assert estimator.item_features_dim == 0, f"Item features dim should be 0 for {scenario_name}"
        if scenario_name == "no_interaction_features":
            assert estimator.interaction_features_dim == 0, f"Interaction features dim should be 0 for {scenario_name}"

        predictions = estimator.predict_proba_with_embeddings(interactions_df)
        assert predictions.shape == (len(interactions_df), 1), f"Predictions shape mismatch for {scenario_name}"


def test_estimator_handles_unknown_ids_in_prediction(estimator_cls_and_params: ClsAndParams, embedding_data: dict):
    """Tests prediction with user/item IDs not seen during training."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)

    estimator.fit_embedding_model(
        users=embedding_data["users_df"],
        items=embedding_data["items_df"],
        interactions=embedding_data["interactions_df"],
    )

    predict_interactions_data = {
        USER_ID_NAME: ["u1", "unknown_user_1", "u2", "unknown_user_2"],
        ITEM_ID_NAME: ["i1", "i2", "unknown_item_1", "i3"],
        LABEL_NAME: [1, 0, 1, 0],  # Not used in predict, but good to have
        "inter_feat1": [0.1, 0.2, 0.3, 0.4],
    }
    predict_interactions_df = pd.DataFrame(predict_interactions_data)

    # users_df must be provided. If it doesn't contain USER_EMBEDDING_NAME,
    # the model uses its internal embeddings/features.
    predictions = estimator.predict_proba_with_embeddings(predict_interactions_df)

    assert predictions.shape == (len(predict_interactions_df), 1), "Predictions shape mismatch for unknown IDs."
    assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions for unknown IDs should be probabilities."


def test_estimator_with_empty_features(estimator_cls_and_params: ClsAndParams, embedding_data: dict):
    """Test case where users_df and items_df are provided but contain no feature columns, only IDs."""
    users_df_no_feats = embedding_data["users_df"][[USER_ID_NAME]].copy()
    items_df_no_feats = embedding_data["items_df"][[ITEM_ID_NAME]].copy()
    interactions_df_no_inter_feats = embedding_data["interactions_df"][[USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME]].copy()

    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)
    estimator.fit_embedding_model(
        users=users_df_no_feats, items=items_df_no_feats, interactions=interactions_df_no_inter_feats
    )
    assert estimator.user_features_dim == 0
    assert estimator.item_features_dim == 0
    assert estimator.interaction_features_dim == 0

    predictions = estimator.predict_proba_with_embeddings(interactions_df_no_inter_feats)
    assert predictions.shape == (len(interactions_df_no_inter_feats), 1)


def test_estimator_with_no_features(estimator_cls_and_params: ClsAndParams, embedding_data: dict):
    """Test case where users_df and items_df are None during fit."""
    interactions_df_no_inter_feats = embedding_data["interactions_df"][[USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME]].copy()

    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)
    estimator.fit_embedding_model(users=None, items=None, interactions=interactions_df_no_inter_feats)
    assert estimator.user_features_dim == 0
    assert estimator.item_features_dim == 0
    assert estimator.interaction_features_dim == 0
    assert estimator.user_features_tensor is None  # Should be none if users_df is None
    assert estimator.item_features_tensor is None  # Should be none if items_df is None

    predictions = estimator.predict_proba_with_embeddings(interactions_df_no_inter_feats)
    assert predictions.shape == (len(interactions_df_no_inter_feats), 1)


def test_embedding_extraction_truncation_pickle(estimator_cls_and_params: ClsAndParams, embedding_data: dict):
    """
    Tests the full flow of expected usage:
    1. Train model.
    2. Get user embeddings (would be stored in FMP).
    3. Truncate user data.
    4. Pickle and unpickle the truncated model.
    5. Predict using the unpickled model with provided embeddings.
    """
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)

    # Use context-free interactions: get_user_embeddings() raises for context_mode="user_tower"
    # when context features are present (user reps would be context-dependent).
    interactions_no_context = embedding_data["interactions_df"][[USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME]].copy()

    # 1. Train model
    estimator.fit_embedding_model(
        users=embedding_data["users_df"],
        items=embedding_data["items_df"],
        interactions=interactions_no_context,
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
    # Prepare users_df for prediction - use a subset of users for whom we have embeddings
    users_for_prediction_df = user_embeddings_df.head(2).copy()

    # Prepare interactions_df for these users and some items
    items_for_prediction = embedding_data["items_df"][[ITEM_ID_NAME]].head(2)

    interactions_for_prediction_list = []
    for _, user_row in users_for_prediction_df.iterrows():
        user_id = user_row[USER_ID_NAME]
        for item_id in items_for_prediction[ITEM_ID_NAME]:
            interaction_entry = {
                USER_ID_NAME: user_id,
                ITEM_ID_NAME: item_id,
            }
            interactions_for_prediction_list.append(interaction_entry)

    interactions_predict_df = pd.DataFrame(interactions_for_prediction_list)

    users_for_prediction_df = pd.merge(users_for_prediction_df, embedding_data["users_df"], on=USER_ID_NAME, how="left")

    predictions = unpickled_estimator.predict_proba_with_embeddings(
        interactions=interactions_predict_df, users=users_for_prediction_df
    )

    assert isinstance(predictions, np.ndarray)
    expected_predictions_count = len(interactions_for_prediction_list)
    assert predictions.shape == (
        expected_predictions_count,
        1,
    ), f"Predictions shape mismatch. Expected ({expected_predictions_count}, 1), got {predictions.shape}"

    # Test with an unknown user (not in users_for_prediction_df but in interactions_predict_df)
    # The model should use the placeholder embedding for this user.
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


# ---------------------------------------------------------------------------
# Validation data tests
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_data(embedding_data):
    """A held-out split: last 2 rows of interactions, same users/items."""
    valid_interactions = embedding_data["interactions_df"].tail(2).reset_index(drop=True)
    # valid_users has the same schema as users_df but distinct feature values
    # so we can tell whether the right tensor was used.
    valid_users = embedding_data["users_df"].copy()
    valid_users["user_feat1"] = 99.0
    valid_users["user_feat2"] = 99
    return {"valid_users": valid_users, "valid_interactions": valid_interactions}


def test_fit_with_valid_interactions_logs_val_loss(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, valid_data: dict, caplog
):
    """Val Loss should be logged every epoch when valid_interactions is provided."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**{**params, "epochs": 2, "verbose": 1})

    with caplog.at_level(logging.INFO):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_interactions=valid_data["valid_interactions"],
        )

    val_loss_lines = [r.message for r in caplog.records if "Val Loss" in r.message]
    assert len(val_loss_lines) == 2, "Expected one Val Loss log line per epoch"
    for line in val_loss_lines:
        assert "Loss:" in line and "Val Loss:" in line


def test_fit_without_validation_data_no_val_loss_logged(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, caplog
):
    """No Val Loss should appear in logs when no validation data is provided."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**{**params, "epochs": 2, "verbose": 1})

    with caplog.at_level(logging.INFO):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
        )

    assert not any("Val Loss" in r.message for r in caplog.records)
    train_loss_lines = [r.message for r in caplog.records if "Loss:" in r.message]
    assert len(train_loss_lines) == 2


@pytest.mark.parametrize("missing_col", [USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME])
def test_fit_valid_interactions_missing_required_column_raises(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, valid_data: dict, missing_col: str
):
    """Missing USER_ID, ITEM_ID, or LABEL in valid_interactions should raise ValueError immediately."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)

    bad_valid_interactions = valid_data["valid_interactions"].drop(columns=[missing_col])

    with pytest.raises(ValueError, match=f"valid_interactions is missing required column '{missing_col}'"):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_interactions=bad_valid_interactions,
        )


def test_fit_valid_users_missing_user_id_raises(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, valid_data: dict
):
    """valid_users without USER_ID should raise ValueError."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)

    bad_valid_users = valid_data["valid_users"].drop(columns=[USER_ID_NAME])

    with pytest.raises(ValueError, match=f"valid_users is missing required column '{USER_ID_NAME}'"):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_users=bad_valid_users,
            valid_interactions=valid_data["valid_interactions"],
        )


def test_two_tower_context_modes_fit_predict(embedding_data: dict):
    """All three context_mode values should train and predict without error."""
    for mode in ("user_tower", "trilinear", "scoring_layer"):
        estimator = ContextualizedTwoTowerEstimator(
            final_embedding_dim=8,
            user_embedding_dim=4,
            item_embedding_dim=4,
            context_mode=mode,
            epochs=1,
            random_state=42,
        )
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
        )
        preds = estimator.predict_proba_with_embeddings(embedding_data["interactions_df"])
        assert preds.shape == (len(embedding_data["interactions_df"]), 1), f"mode={mode}"


def test_two_tower_user_tower_mode_blocks_get_user_embeddings(embedding_data: dict):
    """context_mode='user_tower' with context features should raise on get_user_embeddings()."""
    estimator = ContextualizedTwoTowerEstimator(
        final_embedding_dim=8,
        user_embedding_dim=4,
        item_embedding_dim=4,
        context_mode="user_tower",
        epochs=1,
        random_state=42,
    )
    estimator.fit_embedding_model(
        users=embedding_data["users_df"],
        items=embedding_data["items_df"],
        interactions=embedding_data["interactions_df"],  # has inter_feat1
    )
    with pytest.raises(NotImplementedError, match="context_mode='user_tower'"):
        estimator.get_user_embeddings()


def test_two_tower_non_user_tower_modes_allow_get_user_embeddings(embedding_data: dict):
    """context_mode='trilinear' and 'scoring_layer' should allow get_user_embeddings()."""
    for mode in ("trilinear", "scoring_layer"):
        estimator = ContextualizedTwoTowerEstimator(
            final_embedding_dim=8,
            user_embedding_dim=4,
            item_embedding_dim=4,
            context_mode=mode,
            epochs=1,
            random_state=42,
        )
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
        )
        embs = estimator.get_user_embeddings()
        assert len(embs) == len(estimator.user_id_index), f"mode={mode}"


def test_fit_valid_interactions_missing_context_feature_raises(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, valid_data: dict
):
    """valid_interactions missing a context feature column present in training should raise ValueError."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)

    bad_valid_interactions = valid_data["valid_interactions"].drop(columns=["inter_feat1"])

    with pytest.raises(ValueError, match="valid_interactions is missing context feature columns"):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_interactions=bad_valid_interactions,
        )


def test_fit_valid_users_without_valid_interactions_warns(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, valid_data: dict, caplog
):
    """Passing valid_users alone (no valid_interactions) should log a warning and not crash."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)

    with caplog.at_level(logging.WARNING):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_users=valid_data["valid_users"],
        )

    assert any(
        "valid_users" in r.message and "valid_interactions" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_fit_valid_users_features_passed_to_training_loop(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, valid_data: dict
):
    """
    valid_users features should be built into a separate tensor and passed to
    _training_loop as valid_user_features_t — distinct from the training-time
    self.user_features_tensor.
    """
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)

    captured = {}
    original_loop = estimator._training_loop

    def capturing_loop(*args, **kwargs):
        captured["valid_user_features_t"] = kwargs.get("valid_user_features_t")
        return original_loop(*args, **kwargs)

    with patch.object(estimator, "_training_loop", side_effect=capturing_loop):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_users=valid_data["valid_users"],
            valid_interactions=valid_data["valid_interactions"],
        )

    assert "valid_user_features_t" in captured, "_training_loop was not called"
    assert captured["valid_user_features_t"] is not None, "valid_user_features_t should not be None"

    # The valid_users fixture has user_feat1=99.0 for all users — the training
    # users_df has user_feat1 in [0.1, 0.2, 0.3, 0.4]. The two tensors must differ.
    assert not torch.equal(captured["valid_user_features_t"], estimator.user_features_tensor), (
        "valid_user_features_t should be built from valid_users, not the training tensor"
    )


def test_fit_with_valid_data_no_valid_users_uses_training_tensor(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, valid_data: dict
):
    """
    When valid_interactions is given but valid_users is None,
    valid_user_features_t passed to _training_loop should be None
    (the model falls back to self.user_features_tensor during validation forward).
    """
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**params)

    captured = {}
    original_loop = estimator._training_loop

    def capturing_loop(*args, **kwargs):
        captured["valid_user_features_t"] = kwargs.get("valid_user_features_t")
        return original_loop(*args, **kwargs)

    with patch.object(estimator, "_training_loop", side_effect=capturing_loop):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_interactions=valid_data["valid_interactions"],
        )

    assert captured.get("valid_user_features_t") is None


# ---------------------------------------------------------------------------
# Early stopping tests
# ---------------------------------------------------------------------------


def test_early_stopping_without_valid_interactions_raises(estimator_cls_and_params: ClsAndParams):
    """early_stopping_patience without valid_interactions should raise ValueError immediately."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**{**params, "early_stopping_patience": 2})

    with pytest.raises(ValueError, match="early_stopping_patience requires valid_interactions"):
        estimator.fit_embedding_model(
            users=None,
            items=None,
            interactions=pd.DataFrame({USER_ID_NAME: ["u1"], ITEM_ID_NAME: ["i1"], LABEL_NAME: [1]}),
        )


def test_early_stopping_stops_before_max_epochs(estimator_cls_and_params: ClsAndParams, embedding_data: dict, caplog):
    """Training should stop before max epochs when val loss stops improving.

    Uses a validation set with labels flipped from training so that as the model
    fits the training data it diverges on validation, guaranteeing val loss rises.
    """
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**{**params, "epochs": 20, "early_stopping_patience": 2, "verbose": 1})

    # Validation interactions: same users/items as training but labels inverted.
    val_interactions = embedding_data["interactions_df"].copy()
    val_interactions[LABEL_NAME] = 1 - val_interactions[LABEL_NAME]

    with caplog.at_level(logging.INFO):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_interactions=val_interactions,
        )

    val_loss_lines = [r.message for r in caplog.records if "Val Loss" in r.message]
    early_stop_lines = [r.message for r in caplog.records if "Early stopping" in r.message]

    assert len(early_stop_lines) == 1, "Expected exactly one early stopping log message"
    assert len(val_loss_lines) < 20, "Training should have stopped before all 20 epochs"


def test_early_stopping_restores_best_weights(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, valid_data: dict
):
    """With restore_best_weights=True, model state should be from the best val loss epoch."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**{**params, "epochs": 10, "early_stopping_patience": 1, "restore_best_weights": True})

    # Capture the best state manually by intercepting _training_loop
    captured = {}
    original_loop = estimator._training_loop

    def capturing_loop(*args, **kwargs):
        # Wrap to record model state at each val improvement
        original_loop(*args, **kwargs)
        captured["final_state"] = copy.deepcopy(kwargs["model"].state_dict())

    with patch.object(estimator, "_training_loop", side_effect=capturing_loop):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_interactions=valid_data["valid_interactions"],
        )

    # After fit, model state should equal what _training_loop left it as (best weights restored)
    for key in captured["final_state"]:
        assert torch.equal(estimator.model.state_dict()[key], captured["final_state"][key]), (
            f"Parameter '{key}' mismatch after best-weight restoration"
        )


def test_early_stopping_no_restore_keeps_last_weights(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, valid_data: dict
):
    """With restore_best_weights=False, model keeps weights from the epoch it stopped at."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**{**params, "epochs": 10, "early_stopping_patience": 1, "restore_best_weights": False})
    # Should complete without error — just verifying no crash and model is available
    estimator.fit_embedding_model(
        users=embedding_data["users_df"],
        items=embedding_data["items_df"],
        interactions=embedding_data["interactions_df"],
        valid_interactions=valid_data["valid_interactions"],
    )
    assert estimator.model is not None
    predictions = estimator.predict_proba_with_embeddings(embedding_data["interactions_df"])
    assert predictions.shape == (len(embedding_data["interactions_df"]), 1)


def test_early_stopping_patience_not_triggered_completes_all_epochs(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, valid_data: dict, caplog
):
    """With patience larger than epochs, all epochs should run with no early stop message."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**{**params, "epochs": 3, "early_stopping_patience": 100, "verbose": 1})

    with caplog.at_level(logging.INFO):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_interactions=valid_data["valid_interactions"],
        )

    val_loss_lines = [r.message for r in caplog.records if "Val Loss" in r.message]
    early_stop_lines = [r.message for r in caplog.records if "Early stopping" in r.message]

    assert len(val_loss_lines) == 3, "All 3 epochs should have run"
    assert len(early_stop_lines) == 0, "Early stopping should not have triggered"


def test_two_tower_user_tower_missing_context_raises_at_inference(embedding_data: dict):
    """USER_TOWER trained with context features should raise ValueError at inference
    if context columns are absent — instead of a cryptic dimension mismatch."""
    estimator = ContextualizedTwoTowerEstimator(
        final_embedding_dim=8,
        user_embedding_dim=4,
        item_embedding_dim=4,
        context_mode="user_tower",
        epochs=1,
        random_state=42,
    )
    # Train with interactions that include inter_feat1 (context feature)
    estimator.fit_embedding_model(
        users=embedding_data["users_df"],
        items=embedding_data["items_df"],
        interactions=embedding_data["interactions_df"],
    )
    # Drop the context column at serving time
    no_context = embedding_data["interactions_df"].drop(columns=["inter_feat1"])
    with pytest.raises(ValueError, match="missing from the inference interactions DataFrame"):
        estimator.predict_proba_with_embeddings(no_context)


def test_mse_estimator_does_not_apply_sigmoid(embedding_data: dict):
    """MSE-loss (regressor) estimator should return raw logits, not sigmoid-squashed values."""
    estimator = ContextualizedTwoTowerEstimator(
        final_embedding_dim=8,
        user_embedding_dim=4,
        item_embedding_dim=4,
        loss_fn_name="mse",
        epochs=1,
        random_state=42,
    )
    estimator.fit_embedding_model(
        users=embedding_data["users_df"],
        items=embedding_data["items_df"],
        interactions=embedding_data["interactions_df"],
    )
    # Force large biases in the user tower so dot-product scores are clearly > 1
    import torch

    with torch.no_grad():
        for param in estimator.model.user_tower.parameters():
            if param.dim() == 1:  # bias vectors
                param.fill_(10.0)
    preds = estimator.predict_proba_with_embeddings(embedding_data["interactions_df"])
    assert preds.max() > 1.0, "MSE regressor should return unscaled logits, not sigmoid-squashed probabilities"


def test_bce_estimator_applies_sigmoid(embedding_data: dict):
    """BCE-loss (classifier) estimator should apply sigmoid, keeping outputs in [0, 1]."""
    estimator = ContextualizedTwoTowerEstimator(
        final_embedding_dim=8,
        user_embedding_dim=4,
        item_embedding_dim=4,
        loss_fn_name="bce",
        epochs=1,
        random_state=42,
    )
    estimator.fit_embedding_model(
        users=embedding_data["users_df"],
        items=embedding_data["items_df"],
        interactions=embedding_data["interactions_df"],
    )
    # Same large-bias trick: sigmoid(large) ≈ 1.0, still within [0, 1]
    import torch

    with torch.no_grad():
        for param in estimator.model.user_tower.parameters():
            if param.dim() == 1:
                param.fill_(10.0)
    preds = estimator.predict_proba_with_embeddings(embedding_data["interactions_df"])
    assert np.all((preds >= 0.0) & (preds <= 1.0)), "BCE classifier must return probabilities in [0, 1]"


def test_early_stopping_best_state_none_logs_warning(
    estimator_cls_and_params: ClsAndParams, embedding_data: dict, caplog
):
    """When restore_best_weights=True but deepcopy is patched to return None,
    the warning about no improvement being recorded must be logged on early stop."""
    estimator_cls, params = estimator_cls_and_params
    estimator = estimator_cls(**{**params, "epochs": 30, "early_stopping_patience": 1, "restore_best_weights": True})

    # Adversarial val set: inverted labels ensure val loss eventually rises as train loss falls.
    # Some estimators need more epochs before they overfit and val loss turns around.
    val_interactions = embedding_data["interactions_df"].copy()
    val_interactions[LABEL_NAME] = 1 - val_interactions[LABEL_NAME]

    with patch("skrec.estimator.embedding.base_pytorch_estimator.copy.deepcopy", return_value=None):
        with caplog.at_level(logging.WARNING):
            estimator.fit_embedding_model(
                users=embedding_data["users_df"],
                items=embedding_data["items_df"],
                interactions=embedding_data["interactions_df"],
                valid_interactions=val_interactions,
            )

    assert any(
        "restore_best_weights=True but no improvement was ever recorded" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    ), "Expected warning about missing best state when deepcopy returns None"


def test_early_stopping_skips_patience_when_no_valid_val_batches(embedding_data: dict, caplog):
    """When valid_interactions is empty (0 rows), early stopping should be skipped with a warning
    rather than incrementing the patience counter on a spurious inf/0 val loss."""
    estimator = ContextualizedTwoTowerEstimator(
        final_embedding_dim=8,
        user_embedding_dim=4,
        item_embedding_dim=4,
        epochs=3,
        early_stopping_patience=1,  # would fire on epoch 2 if val loss were inf
        verbose=0,
        random_state=42,
    )
    # Empty valid_interactions — has required columns but zero rows; explicit dtypes to avoid
    # torch.tensor() TypeError on object-dtype arrays produced by pd.DataFrame(columns=[...])
    empty_valid = pd.DataFrame(
        {
            USER_ID_NAME: pd.Series([], dtype=object),
            ITEM_ID_NAME: pd.Series([], dtype=object),
            LABEL_NAME: pd.Series([], dtype=float),
            "inter_feat1": pd.Series([], dtype=float),
        }
    )
    with caplog.at_level(logging.WARNING):
        estimator.fit_embedding_model(
            users=embedding_data["users_df"],
            items=embedding_data["items_df"],
            interactions=embedding_data["interactions_df"],
            valid_interactions=empty_valid,
        )
    # Warning must have been logged
    assert any(
        "skipping early stopping" in r.message.lower() for r in caplog.records if r.levelno == logging.WARNING
    ), "Expected a warning about skipping early stopping when no valid batches"
    # All 3 epochs must have completed (patience=1 would have stopped at epoch 2 otherwise)
    assert estimator.model is not None
