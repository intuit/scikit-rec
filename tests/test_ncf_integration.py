"""
Integration test for NCF with UniversalScorer and RankingRecommender.
This test validates that NCF works end-to-end in the recommender pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.embedding.ncf_estimator import NCFEstimator
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.universal import UniversalScorer


@pytest.fixture
def sample_datasets(tmp_path):
    """Create sample datasets for integration testing."""
    # Create users data
    users_data = {
        USER_ID_NAME: ["u1", "u2", "u3", "u4", "u5"],
        "age": [25, 30, 35, 40, 45],
        "gender": [1, 0, 1, 0, 1],
    }
    users_df = pd.DataFrame(users_data)
    users_file = tmp_path / "users.csv"
    users_df.to_csv(users_file, index=False)

    # Create items data
    items_data = {
        ITEM_ID_NAME: ["i1", "i2", "i3", "i4", "i5"],
        "category": [1, 2, 1, 3, 2],
        "price": [10.0, 20.0, 15.0, 25.0, 30.0],
    }
    items_df = pd.DataFrame(items_data)
    items_file = tmp_path / "items.csv"
    items_df.to_csv(items_file, index=False)

    # Create interactions data (implicit feedback)
    interactions_data = {
        USER_ID_NAME: ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4", "u5", "u5"],
        ITEM_ID_NAME: ["i1", "i2", "i2", "i3", "i1", "i4", "i3", "i5", "i2", "i4"],
        LABEL_NAME: [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    }
    interactions_df = pd.DataFrame(interactions_data)
    interactions_file = tmp_path / "interactions.csv"
    interactions_df.to_csv(interactions_file, index=False)

    # Create dataset objects
    users_dataset = UsersDataset(data_location=str(users_file))
    items_dataset = ItemsDataset(data_location=str(items_file))
    interactions_dataset = InteractionsDataset(data_location=str(interactions_file))

    return {
        "users_dataset": users_dataset,
        "items_dataset": items_dataset,
        "interactions_dataset": interactions_dataset,
        "users_df": users_df,
        "items_df": items_df,
        "interactions_df": interactions_df,
    }


def test_ncf_with_universal_scorer_and_recommender(sample_datasets):
    """Test NCF estimator with UniversalScorer and RankingRecommender."""
    # Create NCF estimator (using NeuMF)
    ncf_estimator = NCFEstimator(
        ncf_type="neumf",
        gmf_embedding_dim=8,
        mlp_embedding_dim=8,
        mlp_layers=[16, 8],
        dropout=0.1,
        learning_rate=0.01,
        epochs=2,
        batch_size=4,
        random_state=42,
        verbose=0,
    )

    # Create scorer with NCF estimator
    scorer = UniversalScorer(estimator=ncf_estimator)

    # Create recommender
    recommender = RankingRecommender(scorer=scorer)

    # Train the recommender
    recommender.train(
        users_ds=sample_datasets["users_dataset"],
        items_ds=sample_datasets["items_dataset"],
        interactions_ds=sample_datasets["interactions_dataset"],
    )

    # Test recommendation (for embedding estimators in batch mode, pass interactions with USER_ID only)
    test_users_df = sample_datasets["interactions_df"][:2][[USER_ID_NAME]].drop_duplicates()
    recommendations = recommender.recommend(
        interactions=test_users_df,
        users=None,  # Batch mode - use learned embeddings
        top_k=3,
    )

    # Verify recommendations shape
    assert recommendations.shape == (
        len(test_users_df),
        3,
    ), f"Expected ({len(test_users_df)}, 3), got {recommendations.shape}"
    assert isinstance(recommendations, np.ndarray)

    # Verify recommendations are valid item IDs
    valid_items = set(sample_datasets["items_df"][ITEM_ID_NAME])
    for user_recs in recommendations:
        for item_id in user_recs:
            assert item_id in valid_items, f"Invalid item ID: {item_id}"


def test_ncf_gmf_variant_with_recommender(sample_datasets):
    """Test NCF GMF variant in the full pipeline."""
    ncf_estimator = NCFEstimator(
        ncf_type="gmf",
        gmf_embedding_dim=8,
        learning_rate=0.01,
        epochs=2,
        batch_size=4,
        random_state=42,
        verbose=0,
    )

    scorer = UniversalScorer(estimator=ncf_estimator)
    recommender = RankingRecommender(scorer=scorer)

    recommender.train(
        users_ds=sample_datasets["users_dataset"],
        items_ds=sample_datasets["items_dataset"],
        interactions_ds=sample_datasets["interactions_dataset"],
    )

    test_users_df = pd.DataFrame({USER_ID_NAME: ["u1"]})
    recommendations = recommender.recommend(
        interactions=test_users_df,
        users=None,  # Batch mode
        top_k=5,
    )

    assert recommendations.shape == (1, 5)
    assert len(np.unique(recommendations[0])) == 5, "Recommendations should be unique"


def test_ncf_mlp_variant_with_recommender(sample_datasets):
    """Test NCF MLP variant in the full pipeline."""
    ncf_estimator = NCFEstimator(
        ncf_type="mlp",
        mlp_embedding_dim=8,
        mlp_layers=[16, 8],
        dropout=0.1,
        learning_rate=0.01,
        epochs=2,
        batch_size=4,
        random_state=42,
        verbose=0,
    )

    scorer = UniversalScorer(estimator=ncf_estimator)
    recommender = RankingRecommender(scorer=scorer)

    recommender.train(
        users_ds=sample_datasets["users_dataset"],
        items_ds=sample_datasets["items_dataset"],
        interactions_ds=sample_datasets["interactions_dataset"],
    )

    # Test score_items functionality
    test_users_df = pd.DataFrame({USER_ID_NAME: ["u1"]})
    scores = recommender.score_items(
        interactions=test_users_df,
        users=None,  # Batch mode
    )

    # Scores should be a DataFrame with user rows and item columns
    assert isinstance(scores, pd.DataFrame)
    assert scores.shape[0] == 1  # 1 user
    assert scores.shape[1] == 5  # 5 items


def test_ncf_with_new_users_and_items(sample_datasets):
    """Test NCF handles new users/items at inference time."""
    ncf_estimator = NCFEstimator(
        ncf_type="neumf",
        gmf_embedding_dim=8,
        mlp_embedding_dim=8,
        mlp_layers=[16, 8],
        epochs=2,
        batch_size=4,
        random_state=42,
        verbose=0,
    )

    scorer = UniversalScorer(estimator=ncf_estimator)
    recommender = RankingRecommender(scorer=scorer)

    recommender.train(
        users_ds=sample_datasets["users_dataset"],
        items_ds=sample_datasets["items_dataset"],
        interactions_ds=sample_datasets["interactions_dataset"],
    )

    # Create data with new (unknown) users
    new_users_df = pd.DataFrame({USER_ID_NAME: ["u_new_1", "u_new_2"]})

    # Should handle unknown users gracefully (they'll use the unknown embedding)
    recommendations = recommender.recommend(
        interactions=new_users_df,
        users=None,  # Batch mode
        top_k=3,
    )

    assert recommendations.shape == (2, 3)


def test_ncf_scoring_consistency(sample_datasets):
    """Test that scoring is consistent and produces valid probabilities."""
    ncf_estimator = NCFEstimator(
        ncf_type="neumf",
        gmf_embedding_dim=8,
        mlp_embedding_dim=8,
        mlp_layers=[16, 8],
        epochs=3,
        batch_size=4,
        random_state=42,
        verbose=0,
    )

    scorer = UniversalScorer(estimator=ncf_estimator)
    recommender = RankingRecommender(scorer=scorer)

    recommender.train(
        users_ds=sample_datasets["users_dataset"],
        items_ds=sample_datasets["items_dataset"],
        interactions_ds=sample_datasets["interactions_dataset"],
    )

    # Score items multiple times - should be deterministic
    test_users_df = pd.DataFrame({USER_ID_NAME: ["u1", "u2"]})

    scores1 = recommender.score_items(
        interactions=test_users_df,
        users=None,  # Batch mode
    )

    scores2 = recommender.score_items(
        interactions=test_users_df,
        users=None,  # Batch mode
    )

    # Scores should be identical (deterministic)
    pd.testing.assert_frame_equal(scores1, scores2)

    # All scores should be probabilities (0-1 range)
    assert (scores1.values >= 0).all() and (scores1.values <= 1).all(), "Scores should be in [0, 1] range"
