"""Integration tests for MatrixFactorizationEstimator with UniversalScorer and RankingRecommender."""

import numpy as np
import pandas as pd
import pytest

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.datatypes import MFAlgorithm
from skrec.estimator.embedding.matrix_factorization_estimator import (
    MatrixFactorizationEstimator,
)
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.universal import UniversalScorer


@pytest.fixture
def sample_cf_datasets(tmp_path):
    """Sample datasets for CF integration (no torch required)."""
    users_data = {
        USER_ID_NAME: ["u1", "u2", "u3", "u4", "u5"],
    }
    users_df = pd.DataFrame(users_data)
    users_file = tmp_path / "users.csv"
    users_df.to_csv(users_file, index=False)

    items_data = {
        ITEM_ID_NAME: ["i1", "i2", "i3", "i4", "i5"],
    }
    items_df = pd.DataFrame(items_data)
    items_file = tmp_path / "items.csv"
    items_df.to_csv(items_file, index=False)

    interactions_data = {
        USER_ID_NAME: ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4", "u5", "u5"],
        ITEM_ID_NAME: ["i1", "i2", "i2", "i3", "i1", "i4", "i3", "i5", "i2", "i4"],
        LABEL_NAME: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    }
    interactions_df = pd.DataFrame(interactions_data)
    interactions_file = tmp_path / "interactions.csv"
    interactions_df.to_csv(interactions_file, index=False)

    return {
        "users_dataset": UsersDataset(data_location=str(users_file)),
        "items_dataset": ItemsDataset(data_location=str(items_file)),
        "interactions_dataset": InteractionsDataset(data_location=str(interactions_file)),
        "users_df": users_df,
        "items_df": items_df,
        "interactions_df": interactions_df,
    }


def test_mf_with_universal_scorer_and_recommender(sample_cf_datasets):
    """MatrixFactorizationEstimator works with UniversalScorer and RankingRecommender."""
    estimator = MatrixFactorizationEstimator(
        n_factors=8,
        regularization=0.01,
        epochs=15,
        random_state=42,
        verbose=0,
    )
    scorer = UniversalScorer(estimator=estimator)
    recommender = RankingRecommender(scorer=scorer)

    recommender.train(
        users_ds=sample_cf_datasets["users_dataset"],
        items_ds=sample_cf_datasets["items_dataset"],
        interactions_ds=sample_cf_datasets["interactions_dataset"],
    )

    test_users_df = sample_cf_datasets["interactions_df"].head(2)[[USER_ID_NAME]].drop_duplicates()
    recommendations = recommender.recommend(
        interactions=test_users_df,
        users=None,
        top_k=3,
    )
    assert recommendations.shape == (len(test_users_df), 3)
    assert isinstance(recommendations, np.ndarray)
    valid_items = set(sample_cf_datasets["items_df"][ITEM_ID_NAME])
    for row in recommendations:
        for item_id in row:
            assert item_id in valid_items, f"Invalid item ID: {item_id}"


def test_mf_score_items_integration(sample_cf_datasets):
    """score_items returns correct shape (n_users x n_items)."""
    estimator = MatrixFactorizationEstimator(
        n_factors=4,
        epochs=10,
        random_state=42,
    )
    scorer = UniversalScorer(estimator=estimator)
    recommender = RankingRecommender(scorer=scorer)
    recommender.train(
        items_ds=sample_cf_datasets["items_dataset"],
        interactions_ds=sample_cf_datasets["interactions_dataset"],
    )

    test_users_df = pd.DataFrame({USER_ID_NAME: ["u1", "u2"]})
    scores = recommender.score_items(
        interactions=test_users_df,
        users=None,
    )
    assert scores.shape == (2, 5), "Expected (n_users=2, n_items=5)"
    assert np.all(np.isfinite(scores.to_numpy()))


def test_mf_recommend_top_k_unique(sample_cf_datasets):
    """Recommendations for one user are top_k and use item set."""
    estimator = MatrixFactorizationEstimator(n_factors=8, epochs=20, random_state=42)
    scorer = UniversalScorer(estimator=estimator)
    recommender = RankingRecommender(scorer=scorer)
    recommender.train(
        items_ds=sample_cf_datasets["items_dataset"],
        interactions_ds=sample_cf_datasets["interactions_dataset"],
    )

    test_users_df = pd.DataFrame({USER_ID_NAME: ["u1"]})
    recs = recommender.recommend(interactions=test_users_df, users=None, top_k=5)
    assert recs.shape == (1, 5)
    assert len(np.unique(recs[0])) == 5, "Top-5 should be 5 distinct items"


def test_mf_sgd_integration(sample_cf_datasets):
    """SGD algorithm works in full pipeline (train + recommend)."""
    estimator = MatrixFactorizationEstimator(
        n_factors=6,
        algorithm=MFAlgorithm.SGD,
        learning_rate=0.02,
        epochs=15,
        random_state=42,
    )
    scorer = UniversalScorer(estimator=estimator)
    recommender = RankingRecommender(scorer=scorer)
    recommender.train(
        items_ds=sample_cf_datasets["items_dataset"],
        interactions_ds=sample_cf_datasets["interactions_dataset"],
    )
    test_users_df = pd.DataFrame({USER_ID_NAME: ["u1"]})
    recs = recommender.recommend(interactions=test_users_df, users=None, top_k=3)
    assert recs.shape == (1, 3)
    assert np.all(np.isin(recs, sample_cf_datasets["items_df"][ITEM_ID_NAME].values))
