"""Integration tests: RankingRecommender + retrievers end-to-end."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.estimator.embedding.matrix_factorization_estimator import (
    MatrixFactorizationEstimator,
)
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.retriever.content_based_retriever import ContentBasedRetriever
from skrec.retriever.embedding_retriever import EmbeddingRetriever
from skrec.retriever.popularity_retriever import PopularityRetriever
from skrec.scorer.universal import UniversalScorer
from tests.utils import MockEstimator


@pytest.fixture
def interactions_df():
    return pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u2", "u2", "u3", "u3", "u1"],
            ITEM_ID_NAME: ["i1", "i2", "i1", "i3", "i2", "i3", "i3"],
            LABEL_NAME: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def items_df():
    return pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2", "i3"],
            "price": [10.0, 20.0, 30.0],
            "rating": [4.0, 3.0, 5.0],
        }
    )


def _make_datasets(interactions_df, items_df):
    interactions_ds = InteractionsDataset(data_location=None)
    interactions_ds.fetch_data = MagicMock(return_value=interactions_df)
    items_ds = ItemsDataset(data_location=None)
    items_ds.fetch_data = MagicMock(return_value=items_df)
    return interactions_ds, items_ds


def test_embedding_retriever_end_to_end(interactions_df, items_df):
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=5, random_state=42)),
        retriever=EmbeddingRetriever(top_k=2),
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
    query = pd.DataFrame({USER_ID_NAME: ["u1"]})
    recs = recommender.recommend(interactions=query, top_k=1)
    assert recs.shape[0] == 1
    assert recs[0][0] in ["i1", "i2", "i3"]


def test_popularity_retriever_end_to_end(interactions_df, items_df):
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=5, random_state=42)),
        retriever=PopularityRetriever(top_k=2),
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
    query = pd.DataFrame({USER_ID_NAME: ["u1"]})
    recs = recommender.recommend(interactions=query, top_k=1)
    assert recs.shape[0] == 1


def test_content_based_retriever_end_to_end(interactions_df, items_df):
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=5, random_state=42)),
        retriever=ContentBasedRetriever(top_k=2, feature_columns=["price", "rating"]),
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
    query = pd.DataFrame({USER_ID_NAME: ["u1"]})
    recs = recommender.recommend(interactions=query, top_k=1)
    assert recs.shape[0] == 1


def test_recommendations_are_subset_of_retrieved_candidates(interactions_df, items_df):
    """Every recommendation for every user must come from that user's retrieved
    candidate set. Checks all top_k slots across all users — not just the first."""
    users = ["u1", "u2", "u3"]
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    retriever = EmbeddingRetriever(top_k=2)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=5, random_state=42)),
        retriever=retriever,
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)

    query = pd.DataFrame({USER_ID_NAME: users})
    recs = recommender.recommend(interactions=query, top_k=2)  # 2 recs per user

    for i, uid in enumerate(users):
        user_candidates = set(str(c) for c in retriever.retrieve([uid], top_k=2)[uid])
        for rec in recs[i]:
            assert str(rec) in user_candidates, f"{uid}: recommendation {rec!r} not in candidate set {user_candidates}"


def test_no_retriever_still_works(interactions_df, items_df):
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=5, random_state=42)),
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
    query = pd.DataFrame({USER_ID_NAME: ["u1", "u2"]})
    recs = recommender.recommend(interactions=query, top_k=2)
    assert recs.shape == (2, 2)


def test_probabilistic_sampling_with_retriever(interactions_df, items_df):
    """Probabilistic sampling must work end-to-end when a retriever is attached."""
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=5, random_state=42)),
        retriever=PopularityRetriever(top_k=3),
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
    query = pd.DataFrame({USER_ID_NAME: ["u1"]})
    recs = recommender.recommend(interactions=query, top_k=1, sampling_temperature=1.0)
    assert recs.shape[0] == 1
    assert recs[0][0] in ["i1", "i2", "i3"]


def test_negative_sampling_temperature_raises(interactions_df, items_df):
    """sampling_temperature < 0 must raise ValueError in the retriever path."""
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=5, random_state=42)),
        retriever=PopularityRetriever(top_k=3),
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
    query = pd.DataFrame({USER_ID_NAME: ["u1"]})
    with pytest.raises(ValueError, match="negative"):
        recommender.recommend(interactions=query, top_k=1, sampling_temperature=-1.0)


def test_multi_user_retriever_result_shape(interactions_df, items_df):
    """recommend() with a retriever and multiple users must return shape (n_users, top_k)."""
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=5, random_state=42)),
        retriever=PopularityRetriever(top_k=3),
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
    query = pd.DataFrame({USER_ID_NAME: ["u1", "u2", "u3"]})
    recs = recommender.recommend(interactions=query, top_k=2)
    assert recs.shape == (3, 2)


def test_content_based_retriever_without_items_ds_raises_clear_error(interactions_df):
    """train() must raise a clear ValueError when ContentBasedRetriever is used
    but items_ds is not passed — not a cryptic error from inside build_index()."""
    interactions_ds = InteractionsDataset(data_location=None)
    interactions_ds.fetch_data = MagicMock(return_value=interactions_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=1, random_state=42)),
        retriever=ContentBasedRetriever(top_k=2),
    )
    with pytest.raises(ValueError, match="items_ds"):
        recommender.train(interactions_ds=interactions_ds)  # no items_ds


def test_popularity_retriever_without_interactions_ds_raises_clear_error(items_df):
    """train() must raise a clear ValueError when PopularityRetriever is used
    but interactions_ds is not passed — not a cryptic error from inside build_index()."""
    items_ds = ItemsDataset(data_location=None)
    items_ds.fetch_data = MagicMock(return_value=items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=1, random_state=42)),
        retriever=PopularityRetriever(top_k=2),
    )
    with pytest.raises(ValueError, match="interactions_ds"):
        recommender.train(items_ds=items_ds)  # no interactions_ds


def test_embedding_retriever_with_non_embedding_estimator_raises_clear_error(interactions_df, items_df):
    """train() must raise a clear ValueError when EmbeddingRetriever is paired with a
    non-embedding estimator — before training starts, matching the behavior of the other
    retriever pre-validation checks."""
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MockEstimator()),
        retriever=EmbeddingRetriever(top_k=2),
    )
    with pytest.raises(ValueError, match="BaseEmbeddingEstimator"):
        recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)


def test_recommend_online_with_retriever_logs_warning(interactions_df, items_df, caplog):
    """recommend_online() must log a warning when a retriever is attached,
    since the retriever is silently bypassed in that code path.

    The MF estimator doesn't support recommend_online() (NotImplementedError),
    but the warning is emitted before that check — so we verify the warning
    was logged regardless of whether the call completes.
    """
    import logging

    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=3, random_state=42)),
        retriever=PopularityRetriever(top_k=3),
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
    query = pd.DataFrame({USER_ID_NAME: ["u1"]})
    with caplog.at_level(logging.WARNING):
        try:
            recommender.recommend_online(interactions=query, top_k=1)
        except NotImplementedError:
            pass  # expected: MF doesn't support recommend_online(); warning is logged before this raises
    assert any("recommend_online" in msg for msg in caplog.messages)


def test_item_subset_conflict_with_retriever_logs_warning(interactions_df, items_df, caplog):
    """Setting item_subset externally while a retriever is attached must log a warning,
    since the retriever will override the external subset silently."""
    import logging

    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=3, random_state=42)),
        retriever=PopularityRetriever(top_k=3),
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
    recommender.set_item_subset(["i1", "i2"])
    query = pd.DataFrame({USER_ID_NAME: ["u1"]})
    with caplog.at_level(logging.WARNING):
        recommender.recommend(interactions=query, top_k=1)
    assert any("item_subset" in msg for msg in caplog.messages)


def test_sampling_temperature_none_treated_as_deterministic(interactions_df, items_df):
    """sampling_temperature=None must behave identically to sampling_temperature=0
    (deterministic ranking) — not raise a TypeError — both with and without a retriever."""
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    query = pd.DataFrame({USER_ID_NAME: ["u1"]})

    # Without retriever
    recommender_no_ret = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=3, random_state=42)),
    )
    recommender_no_ret.train(interactions_ds=interactions_ds, items_ds=items_ds)
    recs_none = recommender_no_ret.recommend(interactions=query, top_k=1, sampling_temperature=None)
    recs_zero = recommender_no_ret.recommend(interactions=query, top_k=1, sampling_temperature=0)
    assert list(recs_none[0]) == list(recs_zero[0])

    # With retriever
    interactions_ds2, items_ds2 = _make_datasets(interactions_df, items_df)
    recommender_ret = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=3, random_state=42)),
        retriever=PopularityRetriever(top_k=3),
    )
    recommender_ret.train(interactions_ds=interactions_ds2, items_ds=items_ds2)
    recs_none_ret = recommender_ret.recommend(interactions=query, top_k=1, sampling_temperature=None)
    recs_zero_ret = recommender_ret.recommend(interactions=query, top_k=1, sampling_temperature=0)
    assert list(recs_none_ret[0]) == list(recs_zero_ret[0])


def test_per_user_recommendations_come_from_own_candidates(interactions_df, items_df):
    """Each row in the output must be a top-k item from that user's retrieved candidates,
    not from another user's candidate set (verifies the per-user loop wires correctly)."""
    interactions_ds, items_ds = _make_datasets(interactions_df, items_df)
    retriever = EmbeddingRetriever(top_k=2)
    recommender = RankingRecommender(
        scorer=UniversalScorer(estimator=MatrixFactorizationEstimator(n_factors=4, epochs=5, random_state=42)),
        retriever=retriever,
    )
    recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)

    query = pd.DataFrame({USER_ID_NAME: ["u1", "u2", "u3"]})
    recs = recommender.recommend(interactions=query, top_k=1)

    for i, uid in enumerate(["u1", "u2", "u3"]):
        user_candidates = retriever.retrieve([uid], top_k=2)[uid]
        assert recs[i][0] in [str(c) for c in user_candidates], (
            f"Recommendation for {uid} ({recs[i][0]!r}) not in their candidate set {user_candidates}"
        )
