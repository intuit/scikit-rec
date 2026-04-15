"""Unit tests for EmbeddingRetriever."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from skrec.constants import (
    ITEM_EMBEDDING_NAME,
    ITEM_ID_NAME,
    LABEL_NAME,
    USER_EMBEDDING_NAME,
    USER_ID_NAME,
)
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.estimator.embedding.matrix_factorization_estimator import (
    MatrixFactorizationEstimator,
)
from skrec.retriever.embedding_retriever import EmbeddingRetriever


@pytest.fixture
def small_interactions():
    return pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u2", "u2", "u3", "u3"],
            ITEM_ID_NAME: ["i1", "i2", "i1", "i3", "i2", "i3"],
            LABEL_NAME: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        }
    )


@pytest.fixture
def small_items():
    return pd.DataFrame({ITEM_ID_NAME: ["i1", "i2", "i3"]})


@pytest.fixture
def fitted_mf(small_interactions, small_items):
    est = MatrixFactorizationEstimator(n_factors=4, epochs=3, random_state=42)
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)
    return est


def test_build_index_stores_correct_shape(fitted_mf, small_items):
    retriever = EmbeddingRetriever(top_k=3)
    retriever.build_index(estimator=fitted_mf)
    assert retriever._item_matrix.shape == (3, 4)
    assert len(retriever._item_ids) == 3


def test_build_index_caches_user_embeddings(fitted_mf):
    """User embeddings must be cached at build time, not fetched on each retrieve() call."""
    retriever = EmbeddingRetriever(top_k=3)
    retriever.build_index(estimator=fitted_mf)
    # 3 users (u1, u2, u3) should be in the cache
    assert retriever._user_emb_by_id is not None
    assert len(retriever._user_emb_by_id) == 3
    assert "u1" in retriever._user_emb_by_id.index


def test_retrieve_returns_correct_count(fitted_mf):
    retriever = EmbeddingRetriever(top_k=2)
    retriever.build_index(estimator=fitted_mf)
    results = retriever.retrieve(["u1", "u2"], top_k=2)
    assert len(results["u1"]) == 2
    assert len(results["u2"]) == 2


def test_retrieve_results_are_subset_of_training_items(fitted_mf):
    retriever = EmbeddingRetriever(top_k=3)
    retriever.build_index(estimator=fitted_mf)
    results = retriever.retrieve(["u1"], top_k=3)
    known_items = set(fitted_mf.item_id_index_)
    assert set(results["u1"]).issubset(known_items)


def test_retrieve_order_is_descending_by_dot_product_score(fitted_mf):
    """Items must be returned in descending dot-product score order, not arbitrary order."""
    retriever = EmbeddingRetriever(top_k=3)
    retriever.build_index(estimator=fitted_mf)
    results = retriever.retrieve(["u1"], top_k=3)

    # Recompute expected ordering independently from raw embeddings.
    user_emb_df = fitted_mf.get_user_embeddings().set_index(USER_ID_NAME)
    item_emb_df = fitted_mf.get_item_embeddings()
    user_vec = np.asarray(user_emb_df.loc["u1", USER_EMBEDDING_NAME], dtype=np.float64).ravel()
    item_matrix = np.stack(item_emb_df[ITEM_EMBEDDING_NAME].values).astype(np.float64)
    item_ids = item_emb_df[ITEM_ID_NAME].values

    scores = item_matrix @ user_vec
    expected = item_ids[np.argsort(scores)[::-1]].tolist()

    assert results["u1"] == expected


def test_retrieve_order_is_descending_deterministic():
    """Items must be returned in descending dot-product score order.

    Uses manually constructed embeddings with a known expected order so the
    assertion doesn't depend on the training outcome of MatrixFactorization.

    Item embeddings (1-D for clarity):  i1=3.0, i2=1.0, i3=2.0
    User embedding:                     u1=1.0
    Dot-product scores:                 i1→3.0, i2→1.0, i3→2.0
    Expected ranking:                   [i1, i3, i2]
    """
    fake_estimator = MagicMock()
    fake_estimator.__class__ = BaseEmbeddingEstimator

    item_embs = np.array([[3.0], [1.0], [2.0]], dtype=np.float64)
    fake_estimator.get_item_embeddings.return_value = pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2", "i3"],
            ITEM_EMBEDDING_NAME: list(item_embs),
        }
    )

    user_embs = np.array([[1.0]], dtype=np.float64)
    fake_estimator.get_user_embeddings.return_value = pd.DataFrame(
        {
            USER_ID_NAME: ["u1"],
            USER_EMBEDDING_NAME: list(user_embs),
        }
    )

    retriever = EmbeddingRetriever(top_k=3)
    retriever.build_index(estimator=fake_estimator)
    results = retriever.retrieve(["u1"], top_k=3)

    assert results["u1"] == ["i1", "i3", "i2"]


def test_cold_start_returns_most_popular_items(fitted_mf):
    """Popular fallback must return items ranked by interaction count, not an
    arbitrary ordering. Uses skewed interactions to make the ordering testable."""
    # i1: 3 interactions, i2: 2, i3: 1 — clear descending popularity.
    skewed = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u1", "u2", "u2", "u3"],
            ITEM_ID_NAME: ["i1", "i1", "i1", "i2", "i2", "i3"],
            LABEL_NAME: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    retriever = EmbeddingRetriever(top_k=2, cold_start_strategy="popular")
    retriever.build_index(estimator=fitted_mf, interactions=skewed)
    results = retriever.retrieve(["unknown_user"], top_k=2)
    # Must be the two most popular items in popularity order.
    assert results["unknown_user"] == ["i1", "i2"]


def test_cold_start_popular_filtered_to_embedding_catalog(fitted_mf):
    """Popular fallback must exclude items absent from the embedding catalog.
    If interactions contain ghost item IDs (e.g. deleted items), the retriever
    must not return them — the scorer has no embeddings for those items."""
    ghost_interactions = pd.DataFrame(
        {
            USER_ID_NAME: ["u_ghost"] * 5,
            ITEM_ID_NAME: ["ghost_item"] * 5,  # not in fitted_mf's item vocabulary
            LABEL_NAME: [1.0] * 5,
        }
    )
    retriever = EmbeddingRetriever(top_k=3, cold_start_strategy="popular")
    retriever.build_index(estimator=fitted_mf, interactions=ghost_interactions)
    results = retriever.retrieve(["unknown_user"], top_k=3)
    known_items = set(fitted_mf.item_id_index_)
    assert set(results["unknown_user"]).issubset(known_items)


def test_cold_start_strategy_zero_returns_candidates(fitted_mf):
    """cold_start_strategy='zero' must return the right number of candidates,
    all of which must belong to the indexed item catalog."""
    retriever = EmbeddingRetriever(top_k=2, cold_start_strategy="zero")
    retriever.build_index(estimator=fitted_mf)
    results = retriever.retrieve(["unknown_user"], top_k=2)
    assert "unknown_user" in results
    assert len(results["unknown_user"]) == 2
    known_items = set(fitted_mf.item_id_index_)
    assert set(results["unknown_user"]).issubset(known_items)


def test_cold_start_popular_no_interactions_falls_back_to_catalog_order(fitted_mf):
    """When cold_start_strategy='popular' and no interactions are passed to
    build_index(), cold-start candidates must still come from the embedding
    catalog (in catalog order). retrieve() must not raise."""
    retriever = EmbeddingRetriever(top_k=3, cold_start_strategy="popular")
    retriever.build_index(estimator=fitted_mf)  # no interactions

    # _popular_items must be set to the full catalog
    assert retriever._popular_items is not None
    known_items = set(fitted_mf.item_id_index_)
    assert set(retriever._popular_items).issubset(known_items)

    results = retriever.retrieve(["unknown_user"], top_k=3)
    assert len(results["unknown_user"]) == 3
    assert set(results["unknown_user"]).issubset(known_items)


def test_cold_start_strategy_invalid_raises():
    with pytest.raises(ValueError, match="cold_start_strategy"):
        EmbeddingRetriever(top_k=2, cold_start_strategy="random")


def test_top_k_larger_than_catalog_returns_all(fitted_mf):
    retriever = EmbeddingRetriever(top_k=100)
    retriever.build_index(estimator=fitted_mf)
    results = retriever.retrieve(["u1"], top_k=100)
    assert len(results["u1"]) == 3  # only 3 items in catalog


def test_non_embedding_estimator_raises_type_error():
    retriever = EmbeddingRetriever(top_k=2)
    fake_estimator = MagicMock()
    fake_estimator.__class__ = object  # not BaseEmbeddingEstimator
    with pytest.raises(TypeError, match="BaseEmbeddingEstimator"):
        retriever.build_index(estimator=fake_estimator)


def test_retrieve_before_build_raises_runtime_error():
    retriever = EmbeddingRetriever(top_k=2)
    with pytest.raises(RuntimeError, match="build_index"):
        retriever.retrieve(["u1"], top_k=2)


def test_zero_top_k_raises():
    with pytest.raises(ValueError, match="top_k"):
        EmbeddingRetriever(top_k=0)


def test_empty_item_embeddings_build_and_retrieve():
    """If the estimator returns no item embeddings, build_index must succeed
    and retrieve() must return an empty list for every user — not raise.

    Note: get_user_embeddings() is NOT called in this path because build_index()
    returns early when get_item_embeddings() is empty.
    """
    fake_estimator = MagicMock()
    fake_estimator.__class__ = BaseEmbeddingEstimator  # passes isinstance check
    fake_estimator.get_item_embeddings.return_value = pd.DataFrame()

    retriever = EmbeddingRetriever(top_k=2)
    retriever.build_index(estimator=fake_estimator)

    assert retriever._item_ids is not None
    assert len(retriever._item_ids) == 0

    results = retriever.retrieve(["u1", "u2"], top_k=2)
    assert results == {"u1": [], "u2": []}

    # Confirm get_user_embeddings was never reached (early-return path)
    fake_estimator.get_user_embeddings.assert_not_called()


def test_no_user_embeddings_uses_zero_vector_fallback():
    """When items are indexed but get_user_embeddings() returns empty and
    cold_start_strategy='zero', every unknown user falls back to the
    zero-vector path. retrieve() must not raise and must return candidates
    that are a subset of the indexed catalog."""
    fake_estimator = MagicMock()
    fake_estimator.__class__ = BaseEmbeddingEstimator
    item_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    fake_estimator.get_item_embeddings.return_value = pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2"],
            ITEM_EMBEDDING_NAME: list(item_embs),
        }
    )
    fake_estimator.get_user_embeddings.return_value = pd.DataFrame()  # no users cached

    retriever = EmbeddingRetriever(top_k=2, cold_start_strategy="zero")
    retriever.build_index(estimator=fake_estimator)

    assert retriever._item_matrix.shape == (2, 2)
    assert len(retriever._user_emb_by_id) == 0  # empty cache

    # Any user triggers zero-vector fallback — must not raise, candidates from catalog
    results = retriever.retrieve(["any_user"], top_k=2)
    assert "any_user" in results
    assert len(results["any_user"]) == 2
    assert set(results["any_user"]).issubset({"i1", "i2"})
