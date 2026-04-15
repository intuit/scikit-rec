"""Unit tests for PopularityRetriever."""

import pandas as pd
import pytest

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME
from skrec.retriever.popularity_retriever import PopularityRetriever


@pytest.fixture
def interactions():
    return pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u2", "u2", "u3", "u3", "u1"],
            ITEM_ID_NAME: ["i1", "i2", "i1", "i3", "i2", "i1", "i1"],
            LABEL_NAME: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_build_index_ranks_by_interaction_count(interactions):
    retriever = PopularityRetriever(top_k=3)
    retriever.build_index(interactions=interactions)
    # i1 appears 4 times, i2 appears 2 times, i3 appears 1 time
    assert retriever._popular_items[0] == "i1"


def test_retrieve_returns_same_for_all_users(interactions):
    retriever = PopularityRetriever(top_k=3)
    retriever.build_index(interactions=interactions)
    results = retriever.retrieve(["u1", "u2", "u3"], top_k=3)
    assert results["u1"] == results["u2"] == results["u3"]


def test_top_k_respected(interactions):
    retriever = PopularityRetriever(top_k=2)
    retriever.build_index(interactions=interactions)
    results = retriever.retrieve(["u1"], top_k=2)
    assert len(results["u1"]) == 2


def test_top_k_larger_than_catalog_returns_all(interactions):
    retriever = PopularityRetriever(top_k=100)
    retriever.build_index(interactions=interactions)
    results = retriever.retrieve(["u1"], top_k=100)
    assert len(results["u1"]) == 3  # only 3 unique items


def test_missing_interactions_raises():
    retriever = PopularityRetriever(top_k=2)
    with pytest.raises(ValueError):
        retriever.build_index(interactions=None)


def test_retrieve_before_build_raises():
    retriever = PopularityRetriever(top_k=2)
    with pytest.raises(RuntimeError, match="build_index"):
        retriever.retrieve(["u1"], top_k=2)


def test_retrieve_returns_independent_lists(interactions):
    """Each user must receive an independent list copy — mutation by one caller
    must not corrupt another user's candidate list."""
    retriever = PopularityRetriever(top_k=3)
    retriever.build_index(interactions=interactions)
    results = retriever.retrieve(["u1", "u2"], top_k=3)
    results["u1"].clear()
    assert len(results["u2"]) == 3


def test_retrieve_ordering_matches_popularity(interactions):
    """Items should be returned in descending interaction-count order."""
    retriever = PopularityRetriever(top_k=3)
    retriever.build_index(interactions=interactions)
    results = retriever.retrieve(["u1"], top_k=3)
    # i1 (4 times) > i2 (2 times) > i3 (1 time)
    assert results["u1"] == ["i1", "i2", "i3"]


def test_retrieve_top_k_larger_than_instance_top_k(interactions):
    """retrieve(top_k=N) where N > self.top_k must still return up to N items —
    the full popularity list is stored at build time, not capped at self.top_k."""
    retriever = PopularityRetriever(top_k=1)
    retriever.build_index(interactions=interactions)
    results = retriever.retrieve(["u1"], top_k=3)
    assert len(results["u1"]) == 3  # 3 unique items available


def test_zero_top_k_raises():
    with pytest.raises(ValueError, match="top_k"):
        PopularityRetriever(top_k=0)


def test_popular_items_filtered_to_catalog_when_items_provided(interactions):
    """When an items DataFrame is supplied, the popular list must only contain
    item IDs present in that DataFrame — unknown items must be excluded."""
    items = pd.DataFrame({ITEM_ID_NAME: ["i1", "i2"]})  # i3 intentionally absent
    retriever = PopularityRetriever(top_k=3)
    retriever.build_index(interactions=interactions, items=items)
    known = set(items[ITEM_ID_NAME])
    assert set(retriever._popular_items).issubset(known)
    assert "i3" not in retriever._popular_items


def test_ghost_interactions_fall_back_to_catalog_order(interactions):
    """If every item in interactions is absent from the items catalog,
    the retriever must fall back to catalog order rather than returning
    ghost IDs the scorer doesn't know about."""
    ghost_only = pd.DataFrame(
        {
            USER_ID_NAME: ["u1"] * 3,
            ITEM_ID_NAME: ["ghost1", "ghost1", "ghost2"],
            LABEL_NAME: [1.0, 1.0, 1.0],
        }
    )
    catalog = pd.DataFrame({ITEM_ID_NAME: ["i1", "i2"]})
    retriever = PopularityRetriever(top_k=2)
    retriever.build_index(interactions=ghost_only, items=catalog)
    known = set(catalog[ITEM_ID_NAME])
    assert set(retriever._popular_items).issubset(known)
