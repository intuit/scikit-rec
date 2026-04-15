"""Unit tests for ContentBasedRetriever."""

import numpy as np
import pandas as pd
import pytest

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME
from skrec.retriever.content_based_retriever import ContentBasedRetriever


@pytest.fixture
def items():
    return pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2", "i3", "i4"],
            "price": [10.0, 20.0, 30.0, 40.0],
            "rating": [4.0, 3.0, 5.0, 2.0],
        }
    )


@pytest.fixture
def interactions():
    return pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u2", "u3"],
            ITEM_ID_NAME: ["i1", "i2", "i3", "i1"],
            LABEL_NAME: [1.0, 5.0, 1.0, 1.0],
        }
    )


def test_build_index_normalizes_item_features(items, interactions):
    retriever = ContentBasedRetriever(top_k=4)
    retriever.build_index(interactions=interactions, items=items)
    norms = np.linalg.norm(retriever._item_matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_retrieve_returns_correct_count(items, interactions):
    retriever = ContentBasedRetriever(top_k=2)
    retriever.build_index(interactions=interactions, items=items)
    results = retriever.retrieve(["u1"], top_k=2)
    assert len(results["u1"]) == 2


def test_retrieve_results_are_subset_of_items(items, interactions):
    retriever = ContentBasedRetriever(top_k=4)
    retriever.build_index(interactions=interactions, items=items)
    results = retriever.retrieve(["u1"], top_k=4)
    known = set(items[ITEM_ID_NAME].values)
    assert set(results["u1"]).issubset(known)


def test_cold_start_user_falls_back_to_popularity(items, interactions):
    retriever = ContentBasedRetriever(top_k=2)
    retriever.build_index(interactions=interactions, items=items)
    results = retriever.retrieve(["unknown_user"], top_k=2)
    assert "unknown_user" in results
    assert len(results["unknown_user"]) == 2
    # Items must come from the items DataFrame — not ghost IDs from interactions.
    known_items = set(items[ITEM_ID_NAME])
    assert set(results["unknown_user"]).issubset(known_items)


def test_popular_fallback_excludes_items_absent_from_catalog():
    """Interactions may reference item IDs not present in the items DataFrame
    (e.g. products deleted after training). The popular fallback must not return
    those ghost IDs — the scorer has no features for them."""
    ghost_items_only = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u1"],
            ITEM_ID_NAME: ["ghost1", "ghost1", "ghost2"],  # neither in catalog
            LABEL_NAME: [1.0, 1.0, 1.0],
        }
    )
    catalog = pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2"],
            "price": [10.0, 20.0],
        }
    )
    retriever = ContentBasedRetriever(top_k=2)
    retriever.build_index(interactions=ghost_items_only, items=catalog)
    results = retriever.retrieve(["unknown_user"], top_k=2)
    known_items = set(catalog[ITEM_ID_NAME])
    assert set(results["unknown_user"]).issubset(known_items)


def test_weight_by_outcome_differs_from_unweighted(items, interactions):
    r_unweighted = ContentBasedRetriever(top_k=4, weight_by_outcome=False)
    r_unweighted.build_index(interactions=interactions, items=items)

    r_weighted = ContentBasedRetriever(top_k=4, weight_by_outcome=True)
    r_weighted.build_index(interactions=interactions, items=items)

    results_uw = r_unweighted.retrieve(["u1"], top_k=4)
    results_w = r_weighted.retrieve(["u1"], top_k=4)
    # u1 interacted with i1 (outcome=1) and i2 (outcome=5) — weighted should differ
    assert results_uw["u1"] != results_w["u1"]


def test_feature_columns_subset(items, interactions):
    retriever = ContentBasedRetriever(top_k=4, feature_columns=["price"])
    retriever.build_index(interactions=interactions, items=items)
    assert retriever._item_matrix.shape == (4, 1)


def test_missing_feature_column_raises(items, interactions):
    retriever = ContentBasedRetriever(top_k=2, feature_columns=["nonexistent"])
    with pytest.raises(ValueError, match="nonexistent"):
        retriever.build_index(interactions=interactions, items=items)


def test_no_numeric_columns_raises():
    items_cat = pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2"],
            "category": ["a", "b"],
        }
    )
    retriever = ContentBasedRetriever(top_k=2)
    with pytest.raises(ValueError, match="numeric"):
        retriever.build_index(items=items_cat)


def test_top_k_larger_than_catalog(items, interactions):
    retriever = ContentBasedRetriever(top_k=100)
    retriever.build_index(interactions=interactions, items=items)
    results = retriever.retrieve(["u1"], top_k=100)
    assert len(results["u1"]) == 4  # only 4 items in catalog


def test_build_without_interactions_uses_all_items(items):
    retriever = ContentBasedRetriever(top_k=4)
    retriever.build_index(items=items)  # no interactions
    results = retriever.retrieve(["unknown_user"], top_k=4)
    assert len(results["unknown_user"]) == 4


def test_zero_outcomes_do_not_produce_nan_profile(items):
    """When all outcomes are 0.0 with weight_by_outcome=True, the profile must not
    contain NaN values — the retriever must fall back to uniform weighting."""
    interactions_zero = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1"],
            ITEM_ID_NAME: ["i1", "i2"],
            LABEL_NAME: [0.0, 0.0],
        }
    )
    retriever = ContentBasedRetriever(top_k=4, weight_by_outcome=True)
    retriever.build_index(interactions=interactions_zero, items=items)
    profile = retriever._build_user_profile("u1")
    assert profile is not None
    assert not np.any(np.isnan(profile))


def test_string_feature_column_raises(items):
    """Passing a non-numeric column via feature_columns must raise ValueError,
    not a cryptic numpy cast error."""
    items_with_cat = items.copy()
    items_with_cat["category"] = ["a", "b", "c", "d"]
    retriever = ContentBasedRetriever(top_k=4, feature_columns=["category"])
    with pytest.raises(ValueError, match="numeric"):
        retriever.build_index(items=items_with_cat)


def test_item_id_in_feature_columns_raises(items):
    """Explicitly passing ITEM_ID_NAME in feature_columns must raise a clear
    ValueError — item ID is an identifier, not a feature."""
    retriever = ContentBasedRetriever(top_k=4, feature_columns=[ITEM_ID_NAME, "price"])
    with pytest.raises(ValueError, match=ITEM_ID_NAME):
        retriever.build_index(items=items)


def test_missing_user_id_in_interactions_raises(items):
    """interactions missing USER_ID_NAME must raise a clear ValueError, not a pandas KeyError."""
    bad_interactions = pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2"],
            LABEL_NAME: [1.0, 1.0],
            # USER_ID_NAME intentionally absent
        }
    )
    retriever = ContentBasedRetriever(top_k=4)
    with pytest.raises(ValueError, match=USER_ID_NAME):
        retriever.build_index(interactions=bad_interactions, items=items)


def test_missing_item_id_in_interactions_raises(items):
    """interactions missing ITEM_ID_NAME must raise a clear ValueError, not a pandas KeyError."""
    bad_interactions = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1"],
            LABEL_NAME: [1.0, 1.0],
            # ITEM_ID_NAME intentionally absent
        }
    )
    retriever = ContentBasedRetriever(top_k=4)
    with pytest.raises(ValueError, match=ITEM_ID_NAME):
        retriever.build_index(interactions=bad_interactions, items=items)


def test_weight_by_outcome_without_label_column_logs_warning(items, caplog):
    """weight_by_outcome=True with no LABEL_NAME column must log a warning and
    fall back to uniform weighting — not silently produce wrong results."""
    import logging

    interactions_no_label = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1"],
            ITEM_ID_NAME: ["i1", "i2"],
            # LABEL_NAME intentionally absent
        }
    )
    retriever = ContentBasedRetriever(top_k=4, weight_by_outcome=True)
    with caplog.at_level(logging.WARNING):
        retriever.build_index(interactions=interactions_no_label, items=items)
    assert any("weight_by_outcome" in msg for msg in caplog.messages)

    # Result must still be valid — uniform weighting used as fallback.
    results = retriever.retrieve(["u1"], top_k=4)
    assert len(results["u1"]) == 4


def test_zero_top_k_raises():
    with pytest.raises(ValueError, match="top_k"):
        ContentBasedRetriever(top_k=0)


def test_nan_in_feature_columns_raises(interactions):
    items_with_nan = pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2", "i3", "i4"],
            "price": [10.0, float("nan"), 30.0, 40.0],
            "rating": [4.0, 3.0, 5.0, 2.0],
        }
    )
    retriever = ContentBasedRetriever(top_k=4)
    with pytest.raises(ValueError, match="NaN"):
        retriever.build_index(interactions=interactions, items=items_with_nan)


def test_all_zero_feature_item_does_not_crash(interactions):
    """An item whose every feature is 0.0 must not cause a division-by-zero
    or NaN in the item matrix — it should simply be retrievable."""
    items_with_zero = pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2", "i3", "i4"],
            "price": [10.0, 0.0, 30.0, 40.0],
            "rating": [4.0, 0.0, 5.0, 2.0],  # i2 is all-zero
        }
    )
    retriever = ContentBasedRetriever(top_k=4)
    retriever.build_index(interactions=interactions, items=items_with_zero)

    # No NaN in the item matrix after normalization.
    assert not np.any(np.isnan(retriever._item_matrix))

    # Retrieval must succeed and return the right number of candidates.
    results = retriever.retrieve(["u1"], top_k=4)
    assert len(results["u1"]) == 4
    assert set(results["u1"]).issubset(set(items_with_zero[ITEM_ID_NAME]))


def test_zero_profile_vector_falls_back_gracefully(interactions):
    """If a user's interaction history produces a zero profile vector
    (e.g. the weighted mean of item vectors cancels to zero), retrieve()
    must fall back to the popularity ranking, not return arbitrary candidates."""
    # Two items that are antipodal after normalization: [1,0] and [-1,0].
    # A user who interacted with both equally ends up with a zero profile.
    items_antipodal = pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2", "i3"],
            "x": [1.0, -1.0, 0.5],
            "y": [0.0, 0.0, 0.5],
        }
    )
    # u1 interacted with i1 twice and i2 once, so popularity order is i1 > i2.
    # i3 has no interactions so it comes last in the popular fallback.
    interactions_balanced = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u1"],
            ITEM_ID_NAME: ["i1", "i2", "i1"],
            LABEL_NAME: [1.0, 1.0, 1.0],
        }
    )
    retriever = ContentBasedRetriever(top_k=3)
    retriever.build_index(interactions=interactions_balanced, items=items_antipodal)

    results = retriever.retrieve(["u1"], top_k=3)
    assert len(results["u1"]) == 3
    # Zero-norm profile must fall back to popular ordering, not arbitrary array order.
    # i1 (2 interactions) must rank ahead of i2 (1 interaction).
    assert results["u1"].index("i1") < results["u1"].index("i2")
