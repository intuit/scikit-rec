"""Unit tests for BaseCandidateRetriever._topk_indices."""

import numpy as np

from skrec.retriever.base_retriever import BaseCandidateRetriever


# ---------------------------------------------------------------------------
# Minimal concrete subclass — lets us instantiate the abstract base class.
# ---------------------------------------------------------------------------
class _StubRetriever(BaseCandidateRetriever):
    def build_index(self, estimator=None, interactions=None, items=None):
        pass

    def retrieve(self, user_ids, top_k):
        return {}


topk = BaseCandidateRetriever._topk_indices  # shorthand for all tests below


# ---------------------------------------------------------------------------
# k = 0
# ---------------------------------------------------------------------------
def test_k_zero_returns_empty():
    scores = np.array([3.0, 1.0, 2.0])
    result = topk(scores, 0)
    assert len(result) == 0
    assert result.dtype == np.intp


# ---------------------------------------------------------------------------
# k = 1
# ---------------------------------------------------------------------------
def test_k_one_returns_highest_score():
    scores = np.array([1.0, 5.0, 3.0])
    result = topk(scores, 1)
    assert list(result) == [1]  # index 1 has score 5.0


# ---------------------------------------------------------------------------
# k = n - 1  (partial sort path)
# ---------------------------------------------------------------------------
def test_k_n_minus_one_returns_all_but_lowest():
    scores = np.array([3.0, 1.0, 4.0, 2.0])  # sorted descending: 4,3,2,1 → idx 2,0,3,1
    result = topk(scores, 3)
    assert len(result) == 3
    assert list(result) == [2, 0, 3]  # top-3 in descending score order


# ---------------------------------------------------------------------------
# k = n  (exact catalog size — full-sort path)
# ---------------------------------------------------------------------------
def test_k_equals_n_returns_all_in_descending_order():
    scores = np.array([1.0, 4.0, 2.0, 3.0])
    result = topk(scores, 4)
    assert list(result) == [1, 3, 2, 0]  # descending: 4,3,2,1


# ---------------------------------------------------------------------------
# k > n  (larger than catalog — must clamp, not error)
# ---------------------------------------------------------------------------
def test_k_greater_than_n_returns_all():
    scores = np.array([2.0, 1.0, 3.0])
    result = topk(scores, 100)
    assert len(result) == 3
    assert list(result) == [2, 0, 1]  # descending: 3,2,1


# ---------------------------------------------------------------------------
# All-equal scores
# ---------------------------------------------------------------------------
def test_all_equal_scores_returns_correct_count():
    scores = np.array([7.0, 7.0, 7.0, 7.0])
    result = topk(scores, 2)
    assert len(result) == 2
    # All scores equal — any 2 indices are valid; no NaN, no crash.
    assert set(result).issubset({0, 1, 2, 3})


# ---------------------------------------------------------------------------
# Return type is always np.intp (suitable for array indexing)
# ---------------------------------------------------------------------------
def test_return_dtype_is_intp():
    scores = np.array([1.0, 2.0, 3.0])
    assert topk(scores, 2).dtype == np.intp
    assert topk(scores, 0).dtype == np.intp
    assert topk(scores, 3).dtype == np.intp


# ---------------------------------------------------------------------------
# Single-element array
# ---------------------------------------------------------------------------
def test_single_element_array():
    scores = np.array([42.0])
    assert list(topk(scores, 1)) == [0]
    assert list(topk(scores, 0)) == []
    assert list(topk(scores, 5)) == [0]  # k > n, clamped


# ---------------------------------------------------------------------------
# Negative scores handled correctly
# ---------------------------------------------------------------------------
def test_negative_scores_ranked_correctly():
    scores = np.array([-1.0, -3.0, -2.0])
    result = topk(scores, 2)
    assert list(result) == [0, 2]  # -1 > -2 > -3
