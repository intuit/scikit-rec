import numpy as np
from numpy.testing import assert_array_almost_equal

from skrec.util.numpy_util import (
    index_rows,
    sample_with_replacement_2d,
    sample_without_replacement_2d,
    softmax_2d,
)


def test_index_rows():
    foo = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    indexes = np.array(
        [
            [2, 0, 1],
            [1, 2, 0],
        ]
    )

    expected = np.array(
        [
            [3, 1, 2],
            [5, 6, 4],
        ]
    )

    result = index_rows(foo, indexes)

    assert np.array_equal(expected, result)


def test_softmax_2d():
    scores = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [-1.0, -2.0, -3.0]])

    # Test with temperature = 1.0
    exp_scores = np.exp(scores)
    expected_temp1 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    result_temp1 = softmax_2d(scores, temperature=1.0)
    assert_array_almost_equal(result_temp1, expected_temp1)

    # Test with temperature = 0.0 (should be one-hot)
    result_temp0 = softmax_2d(scores, temperature=0.0)
    assert np.all(np.sum(result_temp0, axis=1) == 1.0)
    assert np.all((result_temp0 == 0) | (result_temp0 == 1))
    for i in range(scores.shape[0]):
        assert result_temp0[i, np.argmax(scores[i, :])] == 1.0, f"Mismatch in one-hot encoding for row {i}"


def test_sample_with_replacement_2d():
    probas = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
    k = 2

    sampled_indices = sample_with_replacement_2d(probas, k)
    assert sampled_indices.shape == (2, k)
    assert np.all(sampled_indices >= 0)
    assert np.all(sampled_indices < probas.shape[1])

    # Test with k=0
    sampled_indices_k0 = sample_with_replacement_2d(probas, k=0)
    assert sampled_indices_k0.shape == (2, 0)


def test_sample_without_replacement_2d():
    probas = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2], [0.0, 0.0, 1.0]])
    k = 2

    sampled_indices = sample_without_replacement_2d(probas, k)
    assert sampled_indices.shape == (3, k)
    assert np.all(sampled_indices >= 0)
    assert np.all(sampled_indices < probas.shape[1])
    for row in sampled_indices:
        assert len(set(row)) == k

    # Test with k=0
    sampled_indices_k0 = sample_without_replacement_2d(probas, k=0)
    assert sampled_indices_k0.shape == (3, 0)

    # Test with k = n_items
    k_eq_n_items = probas.shape[1]
    sampled_indices_k_eq_n = sample_without_replacement_2d(probas, k_eq_n_items)
    assert sampled_indices_k_eq_n.shape == (3, k_eq_n_items)
    for i in range(probas.shape[0]):
        assert set(sampled_indices_k_eq_n[i]) == set(np.arange(probas.shape[1]))
