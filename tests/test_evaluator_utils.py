import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skrec.evaluator.utils import create_padded_matrix


def test_create_padded_matrix():
    """Tests the create_padded_matrix utility function."""

    # Case 1: Basic padding, infer max_len and dtype (int)
    list1 = [[1, 2], [3, 4, 5]]
    expected1 = np.array([[1, 2, 0], [3, 4, 5]])
    result1 = create_padded_matrix(list1)
    assert_array_equal(result1, expected1)
    assert result1.dtype == int

    # Case 2: Specify max_len
    list2 = [[1, 2], [3, 4, 5]]
    expected2 = np.array([[1, 2, 0, 0], [3, 4, 5, 0]])
    result2 = create_padded_matrix(list2, max_len=4)
    assert_array_equal(result2, expected2)

    # Case 3: Specify dtype (float)
    list3 = [[1, 2], [3, 4, 5]]
    expected3 = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]])
    result3 = create_padded_matrix(list3, dtype=float)
    assert_array_equal(result3, expected3)
    assert result3.dtype == float

    # Case 4: Specify pad_value
    list4 = [[1, 2], [3, 4, 5]]
    expected4 = np.array([[1, 2, -1], [3, 4, 5]])
    result4 = create_padded_matrix(list4, pad_value=-1)
    assert_array_equal(result4, expected4)

    # Case 5: Empty input list
    list5 = []
    expected5 = np.array([])
    result5 = create_padded_matrix(list5)
    assert_array_equal(result5, expected5)
    # Check default dtype for empty case (should be float based on implementation)
    assert result5.dtype == float

    # Case 6: List with empty sequences
    list6 = [[], [1, 2], []]
    expected6 = np.array([[0, 0], [1, 2], [0, 0]])
    result6 = create_padded_matrix(list6)
    assert_array_equal(result6, expected6)

    # Case 7: Mixed types (int and float), should infer float
    list7 = [[1, 2], [3.5, 4.5, 5.0]]
    expected7 = np.array([[1.0, 2.0, 0.0], [3.5, 4.5, 5.0]])
    result7 = create_padded_matrix(list7)
    assert_array_equal(result7, expected7)
    assert result7.dtype == float

    # Case 8: Input contains NumPy arrays
    list8 = [np.array([1, 2]), np.array([3, 4, 5])]
    expected8 = np.array([[1, 2, 0], [3, 4, 5]])
    result8 = create_padded_matrix(list8)
    assert_array_equal(result8, expected8)

    # Case 9: Sequence longer than max_len raises ValueError
    list9 = [[1, 2, 3], [4, 5]]
    with pytest.raises(ValueError, match="Sequence length 3 exceeds max_len 2"):
        create_padded_matrix(list9, max_len=2)

    # Case 10: Specify dtype when input is empty
    list10 = []
    expected10 = np.array([], dtype=np.int32)
    result10 = create_padded_matrix(list10, dtype=np.int32)
    assert_array_equal(result10, expected10)
    assert result10.dtype == np.int32

    # Case 11: Mixed list and numpy array
    list11 = [[1, 2], np.array([3.5, 4.5, 5.0])]
    expected11 = np.array([[1.0, 2.0, 0.0], [3.5, 4.5, 5.0]])
    result11 = create_padded_matrix(list11)
    assert_array_equal(result11, expected11)
    assert result11.dtype == float

    # Case 12: All empty sequences, infer max_len=0, default dtype=float
    list12 = [[], [], []]
    expected12 = np.empty((3, 0), dtype=float)  # Array with shape (3, 0)
    result12 = create_padded_matrix(list12)
    assert_array_equal(result12, expected12)
    assert result12.dtype == float
