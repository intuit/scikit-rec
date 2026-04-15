from typing import List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from skrec.util.logger import get_logger

logger = get_logger(__name__)


def create_padded_matrix(
    list_of_sequences: List[Sequence], max_len: Optional[int] = None, pad_value: Union[int, float] = 0, dtype=None
) -> NDArray:
    """
    Pads sequences (lists or arrays) in a list to max_len and stacks them.

    Args:
        list_of_sequences: A list containing sequences (lists or NumPy arrays).
        max_len: The target length for padding. If None, it's inferred from the
                 longest sequence in the list.
        pad_value: The value used for padding.
        dtype: The desired dtype for the output NumPy array. If None, NumPy's
               default inference mechanisms are used.

    Returns:
        A 2D NumPy array with padded sequences.
    """
    if not list_of_sequences:
        return np.array([], dtype=dtype if dtype is not None else float)

    # Convert inner sequences to numpy arrays.
    arrays = [np.asarray(seq, dtype=dtype) for seq in list_of_sequences]

    if max_len is None:
        max_len = max(len(arr_item) for arr_item in arrays) if arrays else 0

    padded_arrays = []
    for arr_item in arrays:
        current_len = len(arr_item)
        pad_width = max_len - current_len
        if pad_width < 0:
            raise ValueError(f"Sequence length {current_len} exceeds max_len {max_len}")

        # np.pad will use the dtype of arr_item. If arr_item was created with a specific
        # dtype via np.asarray(seq, dtype=dtype), that will be respected.
        # If dtype was None, np.asarray inferred it, and np.pad will use that.
        padded_arr = np.pad(arr_item, (0, pad_width), "constant", constant_values=pad_value)
        padded_arrays.append(padded_arr)

    result_matrix = np.stack(padded_arrays)

    if dtype is not None and result_matrix.dtype != dtype:
        result_matrix = result_matrix.astype(dtype, copy=False)

    return result_matrix


def calculate_propensity_ratio(target_propensities: NDArray, logging_propensities: NDArray) -> NDArray:
    """
    Calculates the propensity ratio (target / logging) with safe division.

    Handles division by zero: 0/0 -> NaN (excluded by downstream nanmean),
    x/0 -> inf (for x!=0).

    Args:
        target_propensities: Array of target policy probabilities.
        logging_propensities: Array of logging policy probabilities.

    Returns:
        An array containing the calculated propensity ratios.  Entries where
        both target and logging propensities are zero are NaN (unknown ratio),
        not 0, to avoid biasing IPS/SNIPS estimates downward.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(target_propensities, logging_propensities)
    return ratio
