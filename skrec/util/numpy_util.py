from typing import Optional

import numpy as np
from numpy.typing import NDArray


def index_rows(array: NDArray, indexes: NDArray) -> NDArray:
    # Indexes from array row-by-row, using an array of indexes
    template = np.tile(np.arange(array.shape[0]).reshape(-1, 1), [1, array.shape[1]])
    return array[template, indexes]


def sample_with_replacement_2d(
    probas: NDArray[np.float64], k: int, rng: Optional[np.random.Generator] = None
) -> NDArray[np.int_]:
    """
    Samples k items for each user with replacement, based on probabilities.

    Args:
        probas (np.ndarray): Matrix of shape (n_users, n_items) with probabilities.
                             Each row must sum to 1.
        k (int): Number of items to sample for each user.
        rng (np.random.Generator, optional): NumPy random number generator.
                                              Defaults to np.random.default_rng().

    Returns:
        np.ndarray: Matrix of shape (n_users, k) with sampled item indices.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_users, n_items = probas.shape
    if k == 0:
        return np.empty((n_users, 0), dtype=int)

    if n_items == 0:  # Cannot sample if there are no items
        return np.empty((n_users, 0), dtype=int)

    cumulative_probas = np.cumsum(probas, axis=1)  # Shape (n_users, n_items)

    # Each random number will select one item for one of the k draws
    random_values = rng.random((n_users, k, 1))  # Shape (n_users, k, 1) for broadcasting

    # Find the first item index where random_value <= cumulative_proba
    sampled_indices = np.argmax(random_values <= cumulative_probas[:, np.newaxis, :], axis=2)

    return sampled_indices


def sample_without_replacement_2d(
    probas: NDArray[np.float64], k: int, rng: Optional[np.random.Generator] = None
) -> NDArray[np.int_]:
    """
    Samples k items for each user without replacement.
    Vectorized over users and iterative k times.

    Args:
        probas (np.ndarray): Matrix of shape (n_users, n_items) with probabilities.
                             Each row must initially sum to 1.
        k (int): Number of items to sample for each user. Must be <= n_items.
        rng (np.random.Generator, optional): NumPy random number generator.
                                              Defaults to np.random.default_rng().

    Returns:
        np.ndarray: Matrix of shape (n_users, k) with sampled item indices.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_users, n_items = probas.shape

    if k < 0:
        raise ValueError("Number of samples k cannot be negative.")
    if k == 0:
        return np.empty((n_users, 0), dtype=int)
    if k > n_items:
        raise ValueError(f"Cannot sample k={k} items without replacement from n_items={n_items} items.")

    # Create a mutable copy of probabilities.
    current_probas = np.array(probas, dtype=np.float64, copy=True)
    result_samples = np.zeros((n_users, k), dtype=int)
    row_indices = np.arange(n_users)

    # Small epsilon to handle no replacement with 0 probas
    epsilon = np.finfo(current_probas.dtype).eps
    current_probas += epsilon

    for i in range(k):
        cumulative_probas = np.cumsum(current_probas, axis=1)

        # sum of remaining positive probabilities (weights)
        current_row_sums = cumulative_probas[:, -1:]

        # Scale random draw by the sum of current positive weights.
        random_draws = rng.random((n_users, 1), dtype=current_probas.dtype) * current_row_sums

        sampled_this_step = np.argmax(random_draws <= cumulative_probas, axis=1)
        result_samples[:, i] = sampled_this_step

        current_probas[row_indices, sampled_this_step] = 0.0

    return result_samples


def softmax_2d(scores: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    """
    Applies softmax to raw scores to produce probability distributions,
    scaled by a temperature parameter.

    Args:
        scores: A 2D array of raw scores (e.g., logits). Shape (N_users, N_items).
        temperature: The temperature for scaling. Must be non-negative.
                        A temperature of 0 leads to one-hot probabilities (max score gets 1, others 0).
                        Higher temperatures lead to softer probability distributions.

    Returns:
        A 2D array of probabilities. Shape (N_users, N_items).

    Raises:
        ValueError: If temperature is negative.
    """
    if temperature < 0:
        raise ValueError("Temperature cannot be negative.")
    if scores.size == 0:
        return np.empty_like(scores)

    if temperature == 0:
        # One-hot encoding of max scores
        probabilities = np.zeros_like(scores)
        if scores.shape[1] > 0:  # Ensure not empty along axis 1 before argmax
            max_indices = np.argmax(scores, axis=1)
            probabilities[np.arange(scores.shape[0]), max_indices] = 1.0
        return probabilities
    else:
        # Temperature-scaled softmax
        # Subtract max for numerical stability (handles potential overflow with exp)
        scores_stable = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_stable / temperature)
        sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
        # Handle cases where sum_exp_scores is zero to avoid division by zero
        # This can happen if all exp_scores are zero (e.g., very negative scores_stable / large temperature)
        # Or if a row in scores was all -inf.
        probabilities = np.divide(exp_scores, sum_exp_scores, out=np.zeros_like(exp_scores), where=sum_exp_scores != 0)
        return probabilities
