"""Train/validation/test splitting utilities for recommendation systems.

The functions here follow scikit-learn's ``model_selection`` philosophy
(functional, DataFrame-in/DataFrame-out, no implicit state) but with splitter
strategies appropriate for recsys: random row splits leak the future into the
past and trivially memorize users, so this module provides splitters that
respect time, user boundaries, or both.

Users typically do::

    from skrec.split import temporal_split

    result = temporal_split(interactions_df, valid_fraction=0.1, test_fraction=0.1)
    train, valid, test = result.train, result.valid, result.test
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from skrec.constants import TIMESTAMP_COL, USER_ID_NAME


@dataclass
class SplitResult:
    """Container for the outputs of a splitter function.

    Attributes:
        train: Training rows.
        valid: Validation rows.
        test: Test rows, or ``None`` when the splitter was asked to produce no
            test split (e.g. ``test_fraction == 0`` or ``n_test == 0``).
        info: Splitter-specific diagnostic fields. See the docstring of each
            splitter for the keys it populates.
    """

    train: pd.DataFrame
    valid: pd.DataFrame
    test: Optional[pd.DataFrame] = None
    info: dict[str, Any] = field(default_factory=dict)


def _validate_non_empty(interactions: pd.DataFrame) -> None:
    if len(interactions) == 0:
        raise ValueError("interactions is empty; cannot produce a split from zero rows.")


def _validate_columns(interactions: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in interactions.columns]
    if missing:
        raise ValueError(f"interactions is missing required column(s): {missing}")


def _validate_no_nan_users(interactions: pd.DataFrame, user_col: str) -> None:
    """Reject NaN values in ``user_col``.

    NaN user IDs are a silent data-loss hazard in this module: ``groupby``
    drops NaN groups by default, and ``.isin([..., NaN, ...])`` never matches
    NaN (since ``NaN != NaN``). Either behavior would cause rows to disappear
    from all three splits without warning, violating the row-preservation
    contract. We fail at the boundary instead.
    """
    n_nan = int(interactions[user_col].isna().sum())
    if n_nan > 0:
        raise ValueError(
            f"interactions has {n_nan} row(s) with NaN in user_col='{user_col}'. "
            "NaN user IDs are not supported because groupby and .isin silently drop "
            "them, which would cause these rows to vanish from every split. "
            "Clean or filter the input before splitting."
        )


def temporal_split(
    interactions: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float = 0.0,
    timestamp_col: str = TIMESTAMP_COL,
) -> SplitResult:
    """Chronological split: oldest rows go to train, newest to test.

    Rows are sorted by ``timestamp_col`` ascending, then sliced into
    train / valid / test by row count. Ties on timestamp are broken by
    input row order (stable sort).

    Args:
        interactions: The input DataFrame. Must contain ``timestamp_col``.
        valid_fraction: Fraction of rows to place in the validation split.
        test_fraction: Fraction of rows to place in the test split. If 0,
            ``SplitResult.test`` is ``None``.
        timestamp_col: Name of the timestamp column.

    Returns:
        A ``SplitResult`` whose ``info`` dict contains ``train_date_range``,
        ``valid_date_range``, ``test_date_range`` (or ``None``), and
        ``n_train``, ``n_valid``, ``n_test``.

    Raises:
        ValueError: If ``interactions`` is empty, if either fraction is
            negative, if ``valid_fraction + test_fraction >= 1.0``, or if
            ``timestamp_col`` is not a column of ``interactions``.
    """
    _validate_non_empty(interactions)
    _validate_columns(interactions, [timestamp_col])
    if valid_fraction < 0 or test_fraction < 0:
        raise ValueError(f"fractions must be non-negative; got valid={valid_fraction}, test={test_fraction}")
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError(f"valid_fraction + test_fraction must be < 1.0; got {valid_fraction} + {test_fraction}")

    sorted_df = interactions.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    n = len(sorted_df)
    n_test = int(n * test_fraction)
    n_valid = int(n * valid_fraction)
    n_train = n - n_valid - n_test

    train = sorted_df.iloc[:n_train].reset_index(drop=True)
    valid = sorted_df.iloc[n_train : n_train + n_valid].reset_index(drop=True)
    test: Optional[pd.DataFrame]
    if test_fraction == 0:
        test = None
    else:
        test = sorted_df.iloc[n_train + n_valid :].reset_index(drop=True)

    def _range(df: pd.DataFrame):
        if len(df) == 0:
            return None
        return (df[timestamp_col].min(), df[timestamp_col].max())

    info = {
        "train_date_range": _range(train),
        "valid_date_range": _range(valid),
        "test_date_range": _range(test) if test is not None else None,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": 0 if test is None else len(test),
    }
    return SplitResult(train=train, valid=valid, test=test, info=info)


def leave_last_n_per_user(
    interactions: pd.DataFrame,
    n_valid: int,
    n_test: int = 0,
    user_col: str = USER_ID_NAME,
    timestamp_col: str = TIMESTAMP_COL,
) -> SplitResult:
    """Hold out each user's last ``n_valid`` (and optionally ``n_test``) rows by time.

    This is the standard evaluation protocol for sequential recommenders: for
    each user, the most recent interactions become validation / test and the
    rest become training. Users with fewer than ``n_valid + n_test`` rows are
    dropped entirely (recorded in ``info["users_dropped"]`` in first-occurrence
    order) because there is no meaningful way to split them.

    Args:
        interactions: The input DataFrame.
        n_valid: Number of most-recent rows per user to place in validation.
            Must be ``>= 1``.
        n_test: Number of most-recent rows per user to place in test (after
            the validation rows). If 0, ``SplitResult.test`` is ``None``.
        user_col: Name of the user ID column.
        timestamp_col: Name of the timestamp column.

    Returns:
        A ``SplitResult`` whose ``info`` dict contains ``n_users_total``,
        ``n_users_kept``, ``users_dropped``, ``n_train``, ``n_valid``,
        ``n_test``.

    Raises:
        ValueError: If ``interactions`` is empty; if ``n_valid < 1`` or
            ``n_test < 0``; if ``user_col`` or ``timestamp_col`` is missing;
            if ``user_col`` contains NaN values; or if no user has enough
            rows to survive the filter.
    """
    _validate_non_empty(interactions)
    _validate_columns(interactions, [user_col, timestamp_col])
    _validate_no_nan_users(interactions, user_col)
    if n_valid < 1:
        raise ValueError(f"n_valid must be >= 1; got {n_valid}")
    if n_test < 0:
        raise ValueError(f"n_test must be >= 0; got {n_test}")

    unique_users = interactions[user_col].unique().tolist()
    hold_per_user = n_valid + n_test

    # Sort by (user, timestamp) with a stable sort so rows for each user are
    # contiguous and ordered oldest→newest. groupby(sort=False) then preserves
    # this ordering for cumcount/transform.
    sorted_df = interactions.sort_values([user_col, timestamp_col], kind="mergesort").reset_index(drop=True)
    counts = sorted_df.groupby(user_col, sort=False)[timestamp_col].transform("size")
    # Rank within user, 0-indexed: 0 = oldest, count-1 = newest.
    within_user_rank = sorted_df.groupby(user_col, sort=False).cumcount()

    # sorted_df, counts, and within_user_rank share sorted_df's RangeIndex;
    # boolean-masking all three by keep_mask preserves that alignment, so
    # per-row comparisons like (kept_ranks < train_cutoff) stay valid.
    keep_mask = counts > hold_per_user
    kept = sorted_df[keep_mask]
    kept_counts = counts[keep_mask]
    kept_ranks = within_user_rank[keep_mask]

    # Train = first (count - hold_per_user) rows; then n_valid; then n_test.
    train_cutoff = kept_counts - hold_per_user
    valid_cutoff = kept_counts - n_test

    train = kept[kept_ranks < train_cutoff].reset_index(drop=True)
    valid = kept[(kept_ranks >= train_cutoff) & (kept_ranks < valid_cutoff)].reset_index(drop=True)
    test: Optional[pd.DataFrame]
    if n_test == 0:
        test = None
    else:
        test = kept[kept_ranks >= valid_cutoff].reset_index(drop=True)

    users_kept_set = set(kept[user_col].unique().tolist())
    users_dropped = [u for u in unique_users if u not in users_kept_set]

    # Deliberate deviation from split_plan.md (which only asks for users_dropped
    # to be populated): if *every* user is dropped we raise instead of returning
    # three empty frames. A silent empty result fails far downstream in HPO with
    # a cryptic "no training data" error; raising here points at the actual cause.
    if len(users_kept_set) == 0:
        raise ValueError(
            f"all {len(unique_users)} users had fewer than n_valid + n_test = {hold_per_user} rows; "
            "no users retained. Lower n_valid/n_test or supply users with more interactions."
        )

    info = {
        "n_users_total": len(unique_users),
        "n_users_kept": len(users_kept_set),
        "users_dropped": users_dropped,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": 0 if test is None else len(test),
    }
    return SplitResult(train=train, valid=valid, test=test, info=info)


def random_split_per_user(
    interactions: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float = 0.0,
    user_col: str = USER_ID_NAME,
    random_state: Optional[int] = None,
) -> SplitResult:
    """Within each user, randomly hold out a fraction of rows.

    Every user appears in at least one split. Because per-user holdout sizes
    are computed with ``int(m * fraction)``, a user with few rows may end up
    exclusively in the train split (when ``int(m * valid_fraction)`` and
    ``int(m * test_fraction)`` both truncate to zero). Separately, any user
    whose holdout sizes consume all their rows — leaving zero training rows —
    is recorded in ``info["users_with_no_train_rows"]``.

    Args:
        interactions: The input DataFrame.
        valid_fraction: Fraction of each user's rows to hold out for validation.
        test_fraction: Fraction of each user's rows to hold out for test.
            If 0, ``SplitResult.test`` is ``None``.
        user_col: Name of the user ID column.
        random_state: Seed for the NumPy RNG. ``None`` is allowed and yields
            a non-deterministic split.

    Returns:
        A ``SplitResult`` whose ``info`` dict contains ``n_users``,
        ``n_train``, ``n_valid``, ``n_test``, and ``users_with_no_train_rows``.

    Raises:
        ValueError: If ``interactions`` is empty; if either fraction is
            negative; if ``valid_fraction + test_fraction >= 1.0``; if
            ``user_col`` is missing; or if ``user_col`` contains NaN values.
    """
    _validate_non_empty(interactions)
    _validate_columns(interactions, [user_col])
    _validate_no_nan_users(interactions, user_col)
    if valid_fraction < 0 or test_fraction < 0:
        raise ValueError(f"fractions must be non-negative; got valid={valid_fraction}, test={test_fraction}")
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError(f"valid_fraction + test_fraction must be < 1.0; got {valid_fraction} + {test_fraction}")

    rng = np.random.default_rng(random_state)

    train_frames: list = []
    valid_frames: list = []
    test_frames: list = []
    users_with_no_train_rows: list = []

    # Preserve the order of first occurrence for determinism across pandas versions.
    n_users = 0
    for user_id, group in interactions.groupby(user_col, sort=False):
        n_users += 1
        m = len(group)
        n_test_user = int(m * test_fraction)
        n_valid_user = int(m * valid_fraction)

        shuffled_positions = rng.permutation(m)
        test_positions = shuffled_positions[:n_test_user]
        valid_positions = shuffled_positions[n_test_user : n_test_user + n_valid_user]
        train_positions = shuffled_positions[n_test_user + n_valid_user :]

        if len(train_positions) == 0:
            users_with_no_train_rows.append(user_id)

        train_frames.append(group.iloc[train_positions])
        valid_frames.append(group.iloc[valid_positions])
        if test_fraction > 0:
            test_frames.append(group.iloc[test_positions])

    train = _concat_preserving_schema(train_frames, interactions).reset_index(drop=True)
    valid = _concat_preserving_schema(valid_frames, interactions).reset_index(drop=True)
    test: Optional[pd.DataFrame]
    if test_fraction == 0:
        test = None
    else:
        test = _concat_preserving_schema(test_frames, interactions).reset_index(drop=True)

    info = {
        "n_users": n_users,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": 0 if test is None else len(test),
        "users_with_no_train_rows": users_with_no_train_rows,
    }
    return SplitResult(train=train, valid=valid, test=test, info=info)


def leave_n_users_out(
    interactions: pd.DataFrame,
    n_valid_users: int,
    n_test_users: int = 0,
    user_col: str = USER_ID_NAME,
    random_state: Optional[int] = None,
) -> SplitResult:
    """Hold out entire users, so train sees zero rows for held-out users.

    This is the only honest way to evaluate cold-start performance: a user in
    the validation or test split never appears in training.

    Args:
        interactions: The input DataFrame.
        n_valid_users: Number of distinct users to route to validation.
        n_test_users: Number of distinct users to route to test. If 0,
            ``SplitResult.test`` is ``None``.
        user_col: Name of the user ID column.
        random_state: Seed for the NumPy RNG.

    Returns:
        A ``SplitResult`` whose ``info`` dict contains ``train_user_ids``,
        ``valid_user_ids``, ``test_user_ids`` (or ``None``), and ``n_train``,
        ``n_valid``, ``n_test``.

    Raises:
        ValueError: If ``interactions`` is empty; if either count is negative;
            if ``n_valid_users + n_test_users`` does not leave at least one
            training user; if ``user_col`` is missing; or if ``user_col``
            contains NaN values.
    """
    _validate_non_empty(interactions)
    _validate_columns(interactions, [user_col])
    _validate_no_nan_users(interactions, user_col)
    if n_valid_users < 0 or n_test_users < 0:
        raise ValueError(f"user counts must be non-negative; got valid={n_valid_users}, test={n_test_users}")

    unique_users = interactions[user_col].unique()
    n_unique = len(unique_users)
    if n_valid_users + n_test_users >= n_unique:
        raise ValueError(
            f"n_valid_users + n_test_users must leave at least one training user; "
            f"got {n_valid_users} + {n_test_users} out of {n_unique} unique users."
        )

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_users)

    # .tolist() is load-bearing: these lists flow into info["train_user_ids"]
    # etc., which the spec types as list (not ndarray) and which downstream
    # consumers may JSON-serialize. Do not "simplify" by passing the numpy
    # slices directly — .isin() accepts both, but info would leak numpy dtypes.
    test_users = shuffled[:n_test_users].tolist()
    valid_users = shuffled[n_test_users : n_test_users + n_valid_users].tolist()
    train_users = shuffled[n_test_users + n_valid_users :].tolist()

    train = interactions[interactions[user_col].isin(train_users)].reset_index(drop=True)
    valid = interactions[interactions[user_col].isin(valid_users)].reset_index(drop=True)
    test: Optional[pd.DataFrame]
    test_user_ids: Optional[list]
    if n_test_users == 0:
        test = None
        test_user_ids = None
    else:
        test = interactions[interactions[user_col].isin(test_users)].reset_index(drop=True)
        test_user_ids = test_users

    info = {
        "train_user_ids": train_users,
        "valid_user_ids": valid_users,
        "test_user_ids": test_user_ids,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": 0 if test is None else len(test),
    }
    return SplitResult(train=train, valid=valid, test=test, info=info)


def random_split(
    interactions: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float = 0.0,
    random_state: Optional[int] = None,
) -> SplitResult:
    """Pure random row split, sklearn-style.

    **Warning:** Random row splits are rarely appropriate for recommendation
    systems. They cause temporal leakage (future events end up in training)
    and user overlap (the same user appears in both train and test), which
    inflates evaluation metrics. Prefer ``temporal_split``,
    ``leave_last_n_per_user``, ``random_split_per_user``, or
    ``leave_n_users_out`` for honest evaluation. Use this function only for
    sanity checks or when you have an explicit reason.

    Args:
        interactions: The input DataFrame.
        valid_fraction: Fraction of rows to place in the validation split.
        test_fraction: Fraction of rows to place in the test split. If 0,
            ``SplitResult.test`` is ``None``.
        random_state: Seed for the NumPy RNG.

    Returns:
        A ``SplitResult`` whose ``info`` dict contains ``n_train``, ``n_valid``,
        ``n_test``.

    Raises:
        ValueError: If ``interactions`` is empty, if either fraction is
            negative, or if ``valid_fraction + test_fraction >= 1.0``.
    """
    _validate_non_empty(interactions)
    if valid_fraction < 0 or test_fraction < 0:
        raise ValueError(f"fractions must be non-negative; got valid={valid_fraction}, test={test_fraction}")
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError(f"valid_fraction + test_fraction must be < 1.0; got {valid_fraction} + {test_fraction}")

    rng = np.random.default_rng(random_state)
    n = len(interactions)
    n_test = int(n * test_fraction)
    n_valid = int(n * valid_fraction)

    shuffled_positions = rng.permutation(n)
    test_positions = shuffled_positions[:n_test]
    valid_positions = shuffled_positions[n_test : n_test + n_valid]
    train_positions = shuffled_positions[n_test + n_valid :]

    train = interactions.iloc[train_positions].reset_index(drop=True)
    valid = interactions.iloc[valid_positions].reset_index(drop=True)
    test: Optional[pd.DataFrame]
    if test_fraction == 0:
        test = None
    else:
        test = interactions.iloc[test_positions].reset_index(drop=True)

    info = {
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": 0 if test is None else len(test),
    }
    return SplitResult(train=train, valid=valid, test=test, info=info)


def _concat_preserving_schema(frames: list, template: pd.DataFrame) -> pd.DataFrame:
    """Concatenate ``frames``; empty input returns an empty frame with the template's schema."""
    non_empty = [f for f in frames if len(f) > 0]
    if not non_empty:
        return template.iloc[0:0].copy()
    return pd.concat(non_empty, ignore_index=False)
