"""Tests for ``skrec.split``."""

import pandas as pd
import pytest

from skrec.split import (
    SplitResult,
    leave_last_n_per_user,
    leave_n_users_out,
    random_split,
    random_split_per_user,
    temporal_split,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_interactions() -> pd.DataFrame:
    """100 interactions, 10 users, 20 items, monotonic timestamps."""
    return pd.DataFrame(
        {
            "USER_ID": [f"u{i % 10}" for i in range(100)],
            "ITEM_ID": [f"i{i % 20}" for i in range(100)],
            "OUTCOME": [1] * 100,
            "TIMESTAMP": pd.to_datetime([f"2026-01-{1 + i // 4}" for i in range(100)]),
        }
    )


@pytest.fixture
def sparse_user_interactions() -> pd.DataFrame:
    """Some users with only 1 row, some with many (10 rows)."""
    rows = [
        {
            "USER_ID": "u_loner",
            "ITEM_ID": "i0",
            "OUTCOME": 1,
            "TIMESTAMP": pd.Timestamp("2026-01-01"),
        }
    ]
    for u in range(5):
        for j in range(10):
            rows.append(
                {
                    "USER_ID": f"u{u}",
                    "ITEM_ID": f"i{j}",
                    "OUTCOME": 1,
                    "TIMESTAMP": pd.Timestamp("2026-01-01") + pd.Timedelta(days=j),
                }
            )
    return pd.DataFrame(rows)


def _assert_schema_preserved(input_df: pd.DataFrame, *outputs: pd.DataFrame) -> None:
    for out in outputs:
        assert list(out.columns) == list(input_df.columns)


def _assert_index_reset(*outputs: pd.DataFrame) -> None:
    for out in outputs:
        if len(out) > 0:
            assert out.index.is_monotonic_increasing
            assert out.index[0] == 0


def _assert_rows_preserved(input_df: pd.DataFrame, *outputs: pd.DataFrame) -> None:
    """The union of splits must equal the input row-wise (up to ordering)."""
    pieces = [out for out in outputs if out is not None and len(out) > 0]
    if not pieces:
        assert len(input_df) == 0
        return
    combined = pd.concat(pieces, ignore_index=True)
    sort_cols = list(input_df.columns)
    left = input_df.sort_values(sort_cols).reset_index(drop=True)
    right = combined.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


# ---------------------------------------------------------------------------
# temporal_split
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    def test_happy_path(self, basic_interactions: pd.DataFrame) -> None:
        result = temporal_split(basic_interactions, valid_fraction=0.1, test_fraction=0.1)
        assert isinstance(result, SplitResult)
        assert len(result.train) == 80
        assert len(result.valid) == 10
        assert len(result.test) == 10
        _assert_rows_preserved(basic_interactions, result.train, result.valid, result.test)

    def test_no_test_split_returns_none(self, basic_interactions: pd.DataFrame) -> None:
        result = temporal_split(basic_interactions, valid_fraction=0.2)
        assert result.test is None
        assert result.info["n_test"] == 0
        assert result.info["test_date_range"] is None

    def test_index_reset(self, basic_interactions: pd.DataFrame) -> None:
        result = temporal_split(basic_interactions, valid_fraction=0.2, test_fraction=0.1)
        _assert_index_reset(result.train, result.valid, result.test)

    def test_schema_preserved(self, basic_interactions: pd.DataFrame) -> None:
        result = temporal_split(basic_interactions, valid_fraction=0.2, test_fraction=0.1)
        _assert_schema_preserved(basic_interactions, result.train, result.valid, result.test)

    def test_info_fields(self, basic_interactions: pd.DataFrame) -> None:
        result = temporal_split(basic_interactions, valid_fraction=0.2, test_fraction=0.1)
        assert result.info["n_train"] == len(result.train)
        assert result.info["n_valid"] == len(result.valid)
        assert result.info["n_test"] == len(result.test)
        train_lo, train_hi = result.info["train_date_range"]
        assert train_lo == result.train["TIMESTAMP"].min()
        assert train_hi == result.train["TIMESTAMP"].max()
        valid_lo, valid_hi = result.info["valid_date_range"]
        assert valid_lo == result.valid["TIMESTAMP"].min()
        assert valid_hi == result.valid["TIMESTAMP"].max()
        test_lo, test_hi = result.info["test_date_range"]
        assert test_lo == result.test["TIMESTAMP"].min()
        assert test_hi == result.test["TIMESTAMP"].max()

    def test_chronological_ordering(self, basic_interactions: pd.DataFrame) -> None:
        result = temporal_split(basic_interactions, valid_fraction=0.2, test_fraction=0.1)
        assert result.train["TIMESTAMP"].max() <= result.valid["TIMESTAMP"].min()
        assert result.valid["TIMESTAMP"].max() <= result.test["TIMESTAMP"].min()

    def test_shuffled_input_still_chronological(self, basic_interactions: pd.DataFrame) -> None:
        shuffled = basic_interactions.sample(frac=1.0, random_state=0).reset_index(drop=True)
        result = temporal_split(shuffled, valid_fraction=0.2, test_fraction=0.1)
        assert result.train["TIMESTAMP"].max() <= result.valid["TIMESTAMP"].min()
        assert result.valid["TIMESTAMP"].max() <= result.test["TIMESTAMP"].min()

    def test_missing_timestamp_col_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="missing required column"):
            temporal_split(basic_interactions.drop(columns=["TIMESTAMP"]), valid_fraction=0.1)

    def test_fractions_sum_to_one_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="< 1.0"):
            temporal_split(basic_interactions, valid_fraction=0.5, test_fraction=0.5)

    def test_negative_fraction_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            temporal_split(basic_interactions, valid_fraction=-0.1)

    def test_empty_raises(self) -> None:
        empty = pd.DataFrame({"TIMESTAMP": pd.to_datetime([])})
        with pytest.raises(ValueError, match="empty"):
            temporal_split(empty, valid_fraction=0.1)

    def test_single_row_empty_valid(self) -> None:
        df = pd.DataFrame({"TIMESTAMP": pd.to_datetime(["2026-01-01"]), "X": [1]})
        result = temporal_split(df, valid_fraction=0.2)
        assert len(result.train) == 1
        assert len(result.valid) == 0
        assert result.test is None

    def test_all_same_timestamp(self) -> None:
        df = pd.DataFrame(
            {
                "TIMESTAMP": [pd.Timestamp("2026-01-01")] * 10,
                "ROW_ORDER": list(range(10)),
            }
        )
        result = temporal_split(df, valid_fraction=0.2, test_fraction=0.2)
        # Load-bearing: the specific ordering [0..5]/[6,7]/[8,9] requires the
        # stable sort kind="mergesort" in temporal_split. If that is changed
        # (e.g. to quicksort), ties will reorder and this test will fail.
        assert list(result.train["ROW_ORDER"]) == [0, 1, 2, 3, 4, 5]
        assert list(result.valid["ROW_ORDER"]) == [6, 7]
        assert list(result.test["ROW_ORDER"]) == [8, 9]


# ---------------------------------------------------------------------------
# leave_last_n_per_user
# ---------------------------------------------------------------------------


class TestLeaveLastNPerUser:
    def test_happy_path(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_last_n_per_user(basic_interactions, n_valid=1, n_test=1)
        # 10 users * 10 rows each, hold 2 per user → 80 train, 10 valid, 10 test.
        assert len(result.train) == 80
        assert len(result.valid) == 10
        assert len(result.test) == 10
        assert result.info["users_dropped"] == []
        _assert_rows_preserved(basic_interactions, result.train, result.valid, result.test)

    def test_no_test_split_returns_none(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_last_n_per_user(basic_interactions, n_valid=2)
        assert result.test is None
        assert result.info["n_test"] == 0

    def test_index_reset(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_last_n_per_user(basic_interactions, n_valid=1, n_test=1)
        _assert_index_reset(result.train, result.valid, result.test)

    def test_schema_preserved(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_last_n_per_user(basic_interactions, n_valid=1, n_test=1)
        _assert_schema_preserved(basic_interactions, result.train, result.valid, result.test)

    def test_info_fields(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_last_n_per_user(basic_interactions, n_valid=1, n_test=1)
        assert result.info["n_users_total"] == 10
        assert result.info["n_users_kept"] == 10
        assert result.info["users_dropped"] == []
        assert result.info["n_train"] == len(result.train)
        assert result.info["n_valid"] == len(result.valid)
        assert result.info["n_test"] == len(result.test)

    def test_per_user_holdout_is_most_recent(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_last_n_per_user(basic_interactions, n_valid=1, n_test=1)
        for user_id in basic_interactions["USER_ID"].unique():
            user_train = result.train[result.train["USER_ID"] == user_id]
            user_valid = result.valid[result.valid["USER_ID"] == user_id]
            user_test = result.test[result.test["USER_ID"] == user_id]
            assert user_train["TIMESTAMP"].max() <= user_valid["TIMESTAMP"].min()
            assert user_valid["TIMESTAMP"].max() <= user_test["TIMESTAMP"].min()

    def test_sparse_users_dropped(self, sparse_user_interactions: pd.DataFrame) -> None:
        result = leave_last_n_per_user(sparse_user_interactions, n_valid=1, n_test=1)
        assert "u_loner" in result.info["users_dropped"]
        assert result.info["n_users_total"] == 6
        assert result.info["n_users_kept"] == 5

    def test_users_dropped_preserves_first_occurrence_order(self) -> None:
        """`users_dropped` order should match first-appearance order in input."""
        df = pd.DataFrame(
            {
                "USER_ID": ["u_c", "u_b", "u_kept", "u_kept", "u_kept", "u_a", "u_b", "u_c"],
                "ITEM_ID": ["i0"] * 8,
                "TIMESTAMP": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-01",
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-03",
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-02",
                    ]
                ),
            }
        )
        result = leave_last_n_per_user(df, n_valid=1, n_test=1)
        # u_c appears first (index 0), then u_b (index 1), then u_a (index 5).
        assert result.info["users_dropped"] == ["u_c", "u_b", "u_a"]

    def test_exact_threshold_drops_user(self) -> None:
        """User with exactly n_valid + n_test rows is dropped (strict >)."""
        df = pd.DataFrame(
            {
                "USER_ID": ["u_kept"] * 3 + ["u_edge", "u_edge"],
                "ITEM_ID": ["i0", "i1", "i2", "i0", "i1"],
                "TIMESTAMP": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01", "2026-01-02"]),
            }
        )
        result = leave_last_n_per_user(df, n_valid=1, n_test=1)
        assert "u_edge" in result.info["users_dropped"]
        assert "u_kept" not in result.info["users_dropped"]
        assert result.info["n_users_kept"] == 1

    def test_all_users_dropped_raises(self) -> None:
        df = pd.DataFrame(
            {
                "USER_ID": ["u0", "u1", "u2"],
                "ITEM_ID": ["i0", "i1", "i2"],
                "TIMESTAMP": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            }
        )
        with pytest.raises(ValueError, match="no users retained"):
            leave_last_n_per_user(df, n_valid=2, n_test=0)

    def test_n_valid_zero_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="n_valid must be >= 1"):
            leave_last_n_per_user(basic_interactions, n_valid=0)

    def test_negative_n_test_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="n_test must be >= 0"):
            leave_last_n_per_user(basic_interactions, n_valid=1, n_test=-1)

    def test_missing_column_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="missing required column"):
            leave_last_n_per_user(basic_interactions.drop(columns=["TIMESTAMP"]), n_valid=1)

    def test_empty_raises(self) -> None:
        empty = pd.DataFrame({"USER_ID": [], "TIMESTAMP": pd.to_datetime([])})
        with pytest.raises(ValueError, match="empty"):
            leave_last_n_per_user(empty, n_valid=1)

    def test_nan_user_ids_raise(self) -> None:
        df = pd.DataFrame(
            {
                "USER_ID": ["u0", None, "u1"],
                "ITEM_ID": ["i0", "i1", "i2"],
                "TIMESTAMP": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            }
        )
        with pytest.raises(ValueError, match="NaN"):
            leave_last_n_per_user(df, n_valid=1)


# ---------------------------------------------------------------------------
# random_split_per_user
# ---------------------------------------------------------------------------


class TestRandomSplitPerUser:
    def test_happy_path(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split_per_user(basic_interactions, valid_fraction=0.2, test_fraction=0.2, random_state=0)
        total = len(result.train) + len(result.valid) + len(result.test)
        assert total == len(basic_interactions)
        _assert_rows_preserved(basic_interactions, result.train, result.valid, result.test)

    def test_no_test_split_returns_none(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split_per_user(basic_interactions, valid_fraction=0.2, random_state=0)
        assert result.test is None
        assert result.info["n_test"] == 0

    def test_index_reset(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split_per_user(basic_interactions, valid_fraction=0.2, test_fraction=0.2, random_state=0)
        _assert_index_reset(result.train, result.valid, result.test)

    def test_schema_preserved(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split_per_user(basic_interactions, valid_fraction=0.2, test_fraction=0.2, random_state=0)
        _assert_schema_preserved(basic_interactions, result.train, result.valid, result.test)

    def test_info_fields(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split_per_user(basic_interactions, valid_fraction=0.2, test_fraction=0.2, random_state=0)
        assert result.info["n_users"] == 10
        assert result.info["users_with_no_train_rows"] == []
        assert result.info["n_train"] == len(result.train)
        assert result.info["n_valid"] == len(result.valid)
        assert result.info["n_test"] == len(result.test)

    def test_all_users_present_in_train(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split_per_user(basic_interactions, valid_fraction=0.1, test_fraction=0.1, random_state=0)
        train_users = set(result.train["USER_ID"].unique())
        all_users = set(basic_interactions["USER_ID"].unique())
        # Each user has 10 rows and we hold out 2 → 8 train rows per user, so all present.
        assert train_users == all_users

    def test_reproducibility(self, basic_interactions: pd.DataFrame) -> None:
        r1 = random_split_per_user(basic_interactions, valid_fraction=0.2, test_fraction=0.2, random_state=42)
        r2 = random_split_per_user(basic_interactions, valid_fraction=0.2, test_fraction=0.2, random_state=42)
        pd.testing.assert_frame_equal(r1.train, r2.train)
        pd.testing.assert_frame_equal(r1.valid, r2.valid)
        pd.testing.assert_frame_equal(r1.test, r2.test)

    def test_random_state_none_does_not_raise(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split_per_user(basic_interactions, valid_fraction=0.2, random_state=None)
        assert isinstance(result, SplitResult)

    def test_fractions_sum_to_one_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="< 1.0"):
            random_split_per_user(basic_interactions, valid_fraction=0.6, test_fraction=0.5)

    def test_negative_fraction_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            random_split_per_user(basic_interactions, valid_fraction=-0.1)

    def test_empty_raises(self) -> None:
        empty = pd.DataFrame({"USER_ID": []})
        with pytest.raises(ValueError, match="empty"):
            random_split_per_user(empty, valid_fraction=0.2)

    def test_nan_user_ids_raise(self) -> None:
        df = pd.DataFrame({"USER_ID": ["u0", None, "u1"], "ITEM_ID": ["i0", "i1", "i2"]})
        with pytest.raises(ValueError, match="NaN"):
            random_split_per_user(df, valid_fraction=0.2)


# ---------------------------------------------------------------------------
# leave_n_users_out
# ---------------------------------------------------------------------------


class TestLeaveNUsersOut:
    def test_happy_path(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_n_users_out(basic_interactions, n_valid_users=2, n_test_users=2, random_state=0)
        total = len(result.train) + len(result.valid) + len(result.test)
        assert total == len(basic_interactions)
        _assert_rows_preserved(basic_interactions, result.train, result.valid, result.test)

    def test_no_test_split_returns_none(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_n_users_out(basic_interactions, n_valid_users=2, random_state=0)
        assert result.test is None
        assert result.info["test_user_ids"] is None

    def test_index_reset(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_n_users_out(basic_interactions, n_valid_users=2, n_test_users=2, random_state=0)
        _assert_index_reset(result.train, result.valid, result.test)

    def test_schema_preserved(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_n_users_out(basic_interactions, n_valid_users=2, n_test_users=2, random_state=0)
        _assert_schema_preserved(basic_interactions, result.train, result.valid, result.test)

    def test_info_fields(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_n_users_out(basic_interactions, n_valid_users=2, n_test_users=2, random_state=0)
        assert len(result.info["train_user_ids"]) == 6
        assert len(result.info["valid_user_ids"]) == 2
        assert len(result.info["test_user_ids"]) == 2
        assert set(result.train["USER_ID"].unique()) == set(result.info["train_user_ids"])
        assert set(result.valid["USER_ID"].unique()) == set(result.info["valid_user_ids"])
        assert set(result.test["USER_ID"].unique()) == set(result.info["test_user_ids"])
        assert result.info["n_train"] == len(result.train)
        assert result.info["n_valid"] == len(result.valid)
        assert result.info["n_test"] == len(result.test)

    def test_user_set_disjoint(self, basic_interactions: pd.DataFrame) -> None:
        result = leave_n_users_out(basic_interactions, n_valid_users=2, n_test_users=2, random_state=0)
        train_users = set(result.train["USER_ID"].unique())
        valid_users = set(result.valid["USER_ID"].unique())
        test_users = set(result.test["USER_ID"].unique())
        assert train_users.isdisjoint(valid_users)
        assert train_users.isdisjoint(test_users)
        assert valid_users.isdisjoint(test_users)

    def test_reproducibility(self, basic_interactions: pd.DataFrame) -> None:
        r1 = leave_n_users_out(basic_interactions, n_valid_users=2, n_test_users=2, random_state=42)
        r2 = leave_n_users_out(basic_interactions, n_valid_users=2, n_test_users=2, random_state=42)
        assert r1.info["valid_user_ids"] == r2.info["valid_user_ids"]
        assert r1.info["test_user_ids"] == r2.info["test_user_ids"]

    def test_too_many_holdout_users_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least one training user"):
            leave_n_users_out(basic_interactions, n_valid_users=5, n_test_users=5)

    def test_negative_counts_raise(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            leave_n_users_out(basic_interactions, n_valid_users=-1)

    def test_missing_user_col_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="missing required column"):
            leave_n_users_out(basic_interactions.drop(columns=["USER_ID"]), n_valid_users=2)

    def test_empty_raises(self) -> None:
        empty = pd.DataFrame({"USER_ID": []})
        with pytest.raises(ValueError, match="empty"):
            leave_n_users_out(empty, n_valid_users=1)

    def test_nan_user_ids_raise(self) -> None:
        df = pd.DataFrame({"USER_ID": ["u0", None, "u1", "u2"], "ITEM_ID": ["i0", "i1", "i2", "i3"]})
        with pytest.raises(ValueError, match="NaN"):
            leave_n_users_out(df, n_valid_users=1)


# ---------------------------------------------------------------------------
# random_split
# ---------------------------------------------------------------------------


class TestRandomSplit:
    def test_happy_path(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split(basic_interactions, valid_fraction=0.2, test_fraction=0.1, random_state=0)
        assert len(result.train) + len(result.valid) + len(result.test) == len(basic_interactions)
        assert len(result.valid) == 20
        assert len(result.test) == 10
        assert len(result.train) == 70
        _assert_rows_preserved(basic_interactions, result.train, result.valid, result.test)

    def test_no_test_split_returns_none(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split(basic_interactions, valid_fraction=0.2, random_state=0)
        assert result.test is None
        assert result.info["n_test"] == 0

    def test_index_reset(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split(basic_interactions, valid_fraction=0.2, test_fraction=0.1, random_state=0)
        _assert_index_reset(result.train, result.valid, result.test)

    def test_schema_preserved(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split(basic_interactions, valid_fraction=0.2, test_fraction=0.1, random_state=0)
        _assert_schema_preserved(basic_interactions, result.train, result.valid, result.test)

    def test_info_fields(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split(basic_interactions, valid_fraction=0.2, test_fraction=0.1, random_state=0)
        assert result.info["n_train"] == len(result.train)
        assert result.info["n_valid"] == len(result.valid)
        assert result.info["n_test"] == len(result.test)

    def test_reproducibility(self, basic_interactions: pd.DataFrame) -> None:
        r1 = random_split(basic_interactions, valid_fraction=0.2, test_fraction=0.1, random_state=42)
        r2 = random_split(basic_interactions, valid_fraction=0.2, test_fraction=0.1, random_state=42)
        pd.testing.assert_frame_equal(r1.train, r2.train)
        pd.testing.assert_frame_equal(r1.valid, r2.valid)
        pd.testing.assert_frame_equal(r1.test, r2.test)

    def test_random_state_none_does_not_raise(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split(basic_interactions, valid_fraction=0.2, random_state=None)
        assert isinstance(result, SplitResult)

    def test_fractions_sum_to_one_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="< 1.0"):
            random_split(basic_interactions, valid_fraction=0.6, test_fraction=0.5)

    def test_negative_fraction_raises(self, basic_interactions: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            random_split(basic_interactions, valid_fraction=-0.1)

    def test_empty_raises(self) -> None:
        empty = pd.DataFrame({"X": []})
        with pytest.raises(ValueError, match="empty"):
            random_split(empty, valid_fraction=0.2)

    def test_single_row(self) -> None:
        df = pd.DataFrame({"X": [1]})
        result = random_split(df, valid_fraction=0.2)
        assert len(result.train) == 1
        assert len(result.valid) == 0

    def test_zero_fractions_returns_all_train(self, basic_interactions: pd.DataFrame) -> None:
        result = random_split(basic_interactions, valid_fraction=0.0, test_fraction=0.0)
        assert len(result.train) == len(basic_interactions)
        assert len(result.valid) == 0
        assert result.test is None
