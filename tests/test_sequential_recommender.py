
"""
Unit tests for SequentialRecommender._build_sequences and related behaviour.
"""
import logging
import os
import tempfile

import pandas as pd
import pytest

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.estimator.sequential.sasrec_estimator import (
    ITEM_SEQUENCE_COL,
    OUTCOME_SEQUENCE_COL,
    SASRecClassifierEstimator,
)
from skrec.recommender.sequential.sequential_recommender import (
    TIMESTAMP_COL,
    SequentialRecommender,
)
from skrec.scorer.sequential import SequentialScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recommender(max_len: int = 5) -> SequentialRecommender:
    est = SASRecClassifierEstimator(hidden_units=16, num_blocks=1, num_heads=1, epochs=1)
    scorer = SequentialScorer(est)
    return SequentialRecommender(scorer, max_len=max_len)


def _interactions(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _build_sequences — column validation
# ---------------------------------------------------------------------------


class TestBuildSequencesValidation:
    def test_missing_timestamp_raises(self):
        rec = _make_recommender()
        df = _interactions([{USER_ID_NAME: "u1", ITEM_ID_NAME: "i1", LABEL_NAME: 1.0}])
        with pytest.raises(ValueError, match="TIMESTAMP"):
            rec._build_sequences(df)

    def test_missing_item_id_raises(self):
        rec = _make_recommender()
        df = _interactions([{USER_ID_NAME: "u1", TIMESTAMP_COL: 1, LABEL_NAME: 1.0}])
        with pytest.raises(ValueError, match=ITEM_ID_NAME):
            rec._build_sequences(df)

    def test_missing_user_id_raises(self):
        rec = _make_recommender()
        df = _interactions([{ITEM_ID_NAME: "i1", TIMESTAMP_COL: 1, LABEL_NAME: 1.0}])
        with pytest.raises(ValueError, match=USER_ID_NAME):
            rec._build_sequences(df)


# ---------------------------------------------------------------------------
# _build_sequences — ordering
# ---------------------------------------------------------------------------


class TestBuildSequencesOrdering:
    def test_items_sorted_by_timestamp(self):
        """Interactions provided in reverse timestamp order must be sorted ascending."""
        rec = _make_recommender()
        df = _interactions(
            [
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "c", TIMESTAMP_COL: 3, LABEL_NAME: 1.0},
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "a", TIMESTAMP_COL: 1, LABEL_NAME: 1.0},
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "b", TIMESTAMP_COL: 2, LABEL_NAME: 1.0},
            ]
        )
        result = rec._build_sequences(df)
        seq = result.loc[result[USER_ID_NAME] == "u1", ITEM_SEQUENCE_COL].iloc[0]
        assert seq == ["a", "b", "c"]

    def test_outcome_sequence_matches_item_order(self):
        """OUTCOME_SEQUENCE must be in the same timestamp-sorted order as ITEM_SEQUENCE."""
        rec = _make_recommender()
        df = _interactions(
            [
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "x", TIMESTAMP_COL: 10, LABEL_NAME: 0.9},
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "y", TIMESTAMP_COL: 5, LABEL_NAME: 0.1},
            ]
        )
        result = rec._build_sequences(df)
        row = result.loc[result[USER_ID_NAME] == "u1"].iloc[0]
        assert row[ITEM_SEQUENCE_COL] == ["y", "x"]
        assert row[OUTCOME_SEQUENCE_COL] == [0.1, 0.9]


# ---------------------------------------------------------------------------
# _build_sequences — truncation
# ---------------------------------------------------------------------------


class TestBuildSequencesTruncation:
    def test_long_history_truncated_to_max_len_plus_one(self):
        """A user with 10 interactions and max_len=3 should keep only 4 (max_len+1) items."""
        rec = _make_recommender(max_len=3)
        items = [str(i) for i in range(10)]
        df = _interactions(
            [
                {USER_ID_NAME: "u1", ITEM_ID_NAME: item, TIMESTAMP_COL: t, LABEL_NAME: 1.0}
                for t, item in enumerate(items)
            ]
        )
        result = rec._build_sequences(df)
        seq = result.loc[result[USER_ID_NAME] == "u1", ITEM_SEQUENCE_COL].iloc[0]
        assert len(seq) == 4  # max_len + 1
        assert seq == items[-4:]  # most recent items kept

    def test_short_history_not_truncated(self):
        """A user with fewer interactions than max_len+1 keeps their full history."""
        rec = _make_recommender(max_len=10)
        items = ["a", "b", "c"]
        df = _interactions(
            [
                {USER_ID_NAME: "u1", ITEM_ID_NAME: item, TIMESTAMP_COL: t, LABEL_NAME: 1.0}
                for t, item in enumerate(items)
            ]
        )
        result = rec._build_sequences(df)
        seq = result.loc[result[USER_ID_NAME] == "u1", ITEM_SEQUENCE_COL].iloc[0]
        assert seq == items

    def test_outcome_sequence_truncated_consistently(self):
        """OUTCOME_SEQUENCE is truncated to the same length as ITEM_SEQUENCE."""
        rec = _make_recommender(max_len=2)
        df = _interactions(
            [{USER_ID_NAME: "u1", ITEM_ID_NAME: str(i), TIMESTAMP_COL: i, LABEL_NAME: float(i)} for i in range(6)]
        )
        result = rec._build_sequences(df)
        row = result.loc[result[USER_ID_NAME] == "u1"].iloc[0]
        assert len(row[ITEM_SEQUENCE_COL]) == len(row[OUTCOME_SEQUENCE_COL]) == 3  # max_len + 1


# ---------------------------------------------------------------------------
# _build_sequences — inference (no OUTCOME column)
# ---------------------------------------------------------------------------


class TestBuildSequencesInference:
    def test_no_outcome_column_omits_outcome_sequence(self):
        """At inference time (no OUTCOME in df), OUTCOME_SEQUENCE_COL must not appear."""
        rec = _make_recommender()
        df = _interactions(
            [
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "a", TIMESTAMP_COL: 1},
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "b", TIMESTAMP_COL: 2},
            ]
        )
        result = rec._build_sequences(df)
        assert OUTCOME_SEQUENCE_COL not in result.columns

    def test_no_outcome_still_builds_item_sequences(self):
        """Item sequences are still built correctly even without an OUTCOME column."""
        rec = _make_recommender()
        df = _interactions(
            [
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "b", TIMESTAMP_COL: 2},
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "a", TIMESTAMP_COL: 1},
            ]
        )
        result = rec._build_sequences(df)
        seq = result.loc[result[USER_ID_NAME] == "u1", ITEM_SEQUENCE_COL].iloc[0]
        assert seq == ["a", "b"]


# ---------------------------------------------------------------------------
# _build_sequences — multiple users
# ---------------------------------------------------------------------------


class TestBuildSequencesMultiUser:
    def test_each_user_gets_own_sequence(self):
        """Sequences must be independent per user."""
        rec = _make_recommender()
        df = _interactions(
            [
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "a", TIMESTAMP_COL: 1, LABEL_NAME: 1.0},
                {USER_ID_NAME: "u2", ITEM_ID_NAME: "x", TIMESTAMP_COL: 1, LABEL_NAME: 1.0},
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "b", TIMESTAMP_COL: 2, LABEL_NAME: 1.0},
                {USER_ID_NAME: "u2", ITEM_ID_NAME: "y", TIMESTAMP_COL: 2, LABEL_NAME: 1.0},
            ]
        )
        result = rec._build_sequences(df)
        u1_seq = result.loc[result[USER_ID_NAME] == "u1", ITEM_SEQUENCE_COL].iloc[0]
        u2_seq = result.loc[result[USER_ID_NAME] == "u2", ITEM_SEQUENCE_COL].iloc[0]
        assert u1_seq == ["a", "b"]
        assert u2_seq == ["x", "y"]

    def test_result_has_one_row_per_user(self):
        rec = _make_recommender()
        df = _interactions(
            [
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "a", TIMESTAMP_COL: 1, LABEL_NAME: 1.0},
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "b", TIMESTAMP_COL: 2, LABEL_NAME: 1.0},
                {USER_ID_NAME: "u2", ITEM_ID_NAME: "c", TIMESTAMP_COL: 1, LABEL_NAME: 1.0},
            ]
        )
        result = rec._build_sequences(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# max_len override warning
# ---------------------------------------------------------------------------


class TestMaxLenOverride:
    def test_warning_when_estimator_max_len_differs(self, caplog, tiny_interactions_df, tiny_items_df):
        """A warning is logged when recommender.max_len != estimator.max_len."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path = os.path.join(tmpdir, "interactions.csv")
            items_path = os.path.join(tmpdir, "items.csv")
            tiny_interactions_df.to_csv(interactions_path, index=False)
            tiny_items_df.to_csv(items_path, index=False)

            est = SASRecClassifierEstimator(hidden_units=16, num_blocks=1, num_heads=1, epochs=1, max_len=99)
            scorer = SequentialScorer(est)
            rec = SequentialRecommender(scorer, max_len=10)  # different from estimator

            with caplog.at_level(logging.WARNING):
                rec.train(
                    items_ds=ItemsDataset(data_location=items_path),
                    interactions_ds=InteractionsDataset(data_location=interactions_path),
                )

            assert any("max_len" in msg for msg in caplog.messages)

    def test_estimator_max_len_synced_after_train(self, tiny_interactions_df, tiny_items_df):
        """After train(), estimator.max_len must equal recommender.max_len."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path = os.path.join(tmpdir, "interactions.csv")
            items_path = os.path.join(tmpdir, "items.csv")
            tiny_interactions_df.to_csv(interactions_path, index=False)
            tiny_items_df.to_csv(items_path, index=False)

            est = SASRecClassifierEstimator(hidden_units=16, num_blocks=1, num_heads=1, epochs=1, max_len=99)
            scorer = SequentialScorer(est)
            rec = SequentialRecommender(scorer, max_len=10)
            rec.train(
                items_ds=ItemsDataset(data_location=items_path),
                interactions_ds=InteractionsDataset(data_location=interactions_path),
            )

            assert est.max_len == 10


# ---------------------------------------------------------------------------
# use_validation threading
# ---------------------------------------------------------------------------


class TestValidInteractionsThreading:
    def test_valid_interactions_passed_to_estimator(self, tiny_interactions_df, tiny_items_df):
        """When use_validation=True, the estimator's fit_embedding_model
        must receive a non-None valid_interactions argument derived from interactions_ds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path = os.path.join(tmpdir, "interactions.csv")
            items_path = os.path.join(tmpdir, "items.csv")

            tiny_interactions_df.to_csv(interactions_path, index=False)
            tiny_items_df.to_csv(items_path, index=False)

            received = {}

            est = SASRecClassifierEstimator(hidden_units=16, num_blocks=1, num_heads=1, epochs=1)
            original_fit = est.fit_embedding_model

            def capturing_fit(users, items, interactions, valid_users=None, valid_interactions=None):
                received["valid_interactions"] = valid_interactions
                return original_fit(users, items, interactions, valid_users, valid_interactions)

            est.fit_embedding_model = capturing_fit  # type: ignore[method-assign]

            scorer = SequentialScorer(est)
            rec = SequentialRecommender(scorer, max_len=5)
            rec.train(
                items_ds=ItemsDataset(data_location=items_path),
                interactions_ds=InteractionsDataset(data_location=interactions_path),
                use_validation=True,
            )

            assert received.get("valid_interactions") is not None, (
                "Estimator should receive non-None valid_interactions when "
                "use_validation=True is passed to recommender.train()."
            )

    def test_no_valid_interactions_ds_passes_none(self, tiny_interactions_df, tiny_items_df):
        """When use_validation is not set, valid_interactions must be None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path = os.path.join(tmpdir, "interactions.csv")
            items_path = os.path.join(tmpdir, "items.csv")
            tiny_interactions_df.to_csv(interactions_path, index=False)
            tiny_items_df.to_csv(items_path, index=False)

            received = {}

            est = SASRecClassifierEstimator(hidden_units=16, num_blocks=1, num_heads=1, epochs=1)
            original_fit = est.fit_embedding_model

            def capturing_fit(users, items, interactions, valid_users=None, valid_interactions=None):
                received["valid_interactions"] = valid_interactions
                return original_fit(users, items, interactions, valid_users, valid_interactions)

            est.fit_embedding_model = capturing_fit  # type: ignore[method-assign]

            scorer = SequentialScorer(est)
            rec = SequentialRecommender(scorer, max_len=5)
            rec.train(
                items_ds=ItemsDataset(data_location=items_path),
                interactions_ds=InteractionsDataset(data_location=interactions_path),
            )

            assert received.get("valid_interactions") is None


# ---------------------------------------------------------------------------
# recommend_online() not supported
# ---------------------------------------------------------------------------


class TestRecommendOnlineNotSupported:
    def test_recommend_online_raises(self):
        """recommend_online() must raise NotImplementedError for SequentialRecommender."""
        rec = _make_recommender()
        with pytest.raises(NotImplementedError, match="recommend_online"):
            rec.recommend_online()


# ---------------------------------------------------------------------------
# Fixtures (local, not shared via conftest)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_interactions_df():
    rows = []
    seqs = {"u1": ["1", "2", "3", "4", "5"], "u2": ["6", "7", "8", "9", "10"]}
    for uid, seq in seqs.items():
        for t, item in enumerate(seq):
            rows.append({USER_ID_NAME: uid, ITEM_ID_NAME: item, LABEL_NAME: 1.0, TIMESTAMP_COL: t})
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_items_df():
    return pd.DataFrame({ITEM_ID_NAME: [str(i) for i in range(1, 11)]})
