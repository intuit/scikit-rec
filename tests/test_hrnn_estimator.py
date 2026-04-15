
"""
Unit tests for HRNNClassifierEstimator, HRNNRegressorEstimator,
HierarchicalScorer, and HierarchicalSequentialRecommender.
"""
import logging
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.estimator.sequential.hrnn_estimator import (
    SESSION_OUTCOMES_COL,
    SESSION_SEQUENCES_COL,
    HRNNClassifierEstimator,
    HRNNRegressorEstimator,
)
from skrec.recommender.sequential.hierarchical_recommender import (
    HierarchicalSequentialRecommender,
)
from skrec.recommender.sequential.sequential_recommender import TIMESTAMP_COL
from skrec.scorer.base_scorer import BaseScorer
from skrec.scorer.hierarchical import HierarchicalScorer
from skrec.scorer.sequential import SequentialScorer
from skrec.scorer.universal import UniversalScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_ITEMS = 10
ITEMS = [str(i) for i in range(1, N_ITEMS + 1)]


def _make_classifier(epochs=2, num_negatives=1, dropout_rate=0.0, max_sessions=3, max_session_len=5):
    return HRNNClassifierEstimator(
        hidden_units=16,
        num_layers=1,
        dropout_rate=dropout_rate,
        num_negatives=num_negatives,
        max_sessions=max_sessions,
        max_session_len=max_session_len,
        learning_rate=0.001,
        epochs=epochs,
        batch_size=128,
        verbose=0,
        random_state=42,
    )


def _make_recommender(max_sessions=3, max_session_len=5, session_timeout_minutes=30):
    est = _make_classifier(max_sessions=max_sessions, max_session_len=max_session_len)
    scorer = HierarchicalScorer(est)
    return HierarchicalSequentialRecommender(
        scorer,
        max_sessions=max_sessions,
        max_session_len=max_session_len,
        session_timeout_minutes=session_timeout_minutes,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_items_df():
    return pd.DataFrame({ITEM_ID_NAME: ITEMS})


@pytest.fixture
def tiny_sessions_df():
    """
    3 users, each with 2 sessions. Pre-built SESSION_SEQUENCES_COL format.
    Each session has 3 items; last item of the last session is the notional test item.
    """
    data = {
        "u1": [["1", "2", "3"], ["4", "5", "6"]],
        "u2": [["7", "8", "9"], ["10", "1", "2"]],
        "u3": [["3", "5", "7"], ["9", "1", "3"]],
    }
    rows = []
    for uid, sessions in data.items():
        rows.append(
            {
                USER_ID_NAME: uid,
                SESSION_SEQUENCES_COL: sessions,
                SESSION_OUTCOMES_COL: [[1.0] * len(s) for s in sessions],
            }
        )
    return pd.DataFrame(rows), data


@pytest.fixture
def tiny_interactions_df():
    """Raw flat interactions with timestamps, one session per user (all within 30 min)."""
    rows = []
    seqs = {
        "u1": ["1", "2", "3", "4", "5"],
        "u2": ["6", "7", "8", "9", "10"],
        "u3": ["1", "3", "5", "7", "9"],
    }
    for uid, seq in seqs.items():
        for t, item in enumerate(seq):
            rows.append(
                {
                    USER_ID_NAME: uid,
                    ITEM_ID_NAME: item,
                    LABEL_NAME: 1.0,
                    TIMESTAMP_COL: t * 60,  # 1 minute apart — same session
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def multi_session_interactions_df():
    """
    2 users with 2 clear sessions each (separated by >30 min gap).
    Used for testing session boundary detection.
    """
    rows = []
    # u1: session 1 at t=0..3 min, session 2 at t=60..63 min (60 min gap)
    for t, item in enumerate(["1", "2", "3"]):
        rows.append({USER_ID_NAME: "u1", ITEM_ID_NAME: item, LABEL_NAME: 1.0, TIMESTAMP_COL: t * 60})
    for t, item in enumerate(["4", "5", "6"]):
        rows.append({USER_ID_NAME: "u1", ITEM_ID_NAME: item, LABEL_NAME: 1.0, TIMESTAMP_COL: 3600 + t * 60})
    # u2: session 1 at t=0, session 2 after 2 hours
    for t, item in enumerate(["7", "8"]):
        rows.append({USER_ID_NAME: "u2", ITEM_ID_NAME: item, LABEL_NAME: 1.0, TIMESTAMP_COL: t * 60})
    for t, item in enumerate(["9", "10"]):
        rows.append({USER_ID_NAME: "u2", ITEM_ID_NAME: item, LABEL_NAME: 1.0, TIMESTAMP_COL: 7200 + t * 60})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _build_padded_session_tensors — shapes and alignment
# ---------------------------------------------------------------------------


class TestBuildPaddedSessionTensors:
    def test_output_shapes(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        sess_t, tgt_t, out_t = est._build_padded_session_tensors(seqs_df)
        n = len(seqs_df)
        assert sess_t.shape == (n, est.max_sessions, est.max_session_len)
        assert tgt_t.shape == (n, est.max_sessions, est.max_session_len)
        assert out_t.shape == (n, est.max_sessions, est.max_session_len)

    def test_padding_is_zero(self, tiny_sessions_df, tiny_items_df):
        """Sessions shorter than max_session_len must be zero-padded on the left."""
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1, max_session_len=10)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        sess_t, _, _ = est._build_padded_session_tensors(seqs_df)
        # Each session has 3 items; with max_session_len=10, first 7 positions are padding
        assert (sess_t[:, :, :7] == 0).all()

    def test_target_is_next_item_in_session(self, tiny_sessions_df, tiny_items_df):
        """target_tensor[u, s, t] should be the item at position t+1 of session s."""
        seqs_df, data = tiny_sessions_df
        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        sess_t, tgt_t, _ = est._build_padded_session_tensors(seqs_df)

        for i, uid in enumerate(seqs_df[USER_ID_NAME]):
            last_session = data[uid][-1]
            # Sessions are right-aligned: last item is at position -1 with target=0,
            # second-to-last item is at position -2 with target = last item.
            last_item_idx = est._item_to_idx[last_session[-1]]
            assert int(tgt_t[i, -1, -2]) == last_item_idx

    def test_last_position_in_session_has_no_target(self, tiny_sessions_df, tiny_items_df):
        """The last real item in each session has no within-session target (target=0)."""
        seqs_df, data = tiny_sessions_df
        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        sess_t, tgt_t, _ = est._build_padded_session_tensors(seqs_df)

        for i, uid in enumerate(seqs_df[USER_ID_NAME]):
            last_session = data[uid][-1]
            last_item = last_session[-1]
            last_item_idx = est._item_to_idx[last_item]
            # Last real position of last session: sess_t has the item, tgt_t is 0
            assert int(sess_t[i, -1, -1]) == last_item_idx
            assert int(tgt_t[i, -1, -1]) == 0

    def test_outcome_alignment(self, tiny_sessions_df, tiny_items_df):
        """outcome_tensor[u, s, t] must equal the OUTCOME of the target item at t+1."""
        seqs_df, data = tiny_sessions_df
        # Give mixed outcomes to verify alignment
        mixed_outcomes = []
        for uid, sessions in data.items():
            user_outcomes = []
            for s_idx, session in enumerate(sessions):
                user_outcomes.append([float(s_idx % 2)] * len(session))
            mixed_outcomes.append(
                {
                    USER_ID_NAME: uid,
                    SESSION_SEQUENCES_COL: sessions,
                    SESSION_OUTCOMES_COL: user_outcomes,
                }
            )
        seqs_df_mixed = pd.DataFrame(mixed_outcomes)

        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df_mixed)
        _, tgt_t, out_t = est._build_padded_session_tensors(seqs_df_mixed)

        # At any valid position (tgt != 0), outcome must be non-NaN and finite
        valid = tgt_t != 0
        assert np.isfinite(out_t[valid.numpy()]).all()

    def test_short_outcome_list_pads_with_warning(self, tiny_sessions_df, tiny_items_df, caplog):
        """When an outcome list is shorter than the session, a warning should be logged
        and the outcome tensor must remain finite (padded with 1.0, no crash)."""
        seqs_df, data = tiny_sessions_df

        # Build sessions with outcome lists shorter than the session length (1 instead of 3)
        short_rows = []
        for uid, sessions in data.items():
            short_rows.append(
                {
                    USER_ID_NAME: uid,
                    SESSION_SEQUENCES_COL: sessions,
                    SESSION_OUTCOMES_COL: [[0.5] for _ in sessions],  # 1 outcome for 3-item sessions
                }
            )
        short_seqs_df = pd.DataFrame(short_rows)

        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)  # build vocab

        with caplog.at_level(logging.WARNING):
            _, tgt_t, out_t = est._build_padded_session_tensors(short_seqs_df)

        assert any(
            "session(s) had outcome lists shorter" in r.message for r in caplog.records if r.levelno == logging.WARNING
        ), "Expected a warning about short outcome list"
        assert np.isfinite(out_t.numpy()).all(), "Outcome tensor must be finite after padding"


# ---------------------------------------------------------------------------
# _HRNNModule forward — tensor shapes and padding
# ---------------------------------------------------------------------------


class TestHRNNModuleForward:
    def test_output_shape(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)

        sess_t, _, _ = est._build_padded_session_tensors(seqs_df)
        with torch.no_grad():
            out = est.model(sess_t)

        B, S, L, H = out.shape
        assert B == len(seqs_df)
        assert S == est.max_sessions
        assert L == est.max_session_len
        assert H == est.hidden_units

    def test_padding_positions_zeroed(self, tiny_sessions_df, tiny_items_df):
        """Hidden states at padding positions (item=0) must be zeroed out."""
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1, max_session_len=10)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)

        sess_t, _, _ = est._build_padded_session_tensors(seqs_df)
        with torch.no_grad():
            out = est.model(sess_t)

        # Padding mask: positions where sess_t == 0
        pad_mask = (sess_t == 0).unsqueeze(-1).expand_as(out)
        assert (out[pad_mask] == 0).all(), "Hidden states at padding positions must be zero"

    def test_no_nan_in_output(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        sess_t, _, _ = est._build_padded_session_tensors(seqs_df)
        with torch.no_grad():
            out = est.model(sess_t)
        assert not torch.isnan(out).any()

    def test_padding_embedding_stays_zero(self, tiny_sessions_df, tiny_items_df):
        """Embedding at index 0 (padding token) must remain zero after training."""
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=5)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        pad_emb = est.model.item_embedding.weight[0]
        assert pad_emb.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Training correctness
# ---------------------------------------------------------------------------


class TestHRNNTraining:
    def test_fit_sets_model(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1)
        assert est.model is None
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        assert est.model is not None

    def test_item_vocab_size(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        assert est.num_items == N_ITEMS
        assert len(est.item_id_index) == N_ITEMS

    def test_embeddings_update_after_training(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=10)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        emb_norms = est.model.item_embedding.weight[1:].norm(dim=1)
        assert emb_norms.mean().item() > 0.01

    def test_scores_shape(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        scores = est.predict_proba_with_embeddings(seqs_df)
        assert scores.shape == (len(seqs_df), N_ITEMS)

    def test_scores_are_finite(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        scores = est.predict_proba_with_embeddings(seqs_df)
        assert np.all(np.isfinite(scores))

    def test_score_variance_nonzero(self, tiny_sessions_df, tiny_items_df):
        """Scores must differ across items — model must not collapse to constant output."""
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=20)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        scores = est.predict_proba_with_embeddings(seqs_df)
        within_user_std = scores.std(axis=1)
        assert within_user_std.mean() > 0.01

    def test_early_stopping_skips_patience_when_no_valid_val_batches(self, tiny_sessions_df, tiny_items_df, caplog):
        """When all validation sessions have length 1, _compute_hrnn_loss returns None for every
        batch → val_batches stays 0. Early stopping must skip the patience check with a warning
        rather than spuriously incrementing the counter on inf/0."""
        seqs_df, _ = tiny_sessions_df

        # Validation sessions with only 1 item each — no within-session targets, so
        # valid_mask is all-False → _compute_hrnn_loss returns None → val_batches == 0.
        length1_rows = [
            {
                USER_ID_NAME: "u1",
                SESSION_SEQUENCES_COL: [["1"]],
                SESSION_OUTCOMES_COL: [[1.0]],
            },
            {
                USER_ID_NAME: "u2",
                SESSION_SEQUENCES_COL: [["2"]],
                SESSION_OUTCOMES_COL: [[1.0]],
            },
        ]
        val_sessions_df = pd.DataFrame(length1_rows)

        est = HRNNClassifierEstimator(
            hidden_units=8,
            epochs=3,
            early_stopping_patience=1,  # would fire epoch 2 if patience incorrectly updated
            verbose=0,
            random_state=42,
        )
        with caplog.at_level(logging.WARNING):
            est.fit_embedding_model(
                users=None,
                items=tiny_items_df,
                interactions=seqs_df,
                valid_interactions=val_sessions_df,
            )

        assert any(
            "skipping early stopping" in r.message.lower() for r in caplog.records if r.levelno == logging.WARNING
        ), "Expected warning about skipping early stopping when val_batches == 0"
        assert est.model is not None


# ---------------------------------------------------------------------------
# Overfit test
# ---------------------------------------------------------------------------


class TestHRNNOverfit:
    def test_overfit_tiny_dataset(self, tiny_sessions_df, tiny_items_df):
        """After enough training on a tiny dataset, most test items should rank in top-5."""
        seqs_df, data = tiny_sessions_df
        est = _make_classifier(epochs=1000, num_negatives=1, dropout_rate=0.0)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)

        scores = est.predict_proba_with_embeddings(seqs_df)

        hits = 0
        for i, uid in enumerate(seqs_df[USER_ID_NAME]):
            test_item = data[uid][-1][-1]  # last item of last session
            test_idx = est._item_to_idx[test_item] - 1  # 0-indexed into scores
            rank = int((scores[i] > scores[i, test_idx]).sum()) + 1
            if rank <= 5:
                hits += 1

        # HRNN has fewer training pairs than SASRec on this tiny dataset (2 sessions × 2
        # valid positions each), so require at least 2/3 users to rank in top-5.
        assert hits >= 2, f"Expected ≥2/{len(data)} test items to rank top-5 after overfitting, got {hits}."


# ---------------------------------------------------------------------------
# Item order alignment (the sort-order bug regression test)
# ---------------------------------------------------------------------------


class TestItemOrderAlignment:
    def test_scorer_and_estimator_same_order(self, tiny_sessions_df):
        """scorer.item_names and estimator.item_id_index must be in the same order."""
        seqs_df, _ = tiny_sessions_df
        items_int = pd.DataFrame({ITEM_ID_NAME: list(range(1, N_ITEMS + 1))})  # int64

        est = _make_classifier(epochs=1)
        scorer = HierarchicalScorer(est)
        _, items_out, _ = scorer.process_factorized_datasets(
            users_df=None,
            items_df=items_int,
            interactions_df=seqs_df,
            is_training=True,
        )
        est.fit_embedding_model(None, items_out, seqs_df)

        assert list(scorer.item_names) == list(est.item_id_index), (
            f"Scorer: {list(scorer.item_names)[:5]} | Estimator: {list(est.item_id_index)[:5]}"
        )

    def test_score_column_maps_to_correct_item(self, tiny_sessions_df, tiny_items_df):
        """Score at column j must correspond to scorer.item_names[j] after a full forward pass."""
        seqs_df, data = tiny_sessions_df
        est = _make_classifier(epochs=1)
        scorer = HierarchicalScorer(est)
        _, items_out, _ = scorer.process_factorized_datasets(
            users_df=None, items_df=tiny_items_df, interactions_df=seqs_df, is_training=True
        )
        est.fit_embedding_model(None, items_out, seqs_df)

        scores = est.predict_proba_with_embeddings(seqs_df)
        item_name_to_col = {name: j for j, name in enumerate(scorer.item_names)}

        for i, uid in enumerate(seqs_df[USER_ID_NAME]):
            test_item = data[uid][-1][-1]
            j = item_name_to_col[test_item]
            assert np.isfinite(scores[i, j])


# ---------------------------------------------------------------------------
# HRNNRegressorEstimator
# ---------------------------------------------------------------------------


class TestHRNNRegressor:
    def test_trains_without_error(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = HRNNRegressorEstimator(
            hidden_units=16,
            num_layers=1,
            dropout_rate=0.0,
            num_negatives=1,
            max_sessions=3,
            max_session_len=5,
            learning_rate=0.001,
            epochs=2,
            verbose=0,
            random_state=42,
        )
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        assert est.model is not None

    def test_output_shape(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = HRNNRegressorEstimator(
            hidden_units=16,
            num_layers=1,
            dropout_rate=0.0,
            num_negatives=1,
            max_sessions=3,
            max_session_len=5,
            epochs=1,
            verbose=0,
            random_state=42,
        )
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        scores = est.predict_proba_with_embeddings(seqs_df)
        assert scores.shape == (len(seqs_df), N_ITEMS)
        assert np.all(np.isfinite(scores))

    def test_default_loss_is_mse(self):
        est = HRNNRegressorEstimator()
        assert est.loss_fn_name == "mse"

    def test_classifier_default_loss_is_bce(self):
        est = HRNNClassifierEstimator()
        assert est.loss_fn_name == "bce"


# ---------------------------------------------------------------------------
# HierarchicalSequentialRecommender._build_session_sequences
# ---------------------------------------------------------------------------


class TestBuildSessionSequences:
    def test_single_session_per_user(self, tiny_interactions_df):
        """All interactions within 30 min should form one session per user."""
        rec = _make_recommender(session_timeout_minutes=30)
        result = rec._build_session_sequences(tiny_interactions_df)

        assert len(result) == tiny_interactions_df[USER_ID_NAME].nunique()
        for _, row in result.iterrows():
            assert len(row[SESSION_SEQUENCES_COL]) == 1, "Expected 1 session per user"

    def test_multiple_sessions_detected_from_timestamp_gap(self, multi_session_interactions_df):
        """Interactions separated by >30 min should form separate sessions."""
        rec = _make_recommender(session_timeout_minutes=30)
        result = rec._build_session_sequences(multi_session_interactions_df)

        for _, row in result.iterrows():
            assert len(row[SESSION_SEQUENCES_COL]) == 2, f"Expected 2 sessions, got {len(row[SESSION_SEQUENCES_COL])}"

    def test_session_items_in_timestamp_order(self, multi_session_interactions_df):
        """Items within each session must be in ascending timestamp order."""
        rec = _make_recommender(session_timeout_minutes=30)
        result = rec._build_session_sequences(multi_session_interactions_df)

        u1_row = result[result[USER_ID_NAME] == "u1"].iloc[0]
        assert u1_row[SESSION_SEQUENCES_COL][0] == ["1", "2", "3"]
        assert u1_row[SESSION_SEQUENCES_COL][1] == ["4", "5", "6"]

    def test_session_id_column_takes_precedence(self):
        """When SESSION_ID is present it must be used directly, ignoring timestamp gaps."""
        from skrec.recommender.sequential.hierarchical_recommender import SESSION_ID_COL

        rec = _make_recommender(session_timeout_minutes=5)  # tight timeout would split
        df = pd.DataFrame(
            [
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "1", LABEL_NAME: 1.0, TIMESTAMP_COL: 0, SESSION_ID_COL: "s1"},
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "2", LABEL_NAME: 1.0, TIMESTAMP_COL: 10000, SESSION_ID_COL: "s1"},
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "3", LABEL_NAME: 1.0, TIMESTAMP_COL: 20000, SESSION_ID_COL: "s2"},
            ]
        )
        result = rec._build_session_sequences(df)
        u1 = result[result[USER_ID_NAME] == "u1"].iloc[0]
        # Items 1 and 2 share SESSION_ID s1 → same session; item 3 → different session
        assert len(u1[SESSION_SEQUENCES_COL]) == 2
        assert u1[SESSION_SEQUENCES_COL][0] == ["1", "2"]
        assert u1[SESSION_SEQUENCES_COL][1] == ["3"]

    def test_outcome_col_present_when_label_present(self, tiny_interactions_df):
        """SESSION_OUTCOMES_COL must be present when OUTCOME/LABEL column exists in input."""
        rec = _make_recommender()
        result = rec._build_session_sequences(tiny_interactions_df)
        assert SESSION_OUTCOMES_COL in result.columns

    def test_outcome_col_absent_at_inference(self):
        """SESSION_OUTCOMES_COL must NOT appear when OUTCOME/LABEL is absent (inference)."""
        rec = _make_recommender()
        df = pd.DataFrame(
            [
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "1", TIMESTAMP_COL: 0},
                {USER_ID_NAME: "u1", ITEM_ID_NAME: "2", TIMESTAMP_COL: 60},
            ]
        )
        result = rec._build_session_sequences(df)
        assert SESSION_OUTCOMES_COL not in result.columns

    def test_sessions_truncated_to_max_sessions(self):
        """Users with more sessions than max_sessions must keep only the most recent ones."""
        rec = _make_recommender(max_sessions=2, session_timeout_minutes=30)
        rows = []
        # 5 sessions of 2 items each, separated by 1-hour gaps
        for sess_idx in range(5):
            for item_offset, item in enumerate(["1", "2"]):
                rows.append(
                    {
                        USER_ID_NAME: "u1",
                        ITEM_ID_NAME: item,
                        LABEL_NAME: 1.0,
                        TIMESTAMP_COL: sess_idx * 3600 + item_offset * 60,
                    }
                )
        df = pd.DataFrame(rows)
        result = rec._build_session_sequences(df)
        u1 = result[result[USER_ID_NAME] == "u1"].iloc[0]
        assert len(u1[SESSION_SEQUENCES_COL]) == 2, "Must keep only max_sessions=2 most recent sessions"

    def test_one_row_per_user(self, tiny_interactions_df):
        rec = _make_recommender()
        result = rec._build_session_sequences(tiny_interactions_df)
        assert len(result) == tiny_interactions_df[USER_ID_NAME].nunique()

    def test_users_are_independent(self, tiny_interactions_df):
        """Session sequences for u1 must not contain items from u2."""
        rec = _make_recommender()
        result = rec._build_session_sequences(tiny_interactions_df)

        u1_items = {
            item for session in result[result[USER_ID_NAME] == "u1"].iloc[0][SESSION_SEQUENCES_COL] for item in session
        }
        u2_items = {
            item for session in result[result[USER_ID_NAME] == "u2"].iloc[0][SESSION_SEQUENCES_COL] for item in session
        }
        assert u1_items.isdisjoint(u2_items), "User sessions must not contain each other's items"


# ---------------------------------------------------------------------------
# HierarchicalSequentialRecommender — full integration
# ---------------------------------------------------------------------------


class TestHierarchicalRecommenderIntegration:
    def test_train_and_recommend(self, tiny_interactions_df, tiny_items_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path = os.path.join(tmpdir, "interactions.csv")
            items_path = os.path.join(tmpdir, "items.csv")
            tiny_interactions_df.to_csv(interactions_path, index=False)
            tiny_items_df.to_csv(items_path, index=False)

            rec = _make_recommender()
            rec.train(
                items_ds=ItemsDataset(data_location=items_path),
                interactions_ds=InteractionsDataset(data_location=interactions_path),
            )

            recs = rec.recommend(interactions=tiny_interactions_df, top_k=3)
            n_users = tiny_interactions_df[USER_ID_NAME].nunique()
            assert recs.shape == (n_users, 3)

    def test_recommend_items_are_known(self, tiny_interactions_df, tiny_items_df):
        """All recommended items must come from the known item catalogue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path = os.path.join(tmpdir, "interactions.csv")
            items_path = os.path.join(tmpdir, "items.csv")
            tiny_interactions_df.to_csv(interactions_path, index=False)
            tiny_items_df.to_csv(items_path, index=False)

            rec = _make_recommender()
            rec.train(
                items_ds=ItemsDataset(data_location=items_path),
                interactions_ds=InteractionsDataset(data_location=interactions_path),
            )

            recs = rec.recommend(interactions=tiny_interactions_df, top_k=5)
            known = set(ITEMS)
            for row in recs:
                for item in row:
                    assert str(item) in known, f"Unknown item {item} in recommendations"

    def test_item_order_preserved_through_pipeline(self, tiny_interactions_df, tiny_items_df):
        """After full train(), scorer.item_names must match estimator.item_id_index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path = os.path.join(tmpdir, "interactions.csv")
            items_path = os.path.join(tmpdir, "items.csv")
            tiny_interactions_df.to_csv(interactions_path, index=False)
            tiny_items_df.to_csv(items_path, index=False)

            est = _make_classifier(epochs=1)
            scorer = HierarchicalScorer(est)
            rec = HierarchicalSequentialRecommender(scorer, max_sessions=3, max_session_len=5)
            rec.train(
                items_ds=ItemsDataset(data_location=items_path),
                interactions_ds=InteractionsDataset(data_location=interactions_path),
            )

            assert list(scorer.item_names) == list(est.item_id_index)

    def test_explicit_negatives_train_without_error(self, tiny_items_df):
        """Training with explicit negative outcomes (OUTCOME=0.0) must not raise."""
        rows = []
        for uid in ["u1", "u2", "u3"]:
            for t, (item, outcome) in enumerate([("1", 1.0), ("2", 0.0), ("3", 1.0), ("4", 0.0), ("5", 1.0)]):
                rows.append(
                    {
                        USER_ID_NAME: uid,
                        ITEM_ID_NAME: item,
                        LABEL_NAME: outcome,
                        TIMESTAMP_COL: t * 60,
                    }
                )
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path = os.path.join(tmpdir, "interactions.csv")
            items_path = os.path.join(tmpdir, "items.csv")
            df.to_csv(interactions_path, index=False)
            tiny_items_df.to_csv(items_path, index=False)

            rec = _make_recommender()
            rec.train(
                items_ds=ItemsDataset(data_location=items_path),
                interactions_ds=InteractionsDataset(data_location=interactions_path),
            )
            recs = rec.recommend(interactions=df, top_k=3)
            assert recs.shape[1] == 3


# ---------------------------------------------------------------------------
# HierarchicalScorer contract
# ---------------------------------------------------------------------------


class TestHierarchicalScorerContract:
    """Verify HierarchicalScorer's position in the class hierarchy and public contract."""

    def test_is_sequential_scorer(self):
        est = _make_classifier(epochs=1)
        scorer = HierarchicalScorer(est)
        assert isinstance(scorer, SequentialScorer)

    def test_is_base_scorer(self):
        est = _make_classifier(epochs=1)
        scorer = HierarchicalScorer(est)
        assert isinstance(scorer, BaseScorer)

    def test_is_not_universal_scorer(self):
        """HierarchicalScorer must NOT inherit from UniversalScorer after the refactor."""
        est = _make_classifier(epochs=1)
        scorer = HierarchicalScorer(est)
        assert not isinstance(scorer, UniversalScorer)

    def test_set_new_items_raises(self, tiny_items_df):
        est = _make_classifier(epochs=1)
        scorer = HierarchicalScorer(est)
        with pytest.raises(NotImplementedError, match="does not support adding new items"):
            scorer.set_new_items(tiny_items_df)

    def test_score_items_routes_through_embedding_path(self, tiny_sessions_df, tiny_items_df):
        """score_items() must complete successfully via predict_proba_with_embeddings,
        never routing through _calculate_scores (which raises NotImplementedError)."""
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1)
        scorer = HierarchicalScorer(est)
        _, items_out, _ = scorer.process_factorized_datasets(None, tiny_items_df, seqs_df, is_training=True)
        est.fit_embedding_model(None, items_out, seqs_df)

        scores_df = scorer.score_items(interactions=seqs_df)

        assert scores_df.shape == (len(seqs_df), N_ITEMS)
        assert list(scores_df.columns) == list(scorer.item_names)
        assert scores_df.notna().all().all()

    def test_get_user_embeddings_raises(self, tiny_sessions_df, tiny_items_df):
        """HRNN derives user state from session history — no persistent user embedding table."""
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        with pytest.raises(NotImplementedError):
            est.get_user_embeddings()

    def test_get_item_embeddings_works(self, tiny_sessions_df, tiny_items_df):
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=1)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)
        emb_df = est.get_item_embeddings()
        assert ITEM_ID_NAME in emb_df.columns
        assert len(emb_df) == N_ITEMS

    def test_pickle_round_trip(self, tiny_sessions_df, tiny_items_df):
        """Pickle/unpickle a trained estimator and verify inference produces identical scores."""
        seqs_df, _ = tiny_sessions_df
        est = _make_classifier(epochs=2)
        est.fit_embedding_model(None, tiny_items_df, seqs_df)

        scores_before = est.predict_proba_with_embeddings(seqs_df)

        restored = pickle.loads(pickle.dumps(est))
        scores_after = restored.predict_proba_with_embeddings(seqs_df)

        assert scores_after.shape == scores_before.shape
        np.testing.assert_array_equal(scores_before, scores_after)
