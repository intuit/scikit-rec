"""Unit tests for native MatrixFactorizationEstimator (collaborative filtering)."""

import numpy as np
import pandas as pd
import pytest

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_EMBEDDING_NAME, USER_ID_NAME
from skrec.estimator.datatypes import MFAlgorithm, MFOutcomeType
from skrec.estimator.embedding.matrix_factorization_estimator import (
    MatrixFactorizationEstimator,
)


@pytest.fixture
def small_interactions():
    """Small (user, item, outcome) data for tests."""
    return pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u2", "u2", "u3", "u3"],
            ITEM_ID_NAME: ["i1", "i2", "i1", "i3", "i2", "i3"],
            LABEL_NAME: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        }
    )


@pytest.fixture
def small_items():
    """Item catalog aligned with interactions."""
    return pd.DataFrame(
        {
            ITEM_ID_NAME: ["i1", "i2", "i3"],
        }
    )


def test_fit_predict_shape(small_interactions, small_items):
    """Fit and predict return correct shapes and dtypes."""
    est = MatrixFactorizationEstimator(
        n_factors=4,
        regularization=0.1,
        epochs=5,
        random_state=42,
    )
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)

    # Predict on same (user, item) pairs
    scores = est.predict_proba_with_embeddings(small_interactions, users=None)
    assert scores.shape == (len(small_interactions),)
    assert scores.dtype == np.float64
    assert np.all(np.isfinite(scores))


def test_fit_without_items_df(small_interactions):
    """Fit works when items DataFrame is None (items inferred from interactions)."""
    est = MatrixFactorizationEstimator(n_factors=4, epochs=3, random_state=42)
    est.fit_embedding_model(users=None, items=None, interactions=small_interactions)
    scores = est.predict_proba_with_embeddings(small_interactions, users=None)
    assert len(scores) == len(small_interactions)


def test_get_user_embeddings(small_interactions, small_items):
    """get_user_embeddings returns DataFrame with USER_ID and EMBEDDING."""
    est = MatrixFactorizationEstimator(n_factors=4, epochs=3, random_state=42)
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)
    emb_df = est.get_user_embeddings()
    assert USER_ID_NAME in emb_df.columns
    assert USER_EMBEDDING_NAME in emb_df.columns
    assert len(emb_df) == est.unknown_user_idx_
    assert emb_df[USER_EMBEDDING_NAME].iloc[0].shape == (est.n_factors,)


def test_predict_with_external_embeddings(small_interactions, small_items):
    """Predict using users DataFrame with precomputed EMBEDDING column."""
    est = MatrixFactorizationEstimator(n_factors=4, epochs=5, random_state=42)
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)
    user_emb = est.get_user_embeddings()

    # One row per (user, item) for scoring
    interactions_for_score = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u2"],
            ITEM_ID_NAME: ["i1", "i2", "i3"],
        }
    )
    scores_with_emb = est.predict_proba_with_embeddings(interactions_for_score, users=user_emb)
    scores_batch = est.predict_proba_with_embeddings(interactions_for_score, users=None)
    np.testing.assert_allclose(scores_with_emb, scores_batch, rtol=1e-4)


def test_unknown_user_item(small_interactions, small_items):
    """Unknown user/item at prediction time use placeholder factors."""
    est = MatrixFactorizationEstimator(n_factors=4, epochs=3, random_state=42)
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)

    # Query with unknown user (not in training)
    inter_unknown_user = pd.DataFrame(
        {
            USER_ID_NAME: ["u_unknown"],
            ITEM_ID_NAME: ["i1"],
        }
    )
    scores = est.predict_proba_with_embeddings(inter_unknown_user, users=None)
    assert scores.shape == (1,)
    assert np.isfinite(scores[0])

    # Query with unknown item
    inter_unknown_item = pd.DataFrame(
        {
            USER_ID_NAME: ["u1"],
            ITEM_ID_NAME: ["i_unknown"],
        }
    )
    scores2 = est.predict_proba_with_embeddings(inter_unknown_item, users=None)
    assert scores2.shape == (1,)
    assert np.isfinite(scores2[0])


def test_raises_when_not_fitted():
    """Predict and get_user_embeddings raise if model not fitted."""
    est = MatrixFactorizationEstimator(n_factors=4, epochs=2)
    inter = pd.DataFrame({USER_ID_NAME: ["u1"], ITEM_ID_NAME: ["i1"]})
    with pytest.raises(RuntimeError, match="not fitted"):
        est.predict_proba_with_embeddings(inter, users=None)
    with pytest.raises(RuntimeError, match="not fitted"):
        est.get_user_embeddings()


def test_fit_raises_missing_columns():
    """fit_embedding_model raises when required columns are missing."""
    est = MatrixFactorizationEstimator(n_factors=4, epochs=2)
    bad = pd.DataFrame({"x": [1], "y": [2]})
    with pytest.raises(ValueError, match=LABEL_NAME):
        est.fit_embedding_model(users=None, items=None, interactions=bad)


def test_algorithm_enum(small_interactions, small_items):
    """Algorithm can be passed as MFAlgorithm enum (default is ALS)."""
    est = MatrixFactorizationEstimator(
        n_factors=4,
        algorithm=MFAlgorithm.ALS,
        epochs=2,
        random_state=42,
    )
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)
    scores = est.predict_proba_with_embeddings(small_interactions, users=None)
    assert len(scores) == len(small_interactions)


def test_algorithm_sgd_fit_predict(small_interactions, small_items):
    """MFAlgorithm.SGD fits and predicts (learning_rate used)."""
    est = MatrixFactorizationEstimator(
        n_factors=4,
        algorithm=MFAlgorithm.SGD,
        learning_rate=0.02,
        epochs=10,
        random_state=42,
    )
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)
    scores = est.predict_proba_with_embeddings(small_interactions, users=None)
    assert scores.shape == (len(small_interactions),)
    assert np.all(np.isfinite(scores))


def test_outcome_type_continuous(small_items):
    """CONTINUOUS outcome type: real-valued rewards, raw scores."""
    interactions = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u2", "u2"],
            ITEM_ID_NAME: ["i1", "i2", "i1", "i3"],
            LABEL_NAME: [2.5, 4.0, 3.0, 1.5],
        }
    )
    est = MatrixFactorizationEstimator(
        n_factors=4,
        outcome_type=MFOutcomeType.CONTINUOUS,
        epochs=5,
        random_state=42,
    )
    est.fit_embedding_model(users=None, items=small_items, interactions=interactions)
    scores = est.predict_proba_with_embeddings(interactions, users=None)
    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))


def test_outcome_type_binary_scores_in_0_1(small_interactions, small_items):
    """BINARY outcome type: predictions are probabilities in [0, 1]."""
    est = MatrixFactorizationEstimator(
        n_factors=4,
        outcome_type=MFOutcomeType.BINARY,
        algorithm=MFAlgorithm.SGD,
        epochs=15,
        random_state=42,
    )
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)
    scores = est.predict_proba_with_embeddings(small_interactions, users=None)
    assert scores.shape == (len(small_interactions),)
    assert np.all(scores >= 0) and np.all(scores <= 1), "Binary should output probabilities in [0,1]"


def test_outcome_type_binary_with_als(small_interactions, small_items):
    """BINARY with ALS: MSE on 0/1, sigmoid at predict so output in [0,1]."""
    est = MatrixFactorizationEstimator(
        n_factors=4,
        outcome_type=MFOutcomeType.BINARY,
        algorithm=MFAlgorithm.ALS,
        epochs=5,
        random_state=42,
    )
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)
    scores = est.predict_proba_with_embeddings(small_interactions, users=None)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_outcome_type_ordinal_1_to_5(small_items):
    """ORDINAL with 1–5 scale: MSE training, predictions clamped to [1, 5]."""
    interactions = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u1", "u2", "u2", "u3"],
            ITEM_ID_NAME: ["i1", "i2", "i1", "i3", "i2"],
            LABEL_NAME: [1, 4, 3, 5, 2],  # ordinal 1–5
        }
    )
    est = MatrixFactorizationEstimator(
        n_factors=4,
        outcome_type=MFOutcomeType.ORDINAL,
        ordinal_min=1.0,
        ordinal_max=5.0,
        epochs=8,
        random_state=42,
    )
    est.fit_embedding_model(users=None, items=small_items, interactions=interactions)
    scores = est.predict_proba_with_embeddings(interactions, users=None)
    assert scores.shape == (5,)
    assert np.all(scores >= 1) and np.all(scores <= 5), "Ordinal 1–5 should be clamped to [1, 5]"


def test_outcome_type_continuous_raw_scores(small_items):
    """CONTINUOUS: real-valued outcomes, no clamp."""
    interactions = pd.DataFrame(
        {
            USER_ID_NAME: ["u1", "u2"],
            ITEM_ID_NAME: ["i1", "i2"],
            LABEL_NAME: [0.5, 10.3],
        }
    )
    est = MatrixFactorizationEstimator(
        n_factors=4,
        outcome_type=MFOutcomeType.CONTINUOUS,
        epochs=5,
        random_state=42,
    )
    est.fit_embedding_model(users=None, items=small_items, interactions=interactions)
    scores = est.predict_proba_with_embeddings(interactions, users=None)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_embedding_estimator_rejected_by_independent_scorer():
    """IndependentScorer raises TypeError when given a BaseEmbeddingEstimator."""
    from skrec.scorer.independent import IndependentScorer

    est = MatrixFactorizationEstimator(n_factors=4, epochs=2)
    with pytest.raises(TypeError, match="IndependentScorer does not support BaseEmbeddingEstimator"):
        IndependentScorer(estimator=est)


def test_embedding_estimator_rejected_by_multiclass_scorer():
    """MulticlassScorer raises TypeError when given a BaseEmbeddingEstimator."""
    from skrec.scorer.multiclass import MulticlassScorer

    est = MatrixFactorizationEstimator(n_factors=4, epochs=2)
    with pytest.raises(TypeError, match="MulticlassScorer does not support BaseEmbeddingEstimator"):
        MulticlassScorer(estimator=est)


def test_predict_raises_on_none_interactions(small_interactions, small_items):
    """predict_proba_with_embeddings raises ValueError when interactions is None."""
    est = MatrixFactorizationEstimator(n_factors=4, epochs=2, random_state=42)
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)
    with pytest.raises(ValueError, match="interactions cannot be None"):
        est.predict_proba_with_embeddings(None, users=None)


@pytest.mark.parametrize("missing_col", [USER_ID_NAME, ITEM_ID_NAME])
def test_predict_raises_on_missing_required_column(missing_col, small_interactions, small_items):
    """predict_proba_with_embeddings raises ValueError when USER_ID or ITEM_ID is absent."""
    est = MatrixFactorizationEstimator(n_factors=4, epochs=2, random_state=42)
    est.fit_embedding_model(users=None, items=small_items, interactions=small_interactions)
    bad = small_interactions.drop(columns=[missing_col])
    with pytest.raises(ValueError, match=f"missing required column '{missing_col}'"):
        est.predict_proba_with_embeddings(bad, users=None)
