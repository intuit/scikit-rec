import warnings

import numpy as np
import pytest

from skrec.evaluator.base_evaluator import BaseRecommenderEvaluator
from skrec.evaluator.direct_method import DirectMethodEvaluator
from skrec.evaluator.doubly_robust import DREvaluator
from skrec.evaluator.inverse_propensity_score import IPSEvaluator
from skrec.evaluator.policy_weighted import PolicyWeightedEvaluator
from skrec.evaluator.replay_match import ReplayMatchEvaluator
from skrec.evaluator.simple import SimpleEvaluator
from skrec.evaluator.snips import SNIPSEvaluator
from skrec.metrics.datatypes import RecommenderMetricType


# --- Fixture ---
@pytest.fixture
def setup_fixture():
    """Provides evaluator inputs as dense NumPy matrices and metric info."""
    test_data = {}
    N = 3
    n_items = 6  # Max item ID is 5, so 6 items total (0-5)
    pad = BaseRecommenderEvaluator.ITEM_PAD_VALUE

    # Logged data (N, L_max=3)
    test_data["logged_items"] = np.array([[1, 2, 4], [2, 3, pad], [5, pad, pad]], dtype=int)
    # Use np.nan for reward/propensity padding
    test_data["logged_rewards"] = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, np.nan], [1.0, np.nan, np.nan]], dtype=float)
    test_data["logging_proba"] = np.array([[0.5, 0.3, 0.1], [0.7, 0.2, np.nan], [0.1, np.nan, np.nan]], dtype=float)

    # Recommendation Scores (N, n_items)
    test_data["recommendation_scores"] = np.array(
        [
            [1.0, 2.0, 8.0, 7.0, 9.0, 6.0],  # Recs: 4, 2, 3, 5, 1, 0
            [1.0, 9.0, 0.0, 8.0, 1.0, 7.0],  # Recs: 1, 3, 5, 0, 4, 2
            [8.0, 9.0, 1.0, 7.0, 2.0, 3.0],  # Recs: 1, 0, 3, 5, 4, 2
        ],
        dtype=float,
    )

    # Expected Rewards (N, n_items) - Required for DR
    expected_rewards_dense = np.full((N, n_items), 0.1, dtype=float)
    expected_rewards_dense[0, [1, 2, 4]] = [0.6, 0.3, 0.2]
    expected_rewards_dense[1, [2, 3]] = [0.5, 0.4]
    expected_rewards_dense[2, 5] = 0.3
    test_data["expected_rewards"] = expected_rewards_dense

    # Calculate recommendation_probas (simplified softmax with T=1.0)
    scores = test_data["recommendation_scores"]
    exp_scores = np.exp(scores)
    recommendation_probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    test_data["recommendation_probas"] = recommendation_probas
    sorted_indices = np.argsort(-scores, axis=1)
    ranks = np.empty_like(sorted_indices)
    np.put_along_axis(ranks, sorted_indices, np.arange(n_items), axis=1)
    test_data["recommendation_ranks"] = ranks

    # Metric info
    test_data["metric_type"] = RecommenderMetricType.PRECISION_AT_K
    test_data["top_k"] = 3

    return test_data


# --- SimpleEvaluator ---


def test_simple(setup_fixture):
    evaluator = SimpleEvaluator()
    expected_metric = (1 / 3 + 0 + 0) / 3

    modified_rewards = evaluator._compute_modified_rewards(
        recommendation_scores=setup_fixture["recommendation_scores"],
        recommendation_probas=setup_fixture["recommendation_probas"],
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
    )
    metric_result = evaluator.evaluate(
        modified_rewards=modified_rewards,
        recommendation_ranks=setup_fixture["recommendation_ranks"],
        recommendation_scores=setup_fixture["recommendation_scores"],
        metric_type=setup_fixture["metric_type"],
        top_k=setup_fixture["top_k"],
    )
    assert metric_result == pytest.approx(expected_metric)


def test_simple_evaluate_is_stateless(setup_fixture):
    """evaluate() called twice with the same args must return the same result."""
    evaluator = SimpleEvaluator()
    modified_rewards = evaluator._compute_modified_rewards(
        recommendation_scores=setup_fixture["recommendation_scores"],
        recommendation_probas=setup_fixture["recommendation_probas"],
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
    )
    kwargs = dict(
        modified_rewards=modified_rewards,
        recommendation_ranks=setup_fixture["recommendation_ranks"],
        recommendation_scores=setup_fixture["recommendation_scores"],
        metric_type=setup_fixture["metric_type"],
        top_k=setup_fixture["top_k"],
    )
    assert evaluator.evaluate(**kwargs) == evaluator.evaluate(**kwargs)


# --- ReplayMatchEvaluator ---


def test_replay_match(setup_fixture):
    evaluator = ReplayMatchEvaluator()
    expected_metric = (1 / 2 + 0) / 2

    modified_rewards = evaluator._compute_modified_rewards(
        recommendation_scores=setup_fixture["recommendation_scores"],
        recommendation_probas=setup_fixture["recommendation_probas"],
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
    )
    metric_result = evaluator.evaluate(
        modified_rewards=modified_rewards,
        recommendation_ranks=setup_fixture["recommendation_ranks"],
        recommendation_scores=setup_fixture["recommendation_scores"],
        metric_type=setup_fixture["metric_type"],
        top_k=setup_fixture["top_k"],
    )
    assert metric_result == pytest.approx(expected_metric)


# --- IPSEvaluator ---


def test_IPS(setup_fixture):
    tp = setup_fixture["recommendation_probas"]

    rips_u0_i2 = 0.0
    rips_u0_i4 = 1.0 * tp[0, 4] / 0.1
    rips_u1_i3 = 0.0

    expected_metric_no_trim = ((rips_u0_i4 + rips_u0_i2) / 2 + (rips_u1_i3) / 1) / 2

    ips_evaluator_no_trim = IPSEvaluator()
    modified_rewards = ips_evaluator_no_trim._compute_modified_rewards(
        recommendation_scores=setup_fixture["recommendation_scores"],
        recommendation_probas=tp,
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
        logging_proba=setup_fixture["logging_proba"],
    )
    metric_result_no_trim = ips_evaluator_no_trim.evaluate(
        modified_rewards=modified_rewards,
        recommendation_ranks=setup_fixture["recommendation_ranks"],
        recommendation_scores=setup_fixture["recommendation_scores"],
        metric_type=setup_fixture["metric_type"],
        top_k=setup_fixture["top_k"],
    )
    assert metric_result_no_trim == pytest.approx(expected_metric_no_trim)

    # --- With Trimming ---
    trim_threshold = 2.0
    assert tp[0, 1] / 0.5 < trim_threshold
    assert tp[0, 4] / 0.1 > trim_threshold
    expected_metric_trim = ((rips_u0_i2) / 1 + (rips_u1_i3) / 1) / 2

    ips_evaluator_trim = IPSEvaluator(trim_threshold=trim_threshold)
    modified_rewards_trim = ips_evaluator_trim._compute_modified_rewards(
        recommendation_scores=setup_fixture["recommendation_scores"],
        recommendation_probas=tp,
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
        logging_proba=setup_fixture["logging_proba"],
    )
    metric_result_trim = ips_evaluator_trim.evaluate(
        modified_rewards=modified_rewards_trim,
        recommendation_ranks=setup_fixture["recommendation_ranks"],
        recommendation_scores=setup_fixture["recommendation_scores"],
        metric_type=setup_fixture["metric_type"],
        top_k=setup_fixture["top_k"],
    )
    assert metric_result_trim == pytest.approx(expected_metric_trim)

    # --- Error Handling ---
    with pytest.raises(ValueError, match="Logging propensities"):
        IPSEvaluator()._compute_modified_rewards(
            recommendation_scores=setup_fixture["recommendation_scores"],
            recommendation_probas=tp,
            logged_items=setup_fixture["logged_items"],
            logged_rewards=setup_fixture["logged_rewards"],
            logging_proba=None,
        )

    with pytest.raises(ValueError, match="trim_threshold must be positive"):
        IPSEvaluator(trim_threshold=0.0)
    with pytest.raises(ValueError, match="trim_threshold must be positive"):
        IPSEvaluator(trim_threshold=-1.0)


# --- DREvaluator ---


def test_DR(setup_fixture):
    tp = setup_fixture["recommendation_probas"]
    expected_rewards = setup_fixture["expected_rewards"]

    dm0 = (tp[0, :] * expected_rewards[0, :]).sum()
    dr02 = dm0 + (tp[0, 2] / 0.3) * (0.0 - 0.3)
    dr04 = dm0 + (tp[0, 4] / 0.1) * (1.0 - 0.2)
    dm1 = (tp[1, :] * expected_rewards[1, :]).sum()
    dr13 = dm1 + (tp[1, 3] / 0.2) * (0.0 - 0.4)

    p3_u0 = (dr04 + dr02) / 2
    p3_u1 = (dr13) / 1
    expected_metric = (p3_u0 + p3_u1) / 2

    dr_evaluator = DREvaluator()
    modified_rewards = dr_evaluator._compute_modified_rewards(
        recommendation_scores=setup_fixture["recommendation_scores"],
        recommendation_probas=tp,
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
        logging_proba=setup_fixture["logging_proba"],
        expected_rewards=expected_rewards,
    )
    metric_result = dr_evaluator.evaluate(
        modified_rewards=modified_rewards,
        recommendation_ranks=setup_fixture["recommendation_ranks"],
        recommendation_scores=setup_fixture["recommendation_scores"],
        metric_type=setup_fixture["metric_type"],
        top_k=setup_fixture["top_k"],
    )
    assert metric_result == pytest.approx(expected_metric, abs=1e-6)

    # --- Error Handling ---
    with pytest.raises(ValueError, match="Expected rewards"):
        DREvaluator()._compute_modified_rewards(
            recommendation_scores=setup_fixture["recommendation_scores"],
            recommendation_probas=tp,
            logged_items=setup_fixture["logged_items"],
            logged_rewards=setup_fixture["logged_rewards"],
            logging_proba=setup_fixture["logging_proba"],
            expected_rewards=None,
        )


# --- _compute_modified_rewards dispatch ---


def test_compute_modified_rewards_probabilistic_uses_probas(setup_fixture):
    """IPSEvaluator must use recommendation_probas as target_proba, not scores."""
    tp = setup_fixture["recommendation_probas"]
    scores = setup_fixture["recommendation_scores"]

    evaluator = IPSEvaluator()

    # Using correct probas
    result_probas = evaluator._compute_modified_rewards(
        recommendation_scores=scores,
        recommendation_probas=tp,
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
        logging_proba=setup_fixture["logging_proba"],
    )

    # Using scores as probas (wrong — should produce different result)
    result_scores_as_probas = evaluator._compute_modified_rewards(
        recommendation_scores=scores,
        recommendation_probas=scores,  # intentionally wrong
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
        logging_proba=setup_fixture["logging_proba"],
    )

    assert not np.allclose(result_probas, result_scores_as_probas, equal_nan=True)


def test_compute_modified_rewards_non_probabilistic_uses_scores(setup_fixture):
    """SimpleEvaluator must use recommendation_scores as target_proba."""
    tp = setup_fixture["recommendation_probas"]
    scores = setup_fixture["recommendation_scores"]

    evaluator = SimpleEvaluator()

    # Probas argument is ignored for non-probabilistic evaluators;
    # result should be the same regardless of what is passed for recommendation_probas.
    result_with_probas = evaluator._compute_modified_rewards(
        recommendation_scores=scores,
        recommendation_probas=tp,
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
    )
    result_with_none_probas = evaluator._compute_modified_rewards(
        recommendation_scores=scores,
        recommendation_probas=None,
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
    )
    np.testing.assert_array_equal(result_with_probas, result_with_none_probas)


# --- Helpers shared by SNIPS and PolicyWeighted tests ---
def _make_all_pad_logged_items(N, L_max):
    pad = BaseRecommenderEvaluator.ITEM_PAD_VALUE
    return np.full((N, L_max), pad, dtype=int)


def _make_target_proba(N, n_items):
    rng = np.random.default_rng(42)
    raw = rng.random((N, n_items))
    return raw / raw.sum(axis=1, keepdims=True)


# --- SNIPS tests ---


def test_snips_empty_logs_returns_all_nan():
    """When every logged_items entry is padding, _calculate_modified_rewards must
    return an all-NaN matrix without raising a RuntimeWarning."""
    N, n_items, L_max = 4, 10, 3
    logged_items = _make_all_pad_logged_items(N, L_max)
    logged_rewards = np.full((N, L_max), np.nan)
    logging_proba = np.full((N, L_max), np.nan)
    target_proba = _make_target_proba(N, n_items)

    evaluator = SNIPSEvaluator()
    with warnings.catch_warnings(record=True) as record:
        result = evaluator._calculate_modified_rewards(
            logged_items=logged_items,
            logged_rewards=logged_rewards,
            target_proba=target_proba,
            logging_proba=logging_proba,
        )

    assert result.shape == (N, n_items)
    assert np.all(np.isnan(result))
    runtime_warnings = [w for w in record if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0, "Expected no RuntimeWarning for empty logs"


def test_snips_basic_correctness(setup_fixture):
    """SNIPS modified rewards satisfy the standard SNIPS formula:
    nanmean(modified_rewards) == sum(r_i * w_i) / sum(w_i)."""
    tp = setup_fixture["recommendation_probas"]
    lp = setup_fixture["logging_proba"]
    logged_items = setup_fixture["logged_items"]
    logged_rewards = setup_fixture["logged_rewards"]

    evaluator = SNIPSEvaluator()
    result = evaluator._calculate_modified_rewards(
        logged_items=logged_items,
        logged_rewards=logged_rewards,
        target_proba=tp,
        logging_proba=lp,
    )

    assert result.shape == tp.shape

    valid_mask = logged_items != BaseRecommenderEvaluator.ITEM_PAD_VALUE
    row_idx, log_col_idx = np.where(valid_mask)
    col_idx = logged_items[row_idx, log_col_idx].astype(int)
    rewards = logged_rewards[row_idx, log_col_idx]
    lp_valid = lp[row_idx, log_col_idx]
    tp_valid = tp[row_idx, col_idx]
    weights = tp_valid / lp_valid
    expected_snips = np.nansum(rewards * weights) / np.nansum(weights)

    assert np.nanmean(result) == pytest.approx(expected_snips, rel=1e-6)


def test_snips_missing_logging_proba_raises(setup_fixture):
    evaluator = SNIPSEvaluator()
    with pytest.raises(ValueError, match="logging_proba"):
        evaluator._calculate_modified_rewards(
            logged_items=setup_fixture["logged_items"],
            logged_rewards=setup_fixture["logged_rewards"],
            target_proba=setup_fixture["recommendation_probas"],
            logging_proba=None,
        )


# --- PolicyWeighted tests ---


def test_policy_weighted_empty_logs_returns_all_nan():
    """When every logged_items entry is padding, _calculate_modified_rewards must
    return an all-NaN matrix without raising a RuntimeWarning."""
    N, n_items, L_max = 4, 10, 3
    logged_items = _make_all_pad_logged_items(N, L_max)
    logged_rewards = np.full((N, L_max), np.nan)
    target_proba = _make_target_proba(N, n_items)

    evaluator = PolicyWeightedEvaluator()
    with warnings.catch_warnings(record=True) as record:
        result = evaluator._calculate_modified_rewards(
            logged_items=logged_items,
            logged_rewards=logged_rewards,
            target_proba=target_proba,
        )

    assert result.shape == (N, n_items)
    assert np.all(np.isnan(result))
    runtime_warnings = [w for w in record if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0, "Expected no RuntimeWarning for empty logs"


def test_policy_weighted_basic_correctness(setup_fixture):
    """PolicyWeighted modified rewards: nanmean == sum(r*tp) / sum(tp)."""
    tp = setup_fixture["recommendation_probas"]
    logged_items = setup_fixture["logged_items"]
    logged_rewards = setup_fixture["logged_rewards"]

    evaluator = PolicyWeightedEvaluator()
    result = evaluator._calculate_modified_rewards(
        logged_items=logged_items,
        logged_rewards=logged_rewards,
        target_proba=tp,
    )

    assert result.shape == tp.shape

    valid_mask = logged_items != BaseRecommenderEvaluator.ITEM_PAD_VALUE
    row_idx, log_col_idx = np.where(valid_mask)
    col_idx = logged_items[row_idx, log_col_idx].astype(int)
    rewards = logged_rewards[row_idx, log_col_idx]
    tp_valid = tp[row_idx, col_idx]
    expected = np.nansum(rewards * tp_valid) / np.nansum(tp_valid)

    assert np.nanmean(result) == pytest.approx(expected, rel=1e-6)


# --- DirectMethod tests ---


def test_direct_method_correctness():
    n_items = 3
    pad = BaseRecommenderEvaluator.ITEM_PAD_VALUE
    logged_items = np.array([[0, 1], [2, pad]], dtype=int)
    logged_rewards = np.array([[1.0, 0.0], [1.0, np.nan]])
    target_proba = np.array([[0.4, 0.4, 0.2], [0.3, 0.5, 0.2]])
    expected_rewards = np.array([[0.5, 0.2, 0.3], [0.1, 0.7, 0.2]])

    result = DirectMethodEvaluator()._calculate_modified_rewards(
        logged_items=logged_items,
        logged_rewards=logged_rewards,
        target_proba=target_proba,
        expected_rewards=expected_rewards,
    )

    # modified_reward[u,i] = n_items * expected_reward[u,i] * target_proba[u,i]
    np.testing.assert_array_almost_equal(result, n_items * expected_rewards * target_proba)


def test_direct_method_missing_expected_raises():
    pad = BaseRecommenderEvaluator.ITEM_PAD_VALUE
    logged_items = np.array([[0, 1], [2, pad]], dtype=int)
    logged_rewards = np.array([[1.0, 0.0], [1.0, np.nan]])
    target_proba = np.array([[0.4, 0.4, 0.2], [0.3, 0.5, 0.2]])

    with pytest.raises(ValueError, match="Expected rewards are required"):
        DirectMethodEvaluator()._calculate_modified_rewards(
            logged_items=logged_items,
            logged_rewards=logged_rewards,
            target_proba=target_proba,
            expected_rewards=None,
        )


# --- Evaluator/metric compatibility guard ---


def test_classification_metric_with_counterfactual_raises(setup_fixture):
    """IPS (PRESERVES_LOGGED_REWARD=False) + ROC AUC must raise ValueError."""
    tp = setup_fixture["recommendation_probas"]
    modified_rewards = IPSEvaluator()._compute_modified_rewards(
        recommendation_scores=setup_fixture["recommendation_scores"],
        recommendation_probas=tp,
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
        logging_proba=setup_fixture["logging_proba"],
    )
    with pytest.raises(ValueError, match="Classification metric"):
        IPSEvaluator().evaluate(
            modified_rewards=modified_rewards,
            recommendation_ranks=setup_fixture["recommendation_ranks"],
            recommendation_scores=setup_fixture["recommendation_scores"],
            metric_type=RecommenderMetricType.ROC_AUC,
            top_k=setup_fixture["top_k"],
        )


def test_classification_metric_with_counterfactual_pr_auc_raises(setup_fixture):
    """SNIPS (PRESERVES_LOGGED_REWARD=False) + PR AUC must raise ValueError."""
    tp = setup_fixture["recommendation_probas"]
    modified_rewards = SNIPSEvaluator()._compute_modified_rewards(
        recommendation_scores=setup_fixture["recommendation_scores"],
        recommendation_probas=tp,
        logged_items=setup_fixture["logged_items"],
        logged_rewards=setup_fixture["logged_rewards"],
        logging_proba=setup_fixture["logging_proba"],
    )
    with pytest.raises(ValueError, match="Classification metric"):
        SNIPSEvaluator().evaluate(
            modified_rewards=modified_rewards,
            recommendation_ranks=setup_fixture["recommendation_ranks"],
            recommendation_scores=setup_fixture["recommendation_scores"],
            metric_type=RecommenderMetricType.PR_AUC,
            top_k=setup_fixture["top_k"],
        )
