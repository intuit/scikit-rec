import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from skrec.metrics.factory import RecommenderMetricFactory


@pytest.fixture
def setup_fixture():
    """Provides metric instances, sample modified rewards, ranks, and scores."""
    test_data = {}
    test_data["precision"] = RecommenderMetricFactory.create("precision_at_k")
    test_data["recall"] = RecommenderMetricFactory.create("recall_at_k")
    test_data["average_reward"] = RecommenderMetricFactory.create("average_reward_at_k")
    test_data["MAP"] = RecommenderMetricFactory.create("MAP_at_k")
    test_data["MRR"] = RecommenderMetricFactory.create("MRR_at_k")
    test_data["NDCG"] = RecommenderMetricFactory.create("NDCG_at_k")
    # Sample matrix: N=2 users, n_items=4 recommendations
    rewards = np.array([[0, np.nan, 1, 1], [1, 0, 1, 0]], dtype=float)
    test_data["rewards"] = rewards

    # Generate ranks and scores based on rewards shape
    N, n_items = rewards.shape
    ranks = np.tile(np.arange(n_items), (N, 1))
    # Create scores inversely proportional to rank (higher score = better rank)
    scores = np.tile(np.arange(n_items, 0, -1), (N, 1)).astype(float)
    test_data["ranks"] = ranks
    test_data["scores"] = scores

    return test_data


def test_precision(setup_fixture):
    modified_rewards = setup_fixture["rewards"]
    precision_metric = setup_fixture["precision"]
    ranks = setup_fixture["ranks"]
    scores = setup_fixture["scores"]
    top_k_full = modified_rewards.shape[1]

    # Expected for k=4: (nanmean([0,nan,1,1]) + nanmean([1,0,1,0])) / 2 = (2/3 + 1/2) / 2 = 7/12
    assert precision_metric.calculate(ranks, modified_rewards, scores, top_k=top_k_full) == pytest.approx(7 / 12)
    # Expected for k=3: (nanmean([0,nan,1]) + nanmean([1,0,1])) / 2 = (1/2 + 2/3) / 2 = 7/12
    assert precision_metric.calculate(ranks, modified_rewards, scores, top_k=3) == pytest.approx(7 / 12)
    # Expected for k=1: (nanmean([0]) + nanmean([1])) / 2 = (0 + 1) / 2 = 1/2
    assert precision_metric.calculate(ranks, modified_rewards, scores, top_k=1) == pytest.approx(1 / 2)
    # Test k=0 edge case
    assert precision_metric.calculate(ranks, modified_rewards, scores, top_k=0) == 0.0


def test_average_reward(setup_fixture):
    """Tests AverageReward@k calculation (alias for Precision@k)."""
    modified_rewards = setup_fixture["rewards"]
    avg_reward_metric = setup_fixture["average_reward"]
    ranks = setup_fixture["ranks"]
    scores = setup_fixture["scores"]
    top_k_full = modified_rewards.shape[1]

    # Should yield same results as precision
    assert avg_reward_metric.calculate(ranks, modified_rewards, scores, top_k=top_k_full) == pytest.approx(7 / 12)
    assert avg_reward_metric.calculate(ranks, modified_rewards, scores, top_k=3) == pytest.approx(7 / 12)
    assert avg_reward_metric.calculate(ranks, modified_rewards, scores, top_k=1) == pytest.approx(1 / 2)
    assert avg_reward_metric.calculate(ranks, modified_rewards, scores, top_k=0) == 0.0


def test_MAP(setup_fixture):
    modified_rewards = setup_fixture["rewards"]
    map_metric = setup_fixture["MAP"]
    ranks = setup_fixture["ranks"]
    scores = setup_fixture["scores"]
    top_k_full = modified_rewards.shape[1]

    # Expected MAP@4: Average of AP@4 for each user.
    # User 0 AP@4=7/12, User 1 AP@4=5/6. MAP = (7/12 + 5/6)/2 = 17/24.
    assert map_metric.calculate(ranks, modified_rewards, scores, top_k=top_k_full) == pytest.approx(17 / 24)
    # Expected MAP@2: User 0 AP@2=0 (no relevant in [0, nan]),
    # User 1 AP@2=1 (relevant at rank 1 in [1, 0]) -> MAP = (0+1)/2 = 1/2
    # k=2, User 0: rewards=[0,nan] Valid=[0] Relevant=[] AP=0.
    # k=2, User 1: rewards=[1,0] Valid=[1,0] Relevant=[1] AP=1/1=1.
    assert map_metric.calculate(ranks, modified_rewards, scores, top_k=2) == pytest.approx(1 / 2)
    # Test k=0 edge case
    assert map_metric.calculate(ranks, modified_rewards, scores, top_k=0) == 0.0


def test_MRR(setup_fixture):
    modified_rewards = setup_fixture["rewards"]
    mrr_metric = setup_fixture["MRR"]
    ranks = setup_fixture["ranks"]
    scores = setup_fixture["scores"]
    top_k_full = modified_rewards.shape[1]

    # Expected MRR@4: User 0 RR=1/3 (first relevant at rank 3),
    # User 1 RR=1/1 (first relevant at rank 1) -> MRR = (1/3 + 1)/2 = 2/3
    assert mrr_metric.calculate(ranks, modified_rewards, scores, top_k=top_k_full) == pytest.approx(2 / 3)
    # Expected MRR@2: User 0 RR=0 (no relevant in [0, nan]),
    # User 1 RR=1/1 (first relevant at rank 1) -> MRR = (0+1)/2 = 1/2
    assert mrr_metric.calculate(ranks, modified_rewards, scores, top_k=2) == pytest.approx(1 / 2)
    # Test k=0 edge case
    assert mrr_metric.calculate(ranks, modified_rewards, scores, top_k=0) == 0.0


def test_recall(setup_fixture):
    modified_rewards = setup_fixture["rewards"]
    recall_metric = setup_fixture["recall"]
    ranks = setup_fixture["ranks"]
    scores = setup_fixture["scores"]

    # rewards = [[0, nan, 1, 1], [1, 0, 1, 0]]
    # User 1: relevant (>0, non-NaN) at cols 2,3 → total_relevant=2
    # User 2: relevant at cols 0,2 → total_relevant=2
    # k=4: (2/2 + 2/2) / 2 = 1.0
    assert recall_metric.calculate(ranks, modified_rewards, scores, top_k=4) == pytest.approx(1.0)
    # k=3: user1 gets col 2 only (1/2), user2 gets cols 0,2 (2/2) → (0.5 + 1.0)/2 = 0.75
    assert recall_metric.calculate(ranks, modified_rewards, scores, top_k=3) == pytest.approx(0.75)
    # k=1: user1 gets col 0 (value=0, not relevant) (0/2), user2 gets col 0 (1/2) → (0+0.5)/2 = 0.25
    assert recall_metric.calculate(ranks, modified_rewards, scores, top_k=1) == pytest.approx(0.25)
    # k=0 edge case
    assert recall_metric.calculate(ranks, modified_rewards, scores, top_k=0) == 0.0

    # All-NaN rewards: no user has relevant items → aggregate recall is undefined (nan)
    all_nan = np.full_like(modified_rewards, np.nan)
    assert np.isnan(recall_metric.calculate(ranks, all_nan, scores, top_k=4))


def test_NDCG(setup_fixture):
    ndcg_metric = setup_fixture["NDCG"]
    # Use a matrix with non-binary rewards
    modified_rewards = np.array([[2, 4, np.nan, 5], [3, 1, 0, 2]], dtype=float)
    top_k_full = modified_rewards.shape[1]
    # Generate ranks and scores specific to this reward matrix shape
    N, n_items = modified_rewards.shape
    ranks = np.tile(np.arange(n_items), (N, 1))
    scores = np.tile(np.arange(n_items, 0, -1), (N, 1)).astype(float)

    # Expected values calculated manually based on DCG/IDCG formulas below
    # Note: Ranks are 0-based, so positions for log are rank+2
    log2_2 = np.log2(2)  # Rank 0 -> Pos 1 -> log2(1+1)
    log2_3 = np.log2(3)  # Rank 1 -> Pos 2 -> log2(2+1)
    log2_4 = np.log2(4)  # Rank 2 -> Pos 3 -> log2(3+1)
    log2_5 = np.log2(5)  # Rank 3 -> Pos 4 -> log2(4+1)

    # k=4 calculations
    # User 0: Ranks=[0,1,2,3], Rewards=[2, 4, nan, 5]. Sorted Rewards=[5, 4, 2, nan]
    dcg_0 = 2 / log2_2 + 4 / log2_3 + 0 / log2_4 + 5 / log2_5  # Use 0 for nan reward
    idcg_0 = 5 / log2_2 + 4 / log2_3 + 2 / log2_4 + 0 / log2_5
    ndcg_0 = dcg_0 / idcg_0 if idcg_0 else 0
    # User 1: Ranks=[0,1,2,3], Rewards=[3, 1, 0, 2]. Sorted Rewards=[3, 2, 1, 0]
    dcg_1 = 3 / log2_2 + 1 / log2_3 + 0 / log2_4 + 2 / log2_5
    idcg_1 = 3 / log2_2 + 2 / log2_3 + 1 / log2_4 + 0 / log2_5
    ndcg_1 = dcg_1 / idcg_1 if idcg_1 else 0
    expected_ndcg_k4 = (ndcg_0 + ndcg_1) / 2

    # k=3 calculations (consider ranks 0, 1, 2)
    # User 0: Ranks=[0,1,2], Rewards=[2, 4, nan]. DCG uses [2, 4, 0].
    #         Ideal sort of [2, 4, nan, 5] is [5, 4, 2, nan]. Top 3 are [5, 4, 2].
    ndcg_0_k3 = (2 + 4 / np.log2(3)) / (5 + 4 / np.log2(3) + 2 / np.log2(4))

    # User 1: Ranks=[0,1,2], Rewards=[3, 1, 0]. DCG uses [3, 1, 0].
    #         Ideal sort of [3, 1, 0, 2] is [3, 2, 1, 0]. Top 3 are [3, 2, 1].
    ndcg_1_k3 = (3 / log2_2 + 1 / log2_3 + 0) / (3 / log2_2 + 2 / log2_3 + 1 / log2_4)
    expected_ndcg_k3 = (ndcg_0_k3 + ndcg_1_k3) / 2

    assert ndcg_metric.calculate(ranks, modified_rewards, scores, top_k=top_k_full) == pytest.approx(expected_ndcg_k4)
    assert ndcg_metric.calculate(ranks, modified_rewards, scores, top_k=3) == pytest.approx(expected_ndcg_k3)
    # Test k=0 edge case
    assert ndcg_metric.calculate(ranks, modified_rewards, scores, top_k=0) == 0.0


def test_MAP_non_binary_raises(setup_fixture):
    map_metric = setup_fixture["MAP"]
    ranks = setup_fixture["ranks"]
    scores = setup_fixture["scores"]
    non_binary = np.array([[0.5, 0.8, 1.0, 0.0], [1.0, 0.3, 0.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="binary rewards"):
        map_metric.calculate(ranks, non_binary, scores, top_k=4)


def test_MRR_non_binary_raises(setup_fixture):
    mrr_metric = setup_fixture["MRR"]
    ranks = setup_fixture["ranks"]
    scores = setup_fixture["scores"]
    non_binary = np.array([[0.5, 0.8, 1.0, 0.0], [1.0, 0.3, 0.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="binary rewards"):
        mrr_metric.calculate(ranks, non_binary, scores, top_k=4)


def test_roc_auc():
    roc_auc = RecommenderMetricFactory.create("roc_auc")
    # N=2, n_items=4: positives at cols 0 and 2, higher scores than negatives
    ranks = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    scores = np.array([[0.9, 0.3, 0.8, 0.1], [0.9, 0.3, 0.8, 0.1]])
    rewards = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])

    # Perfect separation → AUC = 1.0; verify against sklearn directly
    y_true_flat = rewards.ravel()
    y_score_flat = scores.ravel()
    expected = roc_auc_score(y_true_flat.astype(int), y_score_flat)
    assert roc_auc.calculate(ranks, rewards, scores) == pytest.approx(expected)

    # NaN entries ignored: only cols 0 and 2 survive, still perfect separation → 1.0
    rewards_with_nan = np.array([[1.0, np.nan, 0.0, np.nan], [1.0, np.nan, 0.0, np.nan]])
    assert roc_auc.calculate(ranks, rewards_with_nan, scores) == pytest.approx(1.0)

    # All NaN → 0.0
    assert roc_auc.calculate(ranks, np.full_like(rewards, np.nan), scores) == 0.0

    # Single class present → 0.0
    all_positive = np.ones_like(rewards)
    assert roc_auc.calculate(ranks, all_positive, scores) == 0.0

    # top_k ignored (classification metric)
    assert roc_auc.calculate(ranks, rewards, scores, top_k=2) == pytest.approx(expected)

    # Out-of-range rewards raise ValueError
    oor = np.array([[1.5, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="modified_rewards in"):
        roc_auc.calculate(ranks, oor, scores)
    neg = np.array([[-0.1, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="modified_rewards in"):
        roc_auc.calculate(ranks, neg, scores)


def test_pr_auc():
    pr_auc = RecommenderMetricFactory.create("pr_auc")
    ranks = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    scores = np.array([[0.9, 0.3, 0.8, 0.1], [0.9, 0.3, 0.8, 0.1]])
    rewards = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])

    # Verify against sklearn directly
    y_true_flat = rewards.ravel().astype(int)
    y_score_flat = scores.ravel()
    expected = average_precision_score(y_true_flat, y_score_flat)
    assert pr_auc.calculate(ranks, rewards, scores) == pytest.approx(expected)

    # NaN entries ignored
    rewards_with_nan = np.array([[1.0, np.nan, 0.0, np.nan], [1.0, np.nan, 0.0, np.nan]])
    y_true_valid = np.array([1, 0, 1, 0])
    y_score_valid = np.array([0.9, 0.8, 0.9, 0.8])
    expected_nan = average_precision_score(y_true_valid, y_score_valid)
    assert pr_auc.calculate(ranks, rewards_with_nan, scores) == pytest.approx(expected_nan)

    # All NaN → 0.0
    assert pr_auc.calculate(ranks, np.full_like(rewards, np.nan), scores) == 0.0

    # top_k ignored (classification metric)
    assert pr_auc.calculate(ranks, rewards, scores, top_k=2) == pytest.approx(expected)

    # Out-of-range rewards raise ValueError
    oor = np.array([[1.5, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="modified_rewards in"):
        pr_auc.calculate(ranks, oor, scores)
    neg = np.array([[-0.1, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="modified_rewards in"):
        pr_auc.calculate(ranks, neg, scores)
