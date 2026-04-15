import logging

import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from numpy.testing import assert_array_equal
from scipy.stats import binomtest, chisquare

from skrec.recommender.bandits.strategy.epsilon_greedy import EpsilonGreedy
from skrec.recommender.bandits.strategy.static_action import StaticAction


@pytest.fixture
def setup_fixture():
    setup_fixture = {}
    setup_fixture["rng"] = default_rng(1108)
    return setup_fixture


def test_epsilon_greedy(setup_fixture):
    item_names = np.array(["a", "b", "c", "dddd"])

    # Typical scenario
    strategy = EpsilonGreedy(epsilon=0.5)
    scores = np.array(
        [
            [0.8, 0.8, 0.9, 0.7],
            [0.8, 0.8, 0.8, 0.5],
            [0.8, 0.4, 0.8, 0.3],
            [0.5, 0.4, 0.9, 0.4],
            [0.7, 0.9, 0.7, 0.3],
            [0.8, 0.4, 0.1, 0.0],
            [0.5, 0.6, 0.3, 0.1],
            [0.6, 0.6, 0.5, 0.0],
            [0.9, 0.5, 0.2, 0.4],
            [0.5, 0.9, 0.2, 0.5],
            [0.9, 0.2, 0.4, 0.5],
            [0.4, 0.6, 0.1, 0.1],
            [0.1, 0.0, 0.2, 0.1],
            [0.2, 0.9, 0.2, 0.5],
            [0.5, 0.0, 0.1, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )
    actual_rank, flags = strategy.rank(scores, item_names, top_k=2)
    expected = np.array(
        [
            ["c", "b", "exploit"],  # [0.8, 0.8, 0.9, 0.7],
            ["c", "dddd", "explore"],  # [0.8, 0.8, 0.8, 0.5],
            ["a", "c", "exploit"],  # [0.8, 0.4, 0.8, 0.3],
            ["c", "a", "exploit"],  # [0.5, 0.4, 0.9, 0.4],
            ["dddd", "b", "explore"],  # [0.7, 0.9, 0.7, 0.3],
            ["a", "b", "exploit"],  # [0.8, 0.4, 0.1, 0.0],
            ["b", "a", "exploit"],  # [0.5, 0.6, 0.3, 0.1],
            ["b", "a", "exploit"],  # [0.6, 0.6, 0.5, 0.0],
            ["dddd", "c", "explore"],  # [0.9, 0.5, 0.2, 0.4],
            ["dddd", "b", "explore"],  # [0.5, 0.9, 0.2, 0.5],
            ["dddd", "c", "explore"],  # [0.9, 0.2, 0.4, 0.5],
            ["b", "a", "exploit"],  # [0.4, 0.6, 0.1, 0.1],
            ["c", "a", "exploit"],  # [0.1, 0.0, 0.2, 0.1],
            ["b", "dddd", "exploit"],  # [0.2, 0.9, 0.2, 0.5],
            ["b", "dddd", "explore"],  # [0.5, 0.0, 0.1, 0.0],
            ["dddd", "a", "explore"],  # [0.5, 0.5, 0.5, 0.5],
            ["a", "b", "exploit"],  # [0.5, 0.5, 0.5, 0.5],
            ["a", "dddd", "explore"],  # [0.5, 0.5, 0.5, 0.5],
            ["dddd", "b", "exploit"],  # [0.5, 0.5, 0.5, 0.5],
            ["dddd", "b", "exploit"],  # [0.5, 0.5, 0.5, 0.5],
        ]
    )
    assert_array_equal(item_names[actual_rank], expected[:, :2])
    assert_array_equal(flags, expected[:, 2])

    # All explore
    strategy = EpsilonGreedy(epsilon=1, seed=52)
    scores = np.tile([4, 3, 2, 1], [1000, 1])
    actual_rank_indices, flags = strategy.rank(scores, item_names)
    counts = pd.Series(item_names[actual_rank_indices[:, 0]]).value_counts()
    result = chisquare(counts.values)
    assert result.pvalue > 0.05
    assert (flags == "explore").all()

    # All exploit, with 1 obvious answer
    strategy = EpsilonGreedy(epsilon=0, seed=62)
    scores = np.tile([4, 3, 2, 1], [1000, 1])
    actual_rank_indices, flags = strategy.rank(scores, item_names)
    assert (item_names[actual_rank_indices[:, 0]] == "a").all()
    assert (flags == "exploit").all()

    # All exploit, with 2 high scores
    strategy = EpsilonGreedy(epsilon=0, seed=72)
    scores = np.tile([4, 4, 1, 1], [1000, 1])
    actual_rank_indices, flags = strategy.rank(scores, item_names)
    counts = pd.Series(item_names[actual_rank_indices[:, 0]]).value_counts()
    assert set(counts.index.tolist()) == {"a", "b"}
    result = chisquare(counts.values)
    assert result.pvalue > 0.05

    # Check explore-exploit ratio
    eps = 0.2
    strategy = EpsilonGreedy(epsilon=eps, seed=82)
    scores = np.tile([4, 4, 1, 1], [1000, 1])
    ranked_items, flags = strategy.rank(scores, item_names)
    counts = pd.Series(flags).value_counts()
    result = binomtest(k=counts["explore"], n=counts.sum(), p=eps)
    assert result.pvalue > 0.05

    # Test 1D scores
    strategy = EpsilonGreedy(epsilon=0)
    scores = np.array([1, 4, 1, 1])
    actual_rank_indices, flags = strategy.rank(scores, item_names)
    assert_array_equal(item_names[actual_rank_indices], np.array([["b"]]))
    assert_array_equal(flags, np.array(["exploit"]))

    # Test error for item_list
    scores = np.array([[4, 4, 1], [1, 2, 3]])
    with pytest.raises(ValueError) as cm:
        ranked_items, flags = strategy.rank(scores, item_names)
    assert "Shape of scores (2, 3) does not match length of item_names 4" == str(cm.value)


def test_static_action(setup_fixture, caplog):
    item_names = np.array(sorted(["the best", "second best", "third best", "fourth best"]))

    scores = np.array(
        [
            [0.8, 0.8, 0.9, 0.7],
            [0.8, 0.8, 0.8, 0.5],
            [0.8, 0.4, 0.8, 0.3],
            [0.1, 0.4, 0.8, 0.9],
        ]
    )
    strategy = StaticAction(ranked_item_names=np.array(["the best", "second best", "third best", "fourth best"]))
    ranked_items, flags = strategy.rank(scores, item_names, top_k=2)
    expected = np.array(
        [
            ["the best", "second best"],
            ["the best", "second best"],
            ["the best", "second best"],
            ["the best", "second best"],
        ]
    )
    assert_array_equal(item_names[ranked_items], expected)
    assert (flags == "exploit").all()

    caplog.set_level(logging.WARNING, logger="skrec.recommender.bandits.strategy.static_action")
    with caplog.at_level(logging.WARNING, logger="skrec.recommender.bandits.strategy.static_action"):
        strategy = StaticAction(ranked_item_names=np.array(["aaa", "bbb", "ccc"]))
    expected = (
        "For static action strategy, ranked_item_names should be passed in best-to-worst order, "
        "but they are in alphabetical order. "
        "This could be a coincidence, but it is worth checking."
    )
    assert expected in caplog.text
