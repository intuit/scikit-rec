"""Unit tests for :class:`RecommenderEvaluationSession`."""

import numpy as np
import pytest

from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.evaluator.evaluation_session import RecommenderEvaluationSession
from skrec.metrics.datatypes import RecommenderMetricType


def _process_eval_kwargs(eval_kwargs):
    return dict(eval_kwargs) if eval_kwargs else {}


def _build_bundle(_kwargs, temperature):
    del temperature
    scores = np.array([[0.2, 0.5, 0.3], [0.4, 0.3, 0.3]], dtype=np.float64)
    probas = np.ones_like(scores) / scores.shape[1]
    n_items = scores.shape[1]
    recommended_idx = np.tile(np.arange(n_items, dtype=np.int_), (scores.shape[0], 1))
    ranks = np.empty_like(recommended_idx)
    np.put_along_axis(ranks, recommended_idx, np.arange(n_items), axis=1)
    return scores, probas, ranks


def test_session_raises_without_cached_scores():
    session = RecommenderEvaluationSession(process_eval_kwargs=_process_eval_kwargs)
    with pytest.raises(ValueError, match="No cached recommendation scores"):
        session.evaluate(
            eval_type=RecommenderEvaluatorType.SIMPLE,
            metric_type=RecommenderMetricType.PRECISION_AT_K,
            eval_top_k=2,
            temperature=1.0,
            eval_kwargs={"logged_items": np.array([[0], [1]]), "logged_rewards": np.array([[1.0], [1.0]])},
            eval_factory_kwargs=None,
            score_items_kwargs=None,
            build_score_bundle=_build_bundle,
        )


def test_session_evaluate_and_metric_rescore_cache():
    session = RecommenderEvaluationSession(process_eval_kwargs=_process_eval_kwargs)
    eval_kwargs = {
        "logged_items": np.array([[2], [1]], dtype=np.int_),
        "logged_rewards": np.array([[1.0], [1.0]], dtype=np.float64),
    }
    m1 = session.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.PRECISION_AT_K,
        eval_top_k=2,
        temperature=1.0,
        eval_kwargs=eval_kwargs,
        eval_factory_kwargs=None,
        score_items_kwargs={"interactions": "dummy"},
        build_score_bundle=_build_bundle,
    )
    assert isinstance(m1, float)
    m2 = session.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.MRR_AT_K,
        eval_top_k=2,
        temperature=1.0,
        eval_kwargs=None,
        eval_factory_kwargs=None,
        score_items_kwargs=None,
        build_score_bundle=_build_bundle,
    )
    assert isinstance(m2, float)

    m3 = session.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.MRR_AT_K,
        eval_top_k=2,
        temperature=1.0,
        eval_kwargs={},
        eval_factory_kwargs=None,
        score_items_kwargs=None,
        build_score_bundle=_build_bundle,
    )
    assert m3 == m2


def test_session_clear_cache():
    session = RecommenderEvaluationSession(process_eval_kwargs=_process_eval_kwargs)
    eval_kwargs = {
        "logged_items": np.array([[2], [1]], dtype=np.int_),
        "logged_rewards": np.array([[1.0], [1.0]], dtype=np.float64),
    }
    session.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.PRECISION_AT_K,
        eval_top_k=2,
        temperature=1.0,
        eval_kwargs=eval_kwargs,
        eval_factory_kwargs=None,
        score_items_kwargs={"interactions": "dummy"},
        build_score_bundle=_build_bundle,
    )
    assert session.evaluator is not None
    session.clear_cache()
    assert session.evaluator is None
    assert session._recommendation_scores is None


def test_session_switching_eval_type_requires_eval_kwargs():
    session = RecommenderEvaluationSession(process_eval_kwargs=_process_eval_kwargs)
    eval_kwargs = {
        "logged_items": np.array([[2], [1]], dtype=np.int_),
        "logged_rewards": np.array([[1.0], [1.0]], dtype=np.float64),
    }
    session.evaluate(
        eval_type=RecommenderEvaluatorType.SIMPLE,
        metric_type=RecommenderMetricType.PRECISION_AT_K,
        eval_top_k=2,
        temperature=1.0,
        eval_kwargs=eval_kwargs,
        eval_factory_kwargs=None,
        score_items_kwargs={"interactions": "dummy"},
        build_score_bundle=_build_bundle,
    )
    with pytest.raises(ValueError, match="eval_kwargs is required"):
        session.evaluate(
            eval_type=RecommenderEvaluatorType.REPLAY_MATCH,
            metric_type=RecommenderMetricType.PRECISION_AT_K,
            eval_top_k=2,
            temperature=1.0,
            eval_kwargs=None,
            eval_factory_kwargs=None,
            score_items_kwargs=None,
            build_score_bundle=_build_bundle,
        )

    with pytest.raises(ValueError, match="eval_kwargs is required"):
        session.evaluate(
            eval_type=RecommenderEvaluatorType.REPLAY_MATCH,
            metric_type=RecommenderMetricType.PRECISION_AT_K,
            eval_top_k=2,
            temperature=1.0,
            eval_kwargs={},
            eval_factory_kwargs=None,
            score_items_kwargs=None,
            build_score_bundle=_build_bundle,
        )
