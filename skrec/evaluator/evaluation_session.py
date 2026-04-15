from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from skrec.evaluator.base_evaluator import BaseRecommenderEvaluator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.evaluator.factory import RecommenderEvaluatorFactory
from skrec.metrics.datatypes import RecommenderMetricType

ProcessEvalKwargsFn = Callable[[Optional[Mapping[str, Any]]], Dict[str, Any]]

ScoreBundleFromKwargsFn = Callable[
    [Mapping[str, Any], float],
    Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], NDArray[np.int_]],
]


def _eval_kwargs_nonempty(eval_kwargs: Optional[Mapping[str, Any]]) -> bool:
    """True when the caller passed logged data (non-empty mapping). ``None`` and ``{}`` are treated as absent."""
    if eval_kwargs is None:
        return False
    return len(eval_kwargs) > 0


@dataclass
class RecommenderEvaluationSession:
    """Caches recommendation scores, ranks, and evaluator state for repeated metrics.

    Stateless evaluators compute ``modified_rewards`` once per cache generation;
    this session owns the mutable cache and invalidation rules. Scorer-specific
    work is passed via ``process_eval_kwargs`` and ``build_score_bundle``.
    """

    process_eval_kwargs: ProcessEvalKwargsFn

    evaluator: Optional[BaseRecommenderEvaluator] = None
    _last_eval_type: Optional[RecommenderEvaluatorType] = None
    _last_eval_factory_kwargs: Optional[Mapping[str, Any]] = None

    _recommendation_scores: Optional[NDArray[np.float64]] = None
    _recommendation_probas: Optional[NDArray[np.float64]] = None
    _recommendation_ranks: Optional[NDArray[np.int_]] = None
    _modified_rewards: Optional[NDArray[np.float64]] = None

    def clear_cache(self) -> None:
        self._recommendation_scores = None
        self._recommendation_probas = None
        self._recommendation_ranks = None
        self._modified_rewards = None
        self.evaluator = None
        self._last_eval_type = None
        self._last_eval_factory_kwargs = None

    def evaluate(
        self,
        *,
        eval_type: RecommenderEvaluatorType,
        metric_type: RecommenderMetricType,
        eval_top_k: int,
        temperature: float,
        eval_kwargs: Optional[Mapping[str, Any]],
        eval_factory_kwargs: Optional[Mapping[str, Any]],
        score_items_kwargs: Optional[Mapping[str, Any]],
        build_score_bundle: ScoreBundleFromKwargsFn,
    ) -> float:
        if score_items_kwargs:
            scores, probas, ranks = build_score_bundle(score_items_kwargs, temperature)
            self._recommendation_scores = scores
            self._recommendation_probas = probas
            self._recommendation_ranks = ranks
            self._modified_rewards = None

        if self._recommendation_scores is None:
            raise ValueError(
                "No cached recommendation scores available. Provide score_items_kwargs to generate scores."
            )

        eval_config_changed = (
            self.evaluator is None
            or eval_type != self._last_eval_type
            or eval_factory_kwargs != self._last_eval_factory_kwargs
        )
        if eval_config_changed:
            self.evaluator = RecommenderEvaluatorFactory.create(eval_type, **(eval_factory_kwargs or {}))
            self._last_eval_type = eval_type
            self._last_eval_factory_kwargs = eval_factory_kwargs
            self._modified_rewards = None

        need_modified_rewards = self._modified_rewards is None or _eval_kwargs_nonempty(eval_kwargs)
        if need_modified_rewards:
            if not _eval_kwargs_nonempty(eval_kwargs):
                raise ValueError(
                    "eval_kwargs is required to compute modified rewards. "
                    "Provide logged_items, logged_rewards, and any other required arguments."
                )
            processed_eval_kwargs = self.process_eval_kwargs(eval_kwargs)
            self._modified_rewards = self.evaluator._compute_modified_rewards(
                recommendation_scores=self._recommendation_scores,
                recommendation_probas=self._recommendation_probas,
                **processed_eval_kwargs,
            )

        if self.evaluator is None:
            raise RuntimeError("evaluator must be configured before calling evaluate().")
        if self._modified_rewards is None:
            raise RuntimeError("modified rewards were not computed — this is a bug.")
        return self.evaluator.evaluate(
            modified_rewards=self._modified_rewards,
            recommendation_ranks=self._recommendation_ranks,
            recommendation_scores=self._recommendation_scores,
            metric_type=metric_type,
            top_k=eval_top_k,
        )
