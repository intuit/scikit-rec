from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from skrec.evaluator.categories import EvaluatorCategories
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
from skrec.recommender.bandits.datatypes import StrategyType
from skrec.recommender.bandits.factory import StrategyFactory
from skrec.recommender.bandits.strategy.static_action import StaticAction
from skrec.recommender.base_recommender import BaseRecommender
from skrec.scorer.base_scorer import BaseScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class ContextualBanditsRecommender(BaseRecommender):
    """Recommender that combines model scores with an exploration strategy.

    Wraps any scorer and applies a bandit strategy (e.g. epsilon-greedy,
    static action) to balance exploitation of learned scores with exploration
    of under-served items.

    **Evaluation semantics:** Offline ``evaluate()`` scores items with the
    underlying scorer, but **ranking and probabilities follow the bandit policy**
    wherever the strategy is applied (same as ``recommend()``). You must call
    ``set_strategy()`` before ``evaluate()`` when that path needs the strategy
    (e.g. ``STATIC_ACTION`` with Simple / ReplayMatch–style evaluators). Metrics
    then reflect **policy behavior**, not raw argmax-of-scorer behavior. For
    evaluation of the base model only, use ``RankingRecommender`` (or any
    non-bandit recommender that does not wrap scores in a bandit strategy).
    """

    def __init__(
        self,
        scorer: BaseScorer,
        strategy_type: Optional[StrategyType] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(scorer)
        if strategy_type is not None and strategy_params is not None:
            self.strategy = StrategyFactory.create(strategy_type, strategy_params)
            self.latest_flags = None
            self.strategy_type = strategy_type
        else:
            logger.info("Strategy not specified, but Training Complete! Set Strategy Before Inference")
            self.strategy = None
            self.strategy_type = None

    def set_strategy(
        self,
        strategy_type: StrategyType,
        strategy_params: Dict[str, Any],
    ) -> None:
        """Configure the bandit exploration strategy used during recommendation.

        Can be called after training to swap strategies without retraining.

        Args:
            strategy_type: The type of bandit strategy to use (e.g.
                ``StrategyType.EPSILON_GREEDY``, ``StrategyType.STATIC_ACTION``).
            strategy_params: Hyperparameters for the strategy (e.g.
                ``{"epsilon": 0.1}`` for epsilon-greedy).
        """
        self.strategy = StrategyFactory.create(strategy_type, strategy_params)
        self.latest_flags = None
        self.strategy_type = strategy_type

    def _recommend_from_scores(self, scores: NDArray[np.float64], top_k: int = 1) -> NDArray[np.int_]:
        if self.strategy is None:
            raise RuntimeError("Strategy not set. Call set_strategy() before recommend().")
        active_item_names = self._get_item_names()

        recommended_items, self.latest_flags = self.strategy.rank(scores, active_item_names, top_k)
        return recommended_items

    def _get_probabilities_from_scores(self, scores: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
        """
        Overrides BaseRecommender's method to incorporate bandit strategy blending.
        """
        if self.strategy is None:
            raise RuntimeError("Strategy not set. Call set_strategy() before recommend().")
        if isinstance(self.strategy, StaticAction):
            raise NotImplementedError("Static action strategy does not support probabilistic approach")

        base_item_probabilities = super()._get_probabilities_from_scores(scores, temperature)

        return self.strategy.get_blended_probabilities(base_item_probabilities, self._get_item_names())

    def get_latest_strategy_flags(self) -> NDArray:
        """Return the exploration flags from the most recent ``recommend`` call.

        Strategy flags indicate per-recommendation decisions made by the bandit
        strategy (e.g. which recommendations were exploratory vs. exploitative).
        The exact contents depend on the strategy implementation.

        Returns:
            NDArray of flags produced by the strategy's ``rank`` method during
            the last call to ``recommend``.

        Raises:
            ValueError: If ``recommend`` has not been called yet.
        """
        if self.latest_flags is None:
            raise ValueError("Strategy flags not found; recommend must be called before this method")
        return self.latest_flags

    def _build_static_action_eval_score_bundle(
        self, score_items_kwargs: Mapping[str, Any], temperature: float
    ) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], NDArray[np.int_]]:
        """Scores and **policy** ranks for STATIC_ACTION + non-probabilistic evaluators.

        Uses ``_score_items_np`` for logits, then ``_recommend_from_scores`` so
        ranks match the static-action policy (not plain sort-by-score). Requires
        ``set_strategy()`` to be set. No softmax; ``recommendation_probas`` is
        ``None``.
        """
        del temperature  # unused; kept for callable signature parity with :meth:`_build_eval_score_bundle`
        scores = self._score_items_np(**score_items_kwargs)
        probas: Optional[NDArray[np.float64]] = None
        n_items = scores.shape[1]
        recommended_idx = self._recommend_from_scores(scores, top_k=n_items)
        ranks = np.empty_like(recommended_idx)
        np.put_along_axis(ranks, recommended_idx, np.arange(n_items), axis=1)
        return scores, probas, ranks

    def evaluate(
        self,
        eval_type: RecommenderEvaluatorType,
        metric_type: RecommenderMetricType,
        eval_top_k: int,
        temperature: float = 1.0,
        score_items_kwargs: Optional[Mapping[str, DataFrame]] = None,
        eval_kwargs: Optional[Mapping[str, Any]] = None,
        eval_factory_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> float:
        """
        Evaluates the bandits contextual recommender using a specified evaluator and metric.

        **Policy-aligned evaluation:** Wherever this class applies a bandit
        strategy to turn scores into rankings (including the STATIC_ACTION +
        non-probabilistic branch), ``evaluate()`` uses the **same** ranking rule
        as ``recommend()``. A strategy must be configured via ``set_strategy()``
        (or constructor); otherwise a ``RuntimeError`` is raised.

        If the evaluator requires target probabilities (IPS, DR, SNIPS, etc.),
        or the strategy is not ``STATIC_ACTION``, evaluation delegates to
        ``BaseRecommender.evaluate`` (softmax + strategy blending where applicable).

        Args:
            eval_type: The type of evaluator strategy to use (e.g., SIMPLE, IPS, DR).
            metric_type: The metric to calculate (e.g., PRECISION_AT_K).
            eval_top_k: The 'k' value for the evaluation metric.
            temperature: Temperature for softmax conversion of scores to probabilities.
                Defaults to 1.0. A value of 0 leads to one-hot probabilities.
            score_items_kwargs: Keyword arguments to pass to `self.score_items`
                (e.g., `interactions`, `users`) to generate recommendation scores.
                If None, cached scores and probabilities in the evaluator are used.
            eval_kwargs: Keyword arguments for computing modified rewards. Common arguments
                include `logged_items`, `logged_rewards`, `logging_proba`, `expected_rewards`.
                `logged_items` can be provided either as an array of integer item indices
                (dense, 0 to n_items-1) matching the scorer's internal mapping, or as an
                array/list of string item IDs (detected if the array's dtype is `object`).
                If strings are provided, they will be automatically mapped to integers
                using the scorer's `item_names` list.
                If ``None`` or an empty dict ``{}``, previously cached modified rewards are
                reused when still valid — use that when only ``metric_type`` or
                ``eval_top_k`` changes. Any non-empty mapping triggers recomputation; there
                is no identity check on the logged data itself.
            eval_factory_kwargs: Optional keyword arguments for the evaluator factory.

        Returns:
            The calculated metric score.

        Raises:
            ValueError: If required arguments are missing, if string `logged_items` are
                provided but the scorer lacks `item_names`, if a string item ID in
                `logged_items` is not found, or if temperature is negative.
            TypeError: If unexpected keyword arguments are passed via `eval_kwargs`, or if
                `logged_items` is provided in an unexpected format for mapping.
            RuntimeError: If ``recommend()`` or the STATIC_ACTION non-probabilistic
                ``evaluate()`` path runs without ``set_strategy()`` having been called.
        """
        if EvaluatorCategories.requires_probability(eval_type) or self.strategy_type != StrategyType.STATIC_ACTION:
            return super().evaluate(
                eval_type=eval_type,
                metric_type=metric_type,
                eval_top_k=eval_top_k,
                temperature=temperature,
                score_items_kwargs=score_items_kwargs,
                eval_kwargs=eval_kwargs,
                eval_factory_kwargs=eval_factory_kwargs,
            )

        return self._eval_session.evaluate(
            eval_type=eval_type,
            metric_type=metric_type,
            eval_top_k=eval_top_k,
            temperature=temperature,
            eval_kwargs=eval_kwargs,
            eval_factory_kwargs=eval_factory_kwargs,
            score_items_kwargs=dict(score_items_kwargs) if score_items_kwargs is not None else None,
            build_score_bundle=self._build_static_action_eval_score_bundle,
        )
