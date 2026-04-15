import warnings
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from skrec.evaluator.categories import EvaluatorCategories
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.base_metric import (
    BaseClassificationMetric,
    BasePolicyMetric,
    BaseRankingMetric,
    BaseRecommenderMetric,
)
from skrec.metrics.datatypes import RecommenderMetricType
from skrec.metrics.factory import RecommenderMetricFactory


class BaseRecommenderEvaluator(ABC):
    """
    Base class for offline recommender system evaluation using vectorized operations.

    Subclasses implement different evaluation strategies (Simple, IPS, DR, etc.)
    by defining `_calculate_modified_rewards`.

    This class is stateless. Evaluation is a two-step process managed by the caller
    (typically BaseRecommender):

    1. Compute modified rewards (expensive, cache externally):
       ``modified_rewards = evaluator._compute_modified_rewards(scores, probas, logged_items, ...)``

    2. Compute metric (cheap, pure function):
       ``metric_value = evaluator.evaluate(modified_rewards, ranks, scores, metric_type, top_k)``

    Assumes item IDs are dense indices [0, n_items-1].
    Uses a fixed padding value for logged items.
    """

    ITEM_PAD_VALUE = -1
    TYPE: ClassVar[RecommenderEvaluatorType]

    # Class attribute indicating if the evaluator uses logged rewards directly
    # (True, e.g., Simple, ReplayMatch) or modifies/replaces them based on
    # policy probabilities or models (False, e.g., IPS, DR, DirectMethod).
    # Must be defined by subclasses.
    PRESERVES_LOGGED_REWARD: ClassVar[bool]

    def _compute_modified_rewards(
        self,
        recommendation_scores: NDArray[np.float64],
        recommendation_probas: Optional[NDArray[np.float64]],
        logged_items: NDArray,
        logged_rewards: NDArray,
        logging_proba: Optional[NDArray] = None,
        expected_rewards: Optional[NDArray] = None,
    ) -> NDArray[np.float64]:
        """
        Computes the modified rewards matrix for all items.

        Dispatches to `_calculate_modified_rewards` using probabilities as the
        target policy for probability-based evaluators (IPS, DR, SNIPS, PolicyWeighted,
        DirectMethod) or raw scores for non-probabilistic evaluators (Simple, ReplayMatch).

        This is the expensive step. The result should be cached by the caller and
        reused across calls that only vary in metric type or top_k.

        Args:
            recommendation_scores: Raw scores from the model. Shape (N, n_items).
            recommendation_probas: Probabilities derived from scores (e.g. via softmax).
                                   Shape (N, n_items). Required for probabilistic evaluators.
            logged_items: Logged item indices. Shape (N, L_max).
            logged_rewards: Rewards for logged items. Shape (N, L_max).
            logging_proba: Logging policy probabilities. Shape (N, L_max).
                           Required by IPS, SNIPS, and DR evaluators.
            expected_rewards: Model's expected rewards. Shape (N, n_items).
                              Required by DR and DirectMethod evaluators.

        Returns:
            Array of shape (N, n_items) containing modified rewards.
        """
        n_items = recommendation_scores.shape[1]
        if EvaluatorCategories.requires_probability(self.TYPE):
            return self._calculate_modified_rewards(
                logged_items=logged_items,
                logged_rewards=logged_rewards,
                target_proba=recommendation_probas,
                expected_rewards=expected_rewards,
                logging_proba=logging_proba,
            )
        else:
            return self._calculate_modified_rewards(
                logged_items=logged_items,
                logged_rewards=logged_rewards,
                target_proba=recommendation_scores,
                expected_rewards=expected_rewards,
                logging_proba=logging_proba,
                n_items=n_items,
            )

    def evaluate(
        self,
        modified_rewards: NDArray[np.float64],
        recommendation_ranks: NDArray[np.int_],
        recommendation_scores: NDArray[np.float64],
        metric_type: RecommenderMetricType,
        top_k: Optional[int] = None,
    ) -> float:
        """
        Computes a metric from pre-computed modified rewards. Stateless pure function.

        Args:
            modified_rewards: Modified reward matrix from `_compute_modified_rewards`.
                              Shape (N, n_items).
            recommendation_ranks: Rank of each item per user. Shape (N, n_items).
                                  Value at [u, i] is the rank of item i for user u.
            recommendation_scores: Raw model scores. Shape (N, n_items).
                                   Used by classification metrics (ROC AUC, PR AUC).
            metric_type: The type of metric to calculate.
            top_k: Optional cutoff for ranking metrics.

        Returns:
            The calculated metric score.
        """
        metric = RecommenderMetricFactory.create(metric_type)
        self._check_metric_evaluator_warnings(metric=metric, metric_type=metric_type)
        return metric.calculate(
            recommendation_ranks=recommendation_ranks,
            modified_rewards=modified_rewards,
            recommendation_scores=recommendation_scores,
            top_k=top_k,
        )

    # --- Private Helper Methods ---

    def _check_metric_evaluator_warnings(
        self,
        metric: BaseRecommenderMetric,
        metric_type: RecommenderMetricType,
    ) -> None:
        """
        Issues warnings for potentially problematic combinations of metrics and evaluators.
        Called internally by evaluate() before calculating the metric.
        """
        if isinstance(metric, BaseClassificationMetric) and not self.PRESERVES_LOGGED_REWARD:
            raise ValueError(
                f"Classification metric '{metric_type.name}' is not compatible with evaluator "
                f"'{self.__class__.__name__}'. Classification metrics (ROC AUC, PR AUC) require binary "
                "ground truth labels (0/1) and must be used with SimpleEvaluator or ReplayMatchEvaluator."
            )
        if isinstance(metric, BaseRankingMetric) and not self.PRESERVES_LOGGED_REWARD:
            warnings.warn(
                f"Using Ranking Metric '{metric_type.name}' with Evaluator '{self.__class__.__name__}'. "
                "Applying a @k cutoff to counterfactual reward estimates "
                "might not be standard practice for policy evaluation. "
                "Consider using SimpleEvaluator (or rarely ReplayMatchEvaluator).",
                UserWarning,
            )
        if isinstance(metric, BasePolicyMetric) and self.PRESERVES_LOGGED_REWARD:
            warnings.warn(
                f"Using Policy Metric '{metric_type.name}' with Evaluator '{self.__class__.__name__}'. "
                "Policy metrics typically expect a modified reward incorporating policy probability, not raw rewards.",
                UserWarning,
            )

    # --- Abstract Method for Subclasses ---

    @abstractmethod
    def _calculate_modified_rewards(
        self,
        logged_items: NDArray,
        logged_rewards: NDArray,
        target_proba: NDArray,
        expected_rewards: Optional[NDArray] = None,
        logging_proba: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Abstract method for calculating the modified rewards matrix for all items.

        Subclasses must implement this method using vectorized operations. The
        output represents a score or modified reward for *every* possible item,
        based on the specific evaluation strategy (Simple, RM, IPS, DR).

        Args:
            logged_items: Array of shape (N, L_max) with logged item indices
                (0 to n_items-1) or padding (-1).
            logged_rewards: Array of shape (N, L_max) with actual rewards for
                logged items, aligned with `logged_items`.
            target_proba: Array of shape (N, n_items) with the target policy's
                probability P(action | context) for *all* items, calculated
                from recommendation scores and temperature.
            expected_rewards: Optional array of shape (N, n_items) with the model's
                expected reward E[Reward | context, action] for *all* items.
                Required by DREvaluator.
            logging_proba: Optional array of shape (N, L_max) with the logging policy's
                probability P(logged_action | context) for each logged item, aligned
                with `logged_items`. Required by IPSEvaluator and DREvaluator.

        Returns:
            A NumPy array of shape (N, n_items) containing the modified rewards
            (or scores) for each item for each instance.
        """
        pass

    # --- Helper Methods ---

    def _validate_input_shapes(
        self,
        logged_items: NDArray,
        logged_rewards: NDArray,
        target_proba: Optional[NDArray] = None,
        n_items: Optional[int] = None,
        logging_proba: Optional[NDArray] = None,
        expected_rewards: Optional[NDArray] = None,
    ) -> Tuple[int, int, int]:
        """
        Validates the shapes of core input arrays and optional arrays.

        Ensures consistency in N, n_items, L_max. Uses fixed padding value -1.
        Called internally by `_calculate_modified_rewards` implementations.

        Args:
            target_proba: Array of shape (N, n_items). Used to infer N, n_items.
            n_items: Optional int. Used when target_proba is not provided to infer n_items.
            logged_items: Array of shape (N, L_max).
            logged_rewards: Array of shape (N, L_max).
            logging_proba: Optional array of shape (N, L_max).
            expected_rewards: Optional array of shape (N, n_items).

        Returns:
            A tuple (N, n_items, L_max) containing the validated dimensions.

        Raises:
            ValueError: If any dimension inconsistency is found or if item indices
                        in logged_items are out of bounds [0, n_items-1] (excluding -1 padding).
        """
        if target_proba is not None and target_proba.ndim != 2:
            raise ValueError(f"target_proba must be 2D (N, n_items), got shape {target_proba.shape}")

        if target_proba is None and n_items is None:
            raise ValueError("n_items is required when target_proba is not provided")

        if target_proba is not None:
            N, n_items = target_proba.shape
        else:
            N = logged_items.shape[0]

        if logged_items.ndim != 2:
            raise ValueError(f"logged_items must be 2D (N, L_max), got shape {logged_items.shape}")
        _N_log, L_max = logged_items.shape
        if _N_log != N:
            raise ValueError(f"Mismatch in N dimension: target_proba ({N}) vs logged_items ({_N_log})")

        if logged_rewards.shape != (N, L_max):
            raise ValueError(f"Shape mismatch: logged_rewards ({logged_rewards.shape}) vs expected ({N}, {L_max})")

        # Validate shapes of optional arrays
        if logging_proba is not None:
            if logging_proba.shape != (N, L_max):
                raise ValueError(f"Shape mismatch: logging_proba ({logging_proba.shape}) vs expected ({N}, {L_max})")

        if expected_rewards is not None:
            if expected_rewards.shape != (N, n_items):
                raise ValueError(
                    f"Shape mismatch: expected_rewards ({expected_rewards.shape}) vs expected ({N}, {n_items})"
                )

        # Validate logged_items indices (must be in [0, n_items-1] or ITEM_PAD_VALUE)
        valid_mask = logged_items != self.ITEM_PAD_VALUE
        if np.any(valid_mask):  # Only check if there are any non-padded items
            valid_items = logged_items[valid_mask]
            min_item_idx = valid_items.min()
            max_item_idx = valid_items.max()
            if min_item_idx < 0 or max_item_idx >= n_items:
                raise ValueError(
                    f"Found invalid item index in logged_items. Indices must be within "
                    f"[0, {n_items - 1}] or equal to padding value {self.ITEM_PAD_VALUE}. "
                    f"Found min={min_item_idx}, max={max_item_idx} among non-padded items."
                )

        return N, n_items, L_max
