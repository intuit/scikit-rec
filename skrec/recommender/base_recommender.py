from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from skrec.constants import OUTCOME_PREFIX
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.evaluator.base_evaluator import BaseRecommenderEvaluator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.evaluator.evaluation_session import RecommenderEvaluationSession
from skrec.metrics.datatypes import RecommenderMetricType
from skrec.recommender.inference_input import InferenceInputPreparer
from skrec.recommender.training_coordinator import coordinate_training
from skrec.scorer.base_scorer import BaseScorer
from skrec.scorer.independent import IndependentScorer
from skrec.scorer.multioutput import MultioutputScorer
from skrec.util.logger import get_logger
from skrec.util.numpy_util import (
    sample_with_replacement_2d,
    sample_without_replacement_2d,
    softmax_2d,
)

logger = get_logger(__name__)


class BaseRecommender(ABC):
    """Abstract base class for all recommenders.

    Orchestrates the full recommend pipeline: training, scoring, recommendation,
    and evaluation.  Delegates ML work to a ``BaseScorer`` (which in turn
    delegates to a ``BaseEstimator``).  Subclasses implement
    ``_recommend_from_scores`` to define the ranking or sampling strategy.

    Subclasses that define ``__init__`` must call ``super().__init__(scorer)`` so
    ``_eval_session`` and ``_inference`` are initialized.
    """

    def __init__(self, scorer: BaseScorer) -> None:
        self.scorer = scorer
        self.outcome_cols: List[str] = []
        self._eval_session = RecommenderEvaluationSession(process_eval_kwargs=self._process_eval_kwargs)
        self._inference = InferenceInputPreparer(self)

    @property
    def evaluation_session(self) -> RecommenderEvaluationSession:
        """Evaluation cache used by :meth:`evaluate` (scores, ranks, modified rewards)."""
        return self._eval_session

    def clear_evaluation_cache(self) -> None:
        """Clear all state held for :meth:`evaluate` (scores, ranks, modified rewards, evaluator)."""
        self._eval_session.clear_cache()

    @property
    def evaluator(self) -> Optional[BaseRecommenderEvaluator]:
        """Last evaluator used by :meth:`evaluate` (lives on the evaluation session)."""
        return self._eval_session.evaluator

    def train(
        self,
        users_ds: Optional[UsersDataset] = None,
        items_ds: Optional[ItemsDataset] = None,
        interactions_ds: Optional[InteractionsDataset] = None,
        valid_users_ds: Optional[UsersDataset] = None,
        valid_interactions_ds: Optional[InteractionsDataset] = None,
    ) -> None:
        """Fit the recommender on the provided datasets.

        Fetches data from each dataset, then delegates to the underlying scorer
        and estimator.  All arguments are optional so that subclasses can be
        trained on whichever combination of users, items, and interactions they
        require.

        Args:
            users_ds: Dataset containing user features.  Pass ``None`` when the
                model does not use user-level features.
            items_ds: Dataset containing item features and the candidate item
                catalogue.  Pass ``None`` when items are derived solely from
                interactions.
            interactions_ds: Dataset of observed user–item interactions used
                for training.  Required for most recommender types.
            valid_users_ds: Optional user features for the validation split.
                Must be provided whenever ``users_ds`` is provided and
                ``valid_interactions_ds`` is also provided.
            valid_interactions_ds: Optional interactions for the validation
                split.  When provided, training will emit per-epoch validation
                loss and (if configured) apply early stopping.
        """
        if users_ds is not None:
            users_df = users_ds.fetch_data()
            self.users_schema = users_ds.client_schema
        else:
            users_df = None
            self.users_schema = None

        if interactions_ds is not None:
            interactions_df = interactions_ds.fetch_data()
            self.interactions_schema = interactions_ds.client_schema
        else:
            interactions_df = None
            self.interactions_schema = None

        if items_ds is not None:
            items_df = items_ds.fetch_data()
            self.items_schema = items_ds.client_schema
        else:
            items_df = None
            self.items_schema = None

        # Store raw fetched DataFrames so subclasses can access them without
        # a second fetch_data() call (e.g. RankingRecommender.train building retrieval index).
        self._train_interactions_df = interactions_df
        self._train_items_df = items_df

        valid_users_df = valid_users_ds.fetch_data() if valid_users_ds else None
        valid_interactions_df = valid_interactions_ds.fetch_data() if valid_interactions_ds else None

        if valid_users_ds is not None and valid_interactions_ds is None:
            logger.warning(
                "valid_users_ds was provided without valid_interactions_ds — validation data will be ignored."
            )

        if interactions_df is not None:
            self.outcome_cols = [col for col in interactions_df.columns if col.startswith(OUTCOME_PREFIX)]
        interactions_df = self._process_outcome_columns(interactions_df)
        valid_interactions_df = self._process_outcome_columns(valid_interactions_df)

        coordinate_training(
            self,
            users_df=users_df,
            items_df=items_df,
            interactions_df=interactions_df,
            valid_users_df=valid_users_df,
            valid_interactions_df=valid_interactions_df,
            users_ds=users_ds,
            items_ds=items_ds,
            interactions_ds=interactions_ds,
            valid_users_ds=valid_users_ds,
            valid_interactions_ds=valid_interactions_ds,
        )

    def _process_outcome_columns(self, interactions_df: Optional[DataFrame] = None) -> Optional[DataFrame]:
        if interactions_df is None:
            return None
        if not self.outcome_cols:
            return interactions_df
        return interactions_df.drop(columns=self.outcome_cols)

    def set_item_subset(self, item_subset: List[str]) -> None:
        """Restrict recommendations to a subset of the item catalogue.

        Delegates to ``self.scorer.set_item_subset``.  Call
        ``clear_item_subset`` to restore full-catalogue scoring.

        Args:
            item_subset: Non-empty list of item IDs seen during training.
        """
        self.scorer.set_item_subset(item_subset)

    def set_item_specific_features(
        self,
        item_specific_features_users: Dict[str, List[str]],
        item_specific_features_interactions: Dict[str, List[str]],
    ) -> None:
        """Configure per-item feature subsets (``IndependentScorer`` only).

        See ``IndependentScorer.set_item_specific_features`` for full
        documentation.

        Args:
            item_specific_features_users: Mapping from item ID to user feature
                column names.
            item_specific_features_interactions: Mapping from item ID to
                interaction feature column names.

        Raises:
            ValueError: If the scorer is not an ``IndependentScorer``.
        """
        if not isinstance(self.scorer, IndependentScorer):
            raise ValueError("Item specific features can only be set for IndependentScorer")
        self.scorer.set_item_specific_features(item_specific_features_users, item_specific_features_interactions)

    def clear_item_subset(self) -> None:
        """Remove any active item subset and restore full-catalogue scoring."""
        self.scorer.clear_item_subset()

    def set_new_items(self, new_items_df: DataFrame) -> None:
        """Extend the item catalogue with new items without retraining.

        Delegates to ``self.scorer.set_new_items``.  Must be called before
        ``set_item_subset`` if both are used together.

        Args:
            new_items_df: DataFrame with the same columns as the training
                items DataFrame, including ``ITEM_ID_NAME``.
        """
        self.scorer.set_new_items(new_items_df)

    def _build_interactions_schema(self, strip_user_id: bool = False):
        """Returns a trimmed copy of interactions_schema with internal columns removed, or None."""
        return self._inference.build_trimmed_interactions_schema(strip_user_id)

    def _preprocess_inputs(
        self,
        interactions: Optional[DataFrame],
        users: Optional[DataFrame],
    ) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """Validate schema and apply preprocessing shared by score_items and _score_items_np."""
        return self._inference.preprocess_inputs(interactions, users)

    def score_items(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> DataFrame:
        """Score every item for each user after applying schema validation.

        Applies any configured ``interactions_schema`` and ``users_schema``
        before delegating to the underlying scorer.

        Args:
            interactions: DataFrame with interaction context features.  Must
                include ``USER_ID_NAME`` when ``users`` is also provided.
            users: Optional DataFrame with user features.

        Returns:
            DataFrame of shape ``(n_users, n_items)`` with item IDs as columns
            and predicted scores as values.
        """
        interactions, users = self._preprocess_inputs(interactions, users)
        return self.scorer.score_items(interactions, users)

    def _score_items_np(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> NDArray[np.float64]:
        """Internal: returns raw scores array, skipping DataFrame wrapping."""
        interactions, users = self._preprocess_inputs(interactions, users)
        return self.scorer._score_items_np(interactions, users)

    def apply_schema_to_users(self, users: Optional[DataFrame], strip_user_id: bool = False) -> Optional[DataFrame]:
        """Apply the trained users schema to a users DataFrame.

        Selects and type-coerces the columns declared in ``users_schema``.
        When ``strip_user_id=True``, ``USER_ID_NAME`` is removed from the
        schema before applying (used by ``recommend_online`` where the user ID
        has already been consumed for routing).

        Args:
            users: DataFrame to transform, or ``None``.
            strip_user_id: If ``True``, exclude ``USER_ID_NAME`` from the
                applied schema.

        Returns:
            Transformed DataFrame, or ``None`` when ``users`` is ``None`` and
            the scenario is valid (e.g. embedding estimator batch mode).
        """
        return self._inference.apply_users_schema(users, strip_user_id)

    def _get_available_items_count(self) -> int:
        """
        Get the number of available items for recommendation.

        Returns:
            Number of available items after exclusions.
        """
        if self.scorer.item_subset is not None:
            items = self.scorer.item_subset
        else:
            items = self.scorer.item_names if self.scorer.item_names is not None else []

        return len(items)

    def recommend(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
        top_k: int = 1,
        sampling_temperature: float = 0,
        replace: bool = False,
    ) -> NDArray:
        """
        Recommends items for given users based on interactions.

        This method can operate in two modes:
        1. If `sampling_temperature` is set or defaulted to 0: we will go with Deterministic Ranking mode.
           Items are ranked based on their scores, and the top_k items are returned. This uses the
           `_recommend_from_scores` method implemented by subclasses.
        2. If `sampling_temperature` is positive, we will go with Probabilistic Sampling mode.
           Scores are converted into probability distributions using `_get_probabilities_from_scores`.
           Items are then sampled from these distributions with or without replacement.

        """
        if sampling_temperature < 0:
            raise ValueError("sampling_temperature cannot be negative.")

        active_item_names = self._get_item_names()

        scores_np = self._score_items_np(interactions, users)
        recommended_idx: NDArray[np.int_]

        if sampling_temperature == 0:
            # Deterministic ranking
            # Check if top_k exceeds available items and warn if necessary
            available_items = self._get_available_items_count()
            if top_k > available_items:
                logger.warning(
                    f"Requested top_k ({top_k}) is larger than available items ({available_items}). "
                    f"Will return only {available_items} items."
                )
            recommended_idx = self._recommend_from_scores(scores_np, top_k)
        else:
            # Probabilistic sampling
            probabilities = self._get_probabilities_from_scores(scores_np, sampling_temperature)
            recommended_idx = self._sample_from_probabilities(probabilities, top_k, replace)
        return active_item_names[recommended_idx]

    def recommend_online(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
        top_k: int = 1,
    ) -> NDArray:
        """
        Real-time single-user recommendation without join overhead.

        **Compatibility**: works with all non-embedding scorers —
        ``UniversalScorer``, ``MulticlassScorer``, ``MultioutputScorer``,
        and ``IndependentScorer``. NOT supported for:

        - Embedding-based estimators (NCF, Two-Tower, DeepFM) — use
          ``recommend()`` instead.
        - ``SequentialRecommender`` / ``HierarchicalSequentialRecommender`` —
          SASRec/HRNN have their own efficient forward pass; use
          ``recommend()`` directly.

        Skips the pandas merge entirely for low-latency single-user serving.
        Accepts separate single-row interaction and user DataFrames, applies
        schema validation, merges them, and returns the top-k item names.

        **Return-type note for MultioutputScorer**: because ``MultioutputScorer``
        predicts a class label per item category (e.g. genre, device) rather than
        ranking independent items, this method returns a ``DataFrame`` of predicted
        class labels (e.g. ``{ITEM_genre: "action", ITEM_device: "mobile"}``)
        instead of the usual ``NDArray[str]`` of item names.  The ``top_k``
        parameter is ignored in that path.

        Args:
            interactions: Optional single-row DataFrame with interaction features.
            users: Optional single-row DataFrame with user features.
            top_k: Number of top items to return (ignored for MultioutputScorer).

        Returns:
            1D NDArray of top-k item names for the single user, or a DataFrame of
            predicted class labels when using MultioutputScorer.
        """
        interactions = self._process_outcome_columns(interactions)

        interactions_schema = self._inference.build_trimmed_interactions_schema(strip_user_id=True)
        if interactions_schema and interactions is not None:
            interactions = interactions_schema.apply(interactions)

        if self.users_schema:
            users = self._inference.apply_users_schema(users, strip_user_id=True)

        parts = [df for df in [interactions, users] if df is not None]
        if not parts:
            raise ValueError("Both interactions and users are None.")
        features = (
            parts[0].reset_index(drop=True)
            if len(parts) == 1
            else pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
        )

        active_item_names = self._get_item_names()

        if isinstance(self.scorer, MultioutputScorer):
            return self.scorer.score_fast(features)[list(active_item_names)]

        scores_np = self.scorer._score_fast_np(features)
        recommended_idx = self._recommend_from_scores(scores_np, top_k)
        return active_item_names[recommended_idx[0]]

    @abstractmethod
    def _recommend_from_scores(self, scores: NDArray[np.float64], top_k: int = 1) -> NDArray[np.int_]:
        """
        Given a NumPy array of scores (items as columns, users as rows),
        return an NDArray of item *indices* (shape N x top_k) for the top_k recommendations.
        The indices should be relative to the columns of the input scores,
        which are aligned with self._get_item_names().
        """
        pass

    def _get_item_names(self) -> NDArray[np.str_]:
        """
        Returns the active list of item names.
        Uses scorer.item_subset if set, otherwise scorer.item_names.
        """
        if self.scorer.item_subset is not None:
            return np.asarray(self.scorer.item_subset, dtype=np.str_)
        if self.scorer.item_names is None:
            raise ValueError("Scorer's item_names is None. Ensure the scorer is trained or item names are set.")
        return self.scorer.item_names

    def _get_probabilities_from_scores(self, scores: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
        """
        Applies softmax to raw scores to produce probability distributions.
        This method can be overridden by subclasses for custom probability generation.
        """
        return softmax_2d(scores, temperature)

    def _sample_from_probabilities(
        self, probabilities: NDArray[np.float64], top_k: int, replace: bool
    ) -> NDArray[np.int_]:
        """
        Samples item indices from probability distributions.

        Args:
            probabilities: A 2D array where each row is a probability distribution
                           over items for a user. Shape (N_users, N_items).
            top_k: The number of items to sample for each user.
            replace: Whether to sample with replacement.

        Returns:
            A 2D array of item indices. Shape (N_users, top_k).
        """
        _, n_items = probabilities.shape

        if replace:
            sampled_indices = sample_with_replacement_2d(probabilities, top_k)
        else:
            if top_k > n_items:
                logger.warning(
                    f"Cannot sample {top_k} items without replacement from {n_items} available items. "
                    f"Will sample {n_items} items instead."
                )
                top_k = n_items
            sampled_indices = sample_without_replacement_2d(probabilities, top_k)

        return sampled_indices

    def _process_eval_kwargs(self, eval_kwargs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """
        Processes evaluator keyword arguments, specifically mapping string `logged_items`
        to integer indices if necessary using the scorer's item_names.
        Map `None` and empty string `""` to `-1` (`=ITEM_PAD_VALUE`)
        """
        processed_kwargs = dict(eval_kwargs) if eval_kwargs else {}

        if "logged_items" in processed_kwargs:
            logged_items = processed_kwargs["logged_items"]
            logged_items_arr = np.asarray(logged_items)

            # If dtype is object, assume it contains strings needing mapping
            if logged_items_arr.dtype == object:
                if self.scorer.item_names is None:
                    raise ValueError("Scorer's item_names is required for string logged_items but is not available.")

                # Create reverse map for efficient lookup
                item_name_to_id_map = {name: i for i, name in enumerate(self.scorer.item_names)}
                item_name_to_id_map[None] = BaseRecommenderEvaluator.ITEM_PAD_VALUE
                item_name_to_id_map[""] = BaseRecommenderEvaluator.ITEM_PAD_VALUE

                # Find unique string items and the inverse mapping to reconstruct the original shape
                unique_items, inverse_indices = np.unique(logged_items_arr, return_inverse=True)

                # Map only the unique items to integers
                try:
                    mapped_unique_items = np.array([item_name_to_id_map[item] for item in unique_items], dtype=np.int_)
                except KeyError as e:
                    raise ValueError(f"Logged item '{e.args[0]}' not found in scorer's item_names list.") from e

                mapped_items = mapped_unique_items[inverse_indices].reshape(logged_items_arr.shape)
                processed_kwargs["logged_items"] = mapped_items

        return processed_kwargs

    def _build_eval_score_bundle(
        self, score_items_kwargs: Mapping[str, Any], temperature: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
        """Build (scores, probas, full ranking indices) for evaluation caching."""
        scores = self.score_items(**score_items_kwargs).to_numpy()
        probas = self._get_probabilities_from_scores(scores, temperature)
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
        Evaluates the recommender using a specified evaluator and metric.

        Caching strategy:
        - Recommendation scores are recomputed only when `score_items_kwargs` is provided.
        - Modified rewards are recomputed when the evaluator config changes (eval_type or
          eval_factory_kwargs) or when new `eval_kwargs` (logged data) are provided.
        - Calling with only `metric_type` or `eval_top_k` changed reuses all cached values.
        - Call :meth:`clear_evaluation_cache` to drop all cached ``evaluate()`` state
          (e.g. after retraining on the same instance or between experiments).

        Args:
            eval_type: The type of evaluator strategy to use (e.g., SIMPLE, IPS, DR).
            metric_type: The metric to calculate (e.g., PRECISION_AT_K).
            eval_top_k: The 'k' value for the evaluation metric.
            temperature: Temperature for softmax conversion of scores to probabilities.
                Defaults to 1.0. A value of 0 leads to one-hot probabilities.
            score_items_kwargs: Keyword arguments to pass to `self.score_items`
                (e.g., `interactions`, `users`) to generate recommendation scores.
                If None, previously cached scores are used.
            eval_kwargs: Keyword arguments for computing modified rewards. Common arguments
                include `logged_items`, `logged_rewards`, `logging_proba`, `expected_rewards`.
                `logged_items` can be provided as integer item indices (dense, 0 to n_items-1)
                or as string item IDs (mapped automatically via the scorer's `item_names`).
                If ``None`` or an empty dict ``{}``, previously cached modified rewards are
                reused when still valid — use that on subsequent calls when only
                ``metric_type`` or ``eval_top_k`` changes (e.g. sweeping top-k).
                Any non-empty mapping triggers recomputation; there is no identity check
                on the logged data itself.
            eval_factory_kwargs: Optional keyword arguments for the evaluator factory.

        Returns:
            The calculated metric score.

        Raises:
            ValueError: If no cached scores are available and `score_items_kwargs` is not
                provided, if `eval_kwargs` is required but not provided, if string
                `logged_items` are provided but the scorer lacks `item_names`, or if
                temperature is negative.
        """
        return self._eval_session.evaluate(
            eval_type=eval_type,
            metric_type=metric_type,
            eval_top_k=eval_top_k,
            temperature=temperature,
            eval_kwargs=eval_kwargs,
            eval_factory_kwargs=eval_factory_kwargs,
            score_items_kwargs=dict(score_items_kwargs) if score_items_kwargs is not None else None,
            build_score_bundle=self._build_eval_score_bundle,
        )
