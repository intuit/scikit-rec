from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from skrec.constants import USER_ID_NAME
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.base_metric import (
    BaseClassificationMetric,
    BaseRankingMetric,
    BaseRegressionMetric,
)
from skrec.metrics.datatypes import RecommenderMetricType
from skrec.metrics.factory import RecommenderMetricFactory
from skrec.recommender.base_recommender import BaseRecommender
from skrec.retriever.base_retriever import BaseCandidateRetriever
from skrec.retriever.content_based_retriever import ContentBasedRetriever
from skrec.retriever.embedding_retriever import EmbeddingRetriever
from skrec.retriever.popularity_retriever import PopularityRetriever
from skrec.scorer.base_scorer import BaseScorer
from skrec.scorer.multiclass import MulticlassScorer
from skrec.scorer.multioutput import MultioutputScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class RankingRecommender(BaseRecommender):
    """
    Recommender that ranks items by score and returns the top-K.

    Optionally accepts a ``retriever`` to narrow the candidate set before
    ranking. See ``skrec.retriever`` for available retrievers and
    guidance on which to use.

    .. note::
        **Not thread-safe when a retriever is attached.** The per-user
        retrieval loop calls ``scorer.set_item_subset()`` and
        ``scorer.clear_item_subset()`` on shared scorer state. Concurrent
        calls to ``recommend()`` on the same instance will corrupt each
        other's candidate sets. Use one instance per thread, or call
        ``recommend()`` sequentially.

    Example — end-to-end with built-in retrieval::

        from skrec.retriever import EmbeddingRetriever

        recommender = RankingRecommender(
            scorer=UniversalScorer(estimator=MatrixFactorizationEstimator()),
            retriever=EmbeddingRetriever(top_k=200),
        )
        recommender.train(interactions_ds=interactions_ds)
        recommendations = recommender.recommend(interactions=df, top_k=10)

    Example — bring your own retrieval via set_item_subset()::

        candidates = elasticsearch.query(user_id, top_k=500)
        recommender.set_item_subset(candidates)
        recommendations = recommender.recommend(interactions=df, top_k=10)
        recommender.clear_item_subset()
    """

    def __init__(self, scorer: BaseScorer, retriever: Optional[BaseCandidateRetriever] = None) -> None:
        # MultioutputScorer doesn't have a retrieval phase: targets are the
        # ITEM_<name> columns of a wide frame, not entries in an item
        # catalogue, so candidate-retrieval-then-rank doesn't apply. Reject
        # the combination at construction so the retriever's index isn't
        # silently built in train() and discarded at recommend time.
        if retriever is not None and isinstance(scorer, MultioutputScorer):
            raise ValueError(
                "MultioutputScorer does not support a retriever — its targets are "
                "ITEM_<name> columns of a wide frame, not entries in an item catalogue, "
                "so the candidate-retrieval-then-rank pattern doesn't apply. Either "
                "drop the retriever, or melt your data to long format and use "
                "UniversalScorer / IndependentScorer / MulticlassScorer."
            )
        super().__init__(scorer)
        self.retriever = retriever
        # Set once during _recommend_multioutput when degenerate-target
        # filtering applies — gates the warning so it fires once per
        # instance rather than once per call. Declared here (rather than
        # lazily created with `getattr(..., False)`) so subclasses that
        # override __init__ pick up the field reliably.
        #
        # Stickiness caveat: this flag is recommender-instance state but
        # the manifest it gates lives on `scorer.degenerate_targets`.
        # Sharing one scorer across multiple recommender instances: each
        # recommender warns independently (each has its own flag) —
        # uncommon pattern, acceptable. Re-training the same recommender
        # in place: `train()` resets this flag so the new manifest's
        # targets get a fresh warning.
        self._warned_degenerate_recommend: bool = False

    def train(
        self,
        users_ds: Optional[UsersDataset] = None,
        items_ds: Optional[ItemsDataset] = None,
        interactions_ds: Optional[InteractionsDataset] = None,
        valid_users_ds: Optional[UsersDataset] = None,
        valid_interactions_ds: Optional[InteractionsDataset] = None,
    ) -> None:
        """Fit the recommender and (if configured) build the retrieval index.

        Validates retriever/estimator compatibility up front, then delegates to
        ``BaseRecommender.train``.  After the scorer and estimator are fitted,
        calls ``retriever.build_index`` using the DataFrames already loaded
        during training — no second ``fetch_data`` call is made.

        Args:
            users_ds: Optional user features dataset.
            items_ds: Optional items dataset.  Required for
                ``ContentBasedRetriever``.
            interactions_ds: Optional interactions dataset.  Required for
                ``PopularityRetriever``.
            valid_users_ds: Optional validation user features dataset.
            valid_interactions_ds: Optional validation interactions dataset.

        Raises:
            ValueError: If a retriever is attached but its required dataset
                inputs are missing, or if an ``EmbeddingRetriever`` is used
                with a non-embedding estimator.
        """
        # Reset the recommend-time degenerate-target warning so a re-train
        # on the same recommender (with potentially a different degenerate
        # manifest) gets a fresh warning when its first recommend() runs.
        # Without this, the flag stuck from the previous fit would silence
        # the new manifest's warning.
        self._warned_degenerate_recommend = False

        if self.retriever is not None:
            # Validate required inputs before training starts — fail fast with a clear
            # message rather than letting a cryptic error surface from inside build_index().
            if isinstance(self.retriever, ContentBasedRetriever) and items_ds is None:
                raise ValueError(
                    "ContentBasedRetriever requires item features but no items_ds was passed to train(). "
                    "Pass items_ds=<your ItemsDataset> when calling train()."
                )
            if isinstance(self.retriever, PopularityRetriever) and interactions_ds is None:
                raise ValueError(
                    "PopularityRetriever requires interaction history but no interactions_ds was passed to train(). "
                    "Pass interactions_ds=<your InteractionsDataset> when calling train()."
                )
            if isinstance(self.retriever, EmbeddingRetriever) and not isinstance(
                getattr(self.scorer, "estimator", None), BaseEmbeddingEstimator
            ):
                raise ValueError(
                    f"EmbeddingRetriever requires a BaseEmbeddingEstimator, but the scorer's estimator is "
                    f"{type(getattr(self.scorer, 'estimator', None)).__name__}. "
                    "Use PopularityRetriever or ContentBasedRetriever for non-embedding estimators."
                )

        super().train(
            users_ds=users_ds,
            items_ds=items_ds,
            interactions_ds=interactions_ds,
            valid_users_ds=valid_users_ds,
            valid_interactions_ds=valid_interactions_ds,
        )
        if self.retriever is not None:
            # Use getattr so custom scorers without .estimator still work with
            # retrievers that ignore the estimator argument (Popularity, ContentBased).
            estimator = getattr(self.scorer, "estimator", None)

            # Reuse DataFrames already fetched by super().train() — no second fetch.
            self.retriever.build_index(
                estimator=estimator,
                interactions=self._train_interactions_df,
                items=self._train_items_df,
            )

    def recommend(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
        top_k: int = 1,
        sampling_temperature: Optional[float] = 0,
        replace: bool = False,
    ) -> NDArray:
        """
        Recommends items for given users based on interactions.

        This method can operate in two modes:
        1. Deterministic Ranking: If `sampling_temperature` is None or 0, items are ranked
           based on their scores, and the top_k items are returned. This uses the
           `_recommend_from_scores` method implemented by subclasses.
        2. Probabilistic Sampling: If `sampling_temperature` is positive, scores are
           converted into probability distributions using `_get_probabilities_from_scores`.
           Items are then sampled from these distributions with or without replacement.

        **MultioutputScorer exception**: when the scorer is a ``MultioutputScorer``, this
        method returns a ``DataFrame`` of shape ``(n_users, n_items)`` containing the
        predicted class label per item — not an NDArray of top-k item names. ``top_k`` is
        ignored. This is intentional: multi-output classification produces one class
        decision per item rather than a single rankable score, so top-k selection has no
        natural meaning. For per-class probabilities use ``scorer.score_items()``.

        When a retriever is attached, candidates are narrowed per-user before
        ranking: ``retriever.top_k`` controls the candidate pool size and
        ``top_k`` (this parameter) selects the final recommendations from that
        pool. Set ``retriever.top_k`` to at least ``10–20×`` the value of
        ``top_k`` to avoid precision loss from early truncation.

        .. warning::
            **Not thread-safe when a retriever is attached.** The per-user
            retrieval loop mutates shared scorer state via
            ``scorer.set_item_subset()`` / ``scorer.clear_item_subset()``.
            Concurrent calls to ``recommend()`` on the same instance will
            corrupt each other's candidate sets. Use one instance per thread,
            or call ``recommend()`` sequentially.
        """
        if isinstance(self.scorer, MultioutputScorer) or isinstance(self.scorer, MulticlassScorer):
            if users is not None:
                raise ValueError("For this scorer, users should be set to None!")
        if isinstance(self.scorer, MultioutputScorer):
            return self._recommend_multioutput(
                interactions=interactions,
                top_k=top_k,
                sampling_temperature=sampling_temperature or 0,
                replace=replace,
            )

        if self.retriever is None:
            # Normalise None → 0 (deterministic) so BaseRecommender.recommend(),
            # which expects a plain float, doesn't receive None and raise TypeError.
            return super().recommend(interactions, users, top_k, sampling_temperature or 0, replace)

        # Per-user retrieval loop
        return self._recommend_with_retriever(interactions, users, top_k, sampling_temperature, replace)

    def _recommend_multioutput(
        self,
        interactions: Optional[DataFrame],
        top_k: int,
        sampling_temperature: float,
        replace: bool,
    ) -> NDArray:
        """Top-K ranking for ``MultioutputScorer`` with degenerate-target filtering.

        Classifier mode only — predicted values across heterogeneous targets
        aren't comparable in regressor mode, so cross-target ranking isn't
        defined there. Drops degenerate targets (constant predictions would
        tie at the top of every per-user ranking and dominate top-K) before
        ranking; the eval path applies the same filter for the same reason.
        """
        scorer: MultioutputScorer = self.scorer  # type: ignore[assignment]
        if not scorer.is_classifier:
            raise ValueError(
                "recommend() requires a classifier-mode MultioutputScorer (binary targets). "
                "Predicted regression values across heterogeneous targets aren't comparable, "
                "so cross-target ranking isn't well-defined. Use "
                "scorer.predict_targets() for per-target point estimates instead."
            )
        if sampling_temperature < 0:
            raise ValueError("sampling_temperature cannot be negative.")

        score_df = scorer.score_items_per_target(interactions=interactions, users=None)
        # Filter degenerate targets — under DegenerateTargetPolicy.CONSTANT they
        # would emit a constant 1.0 per user and dominate every top-K ranking.
        kept = [c for c in score_df.columns if c not in scorer.degenerate_targets]
        if not kept:
            raise ValueError(
                "All targets are degenerate (single-class in training); recommend() "
                "would return only constant predictions. Drop the degenerate columns or "
                "use scorer.predict_classes() to retrieve the constant per-target labels."
            )
        if scorer.degenerate_targets and not self._warned_degenerate_recommend:
            # Warn once per recommender instance — the degenerate-target
            # manifest is fixed at fit time, so the warning content doesn't
            # change between calls. On a high-QPS serving path the unthrottled
            # version would log identical lines per request.
            logger.warning(
                "Excluding %d degenerate target(s) from recommend() rankings: %s. "
                "Constant predictions on these targets would dominate top-K. "
                "(This warning fires once per recommender; subsequent recommend() "
                "calls on the same instance silently apply the same filter.)",
                len(scorer.degenerate_targets),
                sorted(scorer.degenerate_targets.keys()),
            )
            self._warned_degenerate_recommend = True

        scores_np = score_df[kept].to_numpy()
        item_names_arr = np.array(kept, dtype=object)
        n_items = scores_np.shape[1]
        if top_k > n_items:
            logger.warning(
                "Requested top_k (%d) is larger than non-degenerate targets (%d). Returning %d.",
                top_k,
                n_items,
                n_items,
            )
            top_k = n_items

        if sampling_temperature == 0:
            recommended_idx = self._recommend_from_scores(scores_np, top_k=top_k)
        else:
            # Reuse the base recommender's sampling machinery so the seeding
            # contract (np.random.default_rng) and the without-replacement
            # numerical-stability path (sample_without_replacement_2d, which
            # adds a float-eps to each row to avoid ValueError when softmax
            # underflows produce exact zeros) match the universal/multiclass/
            # independent paths exactly.
            probabilities = self._get_probabilities_from_scores(scores_np, sampling_temperature)
            recommended_idx = self._sample_from_probabilities(probabilities, top_k, replace)
        return item_names_arr[recommended_idx]

    def _recommend_with_retriever(
        self,
        interactions: Optional[DataFrame],
        users: Optional[DataFrame],
        top_k: int,
        sampling_temperature: Optional[float],
        replace: bool,
    ) -> NDArray:
        """Run retrieval then ranking per user, collect results.

        Shape contract
        --------------
        Returns an NDArray of shape ``(n_users, top_k)`` when every user's
        candidate set has at least ``top_k`` items. If ``top_k`` exceeds the
        available candidates for a user, that user's row is shorter and a
        warning is logged — the final ``np.array(results)`` becomes a
        dtype=object array of shape ``(n_users,)`` rather than
        ``(n_users, top_k)``.

        Built-in retrievers always return the same candidate count for every
        user (``min(retriever.top_k, catalog_size)``), so this degenerate case
        cannot arise in normal usage. Custom ``BaseCandidateRetriever``
        implementations must respect the same invariant if a rectangular
        output array is required.
        """
        if sampling_temperature is not None and sampling_temperature < 0:
            raise ValueError("sampling_temperature cannot be negative.")

        if self.scorer.item_subset is not None:
            logger.warning(
                "RankingRecommender: an external item_subset is set AND a retriever is attached. "
                "The retriever will override item_subset for each user and clear_item_subset() will "
                "be called after scoring. Remove the retriever or do not call set_item_subset() when "
                "a retriever is in use — these two features are mutually exclusive."
            )

        # Preprocess once for all users — avoids N redundant schema validations in the loop.
        interactions_proc, users_proc = self._preprocess_inputs(interactions, users)

        # Determine user IDs from preprocessed data.
        # NOTE: when both interactions and users are provided, user IDs come from
        # interactions only. Users present solely in the users DataFrame (no
        # interactions) are not scored in the retriever path. This differs from
        # the non-retriever path, which scores all users via the scorer join.
        if interactions_proc is not None and USER_ID_NAME in interactions_proc.columns:
            user_ids = interactions_proc[USER_ID_NAME].unique().tolist()
        elif users_proc is not None and USER_ID_NAME in users_proc.columns:
            user_ids = users_proc[USER_ID_NAME].unique().tolist()
        else:
            # Fall back to no-retrieval path if we can't identify users.
            logger.warning(
                "RankingRecommender: retriever set but USER_ID not found in inputs — "
                "falling back to full-catalog scoring."
            )
            return super().recommend(interactions_proc, users_proc, top_k, sampling_temperature or 0, replace)

        candidates_per_user = self.retriever.retrieve(user_ids, top_k=self.retriever.top_k)

        # Pre-group once to avoid an O(n_users × n_rows) boolean-index scan per user.
        interaction_groups = (
            dict(tuple(interactions_proc.groupby(USER_ID_NAME, sort=False)))
            if interactions_proc is not None and USER_ID_NAME in interactions_proc.columns
            else {}
        )
        user_groups = (
            dict(tuple(users_proc.groupby(USER_ID_NAME, sort=False)))
            if users_proc is not None and USER_ID_NAME in users_proc.columns
            else {}
        )

        results = []
        for user_id in user_ids:
            candidates = candidates_per_user.get(user_id, [])
            candidates_str = [str(c) for c in candidates]

            user_interactions = interaction_groups.get(user_id)
            user_users = user_groups.get(user_id)

            if candidates_str:
                self.scorer.set_item_subset(candidates_str)
            else:
                logger.warning(
                    "RankingRecommender: retriever returned no candidates for user %s — "
                    "falling back to full-catalog scoring.",
                    user_id,
                )

            try:
                # Call scorer directly — data is already preprocessed above.
                # CONTRACT: _score_items_np must not alter the active item subset;
                # active_item_names must be captured after scoring to reflect the
                # restricted subset set by set_item_subset() above.
                scores_np = self.scorer._score_items_np(user_interactions, user_users)
                active_item_names = self._get_item_names()

                if not sampling_temperature:  # None or 0 → deterministic ranking
                    available = self._get_available_items_count()
                    if top_k > available:
                        logger.warning(
                            "Requested top_k (%d) is larger than available items (%d). Will return only %d items.",
                            top_k,
                            available,
                            available,
                        )
                    recommended_idx = self._recommend_from_scores(scores_np, top_k)
                else:
                    probabilities = self._get_probabilities_from_scores(scores_np, sampling_temperature)
                    recommended_idx = self._sample_from_probabilities(probabilities, top_k, replace)

                # _score_items_np is called with one user at a time, so scores_np is
                # shape (1, n_items), recommended_idx is (1, top_k), and [0] extracts
                # the single row to yield a flat (top_k,) array of item names.
                results.append(active_item_names[recommended_idx][0])
            finally:
                self.scorer.clear_item_subset()

        return np.array(results)

    def recommend_online(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
        top_k: int = 1,
    ) -> NDArray:
        """
        Real-time single-user recommendation without join overhead.

        .. warning::
            **The attached retriever is not used.** ``recommend_online()`` scores
            the full item catalog for low-latency serving and does not run the
            retrieval stage. Use ``recommend()`` if retriever-aware recommendations
            are required.
        """
        if self.retriever is not None:
            logger.warning(
                "recommend_online() does not use the attached retriever — the full item "
                "catalog is scored. Call recommend() instead for retriever-aware recommendations."
            )
        return super().recommend_online(interactions=interactions, users=users, top_k=top_k)

    def _recommend_from_scores(self, scores: NDArray[np.float64], top_k: int = 1) -> NDArray[np.int_]:
        return scores.argsort()[:, ::-1][:, :top_k]

    # ------------------------------------------------------------------
    # MultioutputScorer-aware evaluate
    # ------------------------------------------------------------------
    #
    # Architectural note: MultioutputScorer is multi-label classification
    # (binary-only, enforced at fit time) or multi-target regression in
    # regressor mode. So this override:
    #
    #   1. For classifier mode: ranking metrics (NDCG@K, MRR@K, ...) are
    #      well-defined because every target is binary — collapse to a
    #      ``(N, n_targets)`` per-target positive-class score matrix, drop
    #      degenerate targets (constant predictions would tie at the top
    #      of every ranking and bias the metric), build per-user ranks,
    #      and dispatch to ``metric.calculate``. Classification metrics
    #      (ROC_AUC, PR_AUC) compute per-label and either macro-average
    #      or return per-target dict.
    #   2. For regressor mode: regression metrics (RMSE, MAE) compute
    #      per-target. Ranking and classification metrics are rejected —
    #      predicted values across heterogeneous targets aren't comparable
    #      and binary-classification semantics don't apply.
    #   3. Returns a macro-averaged scalar by default to preserve the
    #      ``evaluate() -> float`` contract callers expect; opt-in
    #      ``per_label=True`` returns a ``Dict[str, float]`` for
    #      diagnostics. ``per_label=True`` is rejected for ranking metrics
    #      (they're inherently cross-target).
    #
    # LSP note: BaseRecommender.evaluate is annotated ``-> float`` but this
    # override widens to ``Union[float, Dict[str, float]]``. This is a
    # deliberate widening — the broader return type only fires when the
    # scorer is MultioutputScorer AND ``per_label=True``. The @overload
    # declarations below carry the precise contract for type-checkers:
    # ``per_label=True`` returns ``Dict[str, float]``; the default
    # ``per_label=False`` returns ``float``, so existing call sites that
    # assign to a float-typed variable continue to type-check cleanly.
    #
    # The base BaseRecommender.evaluate() path is unchanged for every other
    # scorer. This is the only special case.

    @overload
    def evaluate(
        self,
        eval_type: RecommenderEvaluatorType,
        metric_type: RecommenderMetricType,
        eval_top_k: Optional[int] = ...,
        temperature: float = ...,
        score_items_kwargs: Optional[Mapping[str, DataFrame]] = ...,
        eval_kwargs: Optional[Mapping[str, Any]] = ...,
        eval_factory_kwargs: Optional[Mapping[str, Any]] = ...,
        per_label: Literal[False] = ...,
    ) -> float: ...

    @overload
    def evaluate(
        self,
        eval_type: RecommenderEvaluatorType,
        metric_type: RecommenderMetricType,
        eval_top_k: Optional[int] = ...,
        temperature: float = ...,
        score_items_kwargs: Optional[Mapping[str, DataFrame]] = ...,
        eval_kwargs: Optional[Mapping[str, Any]] = ...,
        eval_factory_kwargs: Optional[Mapping[str, Any]] = ...,
        *,
        per_label: Literal[True],
    ) -> Dict[str, float]: ...

    def evaluate(
        self,
        eval_type: RecommenderEvaluatorType,
        metric_type: RecommenderMetricType,
        eval_top_k: Optional[int] = None,
        temperature: float = 1.0,
        score_items_kwargs: Optional[Mapping[str, DataFrame]] = None,
        eval_kwargs: Optional[Mapping[str, Any]] = None,
        eval_factory_kwargs: Optional[Mapping[str, Any]] = None,
        per_label: bool = False,
    ) -> Union[float, Dict[str, float]]:
        """Evaluate the recommender.

        For most scorers this delegates to :meth:`BaseRecommender.evaluate`
        unchanged. For :class:`MultioutputScorer` it switches to
        per-label / per-target classification or regression — see the
        architectural note in the source for why.

        Args:
            eval_type: Evaluator strategy. For ``MultioutputScorer`` only
                ``RecommenderEvaluatorType.SIMPLE`` is supported (the
                others assume a ranking-recommender shape that doesn't
                apply to multi-label classification).
            metric_type: Metric to compute. For ``MultioutputScorer``,
                must be a classification metric (``ROC_AUC`` / ``PR_AUC``)
                in classifier mode or a regression metric (``RMSE`` /
                ``MAE``) in regressor mode. Ranking metrics raise
                ``ValueError``.
            eval_top_k: Cutoff for ranking metrics; ignored for
                classification and regression metrics.
            temperature: Temperature for softmax score-to-probability
                conversion (non-multioutput path only).
            score_items_kwargs: Forwarded to ``score_items``.
            eval_kwargs: Must include ``logged_items`` of shape
                ``(n_users, n_targets)`` carrying target column names and
                ``logged_rewards`` of shape ``(n_users, n_targets)``
                carrying ground-truth values aligned with
                ``self.scorer.item_names``.
            eval_factory_kwargs: Forwarded to the evaluator factory
                (non-multioutput path only).
            per_label: When ``True`` and the scorer is
                ``MultioutputScorer``, return a ``Dict[str, float]``
                mapping each target column to its metric value. When
                ``False`` (default), return a macro-averaged scalar.
                Ignored for non-multioutput scorers.

        Returns:
            ``float`` for non-multioutput scorers. For
            ``MultioutputScorer``: ``float`` (macro-averaged) by default,
            or ``Dict[str, float]`` when ``per_label=True``.
        """
        if isinstance(self.scorer, MultioutputScorer):
            return self._evaluate_multioutput(
                eval_type=eval_type,
                metric_type=metric_type,
                eval_top_k=eval_top_k,
                score_items_kwargs=score_items_kwargs,
                eval_kwargs=eval_kwargs,
                per_label=per_label,
            )
        return super().evaluate(
            eval_type=eval_type,
            metric_type=metric_type,
            eval_top_k=eval_top_k,
            temperature=temperature,
            score_items_kwargs=score_items_kwargs,
            eval_kwargs=eval_kwargs,
            eval_factory_kwargs=eval_factory_kwargs,
        )

    def _evaluate_multioutput(
        self,
        eval_type: RecommenderEvaluatorType,
        metric_type: RecommenderMetricType,
        eval_top_k: Optional[int],
        score_items_kwargs: Optional[Mapping[str, DataFrame]],
        eval_kwargs: Optional[Mapping[str, Any]],
        per_label: bool,
    ) -> Union[float, Dict[str, float]]:
        scorer: MultioutputScorer = self.scorer  # type: ignore[assignment]
        metric = RecommenderMetricFactory.create(metric_type)

        # Mode/metric compatibility checks — ranking metrics need binary
        # classifier mode (the binary-only enforcement at fit time guarantees
        # the per-target positive-class score matrix is well-defined);
        # regression metrics need regressor mode; classification metrics need
        # classifier mode.
        if isinstance(metric, BaseRankingMetric) and not scorer.is_classifier:
            raise ValueError(
                f"Ranking metric {metric_type.name} requires a classifier-mode "
                f"MultioutputScorer (binary targets). Predicted regression values across "
                f"heterogeneous targets aren't comparable, so cross-target ranking isn't "
                f"defined. Use RMSE / MAE for per-target regression metrics, or rebuild "
                f"with a classifier estimator."
            )
        if scorer.is_classifier and isinstance(metric, BaseRegressionMetric):
            raise ValueError(
                f"Regression metric {metric_type.name} requires a regressor estimator, but "
                f"this MultioutputScorer wraps a classifier. Use ROC_AUC / PR_AUC for "
                f"classification metrics, NDCG_AT_K / Precision@K for ranking metrics, or "
                f"rebuild the scorer with a regressor estimator."
            )
        if not scorer.is_classifier and isinstance(metric, BaseClassificationMetric):
            raise ValueError(
                f"Classification metric {metric_type.name} requires a classifier estimator, "
                f"but this MultioutputScorer wraps a regressor. Use RMSE / MAE for regression "
                f"metrics, or rebuild the scorer with a classifier estimator."
            )

        if eval_type != RecommenderEvaluatorType.SIMPLE:
            raise ValueError(
                f"MultioutputScorer evaluation only supports RecommenderEvaluatorType.SIMPLE; "
                f"got {eval_type.name}. Counterfactual evaluators (IPS, DR, SNIPS) assume a "
                f"long-format ranking-recommender shape that doesn't apply to multi-label "
                f"classification or multi-target regression."
            )

        # An active item_subset narrows score_items / score_items_per_target
        # output to subset-only columns, but the eval loop iterates the full
        # scorer.item_names catalogue and indexes logged_rewards against the
        # full catalogue. Mixing the two would produce a KeyError on per-label
        # lookups and an IndexError on the ranking-branch column slicing.
        # Reject explicitly rather than silently swallowing the subset — the
        # caller can `clear_item_subset()` before evaluating, then re-set if
        # needed. Same constraint exists for `recommend()` only when ALL
        # subset members are degenerate; the partial case there works because
        # `_recommend_multioutput` builds `kept` from `score_df.columns` (the
        # subset), not from `scorer.item_names`.
        if scorer.item_subset is not None:
            raise ValueError(
                "MultioutputScorer evaluation does not support an active item_subset. "
                "evaluate() iterates the full target catalogue and indexes "
                "logged_rewards against scorer.item_names, which becomes inconsistent "
                "when the subset narrows score_items output. Call "
                "scorer.clear_item_subset() before evaluate(), or filter your "
                "logged_rewards / logged_items columns yourself to match the subset."
            )

        if not score_items_kwargs:
            raise ValueError(
                "MultioutputScorer evaluation requires score_items_kwargs={'interactions': df} "
                "to compute predictions for the validation slice."
            )
        # MultioutputScorer.score_items / score_items_per_target reject a
        # `users` kwarg outright. Catch it here at the eval boundary so the
        # caller gets a precise error pointing at the right shape rather
        # than an opaque scorer-internal ValueError.
        if "users" in score_items_kwargs and score_items_kwargs["users"] is not None:
            raise ValueError(
                "MultioutputScorer doesn't accept a `users` DataFrame; pass user features "
                "as plain columns inside score_items_kwargs['interactions'] instead, and "
                "drop the `users` key from score_items_kwargs."
            )
        if not eval_kwargs or "logged_items" not in eval_kwargs or "logged_rewards" not in eval_kwargs:
            raise ValueError(
                "MultioutputScorer evaluation requires eval_kwargs with "
                "'logged_items' (n_users, n_targets) carrying target column names and "
                "'logged_rewards' (n_users, n_targets) carrying ground-truth values."
            )

        logged_items = np.asarray(eval_kwargs["logged_items"], dtype=object)
        logged_rewards = np.asarray(eval_kwargs["logged_rewards"], dtype=float)
        if logged_items.shape != logged_rewards.shape:
            raise ValueError(
                f"logged_items shape {logged_items.shape} must match logged_rewards shape {logged_rewards.shape}."
            )
        if logged_rewards.shape[1] != len(scorer.item_names):
            raise ValueError(
                f"logged_rewards has {logged_rewards.shape[1]} target columns but the scorer "
                f"was trained on {len(scorer.item_names)} targets. Ensure logged_rewards "
                f"columns align with scorer.item_names."
            )
        # Row-count alignment between scoring inputs and ground truth: if a
        # caller hands us a 100-row valid_df but a 80-row logged_rewards
        # (or vice versa), per-label slicing silently misaligns or errors
        # deep in numpy. Catch it explicitly.
        interactions_df = score_items_kwargs.get("interactions")
        if interactions_df is not None and len(interactions_df) != logged_rewards.shape[0]:
            raise ValueError(
                f"interactions has {len(interactions_df)} rows but logged_rewards has "
                f"{logged_rewards.shape[0]} rows. Pass the same row-aligned slice for both."
            )

        # The wide-multioutput contract has every row of logged_items equal
        # to the same target-name vector — read the first row to build a
        # name → column-index map (cheap; per-row indexing isn't needed).
        first_row = [str(x) for x in logged_items[0]]
        if len(first_row) != len(set(first_row)):
            # Duplicates in the name vector would make logged_col_index map
            # only the last occurrence and silently miscolumn the rewards.
            duplicates = sorted({n for n in first_row if first_row.count(n) > 1})
            raise ValueError(
                f"logged_items[0] contains duplicate target name(s) {duplicates}. "
                f"Each target must appear exactly once per row."
            )
        if set(first_row) != set(scorer.item_names):
            raise ValueError(
                f"logged_items[0] target names {first_row} do not match scorer.item_names "
                f"{list(scorer.item_names)}. Wide-multioutput evaluation expects every row of "
                f"logged_items to be the scorer's target catalogue."
            )
        logged_col_index = {name: i for i, name in enumerate(first_row)}

        # Reorder logged_rewards columns to scorer.item_names order so all
        # downstream computations index by the canonical catalogue.
        canonical_rewards = np.column_stack([logged_rewards[:, logged_col_index[name]] for name in scorer.item_names])

        # Symmetric binary contract for classifier mode: training y was
        # validated to be {0, 1}, so the held-out logged_rewards must also
        # be {0, 1} (NaN allowed for ignore-mask / not-observed). Without
        # this enforcement, a caller passing continuous-ish values like
        # [0.0, 0.5, 0.0] would slip past the eval-side single-class gate
        # (nunique=2) but downstream metrics binarize via `> 0.5` and
        # silently emit 0.0, poisoning macro means and ranking values.
        # Reject explicitly with the same migration hint as `_validate_targets`.
        # Skip degenerate targets — they're NaN-excluded by the per-label
        # short-circuit below (classification metrics) or filtered out
        # entirely (ranking metrics), so non-binary values on those
        # columns can't poison the metric and don't warrant an error
        # pointing at columns the user can't usefully fix.
        if scorer.is_classifier:
            # item_names is an ndarray after BaseScorer._process_items; use a
            # dict for index lookup (ndarray has no .index() method).
            canonical_index = {name: i for i, name in enumerate(scorer.item_names)}
            bad_rewards: List[Tuple[str, list]] = []
            for label in scorer.item_names:
                if label in scorer.degenerate_targets:
                    continue
                col = canonical_rewards[:, canonical_index[label]]
                col_clean = col[~np.isnan(col)]
                non_binary = set(col_clean.tolist()) - {0.0, 1.0}
                if non_binary:
                    bad_rewards.append((label, sorted(non_binary, key=str)[:5]))
            if bad_rewards:
                details = "; ".join(
                    f"{label!r} (saw values: {vals}{'...' if len(vals) == 5 else ''})" for label, vals in bad_rewards
                )
                raise ValueError(
                    f"Classifier-mode MultioutputScorer evaluation requires logged_rewards "
                    f"to be binary numeric — values strictly in {{0, 1}} (or {{0.0, 1.0}}, "
                    f"NaN allowed for ignore-mask). The binary contract is symmetric: "
                    f"training y was validated as {{0, 1}}, so held-out ground truth "
                    f"must be too. Non-binary values seen in: {details}. Pre-encode at "
                    f"the caller before passing to evaluate(): "
                    f"df_eval[col] = (df_eval[col] == 'yes').astype(float)."
                )

        # Branch 1: ranking metric (classifier mode only — guarded above).
        # Cross-target ranking is well-defined because every target is binary
        # (enforced at fit). Compute (N, n_targets) of P(positive) per target,
        # build per-user ranks, dispatch to metric.calculate.
        if isinstance(metric, BaseRankingMetric):
            if per_label:
                raise ValueError(
                    "per_label=True is incompatible with ranking metrics — they aggregate "
                    "across all targets per user, not per target. Use per_label=True only "
                    "with classification (ROC_AUC / PR_AUC) or regression (RMSE / MAE) metrics."
                )
            # Drop degenerate targets — they always score a constant 1.0 (under
            # CONSTANT policy) and would tie at the top of every per-user
            # ranking, biasing NDCG / Precision@K / etc. toward whatever
            # number of degenerate targets the train slice produced. We
            # exclude them, log a warning, and let the metric run on the
            # informative subset.
            keep_indices = [i for i, name in enumerate(scorer.item_names) if name not in scorer.degenerate_targets]
            if not keep_indices:
                raise ValueError(
                    "All targets are degenerate (single-class in training); cannot compute "
                    "a meaningful ranking metric. Drop the degenerate columns or use "
                    "classification metrics with per_label=True instead."
                )
            score_matrix_full = scorer.score_items_per_target(**score_items_kwargs).to_numpy()
            if scorer.degenerate_targets:
                logger.warning(
                    "Excluding %d degenerate target(s) from ranking metric computation: %s. "
                    "Constant predictions on these targets would bias the metric.",
                    len(scorer.degenerate_targets),
                    sorted(scorer.degenerate_targets.keys()),
                )
            score_matrix = score_matrix_full[:, keep_indices]
            ranking_rewards = canonical_rewards[:, keep_indices]
            sorted_idx = np.argsort(-score_matrix, axis=1)
            ranks = np.empty_like(sorted_idx)
            np.put_along_axis(ranks, sorted_idx, np.arange(score_matrix.shape[1]), axis=1)
            value = metric.calculate(
                recommendation_ranks=ranks,
                modified_rewards=ranking_rewards,
                recommendation_scores=score_matrix,
                top_k=eval_top_k,
            )
            return float(value)

        # Branch 2: per-label classification metric (classifier mode), or
        # per-target regression metric (regressor mode). Computed
        # independently per target and either macro-averaged or returned
        # as a Dict.
        scores_df = scorer.score_items(**score_items_kwargs)
        dummy_ranks = np.empty((logged_rewards.shape[0], 1), dtype=int)
        # item_names becomes an ndarray after BaseScorer._process_items runs;
        # ndarrays don't have .index(), so build a name → position lookup once.
        canonical_index = {name: i for i, name in enumerate(scorer.item_names)}

        per_label_results: Dict[str, float] = {}
        for label in scorer.item_names:
            # Two reasons a per-label classification metric is undefined:
            #   1. Training-slice degeneracy — recorded under CONSTANT policy
            #      in scorer.degenerate_targets; the predictor is constant
            #      so any classification metric is uninformative.
            #   2. Validation-slice degeneracy — y_true has only one class
            #      in the held-out slice. ROCAUCMetric returns 0.0 (line 70)
            #      and PRAUCMetric returns nan_to_num(ap, 0.0) (line 72) —
            #      both poison the macro mean with a 0 instead of signalling
            #      "undefined". We detect both upstream and emit nan so the
            #      macro filter excludes them and per_label=True callers see
            #      the undefined cells explicitly.
            #
            # Skipped only for classification metrics; regression metrics
            # are well-defined on single-valued y_true (they collapse to
            # max-abs-deviation and similar — meaningful for constant targets).
            if scorer.is_classifier and label in scorer.degenerate_targets:
                per_label_results[label] = float("nan")
                continue
            y_true = canonical_rewards[:, canonical_index[label]]
            if scorer.is_classifier:
                # Validation-slice degeneracy gate. The binary-only contract
                # guarantees ITEM_<name> is in {0, 1} at fit time, so
                # validation-slice y_true should be binary too — `nunique < 2`
                # after dropping NaN is the exact condition under which the
                # classification metric is undefined. No threshold-aware
                # binarization needed: there's no ambiguity between
                # "<= threshold" and ">= threshold" when values are
                # already 0/1.
                y_true_clean = y_true[~np.isnan(y_true)]
                if len(np.unique(y_true_clean)) < 2:
                    per_label_results[label] = float("nan")
                    continue
                col_name = scorer.positive_proba_column_name(label)
                y_score = scores_df[col_name].to_numpy(dtype=float)
            else:
                y_score = scores_df[label].to_numpy(dtype=float)
            value = metric.calculate(
                recommendation_ranks=dummy_ranks,
                modified_rewards=y_true.reshape(-1, 1),
                recommendation_scores=y_score.reshape(-1, 1),
                top_k=None,
            )
            per_label_results[label] = float(value)

        if per_label:
            return per_label_results
        values = np.array(list(per_label_results.values()), dtype=float)
        finite = values[~np.isnan(values)]
        if finite.size == 0:
            # Match the ranking branch's behaviour: raise rather than
            # silently return NaN (which can flow into downstream
            # comparisons and silently fail). Caller sees a precise error
            # naming the cause.
            raise ValueError(
                "All targets produced undefined per-label metrics — every column was "
                "either (1) training-slice degenerate (under CONSTANT policy), (2) "
                "single-class in the validation slice after binarization at the "
                "metric's threshold, or (3) all-NaN in the validation slice. The "
                "macro mean is undefined. Drop degenerate columns, ensure the "
                "validation split has both classes per target, drop or impute "
                "NaN-heavy targets, or call evaluate(..., per_label=True) to "
                "inspect the per-target NaN manifest directly and see which "
                "category each target fell into."
            )
        return float(finite.mean())
