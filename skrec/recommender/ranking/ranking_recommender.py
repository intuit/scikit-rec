import copy
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
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
from skrec.scorer.mixed_type_multi_target import (
    TARGET_TYPE_TO_METRICS,
    MixedTypeMultiTargetScorer,
    TargetType,
)
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
        # Symmetric rejection for any per-target scorer (capability-flag
        # check rather than an isinstance ladder — picks up future
        # per-target scorers automatically). recommend() short-circuits
        # to predict_targets for these scorers and never consults the
        # retriever; quietly attaching one would build the index in
        # train() and silently discard it at recommend time.
        #
        # ``is True`` (not truthy) so MagicMock(spec=BaseScorer) tests —
        # which produce a MagicMock for any attribute access — don't
        # trip this guard. Real scorers set the class attribute to
        # literal True/False.
        if retriever is not None and getattr(scorer, "is_per_target_scorer", False) is True:
            raise ValueError(
                f"{type(scorer).__name__} is a per-target scorer and does not "
                f"support a retriever — recommend() short-circuits to "
                f"predict_targets() for per-target output, so the "
                f"candidate-retrieval-then-rank pattern doesn't apply. "
                f"Drop the retriever to use this scorer."
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
        # Same once-per-instance throttle pattern for the per-target
        # ``top_k != 1`` warning emitted from ``recommend()``. Without
        # this, high-QPS callers using a non-1 default top_k would get
        # a warning on every request — symmetric with
        # ``_warned_degenerate_recommend`` for the multioutput path.
        self._warned_per_target_top_k: bool = False

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
        # Reset the per-target top_k throttle for the same reason:
        # symmetry with the degenerate flag (the constructor docstring
        # describes both flags as once-per-instance, and re-training
        # restarts the "instance lifecycle" in user-facing terms). A
        # caller who tunes top_k mid-experiment then retrains expects
        # to see the warning the first time the new training surfaces
        # it. The flag is API-misuse signalling, not data-dependent,
        # but matching the degenerate reset keeps both flags' lifecycle
        # rules identical and avoids the doc/runtime asymmetry the
        # round-4 review called out.
        self._warned_per_target_top_k = False

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
        # ``is True`` (not just truthy) so a MagicMock(spec=BaseScorer)
        # whose attribute access auto-generates a truthy MagicMock
        # doesn't take this branch — consistent with the constructor /
        # preprocess_inputs / score-fast dispatch checks.
        is_per_target = getattr(self.scorer, "is_per_target_scorer", False) is True
        if isinstance(self.scorer, MultioutputScorer) or isinstance(self.scorer, MulticlassScorer) or is_per_target:
            if users is not None:
                raise ValueError("For this scorer, users should be set to None!")
        if isinstance(self.scorer, MultioutputScorer):
            return self._recommend_multioutput(
                interactions=interactions,
                top_k=top_k,
                sampling_temperature=sampling_temperature or 0,
                replace=replace,
            )
        if is_per_target:
            # Per-target scorers predict per-target values, not item ranks.
            # recommend() short-circuits to predict_targets — same pattern
            # as MultioutputScorer but returns the per-target scorer's wide
            # DataFrame of per-target point estimates rather than per-target
            # class labels. top_k is meaningless here; warn rather than fail
            # because callers passing default top_k=1 still expect to get
            # results. Capability-flag dispatch avoids an isinstance ladder
            # — any future per-target scorer that opts in is handled here
            # without a recommender-side edit.
            if top_k is not None and top_k != 1 and not self._warned_per_target_top_k:
                logger.warning(
                    "Per-target scorer.recommend() ignores top_k; the scorer "
                    "returns one prediction per declared target rather than a "
                    "ranked item list. (This warning fires once per recommender "
                    "instance.)"
                )
                self._warned_per_target_top_k = True
            # Route through _preprocess_inputs so the per-target recommend
            # path gets the same schema apply + type coercion that
            # score_items and evaluate apply. Without this, calling
            # recommend() and score_items() on the same frame would
            # produce inconsistently coerced X (object-dtype columns
            # surviving recommend but being float-coerced through
            # score_items). The OBSERVED_* preservation primitive
            # underneath is the same one the other paths use, so
            # conditional inference via recommend() also Just Works.
            interactions_proc, _ = self._preprocess_inputs(interactions, None)
            return self.scorer.predict_targets(interactions=interactions_proc)

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

        # Shallow-copy the scorer so the per-user item_subset mutations below are
        # confined to this call stack and never touch self.scorer. The trained
        # estimator, item_names, and items_df are shared by reference (read-only
        # after training); only item_subset and item_subset_df differ per call.
        # This makes concurrent recommend() calls on the same instance safe:
        # each gets its own local_scorer with independent item_subset state.
        local_scorer = copy.copy(self.scorer)

        results = []
        for user_id in user_ids:
            candidates = candidates_per_user.get(user_id, [])
            candidates_str = [str(c) for c in candidates]

            user_interactions = interaction_groups.get(user_id)
            user_users = user_groups.get(user_id)

            if candidates_str:
                local_scorer.set_item_subset(candidates_str)
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
                scores_np = local_scorer._score_items_np(user_interactions, user_users)
                active_item_names = (
                    np.asarray(local_scorer.item_subset, dtype=np.str_)
                    if local_scorer.item_subset is not None
                    else local_scorer.item_names
                )

                if not sampling_temperature:  # None or 0 → deterministic ranking
                    available = len(local_scorer.item_subset or local_scorer.item_names or [])
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
                local_scorer.clear_item_subset()

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
        # Capability-flag dispatch: per-target scorers route through the
        # per-target evaluate path. The MixedTypeMultiTargetScorer is the
        # only scorer that opts in today; flipping a future scorer's
        # ``is_per_target_scorer`` to True picks up this dispatch
        # automatically. ``is True`` (not truthy) for consistency with
        # the recommend()/recommend_online/__init__ checks.
        if getattr(self.scorer, "is_per_target_scorer", False) is True:
            return self._evaluate_mixed_type_multi_target(
                eval_type=eval_type,
                metric_type=metric_type,
                score_items_kwargs=score_items_kwargs,
                eval_kwargs=eval_kwargs,
            )
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

    # --- MixedTypeMultiTargetScorer-aware evaluate ---------------------- #
    #
    # Per-target dispatch by declared TargetType. Always returns
    # ``Dict[str, float]`` — heterogeneous target types have no honest
    # macro aggregation. Restricted to ``RecommenderEvaluatorType.SIMPLE``;
    # ranking metrics rejected with an explicit pointer at
    # ``score_per_target`` and ``predict_targets``.
    def _evaluate_mixed_type_multi_target(
        self,
        eval_type: RecommenderEvaluatorType,
        metric_type: Union[RecommenderMetricType, Dict[str, RecommenderMetricType]],
        score_items_kwargs: Optional[Mapping[str, DataFrame]],
        eval_kwargs: Optional[Mapping[str, Any]],
    ) -> Dict[str, float]:
        scorer: MixedTypeMultiTargetScorer = self.scorer  # type: ignore[assignment]

        if eval_type != RecommenderEvaluatorType.SIMPLE:
            raise ValueError(
                f"MixedTypeMultiTargetScorer evaluation only supports "
                f"RecommenderEvaluatorType.SIMPLE; got {eval_type.name}. "
                f"Counterfactual evaluators (IPS, DR, SNIPS) assume a long-format "
                f"ranking-recommender shape that does not apply to per-target "
                f"prediction."
            )

        # Reject any non-interactions kwarg (including non-None ``users``)
        # explicitly. MultioutputScorer's evaluate path rejects
        # non-None users for the same reason — per-target / per-label
        # scorers don't merge a separate users frame, so passing one
        # here silently does nothing and misleads the caller. Symmetric
        # rejection avoids that surprise.
        if score_items_kwargs:
            for k, v in score_items_kwargs.items():
                if k == "interactions":
                    continue
                if k == "users" and v is None:
                    continue
                raise ValueError(
                    f"score_items_kwargs[{k!r}] is not supported by "
                    f"per-target MixedTypeMultiTargetScorer.evaluate. "
                    f"Only {{'interactions': df}} is accepted (and an "
                    f"optional users=None, ignored for symmetry with "
                    f"MultioutputScorer)."
                )
        if not score_items_kwargs or "interactions" not in score_items_kwargs:
            raise ValueError(
                "MixedTypeMultiTargetScorer.evaluate requires score_items_kwargs"
                "={'interactions': df} to compute predictions."
            )

        interactions = score_items_kwargs["interactions"]
        if not eval_kwargs or "logged_rewards" not in eval_kwargs:
            raise ValueError(
                "MixedTypeMultiTargetScorer evaluation requires eval_kwargs with "
                "'logged_rewards' (a wide DataFrame matching predict_targets's "
                "output column set — one column per fanned-out target)."
            )
        logged_rewards: DataFrame = eval_kwargs["logged_rewards"]
        if not isinstance(logged_rewards, DataFrame):
            raise TypeError(f"'logged_rewards' must be a wide DataFrame; got {type(logged_rewards).__name__}.")

        required = list(scorer._fanned_out_target_columns)
        missing = set(required) - set(logged_rewards.columns)
        extra = set(logged_rewards.columns) - set(required)
        if missing:
            raise ValueError(f"logged_rewards missing target column(s): {sorted(missing)}.")
        if extra:
            raise ValueError(f"logged_rewards has unknown column(s) not in target_specs: {sorted(extra)}.")
        if len(logged_rewards) != len(interactions):
            raise ValueError(
                f"interactions has {len(interactions)} rows but logged_rewards "
                f"has {len(logged_rewards)} rows. Pass row-aligned slices for both."
            )

        # Hoist the interactions-side validator above the per-column
        # logged_rewards checks. A caller with BOTH a malformed
        # interactions frame (vanilla estimator + OBSERVED_*, orphan
        # ITEM_*, partial multilabel group, …) AND a malformed
        # logged_rewards frame sees the more architectural error first
        # — fixing logged_rewards then discovering the OBSERVED problem
        # on a second run would cost an extra debug cycle. Schema
        # apply + preserve + validate happen in one shot so the loop
        # below already knows the interactions are well-formed.
        interactions_proc, _users_proc = self._preprocess_inputs(interactions, None)
        scorer._validate_inference_interactions(interactions_proc)

        # Per-column type validation against the declared target_specs.
        # Fail-fast at the eval boundary; without this, a string-valued
        # regression column or a stray binary value silently propagates
        # into the metric and poisons the result. NaNs are tolerated for
        # every type (per the v2 plan's "ignore-mask" semantics).
        multiclass_classes = scorer._get_multiclass_classes()
        for fanned_name in required:
            target_type = scorer._target_type_for_fanned(fanned_name)
            col = logged_rewards[fanned_name]
            non_nan = col.dropna()
            if non_nan.empty:
                continue  # all-NaN columns are handled later by per-metric NaN policy
            if target_type in (TargetType.BINARY, TargetType.MULTILABEL):
                # MULTILABEL members are binary at the fanned-out level.
                if not pd.api.types.is_numeric_dtype(non_nan):
                    raise ValueError(
                        f"logged_rewards column {fanned_name!r} (declared "
                        f"{target_type.value}) must be numeric in {{0, 1}}; "
                        f"got dtype {col.dtype}."
                    )
                unique_vals = set(np.asarray(non_nan).tolist())
                allowed = {0, 1, 0.0, 1.0, True, False}
                if not unique_vals.issubset(allowed):
                    raise ValueError(
                        f"logged_rewards column {fanned_name!r} (declared "
                        f"{target_type.value}) has values outside "
                        f"{{0, 1}}: {sorted(unique_vals, key=str)}."
                    )
            elif target_type == TargetType.REGRESSION:
                if not pd.api.types.is_numeric_dtype(col):
                    raise ValueError(
                        f"logged_rewards column {fanned_name!r} (declared "
                        f"REGRESSION) must be numeric; got dtype {col.dtype}."
                    )
                # inf would silently propagate into the metric (e.g. MSE
                # → inf, MAE → inf) and poison every downstream
                # aggregation. NaN stays allowed — that's the "row has no
                # logged outcome" signal — but inf is unambiguously
                # malformed input. Reject upfront with the offending
                # row count so the caller can locate the source.
                col_arr = np.asarray(non_nan, dtype=np.float64)
                inf_count = int(np.isinf(col_arr).sum())
                if inf_count:
                    raise ValueError(
                        f"logged_rewards column {fanned_name!r} (declared "
                        f"REGRESSION) contains {inf_count} non-finite "
                        f"value(s) (inf/-inf). NaN is allowed (treated as "
                        f"'no logged outcome' and masked from the metric); "
                        f"inf is not."
                    )
            elif target_type == TargetType.MULTICLASS:
                catalogue = set(multiclass_classes.get(fanned_name, []))
                if not catalogue:
                    # Estimator wasn't fitted with a multiclass catalogue —
                    # skip (this path is unreachable in practice; defensive).
                    continue
                unknown = set(non_nan.tolist()) - catalogue
                if unknown:
                    raise ValueError(
                        f"logged_rewards column {fanned_name!r} (declared "
                        f"MULTICLASS) contains label(s) not in the training-"
                        f"time class catalogue: {sorted(unknown, key=str)}. "
                        f"Catalogue: {sorted(catalogue, key=str)}."
                    )

        # Compute predictions once. Route through scorer._estimator_predict_proba
        # rather than calling the estimator's predict_proba_dict directly —
        # the wrapper builds the OBSERVED_* → observed dict for conditional
        # estimators (v3). Going direct would silently bypass conditioning
        # at evaluate time even when the caller supplied OBSERVED_* columns
        # in the interactions frame. (interactions_proc was prepared and
        # validated upstream of the per-column type loop above.)
        X_inference = scorer._extract_X_inference(interactions_proc)
        proba_dict = scorer._estimator_predict_proba(X_inference, interactions_proc)

        result: Dict[str, float] = {}
        for fanned_name in required:
            target_type = scorer._target_type_for_fanned(fanned_name)
            # Resolve metric for this fanned-out target.
            if isinstance(metric_type, dict):
                if fanned_name not in metric_type:
                    raise ValueError(
                        f"metric_type dict is missing entry for target "
                        f"{fanned_name!r}. Provide entries for every "
                        f"declared target or pass a single broadcast value."
                    )
                resolved = metric_type[fanned_name]
            else:
                resolved = metric_type
            # Reject ranking metrics with a clear pointer. For multilabel
            # members the lookup must consider BOTH the BINARY head
            # contract (the fanned-out semantics) AND the MULTILABEL
            # group declaration — same precedence rule as
            # _metric_lookup_types_for_fanned uses for score_per_target.
            # Without this union, TARGET_TYPE_TO_METRICS[MULTILABEL]
            # would be unreachable through evaluate even though
            # MULTILABEL metrics happen to coincide with BINARY today.
            lookup_types = scorer._metric_lookup_types_for_fanned(fanned_name)
            compat: tuple = ()
            for tt in lookup_types:
                compat = compat + TARGET_TYPE_TO_METRICS[tt]
            # De-dupe while preserving order.
            seen_metrics: set = set()
            compat = tuple(m for m in compat if not (m in seen_metrics or seen_metrics.add(m)))
            if resolved.value not in compat:
                raise ValueError(
                    f"metric_type {resolved.name} is not compatible with target "
                    f"{fanned_name!r} (declared {target_type.value}). Compatible "
                    f"metrics: {sorted(compat)}. For metrics outside this set, "
                    f"use scorer.score_per_target(metric_callables=...)."
                )
            metric_obj = RecommenderMetricFactory.create(resolved)
            if isinstance(metric_obj, BaseRankingMetric):
                # Defensive: should be unreachable thanks to the compat check
                # above (no TargetType maps to a ranking metric), but pin
                # the contract.
                raise ValueError(
                    "Ranking metrics are not applicable to per-target prediction. "
                    "Use score_per_target or predict_targets for per-target "
                    "evaluation."
                )

            # Slice predictions per target type.
            y_true = logged_rewards[fanned_name].to_numpy()
            preds = proba_dict[fanned_name]
            if target_type in (TargetType.BINARY, TargetType.MULTILABEL):
                # Existing BaseClassificationMetric ravels both inputs and
                # masks NaN ground truth. predict_proba_dict returns (n, 2);
                # pass the positive-class column.
                y_score = preds[:, 1]
            elif target_type == TargetType.REGRESSION:
                y_score = preds  # already (n,)
            elif target_type == TargetType.MULTICLASS:
                # (n, K). MULTICLASS_ACCURACY expects ground-truth class
                # indices into the training-time catalogue.
                classes = scorer._get_multiclass_classes().get(fanned_name, [])
                if not classes:
                    # Fail fast: this means the estimator wasn't fit on this
                    # multiclass target, or the catalogue was wiped post-fit.
                    # Falling back to range(K) would silently produce
                    # nonsense metric values.
                    raise RuntimeError(
                        f"Multiclass target {fanned_name!r} has no class "
                        f"catalogue on the fitted estimator. The estimator "
                        f"must be fit with this target's labels before "
                        f"evaluate() can map ground truth to class indices. "
                        f"Got empty _multiclass_classes[{fanned_name!r}]."
                    )
                label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
                # Unknown labels here are defensive — the per-column
                # validator above already rejects unknown multiclass
                # labels, so reaching this branch with an unknown label
                # means a validator bypass. Warn loudly rather than
                # silently coercing to NaN.
                #
                # NaN ground-truth values are NOT unknown labels — they
                # represent "no logged outcome" and are intentionally
                # masked-out by the metric's NaN handling. Filter them
                # out before the catalogue check so we don't fire a
                # spurious warning every time logged_rewards has missing
                # rows.
                unknown_labels = sorted(
                    {
                        v
                        for v in y_true.tolist()
                        if not (isinstance(v, float) and np.isnan(v)) and v is not None and v not in label_to_idx
                    },
                    key=str,
                )
                if unknown_labels:
                    import warnings as _warnings

                    _warnings.warn(
                        f"Multiclass target {fanned_name!r} has logged_rewards "
                        f"label(s) not in the training-time catalogue: "
                        f"{unknown_labels}. These rows will be masked to NaN "
                        f"and excluded from the metric. (Upstream validator "
                        f"should have caught this — investigate.)",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                y_true = np.array(
                    [
                        np.nan
                        if (isinstance(v, float) and np.isnan(v)) or v is None
                        else (label_to_idx[v] if v in label_to_idx else np.nan)
                        for v in y_true.tolist()
                    ],
                    dtype=float,
                )
                y_score = preds
            else:  # pragma: no cover
                raise AssertionError(f"Unhandled target type {target_type}")

            # Degenerate-target masking (mirrors _evaluate_multioutput's
            # behaviour at ranking_recommender.py ~880): for classification
            # metrics on binary / multilabel-member targets, single-class
            # held-out slices have UNDEFINED metric value — ROC_AUC's
            # sklearn implementation raises ValueError, which the metric
            # class swallows and returns 0.0. A "0.0" looks like a broken
            # model; "NaN" honestly signals "metric undefined for this
            # column," matching the MultioutputScorer semantics callers
            # already rely on.
            if target_type in (TargetType.BINARY, TargetType.MULTILABEL):
                # By the time we reach this point, the per-column type
                # validation above has confirmed y_true is numeric in
                # {0, 1} (with NaN allowed). The dtype-conditional fallback
                # that used to live here is unreachable; drop it.
                non_nan_true = y_true[~np.isnan(y_true.astype(float))]
                unique_classes = set(np.asarray(non_nan_true).tolist())
                # Strip the {0,1} expected set to detect single-class.
                if len(unique_classes & {0, 1, 0.0, 1.0, True, False}) < 2:
                    result[fanned_name] = float("nan")
                    continue

            value = metric_obj.calculate(
                recommendation_ranks=np.empty((len(y_true), 0)),
                modified_rewards=y_true,
                recommendation_scores=y_score,
                top_k=None,
            )
            result[fanned_name] = float(value)
        return result
