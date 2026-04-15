from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from skrec.constants import USER_ID_NAME
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
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
        super().__init__(scorer)
        self.retriever = retriever

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
            logger.warning(
                "For MultioutputScorer, top_k is ignored — all item predictions are returned. "
                "To get per-class probabilities instead, call scorer.score_items() directly."
            )
            interactions, users = self._preprocess_inputs(interactions, users)
            return self.scorer.predict_classes(interactions, users)

        if self.retriever is None:
            # Normalise None → 0 (deterministic) so BaseRecommender.recommend(),
            # which expects a plain float, doesn't receive None and raise TypeError.
            return super().recommend(interactions, users, top_k, sampling_temperature or 0, replace)

        # Per-user retrieval loop
        return self._recommend_with_retriever(interactions, users, top_k, sampling_temperature, replace)

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
