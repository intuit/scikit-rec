from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.recommender.gcsl.inference.base_inference import BaseInference
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.retriever.base_retriever import BaseCandidateRetriever
from skrec.scorer.base_scorer import BaseScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class GcslRecommender(RankingRecommender):
    """Multi-objective recommender based on goal-conditioned supervised learning.

    Extends ``RankingRecommender`` to handle multiple outcome columns
    simultaneously. Rather than dropping outcome columns before training (the
    standard behaviour), ``GcslRecommender`` keeps them as **input features**.
    The model therefore learns:

        P(positive | user, item, context, outcome_1, outcome_2, ...)

    At inference an ``InferenceMethod`` fills in the desired goal values for
    each outcome column before scoring. Items whose feature profile is most
    consistent with the requested goals receive the highest scores.

    Args:
        scorer: The underlying scorer (e.g. ``UniversalScorer``).
        inference_method: Strategy for injecting goal values at inference time.
            May be set after construction via ``set_inference_method()``.
        retriever: Optional candidate retriever to narrow the item set before
            scoring. Goal values are injected before the per-user retrieval loop.

    .. note::
        ``recommend_online()`` is **not supported** — the single-row fast path
        bypasses the goal-injection step. Use ``recommend()`` instead.
    """

    def __init__(
        self,
        scorer: BaseScorer,
        inference_method: Optional[BaseInference] = None,
        retriever: Optional[BaseCandidateRetriever] = None,
    ) -> None:
        super().__init__(scorer, retriever)
        self.inference_method = inference_method
        if inference_method is None:
            logger.info(
                "Inference method not specified. Set one with set_inference_method() before calling recommend()."
            )

    def set_inference_method(self, inference_method: BaseInference) -> None:
        """Replace the active inference method.

        If the recommender has already been trained, ``fit()`` is called
        automatically on the new method using the stored training interactions.
        If ``fit()`` raises, the previous inference method is restored and the
        exception is re-raised — the recommender is never left in a broken state.

        Args:
            inference_method: New inference method to use at scoring time.
        """
        old_method = self.inference_method
        self.inference_method = inference_method
        train_df = getattr(self, "_train_interactions_df", None)
        if train_df is not None and self.outcome_cols:
            try:
                inference_method.fit(train_df, self.outcome_cols)
            except Exception:
                self.inference_method = old_method
                raise

    def train(
        self,
        users_ds: Optional[UsersDataset] = None,
        items_ds: Optional[ItemsDataset] = None,
        interactions_ds: Optional[InteractionsDataset] = None,
        valid_users_ds: Optional[UsersDataset] = None,
        valid_interactions_ds: Optional[InteractionsDataset] = None,
    ) -> None:
        super().train(users_ds, items_ds, interactions_ds, valid_users_ds, valid_interactions_ds)

        if not self.outcome_cols:
            raise ValueError("Extra outcome columns are required for GCSL training")

        if self.inference_method is not None:
            self.inference_method.fit(self._train_interactions_df, self.outcome_cols)
        else:
            logger.info(
                "Inference method not specified — skipping fit(). "
                "Set one with set_inference_method() before calling recommend()."
            )

    def _process_outcome_columns(self, interactions: Optional[DataFrame]) -> Optional[DataFrame]:
        """Keep outcome columns as input features instead of dropping them."""
        return interactions

    def _prepare_interactions(
        self,
        interactions: Optional[DataFrame],
        users: Optional[DataFrame],
    ) -> Tuple[DataFrame, Optional[DataFrame]]:
        if self.inference_method is None:
            raise ValueError("Inference method not set. Call set_inference_method() before recommend().")
        if not self.outcome_cols:
            raise ValueError("Outcome column names are not set.")
        # Full schema validation (interactions + users).  _process_outcome_columns
        # is overridden to be a no-op, so outcome columns survive preprocessing.
        interactions, users = self._preprocess_inputs(interactions, users)
        interactions = self.inference_method.transform(interactions)
        return interactions, users

    def score_items(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> DataFrame:
        interactions_gcsl, users = self._prepare_interactions(interactions, users)
        return self.scorer.score_items(interactions_gcsl, users)

    def _score_items_np(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> NDArray[np.float64]:
        interactions_gcsl, users = self._prepare_interactions(interactions, users)
        return self.scorer._score_items_np(interactions_gcsl, users)

    def _recommend_with_retriever(
        self,
        interactions: Optional[DataFrame],
        users: Optional[DataFrame],
        top_k: int,
        sampling_temperature: Optional[float],
        replace: bool,
    ) -> NDArray:
        """Inject goal values before the per-user retrieval loop."""
        if self.inference_method is None:
            raise ValueError("Inference method not set. Call set_inference_method() before recommend().")
        if not self.outcome_cols:
            raise ValueError("Outcome column names are not set.")
        # transform() is called before _preprocess_inputs here (reversed from
        # _prepare_interactions) because the parent's _recommend_with_retriever
        # calls _preprocess_inputs internally.  This is safe because the
        # interaction schema excludes outcome columns.
        interactions = self.inference_method.transform(interactions)
        return super()._recommend_with_retriever(interactions, users, top_k, sampling_temperature, replace)

    def recommend_online(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
        top_k: int = 1,
    ):
        raise NotImplementedError(
            "recommend_online() is not supported for GcslRecommender. "
            "The goal-injection step cannot be applied in the single-row fast path. "
            "Use recommend() instead."
        )
