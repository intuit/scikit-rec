from typing import Optional

from numpy.typing import NDArray
from pandas import DataFrame

from skrec.constants import (
    ITEM_ID_NAME,
    LABEL_NAME,
    OUTCOME_PREFIX,
    TIMESTAMP_COL,
    USER_ID_NAME,
)
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.sequential.sasrec_estimator import (
    ITEM_SEQUENCE_COL,
    OUTCOME_SEQUENCE_COL,
)
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.sequential import SequentialScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class SequentialRecommender(RankingRecommender):
    """
    Recommender for sequential models (SASRec).

    Owns sequence building: sorts interactions by TIMESTAMP, groups by user,
    truncates to max_len, then passes the resulting sequences to SequentialScorer.

    The scorer and estimator receive pre-built sequences and never see raw
    interaction rows or timestamps — those concerns live here.

    Usage::

        estimator = SASRecClassifierEstimator(hidden_units=50, num_blocks=2)
        scorer = SequentialScorer(estimator)
        recommender = SequentialRecommender(scorer, max_len=50)

        recommender.train(items_ds=items_ds, interactions_ds=interactions_ds)
        items, scores = recommender.recommend(interactions=test_interactions, top_k=10)
    """

    def __init__(self, scorer: SequentialScorer, max_len: int = 50):
        """
        Args:
            scorer: A SequentialScorer wrapping a SASRec estimator.
            max_len: Maximum sequence length. Most recent max_len interactions
                     are kept per user; older ones are dropped.
        """
        super().__init__(scorer)
        self.max_len = max_len

    def train(
        self,
        users_ds: Optional[UsersDataset] = None,
        items_ds: Optional[ItemsDataset] = None,
        interactions_ds: Optional[InteractionsDataset] = None,
        valid_users_ds: Optional[UsersDataset] = None,
        use_validation: bool = False,
    ) -> None:
        """
        Train the sequential recommender.

        interactions_ds must contain USER_ID, ITEM_ID, OUTCOME, and TIMESTAMP columns.
        Sequence order is determined by TIMESTAMP; ties are broken by row order.

        Args:
            users_ds: Optional user features dataset (passed through to scorer/estimator).
            items_ds: Optional items dataset. If provided, defines the item vocabulary.
            interactions_ds: Required. Raw interaction logs with TIMESTAMP.
            valid_users_ds: Ignored (users are not used by SASRec).
            use_validation: If True, derives a leave-one-out validation split from
                interactions_ds (each user's second-to-last interaction becomes the
                validation target). The estimator computes validation loss every epoch
                and can apply early stopping if early_stopping_patience is set.
                No separate validation dataset is needed — the split is always derived
                from interactions_ds itself, which is the standard SASRec protocol.
        """
        if interactions_ds is None:
            raise ValueError("interactions_ds is required for SequentialRecommender.")

        users_df = users_ds.fetch_data() if users_ds else None
        items_df = items_ds.fetch_data() if items_ds else None
        interactions_df = interactions_ds.fetch_data()

        # Store schemas for inference-time validation
        self.users_schema = users_ds.client_schema if users_ds else None
        self.interactions_schema = interactions_ds.client_schema if interactions_ds else None
        self.items_schema = items_ds.client_schema if items_ds else None

        # Remove auxiliary outcome columns (e.g. OUTCOME_uplift), keep main OUTCOME
        self.outcome_cols = [col for col in interactions_df.columns if col.startswith(OUTCOME_PREFIX)]
        interactions_df = self._process_outcome_columns(interactions_df)

        # Propagate max_len from the recommender to the estimator.
        # Both carry max_len as a constructor parameter; the recommender's value wins
        # because it controls data preparation (sequence truncation). Warn when they
        # disagree so users are not surprised by silent overrides.
        estimator = self.scorer.estimator  # type: ignore[union-attr]
        if hasattr(estimator, "max_len") and estimator.max_len != self.max_len:
            logger.warning(
                f"SequentialRecommender.max_len={self.max_len} overrides "
                f"{estimator.__class__.__name__}.max_len={estimator.max_len}. "
                "Pass the same max_len to both, or rely on the recommender's value."
            )
        estimator.max_len = self.max_len

        # Build per-user sequences sorted by timestamp
        sequences_df = self._build_sequences(interactions_df)

        # Build validation sequences when use_validation=True.
        # We remove the last interaction per user from the training data to get
        # "all-except-test" histories, then build sequences from those. The last
        # position in each resulting sequence corresponds to the validation item
        # (second-to-last in the full history), which is the early-stopping target.
        val_sequences_df = None
        if use_validation:
            # Sort by timestamp first so cumcount(ascending=False) assigns rank 0 to the
            # most-recent interaction per user, not just the last row in DataFrame order.
            interactions_df = interactions_df.sort_values([USER_ID_NAME, TIMESTAMP_COL])
            interactions_df["_rank"] = interactions_df.groupby(USER_ID_NAME).cumcount(ascending=False)
            all_except_last_df = interactions_df[interactions_df["_rank"] >= 1].drop(columns=["_rank"])
            interactions_df = interactions_df.drop(columns=["_rank"])
            # Users with exactly one interaction have no rows left after leave-one-out and
            # are naturally excluded from all_except_last_df.  Log so the caller knows.
            n_total_users = interactions_df[USER_ID_NAME].nunique()
            n_val_users = all_except_last_df[USER_ID_NAME].nunique()
            n_excluded = n_total_users - n_val_users
            if n_excluded > 0:
                logger.warning(
                    f"{n_excluded} user(s) had only one interaction and were excluded from the "
                    "validation split (no history remains after leave-one-out). "
                    f"Validation will use {n_val_users}/{n_total_users} users."
                )
            val_sequences_df = self._build_sequences(all_except_last_df)

        # Scorer validates format and sets item_names
        train_users, train_items, train_sequences = self.scorer.process_factorized_datasets(
            users_df=users_df,
            items_df=items_df,
            interactions_df=sequences_df,
            is_training=True,
        )

        self.scorer.train_embedding_model(
            users=train_users,
            items=train_items,
            interactions=train_sequences,
            valid_interactions=val_sequences_df,
        )

    def _prepare_sequences(self, interactions_df: DataFrame) -> DataFrame:
        """Build sequence representations from raw interactions.

        Template method: subclasses override this to change how sequences are
        built (e.g. ``HierarchicalSequentialRecommender`` builds session-level
        sequences instead of flat sequences).
        """
        return self._build_sequences(interactions_df)

    def score_items(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Score all items for users based on their interaction history.

        Args:
            interactions: Raw interaction logs with USER_ID, ITEM_ID, TIMESTAMP.
                          OUTCOME is optional at inference time.
            users: Ignored.
        """
        if interactions is None:
            raise ValueError("interactions is required for score_items().")

        interactions = self._process_outcome_columns(interactions)
        sequences_df = self._prepare_sequences(interactions)

        return self.scorer.score_items(interactions=sequences_df, users=users)

    def _score_items_np(self, interactions=None, users=None):
        if interactions is None:
            raise ValueError("interactions is required for score_items().")
        interactions = self._process_outcome_columns(interactions)
        sequences_df = self._prepare_sequences(interactions)
        return self.scorer._score_items_np(interactions=sequences_df, users=users)

    def recommend(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
        top_k: int = 1,
        sampling_temperature: Optional[float] = 0,
        replace: bool = False,
    ) -> NDArray:
        """
        Recommend top-k items for each user based on their interaction sequence.

        Args:
            interactions: Raw interaction logs with USER_ID, ITEM_ID, TIMESTAMP.
            users: Ignored.
            top_k: Number of items to recommend per user.
            sampling_temperature: 0 for deterministic ranking, >0 for probabilistic sampling.
            replace: Whether to sample with replacement (only relevant if temperature > 0).

        Returns:
            NDArray of shape (num_users, top_k) containing recommended item IDs.
        """
        # Build sequences once and cache for this call — score_items will rebuild,
        # but recommend() calls score_items() via parent, so we pass raw interactions
        # and let score_items() handle sequencing.
        return super().recommend(
            interactions=interactions,
            users=users,
            top_k=top_k,
            sampling_temperature=sampling_temperature,
            replace=replace,
        )

    def recommend_online(self, interactions=None, users=None, top_k: int = 1):
        raise NotImplementedError(
            "recommend_online() is not supported for SequentialRecommender. "
            "SASRec requires the full interaction history to build attention sequences — "
            "use recommend() instead."
        )

    def _build_sequences(self, interactions_df: DataFrame) -> DataFrame:
        """
        Sort interactions by TIMESTAMP per user and aggregate into sequences.

        Args:
            interactions_df: Raw interactions with USER_ID, ITEM_ID, TIMESTAMP.
                             OUTCOME is included when present (training) and excluded
                             when absent (inference).

        Returns:
            DataFrame with one row per user:
                USER_ID, ITEM_SEQUENCE (List[str]), OUTCOME_SEQUENCE (List[float])
            OUTCOME_SEQUENCE is omitted if OUTCOME was not in interactions_df.
        """
        if TIMESTAMP_COL not in interactions_df.columns:
            raise ValueError(
                f"'{TIMESTAMP_COL}' column is required in interactions for SequentialRecommender. "
                "Use interactions_schema_with_timestamp_training.yaml as your dataset schema."
            )
        if ITEM_ID_NAME not in interactions_df.columns:
            raise ValueError(f"'{ITEM_ID_NAME}' column is required in interactions.")
        if USER_ID_NAME not in interactions_df.columns:
            raise ValueError(f"'{USER_ID_NAME}' column is required in interactions.")

        has_outcome = LABEL_NAME in interactions_df.columns

        sorted_df = interactions_df.sort_values([USER_ID_NAME, TIMESTAMP_COL])

        agg_dict = {ITEM_SEQUENCE_COL: (ITEM_ID_NAME, list)}
        if has_outcome:
            agg_dict[OUTCOME_SEQUENCE_COL] = (LABEL_NAME, list)

        sequences_df = sorted_df.groupby(USER_ID_NAME, sort=False).agg(**agg_dict).reset_index()

        # Truncate to max_len+1 (keep one extra so that _build_padded_tensors can form
        # input_seq = full_seq[:-1] with exactly max_len items for long histories,
        # ensuring every positional slot in the model is utilized during training).
        truncate_len = self.max_len + 1
        sequences_df[ITEM_SEQUENCE_COL] = sequences_df[ITEM_SEQUENCE_COL].apply(lambda x: x[-truncate_len:])
        if has_outcome:
            sequences_df[OUTCOME_SEQUENCE_COL] = sequences_df[OUTCOME_SEQUENCE_COL].apply(lambda x: x[-truncate_len:])

        logger.info(
            f"Built sequences for {len(sequences_df)} users (max_len={self.max_len}, has_outcome={has_outcome})."
        )
        return sequences_df
