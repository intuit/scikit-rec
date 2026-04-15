from typing import Optional

import pandas as pd
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
from skrec.estimator.sequential.hrnn_estimator import (
    SESSION_OUTCOMES_COL,
    SESSION_SEQUENCES_COL,
)
from skrec.recommender.sequential.sequential_recommender import SequentialRecommender
from skrec.scorer.hierarchical import HierarchicalScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)

SESSION_ID_COL = "SESSION_ID"

# Intermediate column names used during session grouping (not exported)
_SESSION_ITEM_LIST = "_hrnn_session_items"
_SESSION_OUTCOME_LIST = "_hrnn_session_outcomes"


class HierarchicalSequentialRecommender(SequentialRecommender):
    """
    Recommender for HRNN: the Hierarchical Recurrent Neural Network model.

    Extends SequentialRecommender with session-boundary awareness. Interactions
    are split into sessions before being passed to the HierarchicalScorer.

    Two session detection strategies (tried in order):
    1. Explicit: a SESSION_ID column is present in the interactions DataFrame.
    2. Implicit: session_timeout_minutes is set; sessions are detected from
       TIMESTAMP gaps exceeding the timeout.

    Usage::

        estimator = HRNNEstimator(hidden_units=50, max_sessions=10, max_session_len=20)
        scorer = HierarchicalScorer(estimator)
        recommender = HierarchicalSequentialRecommender(
            scorer, max_sessions=10, max_session_len=20
            # session_timeout_minutes=30 is the default; pass SESSION_ID in data to override
        )

        recommender.train(items_ds=items_ds, interactions_ds=interactions_ds)
        items, scores = recommender.recommend(interactions=test_interactions, top_k=10)
    """

    def __init__(
        self,
        scorer: HierarchicalScorer,
        max_sessions: int = 10,
        max_session_len: int = 20,
        session_timeout_minutes: Optional[float] = 30.0,
    ):
        """
        Args:
            scorer: A HierarchicalScorer wrapping an HRNNEstimator.
            max_sessions: Maximum number of past sessions to retain per user.
                          Older sessions are dropped.
            max_session_len: Maximum number of items per session.
                             Older items within a session are dropped.
            session_timeout_minutes: Inactivity gap (in minutes) that starts a new session
                                     when SESSION_ID is not present in the interactions data.
                                     Defaults to 30 minutes (the standard used by Google Analytics
                                     and most session-based RecSys benchmarks).
                                     If SESSION_ID is present in the data it takes priority and
                                     this parameter is ignored.
        """
        # Pass max_session_len as max_len to the parent (used only by the parent's
        # _build_sequences — which we fully override — and for the max_len warning check).
        super().__init__(scorer, max_len=max_session_len)  # type: ignore[arg-type]
        self.max_sessions = max_sessions
        self.max_session_len = max_session_len
        self.session_timeout_minutes = session_timeout_minutes

    def train(
        self,
        users_ds: Optional[UsersDataset] = None,
        items_ds: Optional[ItemsDataset] = None,
        interactions_ds: Optional[InteractionsDataset] = None,
        valid_users_ds: Optional[UsersDataset] = None,
        use_validation: bool = False,
    ) -> None:
        """
        Train the HRNN recommender.

        interactions_ds must contain USER_ID, ITEM_ID, OUTCOME, and TIMESTAMP columns.
        Optionally SESSION_ID can be included to define explicit session boundaries.

        Args:
            users_ds: Optional user features (currently unused by HRNN).
            items_ds: Optional items dataset. If provided, defines the item vocabulary.
            interactions_ds: Required. Raw interaction logs.
            valid_users_ds: Ignored (HRNN has no user embedding table).
            use_validation: If True, derives a leave-one-out validation split from
                interactions_ds (each user's second-to-last interaction becomes the
                validation target). The estimator computes validation session-sequence
                loss every epoch and can apply early stopping if early_stopping_patience
                is set. No separate validation dataset is needed — the split is always
                derived from interactions_ds itself, which is the standard HRNN protocol.
        """
        if interactions_ds is None:
            raise ValueError("interactions_ds is required for HierarchicalSequentialRecommender.")

        users_df = users_ds.fetch_data() if users_ds else None
        items_df = items_ds.fetch_data() if items_ds else None
        interactions_df = interactions_ds.fetch_data()

        self.users_schema = users_ds.client_schema if users_ds else None
        self.interactions_schema = interactions_ds.client_schema if interactions_ds else None
        self.items_schema = items_ds.client_schema if items_ds else None

        self.outcome_cols = [col for col in interactions_df.columns if col.startswith(OUTCOME_PREFIX)]
        interactions_df = self._process_outcome_columns(interactions_df)

        # Propagate max_sessions and max_session_len to the estimator.
        estimator = self.scorer.estimator  # type: ignore[union-attr]
        if hasattr(estimator, "max_sessions") and estimator.max_sessions != self.max_sessions:
            logger.warning(
                f"HierarchicalSequentialRecommender.max_sessions={self.max_sessions} overrides "
                f"{estimator.__class__.__name__}.max_sessions={estimator.max_sessions}."
            )
        if hasattr(estimator, "max_session_len") and estimator.max_session_len != self.max_session_len:
            logger.warning(
                f"HierarchicalSequentialRecommender.max_session_len={self.max_session_len} overrides "
                f"{estimator.__class__.__name__}.max_session_len={estimator.max_session_len}."
            )
        estimator.max_sessions = self.max_sessions
        estimator.max_session_len = self.max_session_len

        sessions_df = self._build_session_sequences(interactions_df)

        # Build validation session sequences when use_validation=True.
        # We remove the last interaction per user to get "all-except-test" histories,
        # then build session sequences from those. The last real position in each
        # resulting sequence corresponds to the validation item (second-to-last).
        val_sessions_df = None
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
            val_sessions_df = self._build_session_sequences(all_except_last_df)

        train_users, train_items, train_sessions = self.scorer.process_factorized_datasets(
            users_df=users_df,
            items_df=items_df,
            interactions_df=sessions_df,
            is_training=True,
        )

        self.scorer.train_embedding_model(
            users=train_users,
            items=train_items,
            interactions=train_sessions,
            valid_interactions=val_sessions_df,
        )

    def _prepare_sequences(self, interactions_df: DataFrame) -> DataFrame:
        """Override: build session-level sequences instead of flat sequences."""
        return self._build_session_sequences(interactions_df)

    def _build_session_sequences(self, interactions_df: DataFrame) -> DataFrame:
        """
        Detect session boundaries and aggregate interactions into per-user session lists.

        Session detection (in priority order):
        1. Explicit: SESSION_ID column present in interactions_df.
        2. Implicit: session_timeout_minutes set — gaps > timeout start a new session.

        Args:
            interactions_df: Raw interactions with USER_ID, ITEM_ID, TIMESTAMP.
                             OUTCOME is included when present (training).

        Returns:
            DataFrame with one row per user:
                USER_ID,
                SESSION_SEQUENCES: List[List[str]]  — outer = sessions (oldest first),
                                                       inner = item IDs within session
                SESSION_OUTCOMES:  List[List[float]] — omitted if OUTCOME not in interactions_df
        """
        if TIMESTAMP_COL not in interactions_df.columns:
            raise ValueError(
                f"'{TIMESTAMP_COL}' column is required in interactions for "
                "HierarchicalSequentialRecommender. "
                "Use interactions_schema_with_timestamp_training.yaml as your dataset schema."
            )
        if ITEM_ID_NAME not in interactions_df.columns:
            raise ValueError(f"'{ITEM_ID_NAME}' column is required in interactions.")
        if USER_ID_NAME not in interactions_df.columns:
            raise ValueError(f"'{USER_ID_NAME}' column is required in interactions.")

        has_outcome = LABEL_NAME in interactions_df.columns
        has_session_id = SESSION_ID_COL in interactions_df.columns

        sorted_df = interactions_df.sort_values([USER_ID_NAME, TIMESTAMP_COL]).copy()

        # --- Determine session key column ---
        if has_session_id:
            # Use provided session boundaries directly
            session_key_col = SESSION_ID_COL
            logger.info("Using explicit SESSION_ID column for session boundaries.")
        elif self.session_timeout_minutes is not None:
            # Detect session boundaries from timestamp gaps.
            # Numeric timestamps (int/float) are treated as seconds — the standard unit
            # for RecSys datasets (Unix epoch seconds, e.g. MovieLens, Amazon Reviews).
            # Datetime/string timestamps are parsed via pd.to_datetime.
            ts_col = sorted_df[TIMESTAMP_COL]
            if pd.api.types.is_numeric_dtype(ts_col):
                gap = ts_col.groupby(sorted_df[USER_ID_NAME]).diff()
                timeout_val = self.session_timeout_minutes * 60  # minutes → seconds
            else:
                parsed_ts = pd.to_datetime(ts_col)
                gap = parsed_ts.groupby(sorted_df[USER_ID_NAME]).diff().dt.total_seconds()
                timeout_val = self.session_timeout_minutes * 60
            new_session = gap.isna() | (gap > timeout_val)  # True at session start
            # Cumulative sum within user gives a monotonically increasing session index
            sorted_df["_session_key"] = new_session.groupby(sorted_df[USER_ID_NAME]).cumsum()
            session_key_col = "_session_key"
            logger.info(f"Detected session boundaries using timeout={self.session_timeout_minutes} minutes.")
        else:
            raise ValueError(
                "Cannot detect session boundaries: interactions have no SESSION_ID column "
                "and session_timeout_minutes is not set on the recommender. "
                "Either add SESSION_ID to your data or pass session_timeout_minutes."
            )

        # --- Group by (user, session) to collect item lists ---
        agg_spec: dict = {_SESSION_ITEM_LIST: (ITEM_ID_NAME, list)}
        if has_outcome:
            agg_spec[_SESSION_OUTCOME_LIST] = (LABEL_NAME, list)

        session_df = sorted_df.groupby([USER_ID_NAME, session_key_col], sort=False).agg(**agg_spec).reset_index()

        # --- Group by user to collect list-of-session-lists ---
        user_agg: dict = {SESSION_SEQUENCES_COL: (_SESSION_ITEM_LIST, list)}
        if has_outcome:
            user_agg[SESSION_OUTCOMES_COL] = (_SESSION_OUTCOME_LIST, list)

        sequences_df = session_df.groupby(USER_ID_NAME, sort=False).agg(**user_agg).reset_index()

        # --- Truncate: keep last max_sessions sessions, each session to max_session_len items ---
        sequences_df[SESSION_SEQUENCES_COL] = sequences_df[SESSION_SEQUENCES_COL].apply(
            lambda sess_list: [s[-self.max_session_len :] for s in sess_list[-self.max_sessions :]]
        )
        if has_outcome:
            sequences_df[SESSION_OUTCOMES_COL] = sequences_df[SESSION_OUTCOMES_COL].apply(
                lambda out_list: [o[-self.max_session_len :] for o in out_list[-self.max_sessions :]]
            )

        logger.info(
            f"Built session sequences for {len(sequences_df)} users "
            f"(max_sessions={self.max_sessions}, max_session_len={self.max_session_len}, "
            f"has_outcome={has_outcome})."
        )
        return sequences_df
