from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from skrec.constants import ITEM_ID_NAME, USER_ID_NAME
from skrec.estimator.sequential.base_sequential_estimator import SequentialEstimator
from skrec.estimator.sequential.hrnn_estimator import SESSION_SEQUENCES_COL
from skrec.scorer.sequential import SequentialScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class HierarchicalScorer(SequentialScorer):
    """
    Scorer for HRNN — the hierarchical session-based recommendation model.

    Expects pre-built session sequences from HierarchicalSequentialRecommender.
    The interactions DataFrame must already contain USER_ID and SESSION_SEQUENCES columns
    (List[List[str]] per user), not flat ITEM_SEQUENCE lists.

    Inherits item management and score_items routing from SequentialScorer.
    Overrides process_factorized_datasets to validate SESSION_SEQUENCES_COL.
    """

    def __init__(self, estimator: SequentialEstimator) -> None:
        super().__init__(estimator)

    def process_factorized_datasets(
        self,
        users_df: Optional[DataFrame],
        items_df: Optional[DataFrame],
        interactions_df: DataFrame,
        is_training: Optional[bool] = True,
    ):
        """
        Validate and process session-structured data.

        Args:
            users_df: Optional user features (unused by HRNN).
            items_df: DataFrame with ITEM_ID. Used to set item_names.
            interactions_df: sessions_df with USER_ID and SESSION_SEQUENCES columns.
                             Built by HierarchicalSequentialRecommender, not raw interactions.
            is_training: When True, sets self.item_names and self.items_df from items_df.

        Returns:
            (users_df, items_df, interactions_df)
        """
        if interactions_df is None:
            raise ValueError("interactions_df cannot be None.")
        if USER_ID_NAME not in interactions_df.columns:
            raise ValueError(f"'{USER_ID_NAME}' column must exist in interactions_df.")
        if SESSION_SEQUENCES_COL not in interactions_df.columns:
            raise ValueError(
                f"'{SESSION_SEQUENCES_COL}' column must exist in interactions_df. "
                "Session sequences must be built by HierarchicalSequentialRecommender "
                "before calling the scorer."
            )

        if is_training:
            if items_df is not None:
                items_df = items_df.copy()
                items_df[ITEM_ID_NAME] = items_df[ITEM_ID_NAME].astype(str)
                items_df = items_df.sort_values(by=ITEM_ID_NAME).reset_index(drop=True)
                self.item_names = np.array(items_df[ITEM_ID_NAME].values, dtype=np.str_)
                self.items_df = items_df
            else:
                # Derive item vocabulary from session sequences
                all_items = sorted(
                    {
                        str(item)
                        for session_list in interactions_df[SESSION_SEQUENCES_COL]
                        for session in session_list
                        for item in session
                    }
                )
                self.item_names = np.array(all_items, dtype=np.str_)
                self.items_df = pd.DataFrame({ITEM_ID_NAME: all_items})

        return users_df, self.items_df, interactions_df
