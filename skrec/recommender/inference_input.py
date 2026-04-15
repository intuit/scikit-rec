"""Schema trimming and preprocessing for inference (shared by score_items and recommend_online)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Tuple

from pandas import DataFrame

from skrec.constants import (
    ITEM_ID_NAME,
    ITEM_PREFIX,
    LABEL_NAME,
    OUTCOME_PREFIX,
    USER_ID_NAME,
)
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.scorer.multiclass import MulticlassScorer
from skrec.scorer.multioutput import MultioutputScorer
from skrec.util.logger import get_logger

if TYPE_CHECKING:
    from skrec.recommender.base_recommender import BaseRecommender

logger = get_logger(__name__)


class InferenceInputPreparer:
    """Applies client interaction/user schemas and outcome stripping for scoring paths.

    Holds a reference to the owning :class:`~skrec.recommender.base_recommender.BaseRecommender`
    and reads ``scorer``, ``interactions_schema``, ``users_schema``, and ``outcome_cols`` from it
    on each call so state stays in sync after :meth:`~skrec.recommender.base_recommender.BaseRecommender.train`.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: BaseRecommender) -> None:
        self._owner = owner

    def process_outcome_columns(self, interactions_df: Optional[DataFrame] = None) -> Optional[DataFrame]:
        """Delegates so subclasses can override outcome handling.

        For example :class:`~skrec.recommender.gcsl.gcsl_recommender.GcslRecommender`.
        """
        return self._owner._process_outcome_columns(interactions_df)

    def build_trimmed_interactions_schema(self, strip_user_id: bool = False):
        """Return a trimmed copy of ``interactions_schema`` with internal columns removed, or ``None``."""
        if not self._owner.interactions_schema:
            return None
        schema = copy.deepcopy(self._owner.interactions_schema)
        for col in [ITEM_ID_NAME, LABEL_NAME]:
            schema.remove_column(col)
        if strip_user_id:
            schema.remove_column(USER_ID_NAME)
        if isinstance(self._owner.scorer, MultioutputScorer):
            for col in [c for c in schema.columns if ITEM_PREFIX in c]:
                schema.remove_column(col)
        for col in [c for c in schema.columns if c.startswith(OUTCOME_PREFIX)]:
            schema.remove_column(col)
        return schema

    def apply_users_schema(self, users: Optional[DataFrame], strip_user_id: bool = False) -> Optional[DataFrame]:
        if self._owner.users_schema:
            users_schema = copy.deepcopy(self._owner.users_schema)

            if strip_user_id and users_schema.columns:
                users_schema.remove_column(USER_ID_NAME)

            if users is not None:
                logger.info("Applying Schema to Users")
                users = users_schema.apply(users)
            elif isinstance(self._owner.scorer.estimator, BaseEmbeddingEstimator):
                pass
            elif users_schema.columns == [USER_ID_NAME] or (strip_user_id and len(users_schema.columns) == 0):
                logger.warning("There are no real user features! You can set users to None!")
            else:
                raise ValueError(f"Expecting User Columns: {users_schema.columns}")
        return users

    def preprocess_inputs(
        self,
        interactions: Optional[DataFrame],
        users: Optional[DataFrame],
    ) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """Validate schema and apply preprocessing shared by ``score_items`` and ``_score_items_np``."""
        if isinstance(self._owner.scorer, MulticlassScorer) or isinstance(self._owner.scorer, MultioutputScorer):
            if users is not None:
                raise ValueError("This scorer cannot accept Users, set it to None!")

        interactions = self.process_outcome_columns(interactions)

        interactions_schema = self.build_trimmed_interactions_schema(strip_user_id=False)
        if interactions_schema:
            if interactions is not None:
                logger.info("Applying Schema to Interactions")
                interactions = interactions_schema.apply(interactions)
            elif interactions_schema.columns == [USER_ID_NAME]:
                logger.warning("There are no real interactions features. You can set interactions as None!")
            else:
                raise ValueError(f"Expecting Interactions Columns: {interactions_schema.columns}")

        if self._owner.users_schema:
            users = self.apply_users_schema(users)

        return interactions, users
