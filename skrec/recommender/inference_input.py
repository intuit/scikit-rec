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
        # interactions_schema is only set as a side-effect of train(); standalone
        # evaluate / single-row inference paths construct the recommender with
        # just the scorer and never call train. Use getattr so the schema-trim
        # is a no-op (return None) rather than AttributeError in that case —
        # the downstream preprocess_inputs path treats None as "no schema
        # configured, pass interactions through unchanged."
        if not getattr(self._owner, "interactions_schema", None):
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

    def apply_interactions_schema_with_preservation(
        self,
        interactions: Optional[DataFrame],
        strip_user_id: bool = False,
    ) -> Optional[DataFrame]:
        """Apply the client interactions schema, preserving scorer-declared columns.

        Shared primitive used by both ``preprocess_inputs`` (batch scoring) and
        ``BaseRecommender.recommend_online`` (single-row scoring). Without
        this, ``interactions_schema.apply()`` silently drops any column the
        client didn't declare — including ``OBSERVED_*`` columns for the v3
        ``MixedTypeMultiTargetScorer`` conditional path, which the scorer
        declares via :meth:`BaseScorer.preserved_inference_columns`. The
        preservation set-aside / re-attach mirrors the recommend_online
        seam so the same primitive feeds every scoring path.

        Returns ``interactions`` unchanged when no schema is configured.

        Args:
            interactions: Input frame to apply the schema to (may be ``None``).
            strip_user_id: When ``True``, drops ``USER_ID`` from the trimmed
                schema before applying (matches the ``recommend_online``
                contract; ``score_items`` keeps USER_ID and passes
                ``strip_user_id=False``).
        """
        if interactions is None:
            return None
        schema = self.build_trimmed_interactions_schema(strip_user_id=strip_user_id)
        if schema is None:
            return interactions
        # Set aside any columns the scorer wants preserved through the
        # schema's unknown-column strip (e.g. OBSERVED_* for v3 conditional).
        # Snapshot as pandas Series (preserves the original frame index) so
        # apply()-side row filtering / reordering can't silently misalign
        # the re-attached values.
        #
        # Two preservation surfaces:
        #   - preserved_inference_columns: exact-name list (declared
        #     OBSERVED_<suffix> for each target).
        #   - preserved_inference_column_prefixes: prefix list (catches
        #     typo'd OBSERVED_* names so the scorer's orphan-detection
        #     can fire instead of the schema silently dropping the typo).
        preserved_names = set(self._owner.scorer.preserved_inference_columns())
        preserved_prefixes = tuple(self._owner.scorer.preserved_inference_column_prefixes())
        preserved_cols = {}
        for col in interactions.columns:
            if col in preserved_names or (preserved_prefixes and col.startswith(preserved_prefixes)):
                preserved_cols[col] = interactions[col].copy()
        logger.info("Applying Schema to Interactions")
        interactions = schema.apply(interactions)
        if preserved_cols:
            # .apply() may return a view; .copy() makes the column assignments
            # write back onto a frame the scorer will actually see.
            interactions = interactions.copy()
            # DatasetSchema.apply today only column-projects (never filters
            # or adds rows), so the post-apply index matches the input.
            # Before P2-6 the re-attach used .reindex which would silently
            # NaN-fill any new rows that future apply() implementations
            # might inject — exactly the failure mode the OBSERVED_*
            # preservation contract is supposed to prevent. Assert
            # equality on EVERY preserved column's index so a future
            # caller whose preserved columns have divergent indices
            # surfaces the contract break loudly rather than silently
            # NaN-ing observed values. (Earlier sample-only check missed
            # mismatches on non-first columns.)
            for name, series in preserved_cols.items():
                if not series.index.equals(interactions.index):
                    raise RuntimeError(
                        f"Preserved column {name!r} has an index that no "
                        f"longer matches the post-apply interactions "
                        f"index. The preservation contract requires "
                        f"apply() to be column-only — re-attaching by "
                        f"reindex would silently NaN-fill or drop rows. "
                        f"Investigate the schema implementation."
                    )
                # Preserve dtype on re-attach. Using ``series.values``
                # below would propagate the original dtype, but assigning
                # via the Series object itself preserves the index-typed
                # dtype handling pandas applies (e.g. nullable Int64
                # stays Int64 instead of falling back to object). The
                # index equality check above already guarantees alignment
                # so a direct Series assignment is safe.
                interactions[name] = series
        return interactions

    def preprocess_inputs(
        self,
        interactions: Optional[DataFrame],
        users: Optional[DataFrame],
    ) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """Validate schema and apply preprocessing shared by ``score_items`` and ``_score_items_np``."""
        # Reject non-None ``users`` for any scorer that fundamentally
        # doesn't accept user-side merges:
        #   - MulticlassScorer / MultioutputScorer: long-standing
        #     isinstance ladder (pre-existing behavior preserved).
        #   - Any ``is_per_target_scorer=True`` scorer
        #     (MixedTypeMultiTargetScorer today): the per-target wide-
        #     format contract has no place for a separate users frame.
        #     Before this gate, the rejection happened later inside
        #     ``scorer.score_items``, surfacing a confusing error
        #     message; capability-flag dispatch here mirrors the
        #     round-3 isinstance-ladder cleanup.
        cannot_accept_users = (
            isinstance(self._owner.scorer, MulticlassScorer)
            or isinstance(self._owner.scorer, MultioutputScorer)
            # ``is True`` (not just truthy) so MagicMock(spec=BaseScorer)
            # — which auto-generates a truthy MagicMock for any attribute
            # access — doesn't accidentally activate the per-target
            # users-rejection branch. Real scorers set the flag to a
            # literal True/False; matches the constructor-side check in
            # RankingRecommender.__init__.
            or getattr(self._owner.scorer, "is_per_target_scorer", False) is True
        )
        if cannot_accept_users and users is not None:
            raise ValueError("This scorer cannot accept Users, set it to None!")

        interactions = self.process_outcome_columns(interactions)

        interactions_schema = self.build_trimmed_interactions_schema(strip_user_id=False)
        if interactions_schema:
            if interactions is not None:
                # Route through the shared preservation primitive so
                # OBSERVED_* (and any other scorer-declared columns) survive
                # the client schema's unknown-column strip. Previously this
                # path called interactions_schema.apply() directly and
                # silently dropped OBSERVED_* on batch score_items.
                interactions = self.apply_interactions_schema_with_preservation(interactions, strip_user_id=False)
            elif interactions_schema.columns == [USER_ID_NAME]:
                logger.warning("There are no real interactions features. You can set interactions as None!")
            else:
                raise ValueError(f"Expecting Interactions Columns: {interactions_schema.columns}")

        if getattr(self._owner, "users_schema", None):
            users = self.apply_users_schema(users)

        return interactions, users
