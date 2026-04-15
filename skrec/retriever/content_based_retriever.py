"""Content-based candidate retriever using item feature cosine similarity."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME
from skrec.estimator.base_estimator import BaseEstimator
from skrec.retriever.base_retriever import BaseCandidateRetriever
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class ContentBasedRetriever(BaseCandidateRetriever):
    """
    Retrieves candidates by cosine similarity between a user profile vector
    and item feature vectors.

    No ML training is required. The user profile is the (optionally weighted)
    mean of the feature vectors of items the user has interacted with.
    Works with any estimator type, including XGBoost and LightGBM.

    Best for
    --------
    - Cold-start users: a single interaction is enough to build a profile.
    - New items added to the catalog after the ranking model was trained —
      they are immediately available for retrieval as long as their features
      are present in the ``items`` DataFrame.
    - Catalogs that change frequently (new products, news articles, etc.).
    - Any estimator type, including XGBoost and LightGBM.

    Not for
    -------
    - Purely ID-based catalogs with no item metadata.
    - Fully cold-start users (zero interactions) — these fall back to the
      globally popular candidates automatically.

    Feature requirements
    --------------------
    Only **numeric** columns are supported for v1.0. Categorical columns must
    be pre-encoded (e.g. one-hot or ordinal) before passing the ``items``
    DataFrame. This matches the contract already imposed by ``UniversalScorer``.

    If ``feature_columns`` is ``None``, all numeric columns in the ``items``
    DataFrame (excluding ``ITEM_ID_NAME``) are used automatically.

    Example
    -------
    ::

        from skrec.retriever import ContentBasedRetriever

        recommender = RankingRecommender(
            scorer=UniversalScorer(estimator=xgb_estimator),
            retriever=ContentBasedRetriever(
                top_k=200,
                feature_columns=["price", "category_enc", "avg_rating"],
            ),
        )
        recommender.train(interactions_ds=interactions_ds, items_ds=items_ds)
        recommendations = recommender.recommend(interactions=df, top_k=10)
    """

    def __init__(
        self,
        top_k: int = 100,
        feature_columns: Optional[List[str]] = None,
        weight_by_outcome: bool = False,
    ):
        """
        Args:
            top_k: Number of candidate items to retrieve per user.
            feature_columns: Numeric item feature columns to use for
                similarity. If ``None``, all numeric columns in the items
                DataFrame (except ``ITEM_ID_NAME``) are used automatically.
            weight_by_outcome: If ``True``, weight each interacted item's
                feature vector by its ``LABEL_NAME`` outcome value when
                building the user profile. Useful when outcomes are ratings
                (e.g. 1–5 stars) so high-rated items contribute more.
                Outcomes are used as-is: negative values (e.g. explicit
                dislikes) will pull the profile *away* from those items,
                which may or may not be desirable. Ensure outcome semantics
                match your intent, or pre-clip outcomes to ``[0, ∞)``.
        """
        super().__init__(top_k=top_k)
        self.feature_columns = feature_columns
        self.weight_by_outcome = weight_by_outcome

        self._item_matrix: Optional[np.ndarray] = None  # (n_items, d), L2-normalized
        self._item_ids: Optional[np.ndarray] = None
        self._item_id_to_idx: Optional[Dict[Any, int]] = None
        self._user_history: Optional[Dict[Any, List[Any]]] = None  # user_id -> [item_id, ...]
        self._user_outcomes: Optional[Dict[Any, List[float]]] = None  # user_id -> [outcome, ...]
        self._popular_items: Optional[List[Any]] = None  # fallback for cold-start users

    def build_index(
        self,
        estimator: Optional[BaseEstimator] = None,
        interactions: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Build the item feature index and user interaction history.

        Args:
            estimator: Unused.
            interactions: Training interactions DataFrame. Must contain
                ``USER_ID_NAME`` and ``ITEM_ID_NAME``. If
                ``weight_by_outcome=True``, must also contain ``LABEL_NAME``.
            items: Items DataFrame with item features. Must contain
                ``ITEM_ID_NAME`` and at least one numeric feature column.

        Raises:
            ValueError: If ``items`` is None, empty, or has no numeric
                feature columns. Also raised if ``feature_columns`` includes
                ``ITEM_ID_NAME``, contains non-numeric columns, or references
                columns absent from ``items``. Also raised if ``interactions``
                is provided but missing ``USER_ID_NAME`` or ``ITEM_ID_NAME``.
        """
        if items is None or items.empty:
            raise ValueError("ContentBasedRetriever requires a non-empty items DataFrame.")
        if ITEM_ID_NAME not in items.columns:
            raise ValueError(f"items must contain '{ITEM_ID_NAME}' column.")

        # Select feature columns
        if self.feature_columns is not None:
            if ITEM_ID_NAME in self.feature_columns:
                raise ValueError(
                    f"feature_columns must not include '{ITEM_ID_NAME}' — item ID is an identifier, not a feature."
                )
            missing = [c for c in self.feature_columns if c not in items.columns]
            if missing:
                raise ValueError(f"feature_columns not found in items DataFrame: {missing}")
            non_numeric = [c for c in self.feature_columns if not pd.api.types.is_numeric_dtype(items[c])]
            if non_numeric:
                raise ValueError(
                    f"feature_columns must be numeric, but found non-numeric columns: {non_numeric}. "
                    "Encode categorical columns numerically first."
                )
            feat_cols = self.feature_columns
        else:
            feat_cols = [c for c in items.select_dtypes(include=[np.number]).columns if c != ITEM_ID_NAME]

        if not feat_cols:
            raise ValueError(
                "ContentBasedRetriever found no numeric feature columns in items DataFrame. "
                "Encode categorical columns numerically first, or specify feature_columns explicitly."
            )

        # Build normalized item matrix
        self._item_ids = items[ITEM_ID_NAME].values
        raw_matrix = items[feat_cols].values.astype(np.float64)

        nan_mask = np.isnan(raw_matrix)
        if nan_mask.any():
            nan_cols = [col for col, has_nan in zip(feat_cols, nan_mask.any(axis=0)) if has_nan]
            raise ValueError(
                f"ContentBasedRetriever found NaN values in feature columns: {nan_cols}. "
                "Impute or drop missing values before building the index."
            )

        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero for zero-feature items
        self._item_matrix = raw_matrix / norms
        self._item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self._item_ids)}

        # Build user interaction history
        self._user_history = {}
        self._user_outcomes = {}
        if interactions is not None and not interactions.empty:
            for col in (USER_ID_NAME, ITEM_ID_NAME):
                if col not in interactions.columns:
                    raise ValueError(f"ContentBasedRetriever: interactions must contain '{col}' column.")
            has_label = LABEL_NAME in interactions.columns
            if self.weight_by_outcome and not has_label:
                logger.warning(
                    "ContentBasedRetriever: weight_by_outcome=True but interactions has no '%s' column — "
                    "falling back to uniform weighting. Add a '%s' column or set weight_by_outcome=False.",
                    LABEL_NAME,
                    LABEL_NAME,
                )
            for uid, group in interactions.groupby(USER_ID_NAME, sort=False):
                self._user_history[uid] = group[ITEM_ID_NAME].tolist()
                self._user_outcomes[uid] = group[LABEL_NAME].astype(float).tolist() if has_label else [1.0] * len(group)

            # Popularity fallback: items ranked by interaction count, filtered to the
            # current catalog so the scorer always receives candidates it knows about.
            # Items in interactions that are absent from the items DataFrame (e.g.
            # deleted catalog entries) are intentionally excluded here.
            counts = interactions[ITEM_ID_NAME].value_counts()
            self._popular_items = [item_id for item_id in counts.index if item_id in self._item_id_to_idx]
            if not self._popular_items:
                # All interacted items were outside the current catalog — fall back.
                self._popular_items = self._item_ids.tolist()
        else:
            # No interactions — popularity fallback is all items in catalog order.
            self._popular_items = self._item_ids.tolist()

        logger.info(
            "ContentBasedRetriever: indexed %d items with %d features, %d users with history.",
            len(self._item_ids),
            len(feat_cols),
            len(self._user_history),
        )

    def _build_user_profile(self, user_id: Any) -> Optional[np.ndarray]:
        """
        Build a normalized user profile vector from interaction history.

        Returns None if the user has no interactions, no interactions map to
        known items (cold-start), or the weighted mean of item vectors has
        zero norm (e.g. antipodal vectors cancel). In all three cases the
        caller should use the popularity fallback.

        Note:
            Items in the user's history that are absent from the items
            DataFrame (e.g. deleted catalog entries) are silently dropped.
            If ALL of a user's history maps to unknown items, this returns
            None and the caller falls back to popularity-based candidates.
        """
        if not self._user_history or user_id not in self._user_history:
            return None

        history = self._user_history[user_id]
        outcomes = self._user_outcomes[user_id]

        # Single-pass filter: gather positions of known items, dropping any catalog gaps.
        known_pos = [i for i, item_id in enumerate(history) if item_id in self._item_id_to_idx]
        if not known_pos:
            return None

        idxs = np.array([self._item_id_to_idx[history[i]] for i in known_pos])
        item_vecs = self._item_matrix[idxs]  # (n_known, d) — single numpy fancy-index

        weights_arr = (
            np.array([outcomes[i] for i in known_pos], dtype=np.float64)
            if self.weight_by_outcome
            else np.ones(len(known_pos), dtype=np.float64)
        )
        total = weights_arr.sum()
        if total == 0:
            weights_arr = np.ones(len(weights_arr), dtype=np.float64)
            total = float(len(weights_arr))
        weights_arr /= total

        profile = item_vecs.T @ weights_arr  # (d,) weighted mean, no Python loop
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile /= norm
            return profile
        # Zero-norm profile (e.g. antipodal item vectors cancel out) — treat as cold-start
        # so the caller falls back to the popularity ranking.
        return None

    def retrieve(self, user_ids: List[Any], top_k: int) -> Dict[Any, List[Any]]:
        """
        Retrieve top-K candidates for each user by cosine similarity.

        Three cases fall back to globally popular items instead of
        cosine-similarity ranking:

        1. **Fully cold-start users** — user has no interaction history at all.
        2. **Fully unknown history** — user has interactions, but every item
           they interacted with is absent from the current items DataFrame
           (e.g. catalog entries deleted after training). In this case no
           user profile can be built and popular items are returned instead.
        3. **Zero-norm profile** — the weighted mean of the user's item vectors
           has zero norm (e.g. antipodal vectors cancel exactly). No meaningful
           cosine similarity can be computed, so popular items are returned.

        Complexity
        ----------
        ``O(n_items × d)`` per user, where ``d`` is the number of feature
        columns. The dominant cost is the matrix–vector dot product
        ``item_matrix @ profile`` (shape ``(n_items, d) @ (d,)``). For
        catalogs with millions of items or high-dimensional feature spaces
        consider a custom retriever backed by an ANN index (e.g. FAISS).

        Args:
            user_ids: User IDs to retrieve candidates for.
            top_k: Number of candidates per user.

        Returns:
            Dict mapping user_id to list of candidate item IDs.
        """
        if self._item_matrix is None:
            raise RuntimeError("build_index() must be called before retrieve().")

        effective_k = min(top_k, len(self._item_ids))
        results: Dict[Any, List[Any]] = {}

        for user_id in user_ids:
            profile = self._build_user_profile(user_id)
            if profile is None:
                # Cold-start: fall back to popular items, capped at effective_k to stay
                # consistent with the catalog-size cap applied to warm users.
                if not self._user_history or user_id not in self._user_history:
                    logger.warning(
                        "ContentBasedRetriever: user %s has no interaction history (cold-start) — "
                        "falling back to popularity.",
                        user_id,
                    )
                elif not any(item_id in self._item_id_to_idx for item_id in self._user_history[user_id]):
                    logger.warning(
                        "ContentBasedRetriever: user %s has interaction history but every interacted "
                        "item is absent from the current catalog (stale history) — "
                        "falling back to popularity. Consider rebuilding the index after catalog changes.",
                        user_id,
                    )
                else:
                    logger.warning(
                        "ContentBasedRetriever: user %s has a zero-norm profile vector (interacted item "
                        "vectors cancel out) — falling back to popularity.",
                        user_id,
                    )
                results[user_id] = self._popular_items[:effective_k]
                continue

            scores = self._item_matrix @ profile  # cosine similarity (items are L2-normalized)
            top_indices = self._topk_indices(scores, effective_k)
            results[user_id] = self._item_ids[top_indices].tolist()

        return results
