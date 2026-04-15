# code for multioutput scorer for getting data into sklearn MultiOutputClassifier
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder

import skrec.constants as C
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.scorer.base_scorer import BaseScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class MultioutputScorer(BaseScorer):
    def __init__(self, estimator: BaseClassifier) -> None:
        if isinstance(estimator, BaseEmbeddingEstimator):
            raise TypeError(
                "MultioutputScorer does not support BaseEmbeddingEstimator. "
                "Use UniversalScorer for embedding estimators (e.g. MatrixFactorizationEstimator, NCFEstimator)."
            )
        super().__init__(estimator)
        self.label_encoding_transformer = None

    def process_datasets(
        self,
        users_df: Optional[pd.DataFrame] = None,
        items_df: Optional[pd.DataFrame] = None,
        interactions_df: Optional[pd.DataFrame] = None,
        is_training: Optional[bool] = True,
    ) -> Tuple[pd.DataFrame, NDArray]:
        """Validate and prepare wide-format interaction data for multi-output classification.

        ``MultioutputScorer`` expects interactions in wide format: one row per
        user with one ``ITEM_<name>`` column per item to predict.  User and item
        feature DataFrames are not used and must be ``None``.

        On first call (``is_training=True``) fits a ``LabelEncoder`` per item
        column and encodes the target columns in place.

        Args:
            users_df: Must be ``None``.
            items_df: Must be ``None``.
            interactions_df: Wide-format DataFrame with ``USER_ID_NAME`` and at
                least two ``ITEM_*`` columns.  One row per user; no duplicate
                user IDs allowed.
            is_training: When ``True``, fits label encoders and initialises
                item state.

        Returns:
            A tuple ``(X, y)`` where ``X`` contains the non-item feature
            columns and ``y`` is the encoded multi-output label matrix.

        Raises:
            ValueError: If ``users_df`` or ``items_df`` is provided.
        """
        if users_df is not None or items_df is not None:
            raise ValueError("Item Dataset and User Dataset will not be used in MultioutputScorer.")

        # Validate BEFORE label encoding: _fit_label_encoders encodes ITEM columns in-place
        # (string → int), so the null and duplicate checks in _validate_interactions must
        # run on the original string values.  super().process_datasets() will call
        # _validate_interactions a second time as part of its template; the cost is O(N)
        # but correctness requires the first call to precede the mutation.
        self._validate_interactions(interactions_df)
        if is_training:
            self._fit_label_encoders(interactions_df)
        else:
            self._transform_label_encoders(interactions_df)
        return super().process_datasets(
            users_df=None,
            items_df=None,
            interactions_df=interactions_df,
            is_training=is_training,
        )

    def _generate_X_y(self, joined_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        list_labels = [x for x in joined_data.columns if x.startswith(C.ITEM_PREFIX)]

        y = joined_data[list_labels]
        X = joined_data.drop(list_labels + [C.USER_ID_NAME], axis=1, errors="ignore")
        return X, y

    def _process_items(self, items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> Tuple[NDArray, pd.DataFrame]:
        returned_items_df = None
        items_list = np.array([x for x in interactions_df.columns if x.startswith(C.ITEM_PREFIX)])
        return items_list, returned_items_df

    def _join_data_train(
        self, users_df: pd.DataFrame, items_df: pd.DataFrame, interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        return interactions_df.copy()

    def _validate_interactions(self, interactions_df: pd.DataFrame) -> None:
        """
        Full override — wide-format interactions have no single target_col or ITEM_ID column,
        so the base long-format checks do not apply.
        Only accepts:
        USER_ID, ITEM_name1, ITEM_name2, ..., ITEM_nameN, feature1, feature2, ..., featureN
        """
        self._validate_interactions_base(interactions_df)

        if C.USER_ID_NAME not in interactions_df.columns:
            raise ValueError("Interaction Dataset must contain USER_ID column for Multioutput Scorer.")

        item_cols = [col for col in interactions_df.columns if col.startswith(C.ITEM_PREFIX)]

        for col in item_cols:
            null_count = interactions_df[col].isnull().sum()
            if null_count > 0:
                raise ValueError(
                    f"item column '{col}' contains {null_count} null value(s). Remove or impute before training."
                )

        if len(item_cols) < 2:
            raise ValueError("Interaction Dataset must contain at least 2 ITEM columns for Multioutput Scorer.")

        if interactions_df[C.USER_ID_NAME].duplicated().any():
            raise ValueError("Multioutput Scorer only accepts one row per user.")

    def _fit_label_encoders(self, interactions_df: pd.DataFrame) -> None:
        """Fit a LabelEncoder per ITEM column, encode in-place, and initialize item state."""
        self.item_count = 0
        self.item_names = []
        self.label_encoding_transformer = {}
        self.output_classes = {}

        for col in interactions_df.columns:
            if col.startswith(C.ITEM_PREFIX):
                self.item_count += 1
                self.item_names.append(col)
                le = LabelEncoder()
                interactions_df[col] = le.fit_transform(interactions_df[col])
                self.label_encoding_transformer[col] = le
                self.output_classes[col] = le.classes_

    def _transform_label_encoders(self, interactions_df: pd.DataFrame) -> None:
        """Encode ITEM columns in-place using already-fitted LabelEncoders."""
        for col in interactions_df.columns:
            if col.startswith(C.ITEM_PREFIX) and col in self.label_encoding_transformer:
                interactions_df[col] = self.label_encoding_transformer[col].transform(interactions_df[col])

    def _calculate_scores(self, joined: Union[pd.DataFrame, NDArray]) -> List[NDArray[np.float64]]:
        return self.estimator.predict_proba(joined)

    def score_fast(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return predicted class labels for all items for a single pre-merged user row.

        Args:
            features: Single-row DataFrame with interaction features.
                ``USER_ID_NAME``, ``ITEM_ID_NAME``, and ``LABEL_NAME`` columns
                are silently dropped if present.

        Returns:
            DataFrame of shape ``(1, n_items)`` with the predicted class label
            per item column (same as ``predict_classes`` for a single user).

        Raises:
            ValueError: If ``features`` has more than one row.
        """
        if features.shape[0] != 1:
            raise ValueError(
                f"score_fast() expects exactly 1 row, got {features.shape[0]}. Use predict_classes() for batch scoring."
            )
        drop_cols = [col for col in [C.USER_ID_NAME, C.ITEM_ID_NAME, C.LABEL_NAME] if col in features.columns]
        if drop_cols:
            features = features.drop(columns=drop_cols)
        scores = self._calculate_scores(features)
        return self._create_df_from_scores(scores)

    def predict_classes(
        self,
        interactions: Optional[pd.DataFrame] = None,
        users: Optional[pd.DataFrame] = None,  # Should always be None for this scorer
    ) -> pd.DataFrame:
        """
        Returns the predicted class label for each item. Shape: (n_users, n_items).

        For per-class probabilities (e.g. ``ITEM_600_0``, ``ITEM_600_1``),
        call ``score_items()`` instead.
        """
        if users is not None:
            raise ValueError("Multioutput Scorer cannot accept Users Dataframe, set it to None!")
        return super().score_items(interactions, users)

    def score_items(
        self,
        interactions: Optional[pd.DataFrame] = None,
        users: Optional[pd.DataFrame] = None,  # Should always be None for this scorer
    ) -> pd.DataFrame:
        """
        Returns per-class probability scores for all items.

        Unlike ``predict_classes()`` which returns a single predicted class label per item,
        this returns the full probability distribution over classes for each item,
        e.g. columns ``ITEM_600_0``, ``ITEM_600_1`` for a binary item.

        Args:
            interactions: DataFrame with interaction features (must not include users).
            users: Must be None — MultioutputScorer does not accept a users DataFrame.

        Returns:
            DataFrame of shape (n_users, sum of n_classes across items).
        """
        if users is not None:
            raise ValueError("Multioutput Scorer cannot accept Users Dataframe, set it to None!")
        user_interactions_df = self._get_user_interactions_df(interactions=interactions, users=users)
        scores = self._calculate_scores(user_interactions_df)
        return self._create_proba_df(scores)

    def _create_df_from_scores(self, scores: NDArray) -> pd.DataFrame:
        """Returns predicted class per item. Shape: (n_users, n_items)."""
        allowed_items = self.item_subset if self.item_subset else self.item_names
        result = {}
        for col_num, col_name in enumerate(self.item_names):
            if col_name in allowed_items:
                le = self.label_encoding_transformer[col_name]
                predictions = np.argmax(scores[col_num], axis=1)
                result[col_name] = le.inverse_transform(predictions)
        return pd.DataFrame(result)

    def _create_proba_df(self, scores: NDArray) -> pd.DataFrame:
        """Returns per-class probabilities. Shape: (n_users, sum of n_classes across items)."""
        allowed_items = self.item_subset if self.item_subset else self.item_names
        dfs = []
        for col_num, col_name in enumerate(self.item_names):
            if col_name in allowed_items:
                le = self.label_encoding_transformer[col_name]
                col_names = [f"{col_name}_{c}" for c in le.classes_]
                dfs.append(pd.DataFrame(scores[col_num], columns=col_names, dtype=np.float64))
        return pd.concat(dfs, axis=1)
