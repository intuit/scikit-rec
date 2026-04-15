"""
Native matrix factorization (collaborative filtering) estimator.

Uses only NumPy — no Surprise, no PyTorch. Implements Alternating Least Squares (ALS)
and Stochastic Gradient Descent (SGD). Compatible with UniversalScorer and RankingRecommender.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from skrec.constants import (
    ITEM_EMBEDDING_NAME,
    ITEM_ID_NAME,
    LABEL_NAME,
    USER_EMBEDDING_NAME,
    USER_ID_NAME,
)
from skrec.estimator.datatypes import MFAlgorithm, MFOutcomeType
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.util.logger import get_logger


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable sigmoid."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


logger = get_logger(__name__)


class MatrixFactorizationEstimator(BaseEmbeddingEstimator):
    """
    Native collaborative filtering via matrix factorization (ALS or SGD).

    Learns user and item latent factors from (user_id, item_id, outcome) interactions.
    Supports continuous, ordinal (e.g. 1–5 ratings), and binary outcomes. Uses only NumPy;
    compatible with UniversalScorer and RankingRecommender.
    """

    def __init__(
        self,
        n_factors: int = 32,
        algorithm: Union[MFAlgorithm, str] = MFAlgorithm.ALS,
        outcome_type: Union[MFOutcomeType, str] = MFOutcomeType.CONTINUOUS,
        ordinal_min: Optional[float] = None,
        ordinal_max: Optional[float] = None,
        regularization: float = 0.01,
        learning_rate: float = 0.01,
        epochs: int = 20,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        """
        Args:
            n_factors: Latent dimension (rank) of the factorization.
            algorithm: Solver to use (MFAlgorithm.ALS or MFAlgorithm.SGD).
            outcome_type: CONTINUOUS (real-valued, MSE); ORDINAL (e.g. 1–5 ratings,
                MSE, optional clamp); BINARY (0/1, BCE with SGD, sigmoid at predict).
            ordinal_min: For ORDINAL outcome_type, optional lower bound (e.g. 1 for 1–5 stars).
            ordinal_max: For ORDINAL outcome_type, optional upper bound (e.g. 5 for 1–5 stars).
            regularization: L2 regularization for factor updates.
            learning_rate: Step size for SGD (ignored when algorithm is ALS).
            epochs: Number of training iterations (ALS alternations or SGD epochs).
            random_state: Random seed for factor initialization and SGD shuffle.
            verbose: If > 0, log progress every epoch.
        """
        self.n_factors = n_factors
        self.algorithm = MFAlgorithm(algorithm) if isinstance(algorithm, str) else algorithm
        self.outcome_type = MFOutcomeType(outcome_type) if isinstance(outcome_type, str) else outcome_type
        self.ordinal_min = ordinal_min
        self.ordinal_max = ordinal_max
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

        self.user_factors_: Optional[NDArray[np.float64]] = None  # (n_users+1, n_factors)
        self.item_factors_: Optional[NDArray[np.float64]] = None  # (n_items+1, n_factors)
        self.global_mean_: float = 0.0

        self.user_id_index_: Optional[pd.Index] = None
        self.item_id_index_: Optional[pd.Index] = None
        self.unknown_user_idx_: Optional[int] = None
        self.unknown_item_idx_: Optional[int] = None
        self.n_users_: int = 0
        self.n_items_: int = 0

    def _build_id_indices(
        self,
        interactions: pd.DataFrame,
        items: Optional[pd.DataFrame],
    ) -> None:
        """Build user/item id -> index mappings aligned with scorer item order when possible."""
        # Users: from interactions only
        user_ids = interactions[USER_ID_NAME].unique()
        self.user_id_index_ = pd.Index(np.sort(user_ids.astype(str)))
        self.unknown_user_idx_ = len(self.user_id_index_)
        self.n_users_ = self.unknown_user_idx_ + 1

        # Items: from items_df if provided (to match scorer), else from interactions
        if items is not None and not items.empty and ITEM_ID_NAME in items.columns:
            self.item_id_index_ = pd.Index(np.sort(items[ITEM_ID_NAME].astype(str).unique()))
        else:
            item_ids = interactions[ITEM_ID_NAME].unique()
            self.item_id_index_ = pd.Index(np.sort(item_ids.astype(str)))
        self.unknown_item_idx_ = len(self.item_id_index_)
        self.n_items_ = self.unknown_item_idx_ + 1

    def _user_item_label_arrays(
        self,
        interactions: pd.DataFrame,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.float64]]:
        """Map USER_ID and ITEM_ID to indices and return label array."""
        u_str = interactions[USER_ID_NAME].astype(str).values
        i_str = interactions[ITEM_ID_NAME].astype(str).values
        u_idx = self.user_id_index_.get_indexer(u_str)
        u_idx[u_idx == -1] = self.unknown_user_idx_
        i_idx = self.item_id_index_.get_indexer(i_str)
        i_idx[i_idx == -1] = self.unknown_item_idx_
        y = interactions[LABEL_NAME].values.astype(np.float64)
        return u_idx, i_idx, y

    def _fit_als(
        self,
        u_idx: NDArray[np.int_],
        i_idx: NDArray[np.int_],
        y: NDArray[np.float64],
    ) -> None:
        """Train factors using Alternating Least Squares."""
        reg = self.regularization
        reg_I = reg * np.eye(self.n_factors, dtype=np.float64)
        for epoch in range(self.epochs):
            for u in range(self.n_users_):
                mask = u_idx == u
                if not np.any(mask):
                    continue
                i_obs = i_idx[mask]
                r_obs = y[mask] - self.global_mean_
                Q_u = self.item_factors_[i_obs]
                XtX = Q_u.T @ Q_u + reg_I
                Xty = Q_u.T @ r_obs
                self.user_factors_[u] = np.linalg.solve(XtX, Xty)
            for i in range(self.n_items_):
                mask = i_idx == i
                if not np.any(mask):
                    continue
                u_obs = u_idx[mask]
                r_obs = y[mask] - self.global_mean_
                P_i = self.user_factors_[u_obs]
                XtX = P_i.T @ P_i + reg_I
                Xty = P_i.T @ r_obs
                self.item_factors_[i] = np.linalg.solve(XtX, Xty)
            if self.verbose > 0 and (epoch + 1) % max(1, self.epochs // 5) == 0:
                pred = self.global_mean_ + np.sum(self.user_factors_[u_idx] * self.item_factors_[i_idx], axis=1)
                mse = float(np.mean((y - pred) ** 2))
                logger.info("ALS epoch %d / %d — train MSE: %.4f", epoch + 1, self.epochs, mse)

    def _fit_sgd(
        self,
        u_idx: NDArray[np.int_],
        i_idx: NDArray[np.int_],
        y: NDArray[np.float64],
    ) -> None:
        """Train factors using Stochastic Gradient Descent (MSE for continuous/ordinal, BCE for binary)."""
        n_samples = len(y)
        lr = self.learning_rate
        reg = self.regularization
        binary = self.outcome_type == MFOutcomeType.BINARY
        for epoch in range(self.epochs):
            perm = np.arange(n_samples)
            if self.random_state is not None:
                np.random.seed(self.random_state + epoch)
            np.random.shuffle(perm)
            u_perm = u_idx[perm]
            i_perm = i_idx[perm]
            y_perm = y[perm]
            for k in range(n_samples):
                u, i, r = int(u_perm[k]), int(i_perm[k]), y_perm[k]
                dot = np.dot(self.user_factors_[u], self.item_factors_[i])
                pred = self.global_mean_ + dot
                if binary:
                    err = _sigmoid(np.array([pred]))[0] - r
                else:
                    err = r - pred
                p_u = self.user_factors_[u].copy()
                q_i = self.item_factors_[i].copy()
                self.user_factors_[u] += lr * (err * q_i - reg * p_u)
                self.item_factors_[i] += lr * (err * p_u - reg * q_i)
            if self.verbose > 0 and (epoch + 1) % max(1, self.epochs // 5) == 0:
                pred_all = self.global_mean_ + np.sum(self.user_factors_[u_idx] * self.item_factors_[i_idx], axis=1)
                if binary:
                    prob = _sigmoid(pred_all)
                    bce = -np.mean(y * np.log(prob + 1e-12) + (1 - y) * np.log(1 - prob + 1e-12))
                    logger.info("SGD epoch %d / %d — train BCE: %.4f", epoch + 1, self.epochs, bce)
                else:
                    mse = float(np.mean((y - pred_all) ** 2))
                    logger.info("SGD epoch %d / %d — train MSE: %.4f", epoch + 1, self.epochs, mse)

    def fit_embedding_model(
        self,
        users: Optional[pd.DataFrame],
        items: Optional[pd.DataFrame],
        interactions: pd.DataFrame,
        valid_users: Optional[pd.DataFrame] = None,
        valid_interactions: Optional[pd.DataFrame] = None,
    ) -> None:
        if interactions is None or interactions.empty:
            raise ValueError("interactions must be a non-empty DataFrame.")
        if LABEL_NAME not in interactions.columns:
            raise ValueError(f"interactions must contain '{LABEL_NAME}'.")
        if USER_ID_NAME not in interactions.columns or ITEM_ID_NAME not in interactions.columns:
            raise ValueError(f"interactions must contain '{USER_ID_NAME}' and '{ITEM_ID_NAME}'.")

        if users is not None and not users.empty and len(users.columns) > 1:
            logger.warning(
                "Customer (user) features are not supported in this basic collaborative filtering setup. "
                "Only user IDs are used; extra columns in the users DataFrame are ignored."
            )
        if items is not None and not items.empty and len(items.columns) > 1:
            logger.warning(
                "Item features are not supported in this basic collaborative filtering setup. "
                "Only item IDs are used; extra columns in the items DataFrame are ignored."
            )

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._build_id_indices(interactions, items)
        u_idx, i_idx, y = self._user_item_label_arrays(interactions)
        self.global_mean_ = float(np.mean(y))

        self.user_factors_ = np.random.randn(self.n_users_, self.n_factors).astype(np.float64) * 0.01
        self.item_factors_ = np.random.randn(self.n_items_, self.n_factors).astype(np.float64) * 0.01

        if self.algorithm == MFAlgorithm.ALS:
            self._fit_als(u_idx, i_idx, y)
        elif self.algorithm == MFAlgorithm.SGD:
            self._fit_sgd(u_idx, i_idx, y)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Use MFAlgorithm.ALS or MFAlgorithm.SGD.")

    def predict_proba_with_embeddings(
        self,
        interactions: pd.DataFrame,
        users: Optional[pd.DataFrame] = None,
    ) -> NDArray:
        if self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("Model not fitted. Call fit_embedding_model first.")

        if interactions is None:
            raise ValueError("interactions cannot be None.")
        for col in (USER_ID_NAME, ITEM_ID_NAME):
            if col not in interactions.columns:
                raise ValueError(
                    f"interactions is missing required column '{col}'. Got columns: {list(interactions.columns)}"
                )

        n_rows = len(interactions)
        scores = np.zeros(n_rows, dtype=np.float64)

        use_external_embeddings = users is not None and not users.empty and USER_EMBEDDING_NAME in users.columns

        if use_external_embeddings:
            # Real-time: user vector from users DataFrame (indexed by USER_ID)
            users_by_id = users.set_index(USER_ID_NAME)
            for idx in range(n_rows):
                uid = interactions[USER_ID_NAME].iloc[idx]
                iid = interactions[ITEM_ID_NAME].iloc[idx]
                try:
                    user_vec = users_by_id.loc[uid, USER_EMBEDDING_NAME]
                except (KeyError, TypeError):
                    user_vec = self.user_factors_[self.unknown_user_idx_]
                if isinstance(user_vec, pd.Series):
                    user_vec = user_vec.values
                user_vec = np.asarray(user_vec, dtype=np.float64).ravel()
                i_idx = self.item_id_index_.get_indexer([str(iid)])[0]
                if i_idx == -1:
                    i_idx = self.unknown_item_idx_
                item_vec = self.item_factors_[i_idx]
                if user_vec.shape[0] != item_vec.shape[0]:
                    user_vec = self.user_factors_[self.unknown_user_idx_]
                scores[idx] = self.global_mean_ + float(np.dot(user_vec, item_vec))
        else:
            # Batch: use internal user and item factors
            u_str = interactions[USER_ID_NAME].astype(str).values
            i_str = interactions[ITEM_ID_NAME].astype(str).values
            u_idx = self.user_id_index_.get_indexer(u_str)
            u_idx[u_idx == -1] = self.unknown_user_idx_
            i_idx = self.item_id_index_.get_indexer(i_str)
            i_idx[i_idx == -1] = self.unknown_item_idx_
            scores = self.global_mean_ + np.sum(self.user_factors_[u_idx] * self.item_factors_[i_idx], axis=1)

        if self.outcome_type == MFOutcomeType.BINARY:
            scores = _sigmoid(scores)
        elif (
            self.outcome_type == MFOutcomeType.ORDINAL and self.ordinal_min is not None and self.ordinal_max is not None
        ):
            scores = np.clip(scores, self.ordinal_min, self.ordinal_max)
        return scores.astype(np.float64)

    def get_user_embeddings(self) -> pd.DataFrame:
        """Return a DataFrame of user_id -> embedding (user factor row) for real-time use."""
        if self.user_factors_ is None or self.user_id_index_ is None:
            raise RuntimeError("Model not fitted. Call fit_embedding_model first.")
        # Exclude unknown placeholder
        n = self.unknown_user_idx_
        if n == 0:
            return pd.DataFrame(columns=[USER_ID_NAME, USER_EMBEDDING_NAME])
        rows = [{USER_ID_NAME: self.user_id_index_[i], USER_EMBEDDING_NAME: self.user_factors_[i]} for i in range(n)]
        return pd.DataFrame(rows)

    def get_item_embeddings(self) -> pd.DataFrame:
        """Return a DataFrame of item_id -> embedding (item factor row) for retrieval use."""
        if self.item_factors_ is None or self.item_id_index_ is None:
            raise RuntimeError("Model not fitted. Call fit_embedding_model first.")
        n = self.unknown_item_idx_
        if n == 0:
            return pd.DataFrame(columns=[ITEM_ID_NAME, ITEM_EMBEDDING_NAME])
        rows = [{ITEM_ID_NAME: self.item_id_index_[i], ITEM_EMBEDDING_NAME: self.item_factors_[i]} for i in range(n)]
        return pd.DataFrame(rows)
