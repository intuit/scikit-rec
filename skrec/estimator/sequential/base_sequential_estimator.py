from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

from numpy.typing import NDArray

from skrec.constants import ITEM_EMBEDDING_NAME, ITEM_ID_NAME
from skrec.util.torch_device import select_torch_device


class SequentialEstimator(ABC):
    """
    Abstract base class for sequential recommendation estimators.

    Defines the contract for estimators that operate on item sequences (SASRec)
    or session hierarchies (HRNN). Sequential estimators receive full interaction
    histories rather than flat feature matrices, so they intentionally do not
    implement the tabular ``predict(X)`` / ``predict_proba(X)`` interface defined
    by ``BaseEstimator``.

    The canonical training entry point is ``fit_embedding_model()``.
    The canonical inference entry point is ``predict_proba_with_embeddings()``.

    Subclasses must implement:
        - ``_build_pytorch_model`` — construct the PyTorch ``nn.Module``
        - ``fit_embedding_model`` — train on sequence/session data
        - ``predict_proba_with_embeddings`` — return (n_users, n_items) scores

    Subclasses may override:
        - ``get_item_embeddings`` — default reads ``model.item_embedding``; override if
          the model uses a different attribute name
        - ``get_user_embeddings`` — default raises ``NotImplementedError``; override in
          subclasses that learn a persistent per-user embedding table
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        optimizer_name: Union[str, Type[optim.Optimizer]] = "adam",
        loss_fn_name: Union[str, nn.Module] = "bce",
        weight_decay: float = 0.0,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        if torch is None:
            raise ImportError(
                "PyTorch is required for sequential estimators. Install it with: pip install scikit-rec[torch]"
            )
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.loss_fn_name = loss_fn_name
        self.weight_decay = weight_decay
        self.device_param_str = device
        self.random_state = random_state
        self.verbose = verbose
        self._user_data_truncated: bool = False

        self.optimizer_cls = self._get_optimizer_cls(self.optimizer_name)
        self.loss_fn = self._get_loss_fn(self.loss_fn_name)

        self.model: Optional[nn.Module] = None

        self.user_id_index: Optional[pd.Index] = None
        self.item_id_index: Optional[pd.Index] = None
        self.num_users: int = 0
        self.num_items: int = 0
        self.unknown_user_idx: Optional[int] = None
        self.unknown_item_idx: Optional[int] = None

        self.device = select_torch_device(self.device_param_str)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_pytorch_model(self) -> nn.Module:
        """Construct and return the PyTorch model (before moving to device).

        Called internally by ``fit_embedding_model`` and ``__setstate__``
        (pickle restore).
        """

    @abstractmethod
    def fit_embedding_model(
        self,
        users: Optional[pd.DataFrame],
        items: Optional[pd.DataFrame],
        interactions: pd.DataFrame,
        valid_users: Optional[pd.DataFrame] = None,
        valid_interactions: Optional[pd.DataFrame] = None,
    ) -> None:
        """Train on sequence or session-structured interaction data.

        Args:
            users: Optional user-feature DataFrame (ignored by SASRec/HRNN).
            items: DataFrame with ``ITEM_ID`` column. Defines the item vocabulary.
            interactions: Sequence DataFrame produced by the recommender layer.
            valid_users: Optional — ignored by SASRec/HRNN.
            valid_interactions: Optional sequence DataFrame for early stopping.
        """

    @abstractmethod
    def predict_proba_with_embeddings(
        self,
        interactions: pd.DataFrame,
        users: Optional[pd.DataFrame] = None,
    ) -> NDArray:
        """Score all catalogue items for each user.

        Args:
            interactions: DataFrame with ``USER_ID`` and sequence column(s).
            users: Ignored for sequence-only models (SASRec, HRNN).

        Returns:
            ``(n_users, n_items)`` float array, column-aligned with
            ``self.item_id_index``.
        """

    def get_item_embeddings(self) -> pd.DataFrame:
        """Return learned item embeddings.

        The default implementation reads ``self.model.item_embedding``, which is
        the expected attribute name for ``nn.Module`` subclasses built by this
        hierarchy. Override if your model uses a different attribute name.

        Returns:
            DataFrame with ``ITEM_ID`` and ``ITEM_EMBEDDING`` columns,
            one row per item in ``self.item_id_index``.

        Raises:
            RuntimeError: If called before ``fit_embedding_model``.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit_embedding_model() first.")
        # item_embedding rows: 0 = padding, 1..num_items = real items
        weights = self.model.item_embedding.weight.data[1:].cpu().numpy()  # type: ignore[union-attr]
        return pd.DataFrame(
            {ITEM_ID_NAME: list(self.item_id_index), ITEM_EMBEDDING_NAME: list(weights)}  # type: ignore[arg-type]
        )

    def get_user_embeddings(self) -> pd.DataFrame:
        """Return learned user embeddings.

        Sequential models (SASRec, HRNN) derive user representations from
        interaction sequences at inference time and do not maintain a persistent
        per-user embedding table. Override this method in subclasses that do
        learn such a table.

        Raises:
            NotImplementedError: Always, for sequence-only models.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not learn per-user embeddings. "
            "User representations are derived from interaction sequences at inference time."
        )

    def support_batch_training(self) -> bool:
        """Sequential estimators do not support batch training."""
        return False

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _get_optimizer_cls(self, optimizer_name: Union[str, Type[optim.Optimizer]]) -> Type[optim.Optimizer]:
        if not isinstance(optimizer_name, str):
            return optimizer_name
        name = optimizer_name.lower()
        if name == "adam":
            return optim.Adam
        if name == "sgd":
            return optim.SGD
        raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Supported: 'adam', 'sgd'.")

    def _get_loss_fn(self, loss_fn_name: Union[str, nn.Module]) -> nn.Module:
        if not isinstance(loss_fn_name, str):
            return loss_fn_name
        name = loss_fn_name.lower()
        if name == "bce":
            return nn.BCEWithLogitsLoss()
        if name == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unsupported loss function: '{loss_fn_name}'. Supported: 'bce', 'mse'.")

    def get_config(self) -> Dict[str, Any]:
        """Return a dict of the base hyperparameters.

        Subclasses should call ``super().get_config()`` and add their own fields.
        """
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer_name": self.optimizer_name
            if isinstance(self.optimizer_name, str)
            else self.optimizer_name.__name__,
            "loss_fn_name": self.loss_fn_name
            if isinstance(self.loss_fn_name, str)
            else self.loss_fn_name.__class__.__name__,
            "weight_decay": self.weight_decay,
            "device": self.device_param_str,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    # ------------------------------------------------------------------
    # Pickling — store model weights separately from the nn.Module graph
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if self.model is not None:
            state["_model_state_dict"] = self.model.state_dict()
        state.pop("model", None)
        state.pop("loss_fn", None)
        state.pop("optimizer_cls", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.model = self._build_pytorch_model()
        self.model.to(self.device)
        if "_model_state_dict" in state:
            self.model.load_state_dict(self.__dict__.pop("_model_state_dict"), strict=True)
        self.loss_fn = self._get_loss_fn(self.loss_fn_name)
        self.optimizer_cls = self._get_optimizer_cls(self.optimizer_name)
