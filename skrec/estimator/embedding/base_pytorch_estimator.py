from __future__ import annotations

import abc
import copy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from skrec.util.logger import get_logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

from numpy.typing import NDArray

from skrec.constants import (
    ITEM_EMBEDDING_NAME,
    ITEM_ID_NAME,
    LABEL_NAME,
    USER_EMBEDDING_NAME,
    USER_ID_NAME,
)
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.util.torch_device import select_torch_device

logger = get_logger(__name__)

if nn is not None:
    _embedding_module_bases = (nn.Module, abc.ABC)
else:
    _embedding_module_bases = (abc.ABC,)


class BasePyTorchEmbeddingModule(*_embedding_module_bases):
    """
    Base class for PyTorch nn.Module models used within BasePyTorchEmbeddingEstimator.
    Handles common initialization of embedding layers, feature projection layers,
    and provides a structured forward pass.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_embedding_dim: int,
        item_embedding_dim: int,
        user_features_dim: int,
        item_features_dim: int,
        # Full tensors from the estimator, containing features for all users/items known at training time.
        # These are stored here for use during training/inference when specific features aren't passed to forward().
        user_features_tensor: Optional[torch.Tensor],
        item_features_tensor: Optional[torch.Tensor],
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.user_features_dim = user_features_dim
        self.item_features_dim = item_features_dim

        self.user_id_embedding = nn.Embedding(self.num_users, self.user_embedding_dim)
        self.item_id_embedding = nn.Embedding(self.num_items, self.item_embedding_dim)

        self.user_features_tensor = user_features_tensor
        self.item_features_tensor = item_features_tensor

    def _get_user_embeddings(
        self,
        user_indices: torch.Tensor,
        user_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Helper to get user embeddings, either from input or learned layer."""
        if user_embeddings is not None:
            # Inference with provided embeddings
            return user_embeddings[user_indices]
        else:
            # Training or inference using model's own embeddings
            return self.user_id_embedding(user_indices)

    def _get_user_features(
        self,
        user_indices: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Helper to get user features, either from input or stored full tensor."""
        if user_features is not None:
            # Inference with provided features
            if user_features.shape[1] == 0:  # Empty features tensor passed
                return None
            return user_features[user_indices]
        elif self.user_features_tensor is not None and self.user_features_tensor.shape[1] > 0:
            # Training or inference using model's own full feature tensor
            return self.user_features_tensor[user_indices]
        return None

    @abc.abstractmethod
    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        interaction_features: torch.Tensor,
        user_embeddings: Optional[torch.Tensor] = None,
        user_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Main forward pass. Processes embeddings and features, then calls
        the model-specific architecture.
        This method must be implemented by subclasses.

        Args:
            user_indices: (batch_size,), indexing in `user_embeddings` or `self.user_id_embedding`
            item_indices: (batch_size,), indexing in `self.item_id_embedding`
            interaction_features: (batch_size, context_ftr_dim)
            user_embeddings: (n_users_in_ems,) to replace `self.user_id_embedding` during real-time inference
            user_features: (n_users_in_ems,) to replace `self.user_features_tensor` during real-time inference

        Returns:
            torch.Tensor: The output scores/logits from the model.
        """
        pass


class BasePyTorchEmbeddingEstimator(BaseEmbeddingEstimator, abc.ABC):
    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        optimizer_name: Union[str, Type[optim.Optimizer]] = "adam",
        loss_fn_name: Union[str, nn.Module] = "bce",
        weight_decay: float = 0.0,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        early_stopping_patience: Optional[int] = None,
        restore_best_weights: bool = True,
    ):
        """Base class for PyTorch embedding-based estimators (e.g. NCF, Two-Tower, DeepFM).

        Args:
            learning_rate: Step size for the optimizer.
            epochs: Number of full passes over the training data.
            batch_size: Number of samples per gradient-update step.
            optimizer_name: Either a string alias (``"adam"``, ``"sgd"``,
                ``"adagrad"``) or a concrete ``torch.optim.Optimizer`` subclass.
            loss_fn_name: Either a string alias (``"bce"``, ``"mse"``) or an
                instantiated ``torch.nn.Module`` loss.
            weight_decay: L2 regularisation coefficient passed to the optimizer.
            device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).  When
                ``None``, CUDA is used automatically if available, otherwise CPU.
            random_state: Seed for NumPy and PyTorch RNGs.  Set for reproducible
                training; leave ``None`` for non-deterministic runs.
            verbose: Logging verbosity level.  ``0`` = silent, ``1`` = epoch
                summaries, ``2`` = batch-level detail.
            early_stopping_patience: Stop training early if validation loss does
                not improve for this many consecutive epochs.  ``None`` disables
                early stopping.
            restore_best_weights: When ``True`` (default) and early stopping
                triggers, restore model weights from the best-validation-loss
                checkpoint before returning.
        """
        if torch is None:
            raise ImportError(
                "PyTorch is required for embedding estimators. "
                "Install it with: pip install scikit-rec[torch]"
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
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_weights = restore_best_weights
        self._user_data_truncated: bool = False

        self.optimizer_cls = self._get_optimizer_cls(self.optimizer_name)
        self.loss_fn = self._get_loss_fn(self.loss_fn_name)

        self.model: Optional[BasePyTorchEmbeddingModule] = None

        self.user_id_index: Optional[pd.Index] = None
        self.item_id_index: Optional[pd.Index] = None
        self.unknown_user_idx: Optional[int] = None
        self.unknown_item_idx: Optional[int] = None

        self.num_users: int = 0
        self.num_items: int = 0

        self.user_feature_names: List[str] = []
        self.item_feature_names: List[str] = []
        self.interaction_feature_names: List[str] = []

        self.user_features_tensor: Optional[torch.Tensor] = None
        self.item_features_tensor: Optional[torch.Tensor] = None

        self.user_features_dim: int = 0
        self.item_features_dim: int = 0
        self.interaction_features_dim: int = 0

        self.device = select_torch_device(self.device_param_str)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def _get_optimizer_cls(self, optimizer_name: Union[str, Type[optim.Optimizer]]) -> Type[optim.Optimizer]:
        if not isinstance(optimizer_name, str):
            return optimizer_name
        optimizer_name_lower = optimizer_name.lower()
        if optimizer_name_lower == "adam":
            return optim.Adam
        elif optimizer_name_lower == "sgd":
            return optim.SGD
        else:
            raise ValueError(f"Unsupported optimizer string: {optimizer_name}. Supported: 'adam', 'sgd'.")

    def _get_loss_fn(self, loss_fn_name: Union[str, nn.Module]) -> nn.Module:
        if not isinstance(loss_fn_name, str):
            return loss_fn_name
        loss_fn_name_lower = loss_fn_name.lower()
        if loss_fn_name_lower == "bce":
            return nn.BCEWithLogitsLoss()
        elif loss_fn_name_lower == "mse":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function string: {loss_fn_name}. Supported: 'bce', 'mse'.")

    def _get_feature_names_and_dims(
        self,
        interactions_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame],
        items_df: Optional[pd.DataFrame],
    ) -> Tuple[List[str], List[str], List[str], int, int, int]:
        if users_df is not None and not users_df.empty:
            user_feature_names = [col for col in users_df.columns if col != USER_ID_NAME]
        else:
            user_feature_names = []
        user_features_dim = len(user_feature_names)

        if items_df is not None and not items_df.empty:
            item_feature_names = [col for col in items_df.columns if col != ITEM_ID_NAME]
        else:
            item_feature_names = []
        item_features_dim = len(item_feature_names)

        interaction_feature_names = [
            col for col in interactions_df.columns if col not in [USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME]
        ]
        interaction_features_dim = len(interaction_feature_names)
        return (
            user_feature_names,
            item_feature_names,
            interaction_feature_names,
            user_features_dim,
            item_features_dim,
            interaction_features_dim,
        )

    def _get_id_indices_and_counts(
        self, interactions_df: pd.DataFrame
    ) -> Tuple[pd.Index, pd.Index, int, int, int, int]:
        user_id_index = pd.Index(interactions_df[USER_ID_NAME].unique())
        item_id_index = pd.Index(interactions_df[ITEM_ID_NAME].unique())

        unknown_user_idx = len(user_id_index)
        unknown_item_idx = len(item_id_index)

        num_users = len(user_id_index) + 1  # +1 for unknown
        num_items = len(item_id_index) + 1  # +1 for unknown
        return user_id_index, item_id_index, unknown_user_idx, unknown_item_idx, num_users, num_items

    def _create_features_tensor(
        self,
        features_df: Optional[pd.DataFrame],
        id_name_const: str,
        id_index: pd.Index,
        feature_names: List[str],
        num_entities: int,
        features_dim: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if features_dim > 0:
            features_data_tensor = torch.zeros((num_entities, features_dim), dtype=torch.float)
            if features_df is not None and not features_df.empty:
                aligned_features_df = features_df.set_index(id_name_const).reindex(id_index)
                feature_values = aligned_features_df[feature_names].fillna(0).values
                features_data_tensor[: len(id_index), :] = torch.tensor(feature_values, dtype=torch.float)
            return features_data_tensor.to(device)
        return None

    def _initialize_id_indices_and_feature_metadata(
        self,
        interactions_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame],
        items_df: Optional[pd.DataFrame],
    ) -> None:
        """
        Initializes ID indices, infers feature names, dimensions, and creates feature tensors.
        This method should only be called during fitting.
        """
        (
            self.user_feature_names,
            self.item_feature_names,
            self.interaction_feature_names,
            self.user_features_dim,
            self.item_features_dim,
            self.interaction_features_dim,
        ) = self._get_feature_names_and_dims(interactions_df, users_df, items_df)

        (
            self.user_id_index,
            self.item_id_index,
            self.unknown_user_idx,
            self.unknown_item_idx,
            self.num_users,
            self.num_items,
        ) = self._get_id_indices_and_counts(interactions_df)

        self.user_features_tensor = self._create_features_tensor(
            users_df,
            USER_ID_NAME,
            self.user_id_index,
            self.user_feature_names,
            self.num_users,
            self.user_features_dim,
            self.device,
        )
        self.item_features_tensor = self._create_features_tensor(
            items_df,
            ITEM_ID_NAME,
            self.item_id_index,
            self.item_feature_names,
            self.num_items,
            self.item_features_dim,
            self.device,
        )

    def _create_interaction_tensors(
        self, interactions_df: pd.DataFrame, is_fitting: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Maps interaction IDs to indices and creates tensors for user IDs, item IDs,
        interaction features, and labels.
        """
        if (
            self.user_id_index is None
            or self.item_id_index is None
            or self.unknown_user_idx is None
            or self.unknown_item_idx is None
        ):
            raise RuntimeError("ID indices not initialized. Call _initialize_id_indices_and_feature_metadata first.")

        user_indices_mapped = self.user_id_index.get_indexer(interactions_df[USER_ID_NAME])
        user_indices_mapped[user_indices_mapped == -1] = self.unknown_user_idx

        item_indices_mapped = self.item_id_index.get_indexer(interactions_df[ITEM_ID_NAME])
        item_indices_mapped[item_indices_mapped == -1] = self.unknown_item_idx

        user_id_indices_tensor = torch.tensor(user_indices_mapped, dtype=torch.long)
        item_id_indices_tensor = torch.tensor(item_indices_mapped, dtype=torch.long)

        if self.interaction_feature_names:
            interaction_features_np = (
                interactions_df[self.interaction_feature_names].fillna(0).values.astype(np.float32)
            )
        else:
            interaction_features_np = np.empty((len(interactions_df), 0), dtype=np.float32)
        interaction_features_tensor = torch.tensor(interaction_features_np, dtype=torch.float)

        labels_tensor: Optional[torch.Tensor] = None
        if is_fitting:  # Assumes LABEL_NAME is present if fitting
            if LABEL_NAME not in interactions_df.columns:
                raise ValueError(f"'{LABEL_NAME}' column missing from interactions_df during fitting.")
            labels_tensor = torch.tensor(interactions_df[LABEL_NAME].values, dtype=torch.float).unsqueeze(1)
        elif LABEL_NAME in interactions_df.columns:  # For prediction, if labels happen to be there
            labels_tensor = torch.tensor(interactions_df[LABEL_NAME].values, dtype=torch.float).unsqueeze(1)

        return user_id_indices_tensor, item_id_indices_tensor, interaction_features_tensor, labels_tensor

    def _prepare_data_and_metadata(
        self,
        users_df: Optional[pd.DataFrame],
        items_df: Optional[pd.DataFrame],
        interactions_df: pd.DataFrame,
        is_fitting: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if is_fitting:
            self._initialize_id_indices_and_feature_metadata(interactions_df, users_df, items_df)

        return self._create_interaction_tensors(interactions_df, is_fitting)

    @abc.abstractmethod
    def _build_pytorch_model(self) -> BasePyTorchEmbeddingModule:
        pass

    def fit_embedding_model(
        self,
        users: Optional[pd.DataFrame],
        items: Optional[pd.DataFrame],
        interactions: pd.DataFrame,
        valid_users: Optional[pd.DataFrame] = None,
        valid_interactions: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Args:
            users: Optional user feature DataFrame (USER_ID + feature columns).
            items: Optional item feature DataFrame (ITEM_ID + feature columns).
            interactions: Training interactions (USER_ID, ITEM_ID, LABEL + any context features).
            valid_users: Optional user feature DataFrame for validation. If provided, user IDs
                should be a subset of those in `valid_interactions`; IDs present in
                `valid_interactions` but absent here will fall back to OOV embeddings.
            valid_interactions: Optional validation interactions. Required when
                `early_stopping_patience` is set.
        """
        self._user_data_truncated = False
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.random_state)

        if self.early_stopping_patience is not None and valid_interactions is None:
            raise ValueError("early_stopping_patience requires valid_interactions to be provided.")

        if valid_users is not None or valid_interactions is not None:
            if valid_interactions is not None:
                for required_col in (USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME):
                    if required_col not in valid_interactions.columns:
                        raise ValueError(f"valid_interactions is missing required column '{required_col}'.")
            if valid_users is not None:
                if USER_ID_NAME not in valid_users.columns:
                    raise ValueError(f"valid_users is missing required column '{USER_ID_NAME}'.")

        user_idx_t, item_idx_t, interaction_feats_t, labels_t = self._prepare_data_and_metadata(
            users, items, interactions, is_fitting=True
        )
        if labels_t is None:
            raise ValueError(f"interactions must contain a '{LABEL_NAME}' column.")

        if valid_interactions is not None and self.interaction_feature_names:
            missing_context_cols = set(self.interaction_feature_names) - set(valid_interactions.columns)
            if missing_context_cols:
                raise ValueError(
                    f"valid_interactions is missing context feature columns present in training: {missing_context_cols}"
                )

        # Prepare validation tensors after fitting metadata (requires user_id_index / item_id_index)
        valid_user_idx_t: Optional[torch.Tensor] = None
        valid_item_idx_t: Optional[torch.Tensor] = None
        valid_interaction_feats_t: Optional[torch.Tensor] = None
        valid_labels_t: Optional[torch.Tensor] = None
        valid_user_features_t: Optional[torch.Tensor] = None
        if valid_interactions is not None:
            (
                valid_user_idx_t,
                valid_item_idx_t,
                valid_interaction_feats_t,
                valid_labels_t,
            ) = self._prepare_data_and_metadata(None, None, valid_interactions, is_fitting=False)
            if valid_labels_t is None:
                raise ValueError(f"valid_interactions must contain a '{LABEL_NAME}' column.")
            if valid_users is not None:
                valid_user_features_t = self._create_features_tensor(
                    valid_users,
                    USER_ID_NAME,
                    self.user_id_index,
                    self.user_feature_names,
                    self.num_users,
                    self.user_features_dim,
                    self.device,
                )
            # valid_users=None is safe: validation users are a subset of training users,
            # so self.user_features_tensor (built during training) already covers them.
            # Passing user_features=None to the model's forward falls back to that tensor.
            # This is intentionally different from the XGBoost path, which raises if
            # valid_users is absent when training used user features — XGBoost requires
            # column parity in X_valid, whereas embedding models look up features by index.
        elif valid_users is not None:
            logger.warning("valid_users was provided without valid_interactions — validation data will be ignored.")

        if valid_interactions is not None and self.user_id_index is not None:
            unknown_val_interaction_users = set(valid_interactions[USER_ID_NAME].unique()) - set(self.user_id_index)
            if unknown_val_interaction_users:
                logger.warning(
                    f"{len(unknown_val_interaction_users)} user ID(s) in valid_interactions were not seen "
                    f"during training and will use the OOV embedding: {unknown_val_interaction_users}"
                )

        if valid_users is not None and valid_interactions is not None and self.user_id_index is not None:
            unknown_val_users = set(valid_users[USER_ID_NAME].unique()) - set(self.user_id_index)
            if unknown_val_users:
                logger.warning(
                    f"{len(unknown_val_users)} user ID(s) in valid_users were not seen during training "
                    f"and their features will be zeroed out: {unknown_val_users}"
                )

        self.model = self._build_pytorch_model()
        self.model.to(self.device)

        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.to(self.device)

        optimizer = self.optimizer_cls(  # type: ignore[call-arg]
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        self._training_loop(
            model=self.model,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            user_idx_t=user_idx_t,
            item_idx_t=item_idx_t,
            interaction_feats_t=interaction_feats_t,
            labels_t=labels_t,
            valid_user_idx_t=valid_user_idx_t,
            valid_item_idx_t=valid_item_idx_t,
            valid_interaction_feats_t=valid_interaction_feats_t,
            valid_labels_t=valid_labels_t,
            valid_user_features_t=valid_user_features_t,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            random_state=self.random_state,
            verbose=self.verbose,
            early_stopping_patience=self.early_stopping_patience,
            restore_best_weights=self.restore_best_weights,
        )

    def _training_loop(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        user_idx_t: torch.Tensor,
        item_idx_t: torch.Tensor,
        interaction_feats_t: torch.Tensor,
        labels_t: torch.Tensor,
        epochs: int,
        batch_size: int,
        device: torch.device,
        random_state: Optional[int],
        verbose: int,
        valid_user_idx_t: Optional[torch.Tensor] = None,
        valid_item_idx_t: Optional[torch.Tensor] = None,
        valid_interaction_feats_t: Optional[torch.Tensor] = None,
        valid_labels_t: Optional[torch.Tensor] = None,
        valid_user_features_t: Optional[torch.Tensor] = None,
        early_stopping_patience: Optional[int] = None,
        restore_best_weights: bool = True,
    ) -> None:
        """Run the mini-batch gradient-descent training loop.

        Shuffles training data each epoch, iterates over mini-batches, and
        optionally evaluates validation loss for early stopping.

        Args:
            model: The PyTorch module to train.  Must implement the embedding
                model forward signature
                ``(user_indices, item_indices, interaction_features,
                user_embeddings, user_features) -> Tensor``.
            optimizer: Configured optimizer (already wrapping ``model.parameters()``).
            loss_fn: Instantiated loss module (e.g. ``nn.BCEWithLogitsLoss``).
            user_idx_t: Integer tensor of shape ``(N,)`` with user lookup indices.
            item_idx_t: Integer tensor of shape ``(N,)`` with item lookup indices.
            interaction_feats_t: Float tensor of shape ``(N, context_dim)``
                containing per-interaction context features.
            labels_t: Float tensor of shape ``(N, 1)`` with training targets.
            epochs: Number of epochs to train.
            batch_size: Mini-batch size.
            device: Device to move each batch to before the forward pass.
            random_state: RNG seed base; ``None`` means no seeding per epoch.
            verbose: Logging verbosity (``0`` = silent, ``1`` = epoch loss).
            valid_user_idx_t: Validation user indices, or ``None``.
            valid_item_idx_t: Validation item indices, or ``None``.
            valid_interaction_feats_t: Validation context features, or ``None``.
            valid_labels_t: Validation labels, or ``None``.
            valid_user_features_t: Pre-computed user feature tensor for
                validation real-time inference, or ``None``.
            early_stopping_patience: Number of epochs without validation loss
                improvement before halting.  ``None`` disables early stopping.
            restore_best_weights: If ``True``, load the best-checkpoint weights
                when training ends (either via early stopping or epoch limit).
        """
        num_samples = user_idx_t.shape[0]
        sample_indices = np.arange(num_samples)
        has_validation = valid_user_idx_t is not None and valid_labels_t is not None

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state: Optional[Dict[str, Any]] = None
        _warned_no_val_batches = False

        model.train()
        for epoch in range(epochs):
            if random_state is not None:  # Ensure different shuffle per epoch if random_state is set
                np.random.seed(random_state + epoch)
            np.random.shuffle(sample_indices)

            epoch_loss = 0.0
            num_batches = 0
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = sample_indices[start_idx:end_idx]

                batch_user_idx = user_idx_t[batch_indices].to(device)
                batch_item_idx = item_idx_t[batch_indices].to(device)
                batch_interaction_feats = interaction_feats_t[batch_indices].to(device)
                batch_labels = labels_t[batch_indices].to(device)

                optimizer.zero_grad()

                # The model's forward signature is:
                # user_indices, item_indices, interaction_features,
                # user_embeddings (None for training), user_features (None for training)
                # For training, user_embeddings and user_features are None, model uses internal layers/tensors.
                predictions = model(
                    user_indices=batch_user_idx,
                    item_indices=batch_item_idx,
                    interaction_features=batch_interaction_feats,
                    user_embeddings=None,  # During training, model uses its own embedding layer
                    user_features=None,  # During training, model uses self.user_features_tensor
                )

                loss = loss_fn(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            if has_validation:
                model.eval()
                val_loss = 0.0
                val_batches = 0
                num_val_samples = valid_user_idx_t.shape[0]  # type: ignore[union-attr]
                with torch.no_grad():
                    for start_idx in range(0, num_val_samples, batch_size):
                        end_idx = min(start_idx + batch_size, num_val_samples)
                        val_preds = model(
                            user_indices=valid_user_idx_t[start_idx:end_idx].to(device),  # type: ignore[index]
                            item_indices=valid_item_idx_t[start_idx:end_idx].to(device),  # type: ignore[index]
                            interaction_features=valid_interaction_feats_t[start_idx:end_idx].to(device),  # type: ignore[index]
                            user_embeddings=None,
                            user_features=valid_user_features_t,  # None falls back to model's internal tensor
                        )
                        val_loss += loss_fn(val_preds, valid_labels_t[start_idx:end_idx].to(device)).item()  # type: ignore[index]
                        val_batches += 1
                model.train()
                if val_batches == 0:
                    if not _warned_no_val_batches:
                        logger.warning(
                            "No valid validation batches found — skipping early stopping check. "
                            "This warning will not repeat for subsequent epochs."
                        )
                        _warned_no_val_batches = True
                    if verbose > 0:
                        logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Val Loss: N/A")
                    continue
                avg_val_loss = val_loss / val_batches
                if verbose > 0:
                    logger.info(
                        f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                    )

                if early_stopping_patience is not None:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_without_improvement = 0
                        if restore_best_weights:
                            best_state = copy.deepcopy(model.state_dict())
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= early_stopping_patience:
                            logger.info(
                                f"Early stopping at epoch {epoch + 1}/{epochs} — val loss did not improve "
                                f"for {early_stopping_patience} epoch(s). Best val loss: {best_val_loss:.4f}"
                            )
                            if restore_best_weights:
                                if best_state is not None:
                                    model.load_state_dict(best_state)
                                else:
                                    logger.warning(
                                        "restore_best_weights=True but no improvement was ever recorded "
                                        "(val loss never decreased from its initial value). "
                                        "Model weights are from the final epoch."
                                    )
                            break
            elif verbose > 0:
                logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

    def _prediction_loop(
        self,
        model: nn.Module,
        user_idx_t: torch.Tensor,  # Maps interactions to rows in user_embeddings / user_features
        item_idx_t: torch.Tensor,  # Maps interactions to rows in self.item_id_embeddings
        interaction_feats_t: torch.Tensor,
        user_embeddings: Optional[torch.Tensor],  # (n_unique_users + 1_placeholder, emb_dim)
        user_features: Optional[torch.Tensor],  # (n_unique_users + 1_placeholder, user_feat_dim)
        batch_size: int,
        device: torch.device,
    ) -> List[np.ndarray]:
        all_predictions_list = []
        num_samples = user_idx_t.shape[0]
        model.eval()  # Ensure model is in eval mode

        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)

                batch_user_indices = user_idx_t[start_idx:end_idx].to(device)
                batch_item_indices = item_idx_t[start_idx:end_idx].to(device)
                batch_interaction_features = interaction_feats_t[start_idx:end_idx].to(device)

                predictions_logits = model(
                    user_indices=batch_user_indices,
                    item_indices=batch_item_indices,
                    interaction_features=batch_interaction_features,
                    user_embeddings=user_embeddings,
                    user_features=user_features,
                )
                if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
                    predictions_proba = torch.sigmoid(predictions_logits)
                else:
                    predictions_proba = predictions_logits
                all_predictions_list.append(predictions_proba.cpu().numpy())
        return all_predictions_list

    def _prepare_inference_tensors(
        self, interactions: pd.DataFrame, users: Optional[pd.DataFrame]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepares tensors for inference (SIMPLIFIED VERSION).

        Assumes input DataFrames are pre-validated and numeric columns are already processed.
        If `users` DataFrame is provided (real-time mode):
          - Uses embeddings and features from this DataFrame.
          - User IDs in `users` are assumed unique.
        If `users` is None (offline prediction mode):
          - Model's internal user embeddings/features are implied (tensors returned as None).
        """
        if self.item_id_index is None or self.unknown_item_idx is None:
            raise RuntimeError("Estimator has not been fitted or item metadata is missing.")

        user_embeddings_tensor: Optional[torch.Tensor] = None
        user_features_tensor: Optional[torch.Tensor] = None

        if users is not None:
            if USER_ID_NAME not in users.columns or USER_EMBEDDING_NAME not in users.columns:
                raise ValueError(
                    f"`users` DataFrame must contain '{USER_ID_NAME}' and '{USER_EMBEDDING_NAME}' columns."
                )

            # User Embeddings
            stacked_user_embeddings = np.vstack(users[USER_EMBEDDING_NAME].values)
            emb_dim = stacked_user_embeddings.shape[1]
            placeholder_embedding = np.zeros((1, emb_dim), dtype=stacked_user_embeddings.dtype)
            user_embeddings_np = np.vstack([stacked_user_embeddings, placeholder_embedding])
            user_embeddings_tensor = torch.tensor(user_embeddings_np, dtype=torch.float).to(self.device)

            user_id_to_row_idx = pd.Series(data=np.arange(len(users)), index=users[USER_ID_NAME])
            placeholder_idx_for_interactions = len(users)

            # User Features
            input_user_feature_names = [f_name for f_name in self.user_feature_names if f_name in users.columns]
            if input_user_feature_names:
                user_features_data = users[input_user_feature_names].values
                current_user_features_dim = user_features_data.shape[1]
                placeholder_user_features = np.zeros((1, current_user_features_dim), dtype=user_features_data.dtype)
                final_user_features_np = np.vstack([user_features_data, placeholder_user_features])
                user_features_tensor = torch.tensor(final_user_features_np, dtype=torch.float).to(self.device)
            elif self.user_features_dim > 0:
                num_rows_for_features = len(users) + 1
                all_zero_features_np = np.zeros((num_rows_for_features, self.user_features_dim), dtype=np.float32)
                user_features_tensor = torch.tensor(all_zero_features_np, dtype=torch.float).to(self.device)

            interaction_user_indices_mapped = (
                user_id_to_row_idx.reindex(interactions[USER_ID_NAME])
                .fillna(placeholder_idx_for_interactions)
                .astype(np.int64)
                .values
            )
            user_indices_tensor = torch.tensor(interaction_user_indices_mapped, dtype=torch.long).to(self.device)
        else:  # Offline mode
            user_indices_mapped = self.user_id_index.get_indexer(interactions[USER_ID_NAME])
            user_indices_mapped[user_indices_mapped == -1] = self.unknown_user_idx
            user_indices_tensor = torch.tensor(user_indices_mapped, dtype=torch.long).to(self.device)

        # Common: Item Indices and Interaction Features
        item_indices_mapped = self.item_id_index.get_indexer(interactions[ITEM_ID_NAME])
        item_indices_mapped[item_indices_mapped == -1] = self.unknown_item_idx
        item_indices_tensor = torch.tensor(item_indices_mapped, dtype=torch.long).to(self.device)

        if self.interaction_feature_names:
            missing_cols = [c for c in self.interaction_feature_names if c not in interactions.columns]
            if missing_cols:
                raise ValueError(
                    f"Context feature(s) {missing_cols} were used during training but are missing "
                    "from the inference interactions DataFrame."
                )
            interaction_features_np = interactions[self.interaction_feature_names].values.astype(np.float32)
        else:
            interaction_features_np = np.empty((len(interactions), 0), dtype=np.float32)
        interaction_features_tensor = torch.tensor(interaction_features_np, dtype=torch.float).to(self.device)

        return (
            user_indices_tensor,
            item_indices_tensor,
            interaction_features_tensor,
            user_embeddings_tensor,
            user_features_tensor,
        )

    def predict_proba_with_embeddings(
        self,
        interactions: pd.DataFrame,
        users: Optional[pd.DataFrame] = None,
    ) -> NDArray:
        """
        Predicts probabilities for given interactions. Operates in two modes:

        1. Real-time Inference Mode (users DataFrame provided):
           If `users` DataFrame is provided, it MUST contain `USER_ID_NAME` and
           `USER_EMBEDDING_NAME` columns. Pre-computed user embeddings from this
           DataFrame are used. Optionally, if user features are also present in this
           `users` DataFrame and the model was trained with user features, these
           will be used. This mode is for scenarios where user embeddings are
           managed externally (e.g., an embedding store).

        2. Batch Prediction Mode (users is None):
           If `users` is `None`, the model uses its internally learned user embeddings
           (from `self.model.user_id_embedding`) and stored user features
           (`self.user_features_tensor`) derived during training. This is the
           typical mode for batch predictions or when not using an external embedding store.

        Args:
            interactions: DataFrame containing interaction data. Must include
                          `USER_ID_NAME`, `ITEM_ID_NAME`, and any context features
                          the model was trained on.
            users: Optional DataFrame.
                   - If provided (Real-time Mode): Must contain `USER_ID_NAME` and
                     `USER_EMBEDDING_NAME` (NumPy arrays). Can also contain user
                     feature columns if the model uses them.
                   - If `None` (Standard Mode): The model uses its internal, learned
                     user embeddings and features.

        Returns:
            NDArray: A NumPy array of predicted probabilities, with shape (n_interactions, 1).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit_embedding_model() first.")

        if interactions is None:
            raise ValueError("interactions cannot be None.")
        for col in (USER_ID_NAME, ITEM_ID_NAME):
            if col not in interactions.columns:
                raise ValueError(
                    f"interactions is missing required column '{col}'. Got columns: {list(interactions.columns)}"
                )
        if self.interaction_feature_names:
            non_numeric = [
                c
                for c in self.interaction_feature_names
                if c in interactions.columns and not pd.api.types.is_numeric_dtype(interactions[c])
            ]
            if non_numeric:
                raise ValueError(
                    f"Interaction feature column(s) {non_numeric} must be numeric but contain non-numeric data."
                )

        (
            user_indices_t,
            item_indices_t,
            interaction_features_t,
            user_embeddings_t,
            user_features_t,
        ) = self._prepare_inference_tensors(interactions, users)

        all_predictions_list = self._prediction_loop(
            self.model,
            user_indices_t,
            item_indices_t,
            interaction_features_t,
            user_embeddings_t,
            user_features_t,
            self.batch_size,
            self.device,
        )

        if not all_predictions_list:
            return np.array([])

        return np.concatenate(all_predictions_list, axis=0)

    def get_config(self) -> Dict[str, Any]:
        config = {
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
            "early_stopping_patience": self.early_stopping_patience,
            "restore_best_weights": self.restore_best_weights,
        }
        return config

    # _fit_model from BaseEstimator is intentionally not overridden here,
    # as BaseEmbeddingEstimator's _fit_model raises NotImplementedError.
    # This class uses fit_embedding_model.

    def get_user_embeddings(self) -> pd.DataFrame:
        """
        Extracts user embeddings from the trained model.

        Returns:
            pd.DataFrame: DataFrame with USER_ID_NAME and USER_EMBEDDING_NAME columns.
        """
        if self.model is None or not hasattr(self.model, "user_id_embedding"):
            raise RuntimeError("Model is not trained or does not have user_id_embedding layer.")
        if self._user_data_truncated:
            raise RuntimeError("User data has been truncated. Cannot extract full embeddings.")
        if self.user_id_index is None or self.unknown_user_idx is None:  # Added check for unknown_user_idx
            raise RuntimeError("User ID index or unknown_user_idx not available. Fit the model first.")

        user_embedding_weights = self.model.user_id_embedding.weight.data.cpu().numpy()

        embeddings_list = []
        # Iterate up to unknown_user_idx, which is the first placeholder/unknown ID.
        # self.user_id_index contains the actual IDs corresponding to these learned embeddings.
        for i in range(self.unknown_user_idx):  # Iterate only over known, learned users
            user_id = self.user_id_index[i]
            embeddings_list.append({USER_ID_NAME: user_id, USER_EMBEDDING_NAME: user_embedding_weights[i]})

        if not embeddings_list:
            return pd.DataFrame(columns=[USER_ID_NAME, USER_EMBEDDING_NAME])

        return pd.DataFrame(embeddings_list)

    def get_item_embeddings(self) -> pd.DataFrame:
        """
        Extracts item embeddings from the trained model's item embedding layer.

        Returns:
            pd.DataFrame: DataFrame with ITEM_ID_NAME and ITEM_EMBEDDING_NAME columns.
        """
        if self.model is None or not hasattr(self.model, "item_id_embedding"):
            raise RuntimeError("Model is not trained or does not have item_id_embedding layer.")
        if self.item_id_index is None or self.unknown_item_idx is None:
            raise RuntimeError("Item ID index not available. Fit the model first.")

        item_embedding_weights = self.model.item_id_embedding.weight.data.cpu().numpy()

        embeddings_list = []
        for i in range(self.unknown_item_idx):
            item_id = self.item_id_index[i]
            embeddings_list.append({ITEM_ID_NAME: item_id, ITEM_EMBEDDING_NAME: item_embedding_weights[i]})

        if not embeddings_list:
            return pd.DataFrame(columns=[ITEM_ID_NAME, ITEM_EMBEDDING_NAME])

        return pd.DataFrame(embeddings_list)

    def truncate_user_data(self) -> None:
        """
        Modifies the estimator's state for compactness after user embeddings
        have been extracted. The learned placeholder embedding is preserved by
        directly replacing the model's user_id_embedding layer.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained.")

        if self._user_data_truncated:
            logger.info("User data is already truncated.")
            return

        # 1. Extract the learned placeholder embedding's weights
        # It's at self.unknown_user_idx in the original embedding matrix
        with torch.no_grad():
            learned_placeholder_weights = self.model.user_id_embedding.weight[self.unknown_user_idx].clone().detach()

        # 2. Create a new, small embedding layer for the placeholder
        # The embedding_dim should be taken from the model's attribute (e.g., self.model.user_embedding_dim)
        new_user_embedding_layer = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.model.user_embedding_dim,  # Only for the placeholder
        ).to(self.device)

        # Assign the learned placeholder weights to this new layer's single entry
        with torch.no_grad():
            new_user_embedding_layer.weight.data[0] = learned_placeholder_weights

        # 3. Replace the model's user_id_embedding layer directly
        self.model.user_id_embedding = new_user_embedding_layer
        self.model.num_users = 1
        self.num_users = 1
        self.unknown_user_idx = 0
        self.user_id_index = pd.Index([""])  # placeholder
        self.user_features_tensor = None

        self._user_data_truncated = True

    def __getstate__(self) -> Dict[str, Any]:
        """
        Custom state for pickling. Stores the model's state_dict separately.
        """
        state = self.__dict__.copy()
        if self.model is not None:
            state["_model_state_dict"] = self.model.state_dict()

        if "model" in state:
            del state["model"]

        state.pop("loss_fn", None)
        state.pop("optimizer_cls", None)

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Custom state for unpickling. Rebuilds the model and loads its state_dict.
        """
        self.__dict__.update(state)

        self.model = self._build_pytorch_model()
        self.model.to(self.device)

        if "_model_state_dict" in state:
            # If data was truncated, self.model was built with a user_id_embedding of size (1, dim).
            # The _model_state_dict also contains user_id_embedding.weight of size (1, dim)
            # with the learned placeholder values. So, strict=True should work.
            self.model.load_state_dict(state["_model_state_dict"], strict=True)

        self.loss_fn = self._get_loss_fn(self.loss_fn_name)
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.to(self.device)

        self.optimizer_cls = self._get_optimizer_cls(self.optimizer_name)
