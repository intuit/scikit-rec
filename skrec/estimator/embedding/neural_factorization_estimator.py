from __future__ import annotations

from typing import List, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from skrec.estimator.embedding.base_pytorch_estimator import (
    BasePyTorchEmbeddingEstimator,
    BasePyTorchEmbeddingModule,
)
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class _NeuralFactorizationNet(BasePyTorchEmbeddingModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        user_features_dim: int,
        item_features_dim: int,
        interaction_features_dim: int,
        mlp_hidden_dim1: Optional[int],
        mlp_hidden_dim2: Optional[int],
        user_features_tensor: Optional[torch.Tensor],
        item_features_tensor: Optional[torch.Tensor],
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            user_embedding_dim=embedding_dim,
            item_embedding_dim=embedding_dim,
            user_features_dim=user_features_dim,
            item_features_dim=item_features_dim,
            user_features_tensor=user_features_tensor,
            item_features_tensor=item_features_tensor,
        )
        # Common embeddings (user_id_embedding, item_id_embedding) and feature tensors
        # (user_features_tensor, item_features_tensor) are initialized in BasePyTorchEmbeddingModule.

        # Feature projection layers specific to this architecture
        self.user_feature_projection: Optional[nn.Linear] = None
        if user_features_dim > 0:
            self.user_feature_projection = nn.Linear(user_features_dim, embedding_dim)

        self.item_feature_projection: Optional[nn.Linear] = None
        if item_features_dim > 0:
            self.item_feature_projection = nn.Linear(item_features_dim, embedding_dim)

        # Context embedding projection
        self.context_projection: Optional[nn.Linear] = None
        if interaction_features_dim > 0:
            self.context_projection = nn.Linear(interaction_features_dim, embedding_dim)

        mlp_input_dim = 6 * embedding_dim
        mlp_layers: List[nn.Module] = []
        current_dim = mlp_input_dim

        if mlp_hidden_dim1 is not None:
            mlp_layers.append(nn.Linear(current_dim, mlp_hidden_dim1))
            mlp_layers.append(nn.ReLU())
            current_dim = mlp_hidden_dim1
            if mlp_hidden_dim2 is not None:
                mlp_layers.append(nn.Linear(current_dim, mlp_hidden_dim2))
                mlp_layers.append(nn.ReLU())
                current_dim = mlp_hidden_dim2

        mlp_layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        interaction_features: torch.Tensor,
        user_embeddings: Optional[torch.Tensor] = None,  # Provided at inference
        user_features: Optional[torch.Tensor] = None,  # Provided at inference
    ) -> torch.Tensor:
        # Get user embeddings and features from input or trained
        batch_user_emb = self._get_user_embeddings(user_indices, user_embeddings)
        batch_user_features = self._get_user_features(user_indices, user_features)

        # Fuse user features with user embeddings if projection layer exists
        if self.user_feature_projection is not None and batch_user_features is not None:
            projected_user_features = self.user_feature_projection(batch_user_features)
            batch_user_emb = batch_user_emb + projected_user_features

        # Get item embeddings (always from model's layer)
        batch_item_emb = self.item_id_embedding(item_indices)
        # Get item features (always from model's full tensor if available)
        batch_item_features: Optional[torch.Tensor] = None
        if self.item_features_tensor is not None and self.item_features_tensor.shape[1] > 0:
            batch_item_features = self.item_features_tensor[item_indices]

        # Fuse item features with item embeddings if projection layer exists
        if self.item_feature_projection is not None and batch_item_features is not None:
            projected_item_features = self.item_feature_projection(batch_item_features)
            batch_item_emb = batch_item_emb + projected_item_features

        # Context embedding
        if self.context_projection is not None and interaction_features.shape[1] > 0:
            batch_context_emb = self.context_projection(interaction_features)
        else:
            # Ensure context_emb has the correct dimension even if no interaction_features
            batch_context_emb = torch.zeros(user_indices.size(0), self.user_embedding_dim, device=user_indices.device)

        # Pairwise interactions
        ui_interaction = batch_user_emb * batch_item_emb
        uc_interaction = batch_user_emb * batch_context_emb
        ic_interaction = batch_item_emb * batch_context_emb

        # Concatenate all for MLP input
        mlp_input = torch.cat(
            [batch_user_emb, batch_item_emb, batch_context_emb, ui_interaction, uc_interaction, ic_interaction], dim=1
        )

        score = self.mlp(mlp_input)
        return score


class NeuralFactorizationEstimator(BasePyTorchEmbeddingEstimator):
    """
    Neural Factorization Model with Contextual Interactions (NFM-CI).

    This model generates embeddings for users, items, and interaction context.
    User and item features, if provided, are projected and added to their
    respective ID embeddings.
    It explicitly forms element-wise products (Hadamard products) of these
    (potentially feature-enhanced) embeddings to capture second-order interactions.
    These interaction terms, along with the original embeddings, are fed into an
    MLP to learn higher-order interactions and predict a score.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        mlp_hidden_dim1: Optional[int] = None,
        mlp_hidden_dim2: Optional[int] = None,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        optimizer_name: str = "adam",
        loss_fn_name: str = "bce",
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        early_stopping_patience: Optional[int] = None,
        restore_best_weights: bool = True,
    ):
        super().__init__(
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            loss_fn_name=loss_fn_name,
            device=device,
            random_state=random_state,
            verbose=verbose,
            early_stopping_patience=early_stopping_patience,
            restore_best_weights=restore_best_weights,
        )
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim1 = mlp_hidden_dim1 if mlp_hidden_dim1 is not None else embedding_dim * 3
        self.mlp_hidden_dim2 = (
            mlp_hidden_dim2
            if mlp_hidden_dim2 is not None
            else embedding_dim + (embedding_dim // 2)
            if embedding_dim > 1
            else 1
        )

    def _build_pytorch_model(self) -> BasePyTorchEmbeddingModule:
        return _NeuralFactorizationNet(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            user_features_dim=self.user_features_dim,
            item_features_dim=self.item_features_dim,
            interaction_features_dim=self.interaction_features_dim,
            mlp_hidden_dim1=self.mlp_hidden_dim1,
            mlp_hidden_dim2=self.mlp_hidden_dim2,
            user_features_tensor=self.user_features_tensor,
            item_features_tensor=self.item_features_tensor,
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "mlp_hidden_dim1": self.mlp_hidden_dim1,
                "mlp_hidden_dim2": self.mlp_hidden_dim2,
            }
        )
        return config
