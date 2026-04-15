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


_nn_module = nn.Module if nn is not None else object


class _CrossLayer(_nn_module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        x0: initial input to the cross network (batch_size, input_dim)
        xl: input from the previous layer (batch_size, input_dim)
        """
        xl_w_b = self.linear(xl)  # W_l * x_l + b_l
        return x0 * xl_w_b + xl  # x0 .* (W_l * x_l + b_l) + x_l


class _DeepCrossNetworkNet(BasePyTorchEmbeddingModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        user_features_dim: int,
        item_features_dim: int,
        interaction_features_dim: int,
        num_cross_layers: int,
        deep_hidden_dim1: Optional[int],
        deep_hidden_dim2: Optional[int],
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

        # Feature projection layers specific to this architecture
        self.user_feature_projection: Optional[nn.Linear] = None
        if user_features_dim > 0:
            self.user_feature_projection = nn.Linear(user_features_dim, embedding_dim)

        self.item_feature_projection: Optional[nn.Linear] = None
        if item_features_dim > 0:
            self.item_feature_projection = nn.Linear(item_features_dim, embedding_dim)

        self.context_projection: Optional[nn.Linear] = None
        if interaction_features_dim > 0:
            self.context_projection = nn.Linear(interaction_features_dim, embedding_dim)

        # Initial concatenation dimension for x0 (user_emb + item_emb + context_emb)
        self.x0_dim = 3 * embedding_dim

        # Cross Network
        self.cross_network = nn.ModuleList([_CrossLayer(self.x0_dim) for _ in range(num_cross_layers)])

        # Deep Network
        deep_layers: List[nn.Module] = []
        current_deep_dim = self.x0_dim
        if deep_hidden_dim1 is not None:
            deep_layers.append(nn.Linear(current_deep_dim, deep_hidden_dim1))
            deep_layers.append(nn.ReLU())
            current_deep_dim = deep_hidden_dim1
            if deep_hidden_dim2 is not None:
                deep_layers.append(nn.Linear(current_deep_dim, deep_hidden_dim2))
                deep_layers.append(nn.ReLU())
                current_deep_dim = deep_hidden_dim2
        self.deep_network = nn.Sequential(*deep_layers)

        # Combination Layer
        # Output of cross network is x0_dim, output of deep network is current_deep_dim
        combination_input_dim = self.x0_dim + current_deep_dim
        self.combination_layer = nn.Linear(combination_input_dim, 1)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        interaction_features: torch.Tensor,
        user_embeddings: Optional[torch.Tensor] = None,
        user_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # User processing
        batch_user_emb_raw = self._get_user_embeddings(user_indices, user_embeddings)
        batch_user_features_raw = self._get_user_features(user_indices, user_features)

        final_user_emb = batch_user_emb_raw
        if self.user_feature_projection is not None and batch_user_features_raw is not None:
            final_user_emb = final_user_emb + self.user_feature_projection(batch_user_features_raw)

        # Item processing
        batch_item_emb_raw = self.item_id_embedding(item_indices)  # Item embeddings always from model
        batch_item_features_raw: Optional[torch.Tensor] = None
        if self.item_features_tensor is not None and self.item_features_tensor.shape[1] > 0:
            batch_item_features_raw = self.item_features_tensor[item_indices]

        final_item_emb = batch_item_emb_raw
        if self.item_feature_projection is not None and batch_item_features_raw is not None:
            final_item_emb = final_item_emb + self.item_feature_projection(batch_item_features_raw)

        # Context processing
        context_emb: torch.Tensor
        if self.context_projection is not None and interaction_features.shape[1] > 0:
            context_emb = self.context_projection(interaction_features)
        else:
            context_emb = torch.zeros(user_indices.size(0), self.user_embedding_dim, device=user_indices.device)

        x0 = torch.cat([final_user_emb, final_item_emb, context_emb], dim=1)

        # Cross Network
        xl = x0
        for cross_layer in self.cross_network:
            xl = cross_layer(x0, xl)
        x_cross_out = xl

        # Deep Network
        x_deep_out = self.deep_network(x0)

        # Combination
        combined_output = torch.cat([x_cross_out, x_deep_out], dim=1)
        score = self.combination_layer(combined_output)
        return score


class DeepCrossNetworkEstimator(BasePyTorchEmbeddingEstimator):
    """
    Deep & Cross Network (DCN) for Contextual Embedding Interactions.

    DCN learns explicit feature interactions of bounded polynomial degree
    efficiently using a "cross network," while a parallel "deep network"
    learns implicit, complex interactions. This version applies DCN to
    interactions between pre-derived user, item, and context embeddings.
    User/item features are projected and added to their ID embeddings.
    Interaction features are projected to form the context embedding.

    - User, item, and context embeddings form the base input `x0`.
    - Cross Network: `x_{l+1} = x0 * (xl @ W_l + b_l) + xl`
    - Deep Network: Standard MLP.
    - Outputs of cross and deep networks are concatenated and passed to a
      final linear layer for the score.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        num_cross_layers: int = 2,
        deep_hidden_dim1: Optional[int] = None,  # e.g., embedding_dim * 3 (for x0_dim)
        deep_hidden_dim2: Optional[int] = None,  # e.g., embedding_dim * 1.5 (for x0_dim)
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
        self.num_cross_layers = num_cross_layers

        # Default hidden dims for deep network are based on x0_dim = 3 * embedding_dim
        x0_dim_equiv = 3 * embedding_dim
        self.deep_hidden_dim1 = deep_hidden_dim1 if deep_hidden_dim1 is not None else x0_dim_equiv
        self.deep_hidden_dim2 = (
            deep_hidden_dim2 if deep_hidden_dim2 is not None else x0_dim_equiv // 2 if x0_dim_equiv > 1 else 1
        )

    def _build_pytorch_model(self) -> BasePyTorchEmbeddingModule:
        return _DeepCrossNetworkNet(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            user_features_dim=self.user_features_dim,
            item_features_dim=self.item_features_dim,
            interaction_features_dim=self.interaction_features_dim,
            num_cross_layers=self.num_cross_layers,
            deep_hidden_dim1=self.deep_hidden_dim1,
            deep_hidden_dim2=self.deep_hidden_dim2,
            user_features_tensor=self.user_features_tensor,
            item_features_tensor=self.item_features_tensor,
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_cross_layers": self.num_cross_layers,
                "deep_hidden_dim1": self.deep_hidden_dim1,
                "deep_hidden_dim2": self.deep_hidden_dim2,
            }
        )
        return config
