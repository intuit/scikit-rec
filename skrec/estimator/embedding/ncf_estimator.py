from __future__ import annotations

from typing import List, Optional

import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from skrec.constants import USER_EMBEDDING_NAME, USER_ID_NAME
from skrec.estimator.embedding.base_pytorch_estimator import (
    BasePyTorchEmbeddingEstimator,
    BasePyTorchEmbeddingModule,
)
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class _NCFModule(BasePyTorchEmbeddingModule):
    """
    Neural Collaborative Filtering (NCF) module supporting three variants:
    - GMF (Generalized Matrix Factorization): Element-wise product of embeddings
    - MLP (Multi-Layer Perceptron): Deep network on concatenated embeddings
    - NeuMF (Neural Matrix Factorization): Ensemble of GMF and MLP with separate embeddings

    This implementation extends the original NCF paper to support user features,
    item features, and interaction features through projection layers.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        ncf_type: str,
        gmf_embedding_dim: int,
        mlp_embedding_dim: int,
        mlp_layers: List[int],
        dropout: float,
        user_features_dim: int,
        item_features_dim: int,
        interaction_features_dim: int,
        user_features_tensor: Optional[torch.Tensor],
        item_features_tensor: Optional[torch.Tensor],
    ):
        """
        Initialize NCF module.

        Args:
            num_users: Number of unique users (including unknown placeholder)
            num_items: Number of unique items (including unknown placeholder)
            ncf_type: Type of NCF variant - "gmf", "mlp", or "neumf"
            gmf_embedding_dim: Embedding dimension for GMF path
            mlp_embedding_dim: Embedding dimension for MLP path
            mlp_layers: List of hidden layer sizes for MLP (e.g., [64, 32, 16, 8])
            dropout: Dropout rate for regularization
            user_features_dim: Dimension of user features
            item_features_dim: Dimension of item features
            interaction_features_dim: Dimension of interaction/context features
            user_features_tensor: Tensor containing all user features
            item_features_tensor: Tensor containing all item features
        """
        # For NCF, we use different embedding dimensions based on the architecture
        # GMF uses gmf_embedding_dim, MLP uses mlp_embedding_dim
        # NeuMF uses both, so we need to handle this specially
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            user_embedding_dim=gmf_embedding_dim,  # Base class uses this for GMF
            item_embedding_dim=gmf_embedding_dim,
            user_features_dim=user_features_dim,
            item_features_dim=item_features_dim,
            user_features_tensor=user_features_tensor,
            item_features_tensor=item_features_tensor,
        )

        self.ncf_type = ncf_type.lower()
        if self.ncf_type not in ["gmf", "mlp", "neumf"]:
            raise ValueError(f"ncf_type must be 'gmf', 'mlp', or 'neumf', got '{ncf_type}'")

        self.gmf_embedding_dim = gmf_embedding_dim
        self.mlp_embedding_dim = mlp_embedding_dim
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.interaction_features_dim = interaction_features_dim

        # For NeuMF, we need separate embeddings for the MLP path
        # The base class already created user_id_embedding and item_id_embedding for GMF
        if self.ncf_type in ["mlp", "neumf"]:
            self.user_id_embedding_mlp = nn.Embedding(num_users, mlp_embedding_dim)
            self.item_id_embedding_mlp = nn.Embedding(num_items, mlp_embedding_dim)
        else:
            self.user_id_embedding_mlp = None
            self.item_id_embedding_mlp = None

        # Feature projection layers
        self.user_feature_projection: Optional[nn.Linear] = None
        if user_features_dim > 0:
            # Project to the same dimension as embeddings
            target_dim = gmf_embedding_dim if self.ncf_type == "gmf" else mlp_embedding_dim
            self.user_feature_projection = nn.Linear(user_features_dim, target_dim)

        self.item_feature_projection: Optional[nn.Linear] = None
        if item_features_dim > 0:
            target_dim = gmf_embedding_dim if self.ncf_type == "gmf" else mlp_embedding_dim
            self.item_feature_projection = nn.Linear(item_features_dim, target_dim)

        # Context/interaction features projection
        self.interaction_projection: Optional[nn.Linear] = None
        if interaction_features_dim > 0:
            # For interaction features, we'll add them to the MLP input
            # This extends beyond the original NCF paper
            pass  # Will be handled in forward pass

        # Build the MLP tower if needed
        if self.ncf_type in ["mlp", "neumf"]:
            self.mlp_tower = self._build_mlp_tower(mlp_layers, dropout, interaction_features_dim)
        else:
            self.mlp_tower = None

        # Final prediction layer
        if self.ncf_type == "gmf":
            # GMF: element-wise product output
            self.prediction_layer = nn.Linear(gmf_embedding_dim, 1)
        elif self.ncf_type == "mlp":
            # MLP: final layer from last MLP hidden layer
            final_mlp_dim = mlp_layers[-1] if mlp_layers else mlp_embedding_dim * 2
            self.prediction_layer = nn.Linear(final_mlp_dim, 1)
        else:  # neumf
            # NeuMF: concatenate GMF and MLP outputs
            final_mlp_dim = mlp_layers[-1] if mlp_layers else mlp_embedding_dim * 2
            self.prediction_layer = nn.Linear(gmf_embedding_dim + final_mlp_dim, 1)

    def _build_mlp_tower(self, mlp_layers: List[int], dropout: float, interaction_features_dim: int) -> nn.Sequential:
        """
        Build the MLP tower for the MLP and NeuMF architectures.

        Args:
            mlp_layers: List of hidden layer sizes
            dropout: Dropout rate
            interaction_features_dim: Dimension of interaction features (added to input)

        Returns:
            Sequential module representing the MLP tower
        """
        layers = []
        # Input dimension is concatenated user and item embeddings plus interaction features
        input_dim = self.mlp_embedding_dim * 2 + interaction_features_dim

        for hidden_dim in mlp_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        return nn.Sequential(*layers)

    def _get_gmf_embeddings(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        user_embeddings: Optional[torch.Tensor] = None,
        user_features: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Get user and item embeddings for the GMF path."""
        # Get user embeddings (from provided or learned)
        batch_user_emb_gmf = self._get_user_embeddings(user_indices, user_embeddings)

        # Get user features and fuse with embeddings
        batch_user_features = self._get_user_features(user_indices, user_features)
        if self.user_feature_projection is not None and batch_user_features is not None:
            # For GMF, we need features projected to gmf_embedding_dim
            # But user_feature_projection is set up for the appropriate dimension
            if self.ncf_type == "gmf":
                projected_user_features = self.user_feature_projection(batch_user_features)
                batch_user_emb_gmf = batch_user_emb_gmf + projected_user_features

        # Get item embeddings (always from model's layer for GMF path)
        batch_item_emb_gmf = self.item_id_embedding(item_indices)

        # Get item features and fuse with embeddings
        batch_item_features: Optional[torch.Tensor] = None
        if self.item_features_tensor is not None and self.item_features_tensor.shape[1] > 0:
            batch_item_features = self.item_features_tensor[item_indices]

        if self.item_feature_projection is not None and batch_item_features is not None:
            if self.ncf_type == "gmf":
                projected_item_features = self.item_feature_projection(batch_item_features)
                batch_item_emb_gmf = batch_item_emb_gmf + projected_item_features

        return batch_user_emb_gmf, batch_item_emb_gmf

    def _get_mlp_embeddings(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        user_embeddings: Optional[torch.Tensor] = None,
        user_features: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Get user and item embeddings for the MLP path."""
        # For MLP path, we use separate embeddings
        # Note: user_embeddings from external source are assumed to be for GMF path
        # For MLP in NeuMF, we use the learned MLP embeddings
        if user_embeddings is not None and self.ncf_type == "neumf":
            # In real-time inference with NeuMF, we need both GMF and MLP embeddings
            # For simplicity, we'll use the same provided embedding for both paths
            # In production, you might store separate embeddings
            batch_user_emb_mlp = user_embeddings[user_indices]
        else:
            batch_user_emb_mlp = self.user_id_embedding_mlp(user_indices)

        # Get user features and fuse
        batch_user_features = self._get_user_features(user_indices, user_features)
        if self.user_feature_projection is not None and batch_user_features is not None:
            if self.ncf_type in ["mlp", "neumf"]:
                projected_user_features = self.user_feature_projection(batch_user_features)
                batch_user_emb_mlp = batch_user_emb_mlp + projected_user_features

        # Get item embeddings for MLP path
        batch_item_emb_mlp = self.item_id_embedding_mlp(item_indices)

        # Get item features and fuse
        batch_item_features: Optional[torch.Tensor] = None
        if self.item_features_tensor is not None and self.item_features_tensor.shape[1] > 0:
            batch_item_features = self.item_features_tensor[item_indices]

        if self.item_feature_projection is not None and batch_item_features is not None:
            if self.ncf_type in ["mlp", "neumf"]:
                projected_item_features = self.item_feature_projection(batch_item_features)
                batch_item_emb_mlp = batch_item_emb_mlp + projected_item_features

        return batch_user_emb_mlp, batch_item_emb_mlp

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        interaction_features: torch.Tensor,
        user_embeddings: Optional[torch.Tensor] = None,
        user_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for NCF.

        Args:
            user_indices: Tensor of user indices (batch_size,)
            item_indices: Tensor of item indices (batch_size,)
            interaction_features: Tensor of interaction features (batch_size, interaction_features_dim)
            user_embeddings: Optional pre-computed user embeddings for real-time inference
            user_features: Optional user features for real-time inference

        Returns:
            Prediction logits (batch_size, 1)
        """
        if self.ncf_type == "gmf":
            # GMF: element-wise product of user and item embeddings
            batch_user_emb, batch_item_emb = self._get_gmf_embeddings(
                user_indices, item_indices, user_embeddings, user_features
            )
            gmf_output = batch_user_emb * batch_item_emb  # Element-wise product
            prediction = self.prediction_layer(gmf_output)

        elif self.ncf_type == "mlp":
            # MLP: concatenate embeddings and pass through MLP tower
            batch_user_emb, batch_item_emb = self._get_mlp_embeddings(
                user_indices, item_indices, user_embeddings, user_features
            )

            # Concatenate user, item embeddings and interaction features
            if interaction_features.shape[1] > 0:
                mlp_input = torch.cat([batch_user_emb, batch_item_emb, interaction_features], dim=1)
            else:
                mlp_input = torch.cat([batch_user_emb, batch_item_emb], dim=1)

            mlp_output = self.mlp_tower(mlp_input)
            prediction = self.prediction_layer(mlp_output)

        else:  # neumf
            # NeuMF: Combine GMF and MLP paths
            # GMF path
            batch_user_emb_gmf, batch_item_emb_gmf = self._get_gmf_embeddings(
                user_indices, item_indices, user_embeddings, user_features
            )
            gmf_output = batch_user_emb_gmf * batch_item_emb_gmf

            # MLP path
            batch_user_emb_mlp, batch_item_emb_mlp = self._get_mlp_embeddings(
                user_indices, item_indices, user_embeddings, user_features
            )

            # Concatenate user, item embeddings and interaction features for MLP
            if interaction_features.shape[1] > 0:
                mlp_input = torch.cat([batch_user_emb_mlp, batch_item_emb_mlp, interaction_features], dim=1)
            else:
                mlp_input = torch.cat([batch_user_emb_mlp, batch_item_emb_mlp], dim=1)

            mlp_output = self.mlp_tower(mlp_input)

            # Concatenate GMF and MLP outputs
            neumf_input = torch.cat([gmf_output, mlp_output], dim=1)
            prediction = self.prediction_layer(neumf_input)

        return prediction


class NCFEstimator(BasePyTorchEmbeddingEstimator):
    """
    Neural Collaborative Filtering (NCF) Estimator.

    Implements three variants of NCF:
    1. GMF (Generalized Matrix Factorization): Matrix factorization with neural architecture
    2. MLP (Multi-Layer Perceptron): Deep network on user-item embeddings
    3. NeuMF (Neural Matrix Factorization): Ensemble of GMF and MLP

    This implementation extends the original NCF paper (He et al., 2017) to support:
    - User features (projected and fused with embeddings)
    - Item features (projected and fused with embeddings)
    - Interaction/context features (incorporated into MLP path)
    - Embedding extraction and model truncation for production deployment
    - Both implicit feedback (BCE loss) and explicit ratings (MSE loss)

    Reference:
        He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017).
        Neural collaborative filtering. In WWW 2017.
    """

    def __init__(
        self,
        ncf_type: str = "neumf",
        gmf_embedding_dim: int = 32,
        mlp_embedding_dim: int = 32,
        mlp_layers: Optional[List[int]] = None,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        optimizer_name: str = "adam",
        loss_fn_name: str = "bce",
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        """
        Initialize NCF Estimator.

        Args:
            ncf_type: Type of NCF variant - "gmf", "mlp", or "neumf" (default: "neumf")
            gmf_embedding_dim: Embedding dimension for GMF path (default: 32)
            mlp_embedding_dim: Embedding dimension for MLP path (default: 32)
            mlp_layers: List of hidden layer sizes for MLP (default: [64, 32, 16, 8])
                       Only used for "mlp" and "neumf" types
            dropout: Dropout rate for regularization (default: 0.0)
            learning_rate: Learning rate for optimizer (default: 0.001)
            epochs: Number of training epochs (default: 10)
            batch_size: Batch size for training (default: 32)
            optimizer_name: Optimizer to use - "adam" or "sgd" (default: "adam")
            loss_fn_name: Loss function - "bce" for implicit or "mse" for explicit (default: "bce")
            device: Device to use - "cpu", "cuda", or None for auto-detect (default: None)
            random_state: Random seed for reproducibility (default: None)
            verbose: Verbosity level - 0 (silent) or 1 (show training progress) (default: 0)
        """
        super().__init__(
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            loss_fn_name=loss_fn_name,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

        # Validate ncf_type
        self.ncf_type = ncf_type.lower()
        if self.ncf_type not in ["gmf", "mlp", "neumf"]:
            raise ValueError(f"ncf_type must be 'gmf', 'mlp', or 'neumf', got '{ncf_type}'")

        self.gmf_embedding_dim = gmf_embedding_dim
        self.mlp_embedding_dim = mlp_embedding_dim

        # Default MLP layers if not provided
        if mlp_layers is None:
            self.mlp_layers = [64, 32, 16, 8]
        else:
            self.mlp_layers = mlp_layers

        self.dropout = dropout

    def _build_pytorch_model(self) -> BasePyTorchEmbeddingModule:
        """Build and return the NCF PyTorch model."""
        return _NCFModule(
            num_users=self.num_users,
            num_items=self.num_items,
            ncf_type=self.ncf_type,
            gmf_embedding_dim=self.gmf_embedding_dim,
            mlp_embedding_dim=self.mlp_embedding_dim,
            mlp_layers=self.mlp_layers,
            dropout=self.dropout,
            user_features_dim=self.user_features_dim,
            item_features_dim=self.item_features_dim,
            interaction_features_dim=self.interaction_features_dim,
            user_features_tensor=self.user_features_tensor,
            item_features_tensor=self.item_features_tensor,
        )

    def get_config(self) -> dict:
        """
        Get configuration dictionary for the estimator.

        Returns:
            Dictionary containing all hyperparameters
        """
        config = super().get_config()
        config.update(
            {
                "ncf_type": self.ncf_type,
                "gmf_embedding_dim": self.gmf_embedding_dim,
                "mlp_embedding_dim": self.mlp_embedding_dim,
                "mlp_layers": self.mlp_layers,
                "dropout": self.dropout,
            }
        )
        return config

    def truncate_user_data(self) -> None:
        """
        Modifies the estimator's state for compactness after user embeddings
        have been extracted. For NCF, we need to handle both GMF and MLP embeddings.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained.")

        if self._user_data_truncated:
            logger.info("User data is already truncated.")
            return

        # 1. Extract the learned placeholder embedding's weights for GMF path
        with torch.no_grad():
            learned_placeholder_weights_gmf = (
                self.model.user_id_embedding.weight[self.unknown_user_idx].clone().detach()
            )

        # 2. Create a new, small embedding layer for the GMF placeholder
        new_user_embedding_layer_gmf = nn.Embedding(num_embeddings=1, embedding_dim=self.model.user_embedding_dim).to(
            self.device
        )

        with torch.no_grad():
            new_user_embedding_layer_gmf.weight.data[0] = learned_placeholder_weights_gmf

        # 3. Replace the model's user_id_embedding layer
        self.model.user_id_embedding = new_user_embedding_layer_gmf

        # 4. For MLP/NeuMF variants, also handle the MLP user embeddings
        if self.ncf_type in ["mlp", "neumf"] and self.model.user_id_embedding_mlp is not None:
            with torch.no_grad():
                learned_placeholder_weights_mlp = (
                    self.model.user_id_embedding_mlp.weight[self.unknown_user_idx].clone().detach()
                )

            new_user_embedding_layer_mlp = nn.Embedding(num_embeddings=1, embedding_dim=self.mlp_embedding_dim).to(
                self.device
            )

            with torch.no_grad():
                new_user_embedding_layer_mlp.weight.data[0] = learned_placeholder_weights_mlp

            self.model.user_id_embedding_mlp = new_user_embedding_layer_mlp

        # 5. Update metadata
        self.model.num_users = 1
        self.num_users = 1
        self.unknown_user_idx = 0
        self.user_id_index = pd.Index([""])  # placeholder
        self.user_features_tensor = None

        self._user_data_truncated = True

    def get_user_embeddings(self) -> pd.DataFrame:
        """
        Extracts user embeddings from the trained model.
        For NCF, we extract GMF embeddings by default (as they're used for all variants).

        Returns:
            pd.DataFrame: DataFrame with USER_ID_NAME and USER_EMBEDDING_NAME columns.
        """
        if self.model is None or not hasattr(self.model, "user_id_embedding"):
            raise RuntimeError("Model is not trained or does not have user_id_embedding layer.")
        if self._user_data_truncated:
            raise RuntimeError("User data has been truncated. Cannot extract full embeddings.")
        if self.user_id_index is None or self.unknown_user_idx is None:
            raise RuntimeError("User ID index or unknown_user_idx not available. Fit the model first.")

        # Extract GMF embeddings (primary embeddings for all NCF variants)
        user_embedding_weights = self.model.user_id_embedding.weight.data.cpu().numpy()

        embeddings_list = []
        for i in range(self.unknown_user_idx):
            user_id = self.user_id_index[i]
            embeddings_list.append({USER_ID_NAME: user_id, USER_EMBEDDING_NAME: user_embedding_weights[i]})

        if not embeddings_list:
            return pd.DataFrame(columns=[USER_ID_NAME, USER_EMBEDDING_NAME])

        return pd.DataFrame(embeddings_list)
