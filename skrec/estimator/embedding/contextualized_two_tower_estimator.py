from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

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


class ContextMode(str, Enum):
    """Controls how interaction context features are incorporated into the two-tower model.

    Attributes:
        USER_TOWER: Context is concatenated into the user tower input alongside the user
            ID embedding and profile features. The final score is the dot product of the
            user and item tower outputs. User embeddings are context-dependent and must
            be recomputed at request time — they cannot be precomputed offline.
            ANN retrieval is supported: compute the user+context embedding at request
            time, then search the precomputed item index.
            ``get_user_embeddings()`` raises if context features are present.

        TRILINEAR: Context is projected to ``final_embedding_dim`` and applied via an
            elementwise (Hadamard) product with the user tower output before the dot
            product with items: ``score = dot(user_rep * context_emb, item_rep)``.
            The user tower is context-free — user embeddings are precomputable offline.
            At serving time, multiply the cached user embedding by the runtime context
            embedding to get the effective query vector, then run ANN search.

        SCORING_LAYER: Context is projected to ``final_embedding_dim`` and concatenated
            with both tower outputs. A final linear layer maps
            ``[user_rep, item_rep, context_rep]`` to a scalar score. Most expressive
            of the three modes, but scores cannot be decomposed as a dot product so
            ANN retrieval is not supported. User embeddings are precomputable but the
            full model must be run at serving time.
    """

    USER_TOWER = "user_tower"
    TRILINEAR = "trilinear"
    SCORING_LAYER = "scoring_layer"


class _TwoTowerNet(BasePyTorchEmbeddingModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_embedding_dim: int,
        item_embedding_dim: int,
        final_embedding_dim: int,
        user_features_dim: int,
        item_features_dim: int,
        interaction_features_dim: int,
        user_tower_hidden_dim1: Optional[int],
        user_tower_hidden_dim2: Optional[int],
        item_tower_hidden_dim1: Optional[int],
        item_tower_hidden_dim2: Optional[int],
        user_features_tensor: Optional[torch.Tensor],
        item_features_tensor: Optional[torch.Tensor],
        context_mode: ContextMode = ContextMode.USER_TOWER,
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            user_embedding_dim=user_embedding_dim,
            item_embedding_dim=item_embedding_dim,
            user_features_dim=user_features_dim,
            item_features_dim=item_features_dim,
            user_features_tensor=user_features_tensor,
            item_features_tensor=item_features_tensor,
        )
        self.final_embedding_dim = final_embedding_dim
        self.context_mode = context_mode
        self.interaction_features_dim = interaction_features_dim

        # USER_TOWER: context concatenated into user tower input — user reps are context-dependent
        # TRILINEAR / SCORING_LAYER: user tower takes profile features only — context-free user reps
        if context_mode == ContextMode.USER_TOWER:
            user_tower_input_dim = user_embedding_dim + user_features_dim + interaction_features_dim
        else:
            user_tower_input_dim = user_embedding_dim + user_features_dim

        self.user_tower = self._build_tower_mlp(
            input_dim=user_tower_input_dim,
            hidden_dim1=user_tower_hidden_dim1,
            hidden_dim2=user_tower_hidden_dim2,
            output_dim=self.final_embedding_dim,
        )

        # Item tower — always context-free; item embeddings are precomputable offline
        item_tower_input_dim = item_embedding_dim + item_features_dim
        self.item_tower = self._build_tower_mlp(
            input_dim=item_tower_input_dim,
            hidden_dim1=item_tower_hidden_dim1,
            hidden_dim2=item_tower_hidden_dim2,
            output_dim=self.final_embedding_dim,
        )

        # TRILINEAR and SCORING_LAYER both project context to final_embedding_dim first
        self.context_projection: Optional[nn.Linear] = None
        if context_mode in (ContextMode.TRILINEAR, ContextMode.SCORING_LAYER) and interaction_features_dim > 0:
            self.context_projection = nn.Linear(interaction_features_dim, final_embedding_dim)

        # SCORING_LAYER only: linear layer over [user_rep, item_rep, context_rep] → scalar
        self.scoring_layer: Optional[nn.Linear] = None
        if context_mode == ContextMode.SCORING_LAYER:
            scoring_input_dim = final_embedding_dim * 2
            if interaction_features_dim > 0:
                scoring_input_dim += final_embedding_dim
            self.scoring_layer = nn.Linear(scoring_input_dim, 1)

    def _build_tower_mlp(
        self,
        input_dim: int,
        hidden_dim1: Optional[int],
        hidden_dim2: Optional[int],
        output_dim: int,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        current_dim = input_dim
        if hidden_dim1 is not None:
            layers.append(nn.Linear(current_dim, hidden_dim1))
            layers.append(nn.ReLU())
            current_dim = hidden_dim1
            if hidden_dim2 is not None:
                layers.append(nn.Linear(current_dim, hidden_dim2))
                layers.append(nn.ReLU())
                current_dim = hidden_dim2
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        interaction_features: torch.Tensor,
        user_embeddings: Optional[torch.Tensor] = None,
        user_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_user_emb = self._get_user_embeddings(user_indices, user_embeddings)
        batch_user_features = self._get_user_features(user_indices, user_features)

        # User tower: USER_TOWER mode concatenates context here; others use profile only
        # Note: when calling predict_proba_with_embeddings(), _prepare_inference_tensors() raises
        # ValueError first if context columns are missing from the DataFrame. This guard exists
        # as a defence for callers that construct tensors manually and pass them directly to model().
        if self.context_mode == ContextMode.USER_TOWER and self.interaction_features_dim > 0:
            if interaction_features.shape[1] == 0:
                raise ValueError(
                    f"Model was trained with {self.interaction_features_dim} context feature(s) in USER_TOWER mode, "
                    "but no context features were provided at inference time. "
                    "Pass the same context features used during training, or switch to context_mode='trilinear' "
                    "or 'scoring_layer' to precompute context-free user embeddings."
                )
        user_tower_inputs = [batch_user_emb]
        if batch_user_features is not None:
            user_tower_inputs.append(batch_user_features)
        if self.context_mode == ContextMode.USER_TOWER and interaction_features.shape[1] > 0:
            user_tower_inputs.append(interaction_features)
        user_representation = self.user_tower(torch.cat(user_tower_inputs, dim=1))

        # Item tower: always context-free
        batch_item_emb = self.item_id_embedding(item_indices)
        item_tower_inputs = [batch_item_emb]
        if self.item_features_tensor is not None and self.item_features_tensor.shape[1] > 0:
            item_tower_inputs.append(self.item_features_tensor[item_indices])
        item_representation = self.item_tower(torch.cat(item_tower_inputs, dim=1))

        if self.context_mode == ContextMode.USER_TOWER:
            # dot(user_tower(user, context), item_tower(item))
            # ANN-compatible: compute user+context embedding at request time, search item index
            score = torch.sum(user_representation * item_representation, dim=1, keepdim=True)

        elif self.context_mode == ContextMode.TRILINEAR:
            # dot(user_rep * context_emb, item_rep)  — trilinear interaction
            # ANN-compatible: modulate cached user_rep by runtime context_emb, search item index
            if self.context_projection is not None and interaction_features.shape[1] > 0:
                context_emb = self.context_projection(interaction_features)
                modulated_user = user_representation * context_emb  # elementwise Hadamard
            else:
                modulated_user = user_representation
            score = torch.sum(modulated_user * item_representation, dim=1, keepdim=True)

        else:  # ContextMode.SCORING_LAYER
            # Linear([user_rep, item_rep, context_rep]) → scalar
            # Not ANN-compatible: full model must run at serving time
            scoring_inputs = [user_representation, item_representation]
            if self.context_projection is not None and interaction_features.shape[1] > 0:
                scoring_inputs.append(self.context_projection(interaction_features))
            score = self.scoring_layer(torch.cat(scoring_inputs, dim=1))  # type: ignore[misc]

        return score


class ContextualizedTwoTowerEstimator(BasePyTorchEmbeddingEstimator):
    """
    A Contextualized Two-Tower Estimator with three selectable context integration modes.

    Both towers (user and item) produce a ``final_embedding_dim`` representation.
    How interaction context features are incorporated is controlled by ``context_mode``:

    ``"user_tower"`` (default — industry standard, ANN-compatible):
        Context is concatenated into the user tower alongside the user ID embedding and
        profile features. The score is the dot product of user and item tower outputs.
        User embeddings are context-dependent and must be recomputed at request time
        (cannot be precomputed offline). Item embeddings are static and precomputable.
        ANN retrieval (e.g., FAISS) is supported: compute the user+context embedding
        at request time, then search the precomputed item index.
        ``get_user_embeddings()`` raises if the model was trained with context features.

    ``"trilinear"`` (ANN-compatible, user embeddings precomputable):
        Context is projected to ``final_embedding_dim`` and applied via an elementwise
        (Hadamard) product with the user tower output before the dot product with items.
        ``score = dot(user_rep * context_emb, item_rep)``
        User tower is context-free — user embeddings can be precomputed offline.
        At serving time, multiply the cached user embedding by the runtime context
        embedding before searching the item index.

    ``"scoring_layer"`` (most expressive, not ANN-compatible):
        Context is projected to ``final_embedding_dim`` and concatenated with both tower
        outputs. A final linear layer maps ``[user_rep, item_rep, context_rep]`` to a
        scalar score. User embeddings are context-free and precomputable, but scores
        cannot be decomposed as a dot product, so ANN retrieval is not supported.

    Assumes that categorical features in users_df, items_df, and interactions_df
    have already been one-hot encoded if they are to be used as numerical features.
    """

    def __init__(
        self,
        user_embedding_dim: int = 64,
        item_embedding_dim: int = 64,
        final_embedding_dim: int = 32,
        context_mode: Union[ContextMode, str] = ContextMode.USER_TOWER,
        user_tower_hidden_dim1: Optional[int] = None,
        user_tower_hidden_dim2: Optional[int] = None,
        item_tower_hidden_dim1: Optional[int] = None,
        item_tower_hidden_dim2: Optional[int] = None,
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
        try:
            self.context_mode = ContextMode(context_mode)
        except ValueError:
            raise ValueError(
                f"Invalid context_mode '{context_mode}'. Must be one of: {[m.value for m in ContextMode]}."
            )
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.final_embedding_dim = final_embedding_dim
        self.user_tower_hidden_dim1 = (
            user_tower_hidden_dim1 if user_tower_hidden_dim1 is not None else final_embedding_dim * 2
        )
        self.user_tower_hidden_dim2 = user_tower_hidden_dim2
        self.item_tower_hidden_dim1 = (
            item_tower_hidden_dim1 if item_tower_hidden_dim1 is not None else final_embedding_dim * 2
        )
        self.item_tower_hidden_dim2 = item_tower_hidden_dim2

    def _build_pytorch_model(self) -> BasePyTorchEmbeddingModule:
        if self.context_mode == ContextMode.USER_TOWER and self.interaction_features_dim > 0:
            logger.warning(
                "context_mode='user_tower' with %d context feature(s): user embeddings are "
                "context-dependent and cannot be precomputed. get_user_embeddings() will raise. "
                "Use predict_proba_with_embeddings() at serving time, or switch to "
                "context_mode='trilinear' or 'scoring_layer' for precomputable user embeddings.",
                self.interaction_features_dim,
            )
        return _TwoTowerNet(
            num_users=self.num_users,
            num_items=self.num_items,
            user_embedding_dim=self.user_embedding_dim,
            item_embedding_dim=self.item_embedding_dim,
            final_embedding_dim=self.final_embedding_dim,
            user_features_dim=self.user_features_dim,
            item_features_dim=self.item_features_dim,
            interaction_features_dim=self.interaction_features_dim,
            user_tower_hidden_dim1=self.user_tower_hidden_dim1,
            user_tower_hidden_dim2=self.user_tower_hidden_dim2,
            item_tower_hidden_dim1=self.item_tower_hidden_dim1,
            item_tower_hidden_dim2=self.item_tower_hidden_dim2,
            user_features_tensor=self.user_features_tensor,
            item_features_tensor=self.item_features_tensor,
            context_mode=self.context_mode,
        )

    def get_user_embeddings(self):
        if self.context_mode == ContextMode.USER_TOWER and self.interaction_features_dim > 0:
            raise NotImplementedError(
                "Cannot precompute user embeddings when context_mode='user_tower' and the model "
                "was trained with context features — user representations depend on request-time "
                "context. Use predict_proba_with_embeddings() directly at serving time, or switch "
                "to context_mode='trilinear' or 'scoring_layer' to precompute context-free user embeddings."
            )
        return super().get_user_embeddings()

    def get_item_embeddings(self):
        """
        Extracts item embeddings by running all items through the item tower.

        Unlike other estimators that return raw embedding layer weights, Two-Tower
        returns the item tower output — the same representation used for scoring.
        These embeddings are suitable for ANN retrieval via dot-product search.

        Returns:
            pd.DataFrame: DataFrame with ITEM_ID_NAME and ITEM_EMBEDDING_NAME columns.
        """
        import pandas as pd
        import torch

        from skrec.constants import ITEM_EMBEDDING_NAME, ITEM_ID_NAME

        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit_embedding_model first.")
        if self.item_id_index is None or self.unknown_item_idx is None:
            raise RuntimeError("Item ID index not available. Fit the model first.")

        self.model.eval()
        n = self.unknown_item_idx
        all_item_indices = torch.arange(n, dtype=torch.long).to(self.device)

        with torch.no_grad():
            item_embs = self.model.item_id_embedding(all_item_indices)
            if self.model.item_features_tensor is not None and self.model.item_features_dim > 0:
                item_feats = self.model.item_features_tensor[all_item_indices]
                tower_input = torch.cat([item_embs, item_feats], dim=1)
            else:
                tower_input = item_embs
            item_representations = self.model.item_tower(tower_input).cpu().numpy()

        rows = [{ITEM_ID_NAME: self.item_id_index[i], ITEM_EMBEDDING_NAME: item_representations[i]} for i in range(n)]
        if not rows:
            return pd.DataFrame(columns=[ITEM_ID_NAME, ITEM_EMBEDDING_NAME])
        return pd.DataFrame(rows)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "user_embedding_dim": self.user_embedding_dim,
                "item_embedding_dim": self.item_embedding_dim,
                "final_embedding_dim": self.final_embedding_dim,
                "context_mode": self.context_mode.value,
                "user_tower_hidden_dim1": self.user_tower_hidden_dim1,
                "user_tower_hidden_dim2": self.user_tower_hidden_dim2,
                "item_tower_hidden_dim1": self.item_tower_hidden_dim1,
                "item_tower_hidden_dim2": self.item_tower_hidden_dim2,
            }
        )
        return config
