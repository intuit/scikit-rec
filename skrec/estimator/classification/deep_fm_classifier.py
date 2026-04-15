from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    AdamW = None  # type: ignore[assignment]
    CosineAnnealingLR = None  # type: ignore[assignment]

from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.util.logger import get_logger
from skrec.util.torch_device import select_torch_device

logger = get_logger(__name__)


# === core PyTorch network ===


_nn_module = nn.Module if nn is not None else object


class CrossNetwork(_nn_module):
    """
    Implements a set of cross layers that help learn higher-order feature interactions.
    x_{l+1} = x_l + x0 * (w_l^T x_l + b_l), for l = 0..L-1.
    """

    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        # Each cross layer has a weight vector (input_dim,) and a bias vector (input_dim,)
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim, dtype=torch.float32)) for _ in range(num_layers)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim, dtype=torch.float32)) for _ in range(num_layers)]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.num_layers):
            nn.init.xavier_uniform_(self.weights[i].unsqueeze(0))  # shape [1, input_dim]
            nn.init.zeros_(self.biases[i])  # shape [input_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_dim]
        x0 = x
        x_l = x
        for i in range(self.num_layers):
            # (x_l * w_i) -> shape [batch_size], then unsqueeze -> [batch_size, 1]
            cross_term = torch.sum(x_l * self.weights[i], dim=1, keepdim=True)
            # cross_term times x0 -> [batch_size, input_dim], then add bias
            x_l = x_l + x0 * cross_term + self.biases[i]
        return x_l


class DeepFactorizationMachineNetwork(_nn_module):
    """
    A factorization-machine-inspired network with:
      - An embedding table for first/second-order interactions.
      - An optional cross network for higher-order interactions.
      - An MLP for additional transformations.
      - Output: 1 logit for binary classification (for BCEWithLogitsLoss).
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: List[int],
        dropout: float,
        use_cross_layer: bool = False,
        num_cross_layers: int = 2,
        use_batch_norm: bool = False,
        bn_momentum: float = 0.1,
    ):
        super().__init__()

        # Embedding matrix for factorization machine part (float32).
        self.embedding_table = nn.Parameter(torch.empty(input_dim, embedding_dim, dtype=torch.float32))
        # Linear weights/bias for the FM linear term.
        self.linear_weight = nn.Parameter(torch.empty(input_dim, 1, dtype=torch.float32))
        self.linear_bias = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.use_cross_layer = use_cross_layer
        self.num_cross_layers = num_cross_layers
        self.use_batch_norm = use_batch_norm
        self.bn_momentum = bn_momentum

        # Optional cross network
        self.cross_net = None
        if self.use_cross_layer and num_cross_layers > 0:
            self.cross_net = CrossNetwork(input_dim=input_dim, num_layers=num_cross_layers)

        # Build MLP layers
        # The MLP input dimension depends on whether cross_net is used:
        #   second-order FM interactions have shape [batch_size, embedding_dim]
        #   cross network output has shape [batch_size, input_dim]
        # We'll concatenate them if cross_net is present.
        mlp_input_dim = embedding_dim if not self.use_cross_layer else (embedding_dim + input_dim)

        layers: List[nn.Module] = []
        prev_dim = mlp_input_dim
        for hd in hidden_dims:
            linear_layer = nn.Linear(prev_dim, hd, bias=True)
            # Optional BN
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(prev_dim, momentum=self.bn_momentum))
            layers.append(linear_layer)
            layers.append(nn.LeakyReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hd

        # Final layer: output 1 logit
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(prev_dim, momentum=self.bn_momentum))
        final_layer = nn.Linear(prev_dim, 1, bias=True)
        layers.append(final_layer)

        self.mlp = nn.Sequential(*layers)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        # Embedding table
        nn.init.xavier_uniform_(self.embedding_table)
        # Linear weights
        nn.init.xavier_uniform_(self.linear_weight)
        nn.init.zeros_(self.linear_bias)

        # MLP layers
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X shape: [batch_size, input_dim].
        Each column is treated as a distinct (possibly one-hot) feature.

        Factorization-machine second-order term:
          v = X @ E  --> shape [batch_size, embedding_dim]
          v_sq = v*v
          X_sq = X*X
          E_sq = E*E
          interactions = 0.5 * (v_sq - (X_sq @ E_sq))

        Then linear_part = X @ linear_weight + linear_bias

        If cross network is used, cross_out = CrossNetwork(X).
        We'll concatenate cross_out with interactions for the MLP.

        Final logit = linear_part + MLP(...).
        """
        # FM second-order
        v = X @ self.embedding_table
        v_square = (X * X) @ (self.embedding_table * self.embedding_table)
        interactions = 0.5 * (v * v - v_square)

        linear_part = X @ self.linear_weight + self.linear_bias

        # Optional cross network
        cross_out = None
        if self.cross_net is not None:
            cross_out = self.cross_net(X)  # shape [batch_size, input_dim]

        if cross_out is not None:
            mlp_input = torch.cat([interactions, cross_out], dim=1)
        else:
            mlp_input = interactions

        mlp_out = self.mlp(mlp_input)  # shape [batch_size, 1]
        logits = linear_part + mlp_out
        return logits


# === Classifier interface ===


class DeepFMClassifier(BaseClassifier):
    """
    Factorization-machine style network with optional cross layers

    Hyperparameters are specified in a flat params dict, e.g.:
        params = {
            "embedding_dim": 16,
            "hidden_dim1": 128,
            "hidden_dim2": 64,   # can add more up to hidden_dim10
            "batch_size": 1024,
            "epochs": 5,
            "lr": 1e-3,
        }
    """

    DEFAULT_PARAMS = {
        # Network hyperparams:
        "embedding_dim": 16,
        "hidden_dim1": 128,
        "hidden_dim2": 64,
        "batch_size": 1024,
        "epochs": 5,
        "lr": 1e-3,
        "l1_reg": 0.0,
        "l2_reg": 0.0,
        "dropout": 0.0,
        "cosine_tmax": 5,
        "device": None,
        # Additional:
        "use_cross_layer": False,
        "num_cross_layers": 2,
        "use_batch_norm": False,
        "bn_momentum": 0.1,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialise a DeepFMClassifier.

        Args:
            params: Flat dict of hyperparameters that override ``DEFAULT_PARAMS``.
                Recognised keys include ``embedding_dim``, ``hidden_dim1`` …
                ``hidden_dim10``, ``batch_size``, ``epochs``, ``lr``,
                ``l1_reg``, ``l2_reg``, ``dropout``, ``cosine_tmax``,
                ``device``, ``use_cross_layer``, ``num_cross_layers``,
                ``use_batch_norm``, and ``bn_momentum``.  Any key not present
                falls back to its default value.
        """
        if params is None:
            params = {}
        self.params = {**self.DEFAULT_PARAMS, **params}
        self._model: Optional[DeepFactorizationMachineNetwork] = None

    def _fit_model(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Build (if needed) and train the DeepFM network.

        Args:
            X: Feature matrix of shape ``(n_train, n_features)``.
            y: Binary labels of shape ``(n_train,)``.
            X_valid: Optional validation feature matrix.  When provided
                together with ``y_valid``, validation loss is logged each
                epoch.
            y_valid: Optional validation labels.
            dtype: PyTorch floating-point dtype to use for all tensors
                (default ``torch.float32``).
        """
        if dtype is None:
            dtype = torch.float32
        # Coerce y and y_valid to be numpy arrays
        if isinstance(y, Series):
            y = y.values

        if isinstance(y_valid, Series):
            y_valid = y_valid.values

        device = select_torch_device(self.params["device"])

        # Convert to PyTorch Tensors
        X_train_torch = torch.from_numpy(X.values).to(device=device, dtype=dtype)
        y_train_torch = torch.from_numpy(y.squeeze()).to(device=device, dtype=dtype)

        # Validation Tensors
        X_val_torch, y_val_torch = None, None
        if X_valid is not None and y_valid is not None:
            X_val_torch = torch.from_numpy(X_valid.values).to(device=device, dtype=dtype)
            y_val_torch = torch.from_numpy(y_valid.squeeze()).to(device=device, dtype=dtype)

        # Build the DeepFactorizationMachineNetwork
        hidden_dims = [
            int(self.params[f"hidden_dim{i}"])
            for i in range(1, 11)  # up to 10 hidden layers if specified
            if f"hidden_dim{i}" in self.params
        ]

        if self._model is None:
            input_dim = X_train_torch.shape[1]
            self._model = DeepFactorizationMachineNetwork(
                input_dim=input_dim,
                embedding_dim=int(self.params["embedding_dim"]),
                hidden_dims=hidden_dims,
                dropout=float(self.params["dropout"]),
                use_cross_layer=bool(self.params["use_cross_layer"]),
                num_cross_layers=int(self.params["num_cross_layers"]),
                use_batch_norm=bool(self.params["use_batch_norm"]),
                bn_momentum=float(self.params["bn_momentum"]),
            ).to(device)

        # Set up optimizer & optional scheduler
        optimizer = AdamW(
            self._model.parameters(), lr=float(self.params["lr"]), weight_decay=float(self.params["l2_reg"])
        )
        scheduler = None
        if self.params["cosine_tmax"] > 0:
            scheduler = CosineAnnealingLR(optimizer, T_max=int(self.params["cosine_tmax"]))

        criterion = nn.BCEWithLogitsLoss()

        batch_size = int(self.params["batch_size"])
        epochs = int(self.params["epochs"])
        l1_reg = float(self.params["l1_reg"])

        self._model.train()
        for epoch in range(epochs):
            # Shuffle training data each epoch
            indices = torch.randperm(X_train_torch.size(0))
            X_train_shuffled = X_train_torch[indices].to(device)
            y_train_shuffled = y_train_torch[indices].to(device)

            total_loss = 0.0
            for i in range(0, X_train_shuffled.size(0), batch_size):
                batch_X = X_train_shuffled[i : i + batch_size]
                batch_y = y_train_shuffled[i : i + batch_size]

                optimizer.zero_grad()
                logits = self._model(batch_X).view(-1)
                loss = criterion(logits, batch_y)

                # Optional L1 regularization
                if l1_reg > 0.0:
                    l1_loss = torch.tensor(0.0, device=device)
                    for param in self._model.parameters():
                        l1_loss += torch.sum(torch.abs(param))
                    loss = loss + l1_reg * l1_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            avg_train_loss = total_loss / (X_train_shuffled.size(0) // batch_size)
            logger.info(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

            # Validation step
            if X_val_torch is not None and y_val_torch is not None:
                val_loss = self._eval_loss(X_val_torch, y_val_torch, criterion, batch_size, l1_reg)
                logger.info(f"  Validation Loss: {val_loss:.4f}")

    def _eval_loss(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        criterion: nn.Module,
        batch_size: int,
        l1_reg: float,
    ) -> float:
        """
        Evaluate on validation data in mini-batches, including L1 if needed.
        """
        if self._model is None:
            raise RuntimeError("Model is not trained yet. Call fit(...) first.")

        self._model.eval()
        total_loss = 0.0

        with torch.inference_mode():
            for i in range(0, X_val.shape[0], batch_size):
                batch_X = X_val[i : i + batch_size]
                batch_y = y_val[i : i + batch_size]
                logits = self._model(batch_X).view(-1)
                loss = criterion(logits, batch_y)

                # Typically we do not penalize L1 on validation,
                # but it can be included if desired
                if l1_reg > 0.0:
                    device = next(self._model.parameters()).device
                    l1_loss = torch.tensor(0.0, device=device)
                    for param in self._model.parameters():
                        l1_loss += torch.sum(torch.abs(param))
                    loss += l1_reg * l1_loss

                total_loss += loss.item()

        self._model.train()
        return total_loss / (X_val.size(0) // batch_size)

    def _predict_proba_model(
        self,
        X: DataFrame,
        dtype: Optional[torch.dtype] = None,
    ) -> NDArray:
        """
        Runs the network to get probabilities.

        Output shape (N, 2): [ P(class=0), P(class=1) ].
        """
        if dtype is None:
            dtype = torch.float32
        if self._model is None:
            raise RuntimeError("Model is not trained yet. Call fit(...) first.")

        device = select_torch_device(self.params["device"])
        self._model.eval()

        X_torch = torch.from_numpy(X.values).to(device=device, dtype=dtype)
        batch_size = int(self.params["batch_size"])

        probs_list = []
        with torch.inference_mode():
            for start_idx in range(0, X_torch.size(0), batch_size):
                end_idx = start_idx + batch_size
                batch_X = X_torch[start_idx:end_idx]
                logits = self._model(batch_X)
                # For BCEWithLogits, sigmoid(logit) is P(y=1).
                batch_probs = torch.sigmoid(logits)

                # Convert shape [B,1] to two-column [P(class=0), P(class=1)]
                batch_probs_2col = torch.cat([1.0 - batch_probs, batch_probs], dim=1)
                probs_list.append(batch_probs_2col.cpu().numpy())

        return np.concatenate(probs_list, axis=0)
