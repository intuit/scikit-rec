# Encoders for the joint multi-target estimator family.
#
# Both encoders satisfy the (internal) JointMultiTargetEncoder Protocol:
#   - forward(X, label_inputs=None) -> hidden representation of shape
#     (batch, hidden_dim).
#   - hidden_dim: int property.
#
# label_input_dim is a v3 hook (kept at 0 in v2). Conditional estimators in
# v3 pass label_input_dim > 0 and feed per-target label encodings into the
# encoder. v2 forwards ignore label_inputs (always None) — encoders treat
# label_input_dim=0 as the feature-only case.
#
# MLPEncoder: feed-forward MLP with dropout and optional batch norm. Skip
# connections between hidden layers when their widths match.
#
# TransformerEncoder: FT-Transformer-style feature tokenization (one
# learnable linear projection per feature, plus a CLS token). Multi-head
# self-attention blocks pool through CLS. Numeric-features-only in v2;
# categorical features are pre-encoded by the caller.

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


_nn_module = nn.Module if nn is not None else object


class MLPEncoder(_nn_module):
    """Feed-forward MLP encoder.

    Args:
        n_features: Number of input feature columns.
        label_input_dim: v3 hook; v2 must pass 0.
        hidden_dim: Width of every hidden layer.
        num_layers: Number of hidden layers (each = Linear + activation + dropout).
        dropout: Dropout probability after each hidden activation.
        use_batch_norm: Whether to apply BatchNorm1d after each Linear.
    """

    def __init__(
        self,
        n_features: int,
        label_input_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if label_input_dim < 0:
            raise ValueError("label_input_dim must be non-negative.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self._hidden_dim = hidden_dim

        in_dim = n_features + label_input_dim
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.layers = nn.Sequential(*layers)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(self, X: "torch.Tensor", label_inputs: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        if label_inputs is not None:
            X = torch.cat([X, label_inputs], dim=1)
        return self.layers(X)


class _FeatureTokenizer(_nn_module):
    """Per-feature linear projection to ``d_model`` + learnable bias.

    Equivalent to FT-Transformer's numerical feature tokenizer.
    Output shape: ``(batch, n_features, d_model)``.
    """

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)

    def forward(self, X: "torch.Tensor") -> "torch.Tensor":
        # X: (batch, n_features)
        # output: (batch, n_features, d_model) = X.unsqueeze(-1) * weight + bias
        return X.unsqueeze(-1) * self.weight + self.bias


class _CustomTransformerEncoderLayer(_nn_module):
    """Pre-norm Transformer block with SEPARATE attn / ffn dropouts.

    ``nn.TransformerEncoderLayer`` exposes a single ``dropout`` knob that
    drives both the FFN sub-layer dropout AND the dropout applied to
    attention outputs — and the ``nn.MultiheadAttention`` internal
    dropout (on attention weights) is fixed at construction. Post-hoc
    assignment to ``self_attn.dropout`` doesn't reliably take effect
    across torch versions.

    To honor the plan's separate ``attn_dropout`` / ``ffn_dropout`` knobs,
    we build the layer ourselves with both dropouts wired through their
    respective sub-modules. Pre-norm (norm-first) layout matches the
    rest of the FT-Transformer-style encoder.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        attn_dropout: float,
        ffn_dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        # MultiheadAttention's ``dropout`` is the attention-weight dropout —
        # the plan's ``attn_dropout``. Construct it correctly here; do NOT
        # mutate post-hoc.
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        # ffn_dropout: applied between FFN sub-layers + on the FFN residual.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_dim, d_model),
        )
        # Separate residual dropouts: ``attn_dropout`` drives BOTH the
        # attention-weight dropout (inside MultiheadAttention above) AND
        # the attention-output residual; ``ffn_dropout`` drives BOTH the
        # FFN-sub-layer dropout AND the FFN-output residual. Before this
        # split, both residuals shared a single ``ffn_dropout``-rated
        # Dropout module — so a user passing attn_dropout=0.5 + ffn_dropout=0.1
        # silently got 0.1 on the attention residual, not 0.5. Honoring
        # the plan's separation of concerns means each knob controls its
        # own residual too.
        self.dropout_attn_residual = nn.Dropout(attn_dropout)
        self.dropout_ffn_residual = nn.Dropout(ffn_dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Pre-norm self-attention.
        h = self.norm_attn(x)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.dropout_attn_residual(attn_out)
        # Pre-norm FFN.
        h = self.norm_ffn(x)
        x = x + self.dropout_ffn_residual(self.ffn(h))
        return x


class TransformerEncoder(_nn_module):
    """FT-Transformer-style encoder.

    Each input feature is tokenized into a ``d_model``-dimensional vector. A
    learnable CLS token is prepended. The sequence passes through ``n_layers``
    custom encoder blocks (multi-head self-attention with separate
    ``attn_dropout`` + FFN with ``ffn_dropout``). The CLS token output is
    the final hidden representation.

    Args:
        n_features: Number of input feature columns.
        label_input_dim: v3 hook; v2 must pass 0.
        d_model: Per-token embedding dimension (also the encoder's hidden_dim).
        n_heads: Number of attention heads.
        n_layers: Number of stacked encoder blocks.
        ffn_dim: Width of the FFN inside each block.
        attn_dropout: Dropout on attention weights (passed to
            ``nn.MultiheadAttention(dropout=...)`` at construction).
        ffn_dropout: Dropout in FFN sub-layers + residual.
    """

    def __init__(
        self,
        n_features: int,
        label_input_dim: int = 0,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 128,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if label_input_dim < 0:
            raise ValueError("label_input_dim must be non-negative.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self._hidden_dim = d_model
        self._n_features = n_features
        self._label_input_dim = label_input_dim

        # Feature tokens.
        self.feature_tokenizer = _FeatureTokenizer(n_features, d_model)
        # Optional label tokens (v3 hook).
        # NOTE: v3 conditional path replaces this with per-target tokens
        # via ConditionalJointMultiTargetBaseEstimator's label encoder
        # output. The scalar-token tokenizer below is kept for v2's
        # label_input_dim=0 case (where it's a no-op) and as a fallback;
        # the conditional Transformer estimator wires its own per-target
        # token sequence and appends them directly to the feature
        # sequence in forward().
        if label_input_dim > 0:
            self.label_tokenizer = _FeatureTokenizer(label_input_dim, d_model)
        else:
            self.label_tokenizer = None  # type: ignore[assignment]
        # CLS token.
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Custom Transformer blocks with SEPARATE attn / ffn dropouts.
        # nn.TransformerEncoderLayer's single-dropout knob doesn't let us
        # honor the plan's two-knob contract — see
        # _CustomTransformerEncoderLayer docstring above for the why.
        self.blocks = nn.ModuleList(
            [
                _CustomTransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                )
                for _ in range(n_layers)
            ]
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(
        self,
        X: "torch.Tensor",
        label_inputs: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Forward pass.

        ``label_inputs`` shapes accepted:
          - 2-D ``(batch, label_input_dim)``: scalar-token path (legacy /
            MLP-symmetric); each scalar becomes one token via the label
            tokenizer.
          - 3-D ``(batch, num_label_tokens, d_model)``: per-token path
            used by the conditional Transformer estimator — the tokens
            are already d_model-wide (one per declared target_specs
            entry) and append directly to the sequence.
        """
        batch = X.shape[0]
        feature_tokens = self.feature_tokenizer(X)  # (batch, n_features, d_model)
        cls_token = self.cls_token.expand(batch, -1, -1)  # (batch, 1, d_model)
        tokens = [cls_token, feature_tokens]
        if label_inputs is not None:
            if label_inputs.dim() == 3:
                # Per-target tokens at d_model — append directly. The
                # caller's contract is "if you pass 3-D label_inputs,
                # you produced them at d_model width yourself via the
                # ConditionalLabelEncoder's per-target projections, and
                # this encoder's own label_tokenizer is unused." If a
                # future caller wires label_input_dim > 0 AND passes
                # 3-D label_inputs, ``self.label_tokenizer`` will still
                # exist but its weights will train without ever being
                # consumed — dead weights silently inflating the
                # parameter count. Fail loudly so the misconfiguration
                # surfaces at the first forward pass.
                if self.label_tokenizer is not None:
                    raise RuntimeError(
                        "TransformerEncoder received 3-D label_inputs "
                        "(per-target tokens at d_model) AND was built "
                        "with label_input_dim > 0 (scalar-token path). "
                        "Pick one: either set label_input_dim=0 (the "
                        "conditional Transformer convention — see "
                        "ConditionalJointMultiTargetTransformerEstimator)"
                        " or pass 2-D label_inputs to use the scalar "
                        "tokenizer. Mixing both leaves the scalar "
                        "tokenizer training as dead weights."
                    )
                tokens.append(label_inputs)
            elif self.label_tokenizer is not None:
                # Scalar-token fallback — each scalar becomes one token.
                tokens.append(self.label_tokenizer(label_inputs))
            else:
                raise RuntimeError(
                    "label_inputs supplied but encoder has no label_tokenizer "
                    "and label_inputs is not 3-D (per-target tokens). Build "
                    "the encoder with label_input_dim > 0 for the scalar path."
                )
        seq = torch.cat(tokens, dim=1)
        for block in self.blocks:
            seq = block(seq)
        return seq[:, 0, :]  # CLS pooling
