# Conditional joint multi-target Transformer estimator (v3).
#
# Thin subclass over ConditionalJointMultiTargetBaseEstimator that supplies
# a TransformerEncoder. Same architectural surface as
# JointMultiTargetTransformerEstimator plus the label channel.

from __future__ import annotations

from skrec.estimator.classification._conditional_joint_multi_target_base import (
    ConditionalJointMultiTargetBaseEstimator,
)
from skrec.estimator.classification._joint_multi_target_base import (
    JointMultiTargetEncoder,
)
from skrec.estimator.classification._joint_multi_target_encoders import (
    TransformerEncoder,
)


class ConditionalJointMultiTargetTransformerEstimator(ConditionalJointMultiTargetBaseEstimator):
    """v3 conditional joint Transformer — accepts ``OBSERVED_*`` at inference.

    See :class:`ConditionalJointMultiTargetMLPEstimator` for the conditional
    contract; this class swaps the MLP encoder for FT-Transformer-style
    feature tokenization + CLS pooling.
    """

    DEFAULT_PARAMS = {
        **ConditionalJointMultiTargetBaseEstimator.DEFAULT_PARAMS,
        # Transformer-specific (mirrors JointMultiTargetTransformerEstimator).
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "ffn_dim": 128,
        "attn_dropout": 0.1,
        "ffn_dropout": 0.1,
        # Plan risk #3 mitigations — same Transformer stability knobs as the
        # vanilla joint Transformer (LR warmup + gradient clipping).
        "warmup_steps": 100,
        "grad_clip_norm": 1.0,
    }

    # --- Label-channel shape overrides for FT-Transformer-style tokens ---
    #
    # The base's default plumbing flat-concats per-target projections into a
    # (batch, num_targets * label_embedding_dim) vector — fine for MLP but
    # wrong for the Transformer, where each declared target should be ONE
    # token at d_model width, appended to the feature-token sequence. The
    # plan's FT-Transformer-style "feature-token + per-target label-token"
    # architecture lives here; without the overrides the plumbing falls
    # through to scalar tokens (one token per scalar dim) and inflates
    # the sequence to num_targets * label_embedding_dim extra tokens
    # instead of num_targets.

    def _label_token_dim(self) -> int:
        # Each per-target label token is d_model-wide so it slots directly
        # into the Transformer's token sequence at the right embedding dim.
        return int(self.params["d_model"])

    def _encoder_label_input_dim(self) -> int:
        # Per-target tokens are appended in forward via the 3-D label_inputs
        # path; the encoder's scalar-token tokenizer is not used.
        return 0

    def _format_label_inputs(self, raw_chunks):
        # Per-target token stack: (batch, num_targets, d_model). The
        # TransformerEncoder's forward detects the 3-D shape and appends
        # the tokens directly to the feature-token sequence.
        return self._label_encoder.encode_per_target(raw_chunks)

    def _build_encoder(self, input_dim: int, label_input_dim: int = 0) -> JointMultiTargetEncoder:
        return TransformerEncoder(
            n_features=input_dim,
            label_input_dim=label_input_dim,
            d_model=int(self.params["d_model"]),
            n_heads=int(self.params["n_heads"]),
            n_layers=int(self.params["n_layers"]),
            ffn_dim=int(self.params["ffn_dim"]),
            attn_dropout=float(self.params["attn_dropout"]),
            ffn_dropout=float(self.params["ffn_dropout"]),
        )
