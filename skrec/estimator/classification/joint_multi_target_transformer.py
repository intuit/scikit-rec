# Joint multi-target Transformer estimator (FT-Transformer-style).
#
# Thin subclass over JointMultiTargetBaseEstimator that supplies a
# TransformerEncoder. All training / prediction logic lives in the base
# class — this file is the family entry point and the seam where v3's
# conditional Transformer subclass extends.

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from skrec.estimator.classification._joint_multi_target_base import (
    JointMultiTargetBaseEstimator,
    JointMultiTargetEncoder,
)
from skrec.estimator.classification._joint_multi_target_encoders import (
    TransformerEncoder,
)
from skrec.estimator.classification._multi_target_protocol import (
    TargetGroupSpec,
    TargetType,
)


class JointMultiTargetTransformerEstimator(JointMultiTargetBaseEstimator):
    """Joint multi-target estimator with an FT-Transformer-style encoder.

    Each feature is tokenized into a ``d_model`` vector; a CLS token pools
    the sequence after ``n_layers`` Transformer blocks. Suited to tabular
    data where pairwise feature interactions matter (FT-Transformer's
    motivation).

    Implements :class:`MultiTargetEstimator`.

    Args:
        target_specs: Per-target schema; must match the scorer's.
        params: Flat dict of hyperparameters. Recognized: ``d_model``,
            ``n_heads``, ``n_layers``, ``ffn_dim``, ``attn_dropout``,
            ``ffn_dropout``, ``warmup_steps`` (reserved for scheduler), plus
            base ``batch_size``, ``epochs``, ``lr``, ``weight_decay``,
            ``regression_normalize``, ``device``, ``seed``.
    """

    DEFAULT_PARAMS = {
        **JointMultiTargetBaseEstimator.DEFAULT_PARAMS,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "ffn_dim": 128,
        "attn_dropout": 0.1,
        "ffn_dropout": 0.1,
        # Plan risk #3 mitigations. Warmup smooths the early-step learning-
        # rate ramp (Transformers are sensitive to large LR at step 0);
        # gradient clipping caps NaN-gradient blow-ups on the attention path.
        "warmup_steps": 100,
        "grad_clip_norm": 1.0,
    }

    def __init__(
        self,
        target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(target_specs=target_specs, params=params)

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
