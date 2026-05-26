# Joint multi-target MLP estimator.
#
# Thin subclass over JointMultiTargetBaseEstimator that supplies an
# MLPEncoder. All training / prediction logic lives in the base class —
# this file is the family entry point and the seam where v3's conditional
# MLP subclass extends.

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from skrec.estimator.classification._joint_multi_target_base import (
    JointMultiTargetBaseEstimator,
    JointMultiTargetEncoder,
)
from skrec.estimator.classification._joint_multi_target_encoders import MLPEncoder
from skrec.estimator.classification._multi_target_protocol import (
    TargetGroupSpec,
    TargetType,
)


class JointMultiTargetMLPEstimator(JointMultiTargetBaseEstimator):
    """Joint multi-target estimator with a shared MLP encoder.

    Trains one MLP feature encoder + per-target heads jointly via summed
    per-type losses. Suited to feature-rich tabular data with moderate
    target counts.

    Implements :class:`MultiTargetEstimator`.

    Args:
        target_specs: Per-target schema; must match the scorer's.
        params: Flat dict of hyperparameters overlaying ``DEFAULT_PARAMS``.
            Recognized keys: ``hidden_dim``, ``num_layers``, ``dropout``,
            ``use_batch_norm``, plus the base ``batch_size``, ``epochs``,
            ``lr``, ``weight_decay``, ``regression_normalize``, ``device``,
            ``seed``.
    """

    DEFAULT_PARAMS = {
        **JointMultiTargetBaseEstimator.DEFAULT_PARAMS,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
        "use_batch_norm": False,
    }

    def __init__(
        self,
        target_specs: Dict[str, Union[TargetType, TargetGroupSpec]],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(target_specs=target_specs, params=params)

    def _build_encoder(self, input_dim: int, label_input_dim: int = 0) -> JointMultiTargetEncoder:
        return MLPEncoder(
            n_features=input_dim,
            label_input_dim=label_input_dim,
            hidden_dim=int(self.params["hidden_dim"]),
            num_layers=int(self.params["num_layers"]),
            dropout=float(self.params["dropout"]),
            use_batch_norm=bool(self.params["use_batch_norm"]),
        )
