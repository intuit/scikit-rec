# Conditional joint multi-target MLP estimator (v3).
#
# Thin subclass over ConditionalJointMultiTargetBaseEstimator that supplies
# an MLPEncoder. Same architectural surface as JointMultiTargetMLPEstimator
# plus the label channel — implements both MultiTargetEstimator AND
# ConditionalMultiTargetEstimator.

from __future__ import annotations

from skrec.estimator.classification._conditional_joint_multi_target_base import (
    ConditionalJointMultiTargetBaseEstimator,
)
from skrec.estimator.classification._joint_multi_target_base import (
    JointMultiTargetEncoder,
)
from skrec.estimator.classification._joint_multi_target_encoders import MLPEncoder


class ConditionalJointMultiTargetMLPEstimator(ConditionalJointMultiTargetBaseEstimator):
    """v3 conditional joint MLP — accepts ``OBSERVED_*`` at inference.

    Implements :class:`ConditionalMultiTargetEstimator` (and its base
    :class:`MultiTargetEstimator`). Inherits ``_build_encoder``-based
    composition from the joint base; conditional masking + label encoder
    plumbing from the conditional base.

    Args:
        target_specs: Per-target schema; must match the scorer's.
        params: Flat dict of hyperparameters overlaying ``DEFAULT_PARAMS``.
            Recognized (in addition to the vanilla MLP params):
              - ``mask_prob`` (float, default 0.5): Bernoulli mask probability
                per (row, target_specs entry) at training. Must be in (0, 1].
              - ``label_embedding_dim`` (int, default 8): width each
                per-target label chunk is projected to before concatenation.
    """

    DEFAULT_PARAMS = {
        **ConditionalJointMultiTargetBaseEstimator.DEFAULT_PARAMS,
        # MLP-specific (mirrors JointMultiTargetMLPEstimator).
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
        "use_batch_norm": False,
    }

    def _build_encoder(self, input_dim: int, label_input_dim: int = 0) -> JointMultiTargetEncoder:
        return MLPEncoder(
            n_features=input_dim,
            label_input_dim=label_input_dim,
            hidden_dim=int(self.params["hidden_dim"]),
            num_layers=int(self.params["num_layers"]),
            dropout=float(self.params["dropout"]),
            use_batch_norm=bool(self.params["use_batch_norm"]),
        )
