# Single joint XGBoost booster over N continuous targets, behind MultioutputScorer
# (regressor mode). The regressor analogue of
# skrec.estimator.classification.joint_xgb_multioutput.JointXGBMultiOutputClassifierEstimator.
#
# Even thinner than the classifier: MultioutputScorer regressor mode consumes the
# estimator's predict() output directly (np.asarray(predict(...)) -> _create_value_df,
# which accepts an (n, N) array), and a multi-output XGBRegressor produces (n, N)
# natively. So no reshape is needed — this class only pins the multi_strategy
# default and carries the same cross-label / CPU-only guidance as the classifier.
#
# Cross-label note (identical to the classifier): 'one_output_per_tree' (default)
# does NOT share structure across targets; 'multi_output_tree' (vector leaf) does,
# but is CPU-only. See the classifier module docstring for the full explanation.

from typing import Optional

from skrec.estimator._fit_params_mixin import SampleWeightStrategy
from skrec.estimator.regression.xgb_regressor import XGBRegressorEstimator
from skrec.util.logger import get_logger

logger = get_logger(__name__)

_GPU_DEVICE_TOKENS = ("cuda", "gpu")


def _requests_gpu(params: dict) -> bool:
    device = str(params.get("device", "")).lower()
    tree_method = str(params.get("tree_method", "")).lower()
    return any(tok in device for tok in _GPU_DEVICE_TOKENS) or "gpu" in tree_method


class JointXGBMultiOutputRegressorEstimator(XGBRegressorEstimator):
    """A single joint XGBoost booster over N continuous targets, for
    ``MultioutputScorer`` in regressor mode.

    Trains one ``XGBRegressor`` on a 2-D ``y`` (shape ``(n, N)``); ``predict``
    returns ``(n, N)``, which ``MultioutputScorer`` regressor mode already
    accepts — no reshape needed (unlike the classifier).

    ``multi_strategy`` (in ``params``) controls cross-target learning:

    - ``'one_output_per_tree'`` (default): separate trees per target in one
      jointly-boosted model; GPU-capable; **no** cross-target structure sharing.
    - ``'multi_output_tree'``: shared vector-leaf splits across targets;
      **genuine cross-target learning**; CPU-only.

    Inherits the fit-time passthrough (``fit_params`` / ``sample_weight``) from
    :class:`XGBRegressorEstimator`.
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        fit_params: Optional[dict] = None,
        sample_weight: SampleWeightStrategy = None,
    ):
        params = dict(params or {})
        params.setdefault("multi_strategy", "one_output_per_tree")

        if params["multi_strategy"] == "multi_output_tree":
            logger.info(
                "JointXGBMultiOutputRegressorEstimator: multi_strategy='multi_output_tree' "
                "(vector leaf) is active — targets share tree structure (cross-target "
                "learning). Note: vector-leaf trees are CPU-only in XGBoost."
            )
            if _requests_gpu(params):
                logger.warning(
                    "multi_strategy='multi_output_tree' was requested together with a GPU "
                    "device/tree_method (%r / %r). XGBoost cannot build vector-leaf trees on "
                    "GPU; this will error or silently fall back. Use tree_method='hist', "
                    "device='cpu' for cross-target learning, or multi_strategy="
                    "'one_output_per_tree' to keep the GPU.",
                    params.get("device"),
                    params.get("tree_method"),
                )

        super().__init__(params=params, fit_params=fit_params, sample_weight=sample_weight)

    # _fit_model and _predict_model are inherited unchanged: XGBRegressor.fit
    # accepts 2-D y (fit_kwargs flow through), and predict returns (n, N) — the
    # exact shape MultioutputScorer regressor mode expects.
