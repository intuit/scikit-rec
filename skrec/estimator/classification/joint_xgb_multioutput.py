# Single joint XGBoost booster over N binary labels, behind MultioutputScorer.
#
# Why this exists:
#   A production "joint num_target=N" XGBoost booster (e.g. the intent ranker)
#   trains one model over all N binary labels. sklearn's XGBClassifier produces
#   this natively from a 2-D binary y — but its predict_proba returns an
#   (n, N) array (one P(label=1) column per label), whereas MultioutputScorer's
#   _calculate_scores is written against sklearn.MultiOutputClassifier's
#   list-of-N-(n,2)-blocks layout. This estimator bridges that single shape
#   gap so the joint booster scores through MultioutputScorer with no scorer
#   change. It replaces the per-migration notebook-local adapter.
#
# IMPORTANT — "joint" does NOT automatically mean cross-label learning:
#   XGBoost's ``multi_strategy`` decides whether labels share structure:
#     - 'one_output_per_tree' (DEFAULT here): separate trees per label inside one
#       jointly-boosted model. GPU-capable. Cross-label coupling is limited to
#       shared hyperparameters + the shared boosting loop — modeling-wise this is
#       close to per-label boosting. (The production intent recipe uses this on
#       GPU, so the production "joint" booster is itself NOT doing cross-label
#       learning.)
#     - 'multi_output_tree' (vector leaf): ONE tree with vector leaves; splits are
#       chosen on the summed gradient/hessian across all labels, so labels inform
#       each other's structure. This is genuine cross-label learning — but it is
#       CPU-only in XGBoost.
#   Pass ``params={'multi_strategy': 'multi_output_tree'}`` to opt into cross-label
#   learning; the constructor logs that it is active + CPU-only and warns if a GPU
#   device/tree_method was also requested (XGBoost can't run vector-leaf on GPU).

from typing import Optional

import numpy as np
from pandas import DataFrame

from skrec.estimator._fit_params_mixin import SampleWeightStrategy
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.util.logger import get_logger

logger = get_logger(__name__)

# Substrings that indicate a GPU was requested via `device` / `tree_method`.
_GPU_DEVICE_TOKENS = ("cuda", "gpu")


def _requests_gpu(params: dict) -> bool:
    device = str(params.get("device", "")).lower()
    tree_method = str(params.get("tree_method", "")).lower()
    return any(tok in device for tok in _GPU_DEVICE_TOKENS) or "gpu" in tree_method


class JointXGBMultiOutputClassifierEstimator(XGBClassifierEstimator):
    """A single joint XGBoost booster over N binary labels, for ``MultioutputScorer``.

    Trains one ``XGBClassifier`` on a 2-D binary ``y`` (shape ``(n, N)``) and
    reshapes the ``(n, N)`` ``predict_proba`` output into the list of N
    ``(n, 2)`` blocks that :class:`~skrec.scorer.multioutput.MultioutputScorer`
    expects in classifier mode. Binary-multilabel only (the scorer enforces the
    ``{0, 1}`` target contract).

    ``multi_strategy`` (passed in ``params``) controls cross-label learning:

    - ``'one_output_per_tree'`` (default): separate trees per label in one
      jointly-boosted model; GPU-capable; **no** real cross-label learning.
    - ``'multi_output_tree'``: shared vector-leaf splits across labels;
      **genuine cross-label learning**; CPU-only.

    Inherits the fit-time passthrough (``fit_params`` / ``sample_weight``) from
    :class:`XGBClassifierEstimator`.

    Args:
        params: XGBoost params. ``multi_strategy`` defaults to
            ``'one_output_per_tree'`` if unset.
        fit_params: Static fit kwargs (see ``SklearnFitParamsMixin``).
        sample_weight: Row-weight strategy (``'balanced'`` / callable / array).
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
                "JointXGBMultiOutputClassifierEstimator: multi_strategy='multi_output_tree' "
                "(vector leaf) is active — labels share tree structure, so this enables "
                "genuine cross-label learning. Note: vector-leaf trees are CPU-only in XGBoost."
            )
            if _requests_gpu(params):
                logger.warning(
                    "multi_strategy='multi_output_tree' was requested together with a GPU "
                    "device/tree_method (%r / %r). XGBoost cannot build vector-leaf trees on "
                    "GPU; this will error or silently fall back. Drop the GPU setting (use "
                    "tree_method='hist', device='cpu') for cross-label learning, or switch to "
                    "multi_strategy='one_output_per_tree' to keep the GPU.",
                    params.get("device"),
                    params.get("tree_method"),
                )

        super().__init__(params=params, fit_params=fit_params, sample_weight=sample_weight)

    # _fit_model is inherited from XGBClassifierEstimator: XGBClassifier.fit
    # accepts the 2-D binary y, and the resolved fit_kwargs (sample_weight etc.)
    # flow through unchanged.

    def _predict_proba_model(self, X) -> list:
        """Reshape the joint ``(n, N)`` proba into a list of N ``(n, 2)`` blocks.

        XGBoost multilabel ``predict_proba`` returns ``(n, N)`` where column j is
        ``P(label_j = 1)``. ``MultioutputScorer._calculate_scores`` expects one
        ``(n, 2)`` block per label (``[P(0), P(1)]``), as produced by
        ``sklearn.MultiOutputClassifier``. We override fully — the base class's
        ``inplace_predict`` + ``column_stack`` path assumes a single ``(n,)``
        column and would mis-shape the multi-output array.
        """
        X_np = X.to_numpy() if isinstance(X, DataFrame) else np.asarray(X)
        P = np.asarray(self._model.predict_proba(X_np))
        # Multi-output XGBClassifier (N >= 2 binary labels) always returns
        # (n, N) where column j is P(label_j = 1). MultioutputScorer guarantees
        # >= 2 ITEM_ columns, so this is the only shape we expect. Reshape each
        # column into a (n, 2) [P(0), P(1)] block. (We deliberately do NOT try
        # to special-case an (n, 2) output as a single binary label — with
        # N == 2 labels that shape is genuinely two labels, and conflating the
        # two would silently drop a label.)
        if P.ndim != 2:
            raise RuntimeError(
                f"JointXGBMultiOutputClassifierEstimator expected a 2-D (n, N) "
                f"predict_proba from the joint booster, got shape {P.shape}. This "
                f"estimator is intended for multi-output (>=2 binary labels) use "
                f"with MultioutputScorer."
            )
        return [np.column_stack((1.0 - P[:, j], P[:, j])) for j in range(P.shape[1])]
