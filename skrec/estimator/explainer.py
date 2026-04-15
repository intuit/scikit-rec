from numpy.typing import NDArray

from skrec.estimator.base_estimator import BaseEstimator
from skrec.estimator.constants import TREE_EXPLAINERS_ALLOWED_ESTIMATORS
from skrec.estimator.tuned_estimator import TunedEstimator


class Explainer:
    def __init__(self, base_estimator: BaseEstimator):
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "shap is required for Explainer. Install it with: pip install scikit-rec[explain]"
            ) from exc

        self.explainer = None
        if isinstance(base_estimator, tuple(TREE_EXPLAINERS_ALLOWED_ESTIMATORS)):
            if isinstance(base_estimator, TunedEstimator):
                self.explainer = shap.TreeExplainer(
                    base_estimator._model.best_estimator_, feature_names=base_estimator.feature_names
                )
            else:
                self.explainer = shap.TreeExplainer(base_estimator._model, feature_names=base_estimator.feature_names)

        if self.explainer is None:
            raise NotImplementedError(f"Explainer not implemented for {type(base_estimator)}")

    def get_explanation(self, X: NDArray):
        explanation = self.explainer(X)
        return explanation
