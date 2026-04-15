from skrec.estimator.classification.xgb_classifier import (
    TunedXGBClassifierEstimator,
    WeightedXGBClassifierEstimator,
    XGBClassifierEstimator,
)
from skrec.estimator.regression.xgb_regressor import (
    TunedXGBRegressorEstimator,
    XGBRegressorEstimator,
)

TREE_EXPLAINERS_ALLOWED_ESTIMATORS = [
    TunedXGBClassifierEstimator,
    WeightedXGBClassifierEstimator,
    XGBClassifierEstimator,
    TunedXGBRegressorEstimator,
    XGBRegressorEstimator,
]
