from typing import Dict, Optional

from lightgbm import LGBMClassifier
from pandas import DataFrame

from skrec.estimator._fit_params_mixin import SampleWeightStrategy
from skrec.estimator.classification.sklearn_universal_classifier import (
    SklearnUniversalClassifierEstimator,
    TunedSklearnUniversalClassifierEstimator,
)
from skrec.estimator.datatypes import HPOType


class LightGBMClassifierEstimator(SklearnUniversalClassifierEstimator):
    def __init__(
        self,
        params: Optional[dict] = None,
        train_params: Optional[dict] = None,
        fit_params: Optional[dict] = None,
        sample_weight: SampleWeightStrategy = None,
    ):
        params = params or {}
        self._model = LGBMClassifier(**params)
        # train_params is the legacy LightGBM-specific static-kwargs dict; kept
        # for back-compat (set_training_params). The generic fit_params + the
        # sample_weight strategy come from the mixin and merge at fit time.
        self._train_params: Dict = train_params or {}
        self._init_fit_params(fit_params, sample_weight)

    def _fit_model(
        self, X: DataFrame, y: DataFrame, X_valid: Optional[DataFrame] = None, y_valid: Optional[DataFrame] = None
    ):
        # LightGBM weights the train set only in v1: its eval-set weight kwarg is
        # `eval_sample_weight` (a list), not XGBoost's `sample_weight_eval_set`,
        # so we resolve without eval weights. train_params (legacy) is merged
        # with fit_params + the resolved sample_weight (the latter win on key
        # collisions).
        fit_kw = {**self._train_params, **self._resolve_fit_kwargs(X, y)}
        if X_valid is not None and y_valid is not None:
            self._model.fit(X, y, eval_set=[(X_valid, y_valid)], **fit_kw)
        else:
            self._model.fit(X, y, **fit_kw)

    def set_training_params(self, train_params: dict):
        self._train_params = train_params


class TunedLightGBMClassifierEstimator(TunedSklearnUniversalClassifierEstimator, LightGBMClassifierEstimator):
    def __init__(
        self,
        hpo_method: HPOType,
        param_space: dict,
        optimizer_params: dict,
        fit_params: Optional[dict] = None,
        sample_weight: SampleWeightStrategy = None,
    ):
        self._train_params: Dict = {}
        super().__init__(
            LGBMClassifier,
            hpo_method,
            param_space,
            optimizer_params,
            fit_params=fit_params,
            sample_weight=sample_weight,
        )
