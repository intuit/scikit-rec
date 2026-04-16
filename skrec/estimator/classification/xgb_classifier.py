import copy
from typing import Dict, Optional, Union

import numpy as np
import xgboost
from numpy.typing import NDArray
from pandas import DataFrame, Series
from xgboost import XGBClassifier

from skrec.constants import ITEM_ID_NAME
from skrec.dataset.batch_training_dataset import BatchTrainingDataset
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.datatypes import HPOType
from skrec.estimator.numpy_predictor import NumpyPredictorMixin
from skrec.estimator.tuned_estimator import TunedEstimator
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class XGBClassifierEstimator(NumpyPredictorMixin, BaseClassifier):
    def __init__(self, params: Optional[dict] = None):
        params = params or {}
        params.setdefault("base_score", 0.5)
        self._model = XGBClassifier(**params)
        self._use_inplace_predict = True

    def set_inplace_predict(self, value: bool):
        self._use_inplace_predict = value

    def _fit_model(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ):
        if X_valid is not None:
            self._model.fit(X, y, eval_set=[(X_valid, y_valid)])
        else:
            self._model.fit(X, y)

    def _predict_proba_model(
        self,
        X: Union[DataFrame, NDArray],
    ) -> NDArray:
        # Dataframe is very slow. Convert to numpy array
        X_np = X.to_numpy() if isinstance(X, DataFrame) else X
        if (
            self._use_inplace_predict
            and hasattr(self._model, "get_booster")
            and callable(getattr(self._model, "get_booster"))
        ):
            pred = self._model.get_booster().inplace_predict(X_np, validate_features=False)
            # We don't need to validate features because it has already happened.
            return np.column_stack((1 - pred, pred))
        else:
            return self._model.predict_proba(X_np)


class BatchXGBClassifierEstimator(BaseClassifier):
    def __init__(self, params: Optional[dict] = None, epochs=1):
        params = params or {}
        self.epochs = epochs
        self.params = params

    def _fit_model(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ):
        raise RuntimeError(
            "BatchXGBClassifierEstimator does not support single-pass fit(). "
            "Use batch_train_model() via BaseRecommender.train() with a partitioned dataset."
        )

    def _batch_fit_model(
        self, train_data_iterator: BatchTrainingDataset, valid_data_iterator: Optional[BatchTrainingDataset] = None
    ) -> None:
        kwargs = {}
        kwargs["dtrain"] = xgboost.DMatrix(train_data_iterator)
        self._dmatrix_col_names = kwargs["dtrain"].feature_names

        if valid_data_iterator is not None:
            valid = xgboost.DMatrix(valid_data_iterator)
            kwargs["evals"] = [(valid, "validation")]  # type: ignore

        params = copy.deepcopy(self.params)

        if "tree_method" not in params:
            # https://xgboost.readthedocs.io/en/latest/tutorials/external_memory.html#using-xgboost-external-memory-version
            params["tree_method"] = "hist"
        if params["tree_method"] not in {"hist", "approx"}:
            raise RuntimeError("Only hist or approx tree methods are supported for batch training.")

        # https://stackoverflow.com/questions/48051749/what-is-the-difference-between-num-boost-round-and-n-estimators
        if "n_estimators" in params:
            kwargs["num_boost_round"] = params.pop("n_estimators", 100)

        move_parameter_names = ["num_boost_round", "maximize", "early_stopping_rounds"]
        for name in move_parameter_names:
            if name in params:
                kwargs[name] = params.pop(name)
        kwargs["params"] = params  # type: ignore

        self._model = xgboost.train(**kwargs)

    def _predict_proba_model(self, X: DataFrame) -> NDArray:
        X = xgboost.DMatrix(X, feature_names=self._dmatrix_col_names)  # type: ignore
        class1 = self._model.predict(X)  # type: ignore
        class0 = 1 - class1
        return np.column_stack([class0, class1])


class TunedXGBClassifierEstimator(TunedEstimator, XGBClassifierEstimator):
    def __init__(self, hpo_method: HPOType, param_space: dict, optimizer_params: dict, base_score: float = 0.5):
        super().__init__(XGBClassifier, hpo_method, param_space, optimizer_params, {"base_score": base_score})
        self._use_inplace_predict = True


class WeightedXGBClassifierEstimator(XGBClassifierEstimator):
    def __init__(
        self,
        params: Optional[dict] = None,
        action_weight: float = 1,
        item_sample_weights: Optional[Dict[str, float]] = None,
    ):
        params = params or {}

        if action_weight != 1 and ("colsample_bynode" not in params or params["colsample_bynode"] == 1):
            raise ValueError("Action weighting requires colsample_bynode < 1")

        if action_weight == 1 and not item_sample_weights:
            logger.warning("No custom weights are being used, so this will act like a normal XGBClassifierEstimator")

        super().__init__(params)
        self.action_weight = action_weight
        self.item_sample_weights = item_sample_weights
        self._use_inplace_predict = True

    def _fit_model(
        self,
        X: DataFrame,
        y: Union[NDArray, Series],
        X_valid: Optional[DataFrame] = None,
        y_valid: Optional[Union[NDArray, Series]] = None,
    ):
        feature_weights = np.ones(X.shape[1])
        sample_weights = np.ones(X.shape[0])

        if self.action_weight != 1:
            logger.info(f"Feature weighting is being used with action_weight = {self.action_weight}")
            names = np.array(self.feature_names).astype(str)
            action_idx = np.where(np.char.startswith(names, ITEM_ID_NAME + "="))[0]
            if action_idx.size == 0:
                raise ValueError(f"No action columns found matching item id name {ITEM_ID_NAME}")

            feature_weights[action_idx] = self.action_weight

        if self.item_sample_weights:
            logger.info("Sample weighting is being used")
            for item, weight in self.item_sample_weights.items():
                col_idx = np.array(self.feature_names) == f"{ITEM_ID_NAME}={item}"
                if not col_idx.any():
                    raise ValueError(f"No column found with item named {item}")
                sample_idx = (X.values[:, col_idx] == 1).flatten()
                sample_weights[sample_idx] = weight

        if X_valid is not None:
            self._model.fit(
                X, y, feature_weights=feature_weights, sample_weight=sample_weights, eval_set=[(X_valid, y_valid)]
            )
        else:
            self._model.fit(X, y, feature_weights=feature_weights, sample_weight=sample_weights)
