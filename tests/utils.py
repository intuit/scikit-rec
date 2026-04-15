import os
import shutil
from typing import Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from skrec.estimator.base_estimator import BaseEstimator
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.scorer.base_scorer import BaseScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


def copy_files_to_folder(src_folder, destination_folder):
    for root, _, files in os.walk((os.path.normpath(src_folder)), topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            shutil.copy2(file_path, destination_folder)


def parse_config(config: dict, key: str):
    logreg_params = config[key]
    hpo_method = logreg_params["hyperparameter_tuning"]["method"]
    optimizer_params = logreg_params["hyperparameter_tuning"]["cv_params"]
    del logreg_params["hyperparameter_tuning"]
    first_params = {k: v[0] for k, v in logreg_params.items()}

    return logreg_params, hpo_method, optimizer_params, first_params


def parse_propensity_data_config(config: dict, key: str):
    filename_params = config[key]
    users_filename = filename_params["users_dataset"]
    items_filename = filename_params["items_dataset"]
    interactions_filenames = filename_params["interactions_dataset"]
    return users_filename, items_filename, interactions_filenames


class MockEstimator(BaseEstimator):
    def __init__(self):
        pass

    def _fit_model(self, X: DataFrame, y: DataFrame):
        pass

    def predict(self, X: DataFrame) -> NDArray:
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        return X.sum(axis=1)


class MockClassifier(BaseClassifier):
    def __init__(self):
        pass

    def _fit_model(self, X: DataFrame, y: DataFrame):
        pass

    def _predict_proba_model(self, X: Union[NDArray, DataFrame]) -> NDArray:
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        result0 = X.sum(axis=1)
        result1 = X.mean(axis=1)
        return np.vstack((result0, result1)).T


class MockClassifier_v2(MockClassifier):
    def __init__(self):
        pass

    def _predict_proba_model(self, X: Union[NDArray, DataFrame]) -> NDArray:
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        result0 = X.mean(axis=1)
        result1 = X.sum(axis=1)
        return np.vstack((result0, result1)).T


class MockScorerWithoutCalculateScores(BaseScorer):
    pass


class MockScorer(BaseScorer):
    def _calculate_scores(self, joined: DataFrame, item_subset: list = None) -> NDArray:
        scores = self.estimator.predict(joined)
        if item_subset is not None:
            relevant_indices = self.get_item_indices(item_subset)
            scores = scores[:, relevant_indices]

        self.item_names = [f"col_{i}" for i in range(0, len(scores))]
        return scores.reshape(1, -1)
