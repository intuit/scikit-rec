import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from xgboost import XGBClassifier

from skrec.estimator.classification.multioutput_classifier import (
    MultiOutputClassifierEstimator,
)
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.estimator.regression.xgb_regressor import XGBRegressorEstimator
from skrec.util.config_loader import load_config
from skrec.util.logger import get_logger
from tests.utils import parse_config

logger = get_logger(__name__)


@pytest.fixture
def setup_fixture(setup_small_datasets):
    files_path = Path.cwd() / "skrec/examples/estimators"
    reward_model_df = pd.read_csv(files_path / "reward_model_data_classification.csv")
    estimator_config = load_config(files_path / "estimator_hyperparameters.yaml")
    _, _, _, setup_small_datasets["xgb_params"] = parse_config(estimator_config, "XGBoostClassifier")
    _, _, _, setup_small_datasets["logreg_params"] = parse_config(estimator_config, "LogisticRegression")
    setup_small_datasets["y"] = reward_model_df["y"].to_numpy()
    setup_small_datasets["y_multi"] = reward_model_df[["y", "y"]].to_numpy()
    setup_small_datasets["X"] = reward_model_df.drop(columns=["y"])
    return setup_small_datasets


def test_multioutput_estimator(setup_fixture):
    multioutput_classifier_estimator = MultiOutputClassifierEstimator(XGBClassifier, setup_fixture["xgb_params"])
    with pytest.raises(ValueError):
        multioutput_classifier_estimator.fit(setup_fixture["X"], setup_fixture["y"])
    multioutput_classifier_estimator.fit(setup_fixture["X"], setup_fixture["y_multi"])


def test_estimator_attributes_are_equal(setup_fixture, caplog):
    xgb_classifier_estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])
    xgb_classifier_estimator_copy = deepcopy(xgb_classifier_estimator)
    logreg_regressor_estimator = XGBRegressorEstimator(setup_fixture["logreg_params"])
    xgb_classifier_sklearn = XGBClassifier()

    for estimator in [
        xgb_classifier_estimator,
        logreg_regressor_estimator,
        xgb_classifier_sklearn,
    ]:
        estimator.fit(setup_fixture["X"], setup_fixture["y"])
    # The estimator is not a BaseEstimator.
    expected_msg = "The estimator to compare is not a BaseEstimator."
    with pytest.raises(TypeError, match=expected_msg):
        attr_equal = xgb_classifier_estimator.estimator_attributes_are_equal(
            xgb_classifier_sklearn, ["feature_importances_"]
        )
    # Some attribute within the list of attributes are not available for the estimators.
    expected_msg = "Estimator does not have the attribute abc"
    with pytest.raises(AttributeError, match=expected_msg):
        attr_equal = xgb_classifier_estimator.estimator_attributes_are_equal(
            logreg_regressor_estimator, ["n_features_in_", "abc"]
        )
    # Different types of estimators
    expected_msg = (
        "Values of the n_features_in_ are equal for these estimators. "
        "This might be caused by insufficient amount of data (attributes equal to the initial values), "
        "models trained on the same dataset or the same model retrained on multiple datasets."
    )
    caplog.set_level(logging.WARNING, logger="skrec.estimator.base_estimator")
    with caplog.at_level(logging.WARNING, logger="skrec.estimator.base_estimator"):
        attr_equal = xgb_classifier_estimator.estimator_attributes_are_equal(
            logreg_regressor_estimator, ["n_features_in_"]
        )
    assert expected_msg in caplog.text
    assert attr_equal == [True]
    # Same types of estimators trained on different datasets
    expected_msg = (
        "Values of the feature_importances_ are different for these estimators. Finished comparing these estimators."
    )
    array_start = int(len(setup_fixture["X"]) / 2)
    xgb_classifier_estimator_copy.fit(setup_fixture["X"].iloc[array_start:, :], setup_fixture["y"][array_start:])
    caplog.set_level(logging.INFO, logger="skrec.estimator.base_estimator")
    with caplog.at_level(logging.INFO, logger="skrec.estimator.base_estimator"):
        attr_equal = xgb_classifier_estimator.estimator_attributes_are_equal(
            xgb_classifier_estimator_copy, ["feature_importances_", "n_features_in_"]
        )
    assert expected_msg in caplog.text
    assert attr_equal == [False, True]


def test_process_for_predict(setup_fixture):
    xgb_classifier_estimator = XGBClassifierEstimator(setup_fixture["xgb_params"])

    expected_msg = "Estimator did not store column names during training"
    with pytest.raises(AttributeError, match=expected_msg):
        xgb_classifier_estimator._process_for_predict(setup_fixture["X"])

    xgb_classifier_estimator.fit(setup_fixture["X"], setup_fixture["y"])

    expected_msg = r"\['x1'\] not in index"
    with pytest.raises(KeyError, match=expected_msg):
        xgb_classifier_estimator._process_for_predict(setup_fixture["X"].drop(columns=["x1"]))

    xgb_classifier_estimator._process_for_predict(setup_fixture["X"].assign(abc=0))
    assert True

    xgb_classifier_estimator._process_for_predict(setup_fixture["X"][setup_fixture["X"].columns[::-1]])
    assert True

    X_correct = setup_fixture["X"].values
    X = xgb_classifier_estimator._process_for_predict(setup_fixture["X"])
    assert (X == X_correct).all()
