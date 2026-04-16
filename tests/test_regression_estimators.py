from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from skrec.estimator.datatypes import HPOType
from skrec.estimator.explainer import Explainer
from skrec.estimator.regression.lightgbm_regressor import (
    LightGBMRegressorEstimator,
    TunedLightGBMRegressorEstimator,
)
from skrec.estimator.regression.xgb_regressor import (
    TunedXGBRegressorEstimator,
    XGBRegressorEstimator,
)
from skrec.estimator.tuned_estimator import TunedEstimator
from skrec.util.config_loader import load_config
from skrec.util.logger import get_logger
from tests.utils import parse_config

logger = get_logger(__name__)


@pytest.fixture
def setup_fixture():
    test_data = {}
    files_path = Path.cwd() / "skrec/examples/estimators"

    test_data["reward_model_df"] = pd.read_csv(files_path / "reward_model_data_regression.csv")
    test_data["estimator_config"] = load_config(files_path / "estimator_hyperparameters.yaml")

    test_data["reward_model_y"] = test_data["reward_model_df"]["y"].to_numpy()
    test_data["reward_model_x"] = test_data["reward_model_df"].drop(columns=["y"])
    test_data["x_column_names"] = test_data["reward_model_df"].drop(columns=["y"]).columns.tolist()
    return test_data


def test_xgb_with_none_input(setup_fixture):
    first_params = parse_config(setup_fixture["estimator_config"], "XGBoostRegressor")
    first_params = first_params[3]
    estimator = XGBRegressorEstimator(first_params)
    estimator.fit(setup_fixture["reward_model_x"], setup_fixture["reward_model_y"])
    x_with_none = setup_fixture["reward_model_x"].copy()
    x_with_none.iloc[0, 0] = None
    x_with_none = x_with_none.replace({np.nan: None})
    estimator.predict(x_with_none)
    print("done")


def test_xgb_reward_model(setup_fixture):
    xgb_params, hpo_method, optimizer_params, first_params = parse_config(
        setup_fixture["estimator_config"], "XGBoostRegressor"
    )
    # No HPO
    estimator = XGBRegressorEstimator(first_params)
    estimator.fit(setup_fixture["reward_model_x"], setup_fixture["reward_model_y"])
    estimator.predict(setup_fixture["reward_model_x"])

    try:
        explainer = Explainer(estimator)
        explanation = explainer.get_explanation(setup_fixture["reward_model_x"])
        assert explanation.values.shape == setup_fixture["reward_model_x"].shape
    except ImportError:
        pass  # shap not installed — skip explainer assertions

    # With HPO
    estimator = TunedXGBRegressorEstimator(hpo_method, xgb_params, optimizer_params)
    estimator.fit(setup_fixture["reward_model_x"], setup_fixture["reward_model_y"])
    estimator.predict(setup_fixture["reward_model_x"])

    try:
        explainer = Explainer(estimator)
        explanation = explainer.get_explanation(setup_fixture["reward_model_x"])
        assert explanation.values.shape == setup_fixture["reward_model_x"].shape
    except ImportError:
        pass  # shap not installed — skip explainer assertions


def test_estimators_with_validation_set(setup_fixture):
    xgb_params, hpo_method, optimizer_params, first_params = parse_config(
        setup_fixture["estimator_config"], "XGBoostRegressor"
    )
    X, y = setup_fixture["reward_model_x"], setup_fixture["reward_model_y"]
    X_valid = X[1:100]
    y_valid = y[1:100]
    estimator = XGBRegressorEstimator(first_params)
    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    estimator.predict(X)

    estimator = TunedXGBRegressorEstimator(hpo_method, xgb_params, optimizer_params)
    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    estimator.predict(X)

    first_params["early_stopping_rounds"] = 1

    estimator = XGBRegressorEstimator(first_params)
    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    estimator.predict(X)

    estimator = TunedXGBRegressorEstimator(hpo_method, xgb_params, optimizer_params)
    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    estimator.predict(X)


def test_lightgbm_regressor_estimator(setup_fixture):
    X = setup_fixture["reward_model_x"]
    y = setup_fixture["reward_model_y"]
    X_valid = X[1:100]
    y_valid = y[1:100]

    estimator = LightGBMRegressorEstimator()

    # test fit with eval set
    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    estimator.feature_names = X.columns
    ans = estimator.predict(X)
    assert ans.shape == (X.shape[0],)

    # test fit without eval set
    estimator.fit(X, y)
    estimator.feature_names = X.columns
    ans = estimator.predict(X)
    assert ans.shape == (X.shape[0],)

    # test fit with training params and model params
    model_params = {"num_boost_round": 3, "objective": "regression", "early_stopping_rounds": 50}
    training_params = {
        "eval_set": [(X_valid, y_valid)],
        "callbacks": [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=1)],
    }
    estimator = LightGBMRegressorEstimator(model_params)
    estimator.set_training_params(training_params)
    estimator.fit(X, y)
    estimator.feature_names = X.columns
    ans = estimator.predict(X)
    assert ans.shape == (X.shape[0],)


def test_tuned_estimator_standalone_regressor():
    """TunedEstimator can be used directly for regression — not just as a mixin.

    This was previously broken: TunedEstimator inherited BaseEstimator but did not
    implement the abstract _fit_model, so instantiation raised TypeError.
    """
    from sklearn.dummy import DummyRegressor

    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "f2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]})
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    estimator = TunedEstimator(DummyRegressor, HPOType.GRID_SEARCH_CV, {}, {"cv": 2})
    estimator.fit(X, y)

    preds = estimator.predict(X)
    assert preds.shape == (len(X),)


def test_tuned_lightgbm_regressor_estimator(setup_fixture):
    X = setup_fixture["reward_model_x"]
    y = setup_fixture["reward_model_y"]
    X_valid = X[1:]
    y_valid = y[1:]

    hp_method = HPOType.GRID_SEARCH_CV

    hp_params = {
        "num_boost_round": [1, 2, 3],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.2],
    }
    optimizer_params = {"n_jobs": 1, "cv": 3, "verbose": 10}

    # test fit without validation set, no training params
    estimator = TunedLightGBMRegressorEstimator(hp_method, hp_params, optimizer_params)
    estimator.fit(X, y)
    estimator.feature_names = X.columns
    ans = estimator.predict(X)
    assert ans.shape == (X.shape[0],)

    # test fit without validation set, with training params
    training_params = {
        "callbacks": [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=1)],
    }
    estimator = TunedLightGBMRegressorEstimator(hp_method, hp_params, optimizer_params)
    estimator.set_training_params(training_params)

    # test fit with validation set, no training params
    estimator = TunedLightGBMRegressorEstimator(hp_method, hp_params, optimizer_params)
    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    estimator.feature_names = X.columns
    ans = estimator.predict(X)
    assert ans.shape == (X.shape[0],)

    # test fit with validation set, with training params
    estimator = TunedLightGBMRegressorEstimator(hp_method, hp_params, optimizer_params)
    estimator.set_training_params(training_params)
    training_params = {
        "eval_set": [(X_valid, y_valid)],
        "callbacks": [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=1)],
    }
    estimator.set_training_params(training_params)
    estimator.fit(X, y)
    estimator.feature_names = X.columns
    ans = estimator.predict(X)
    assert ans.shape == (X.shape[0],)
