import logging
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from skrec.dataset.batch_training_dataset import BatchTrainingDataset
from skrec.dataset.schema import DatasetSchema
from skrec.estimator.classification.lightgbm_classifier import (
    LightGBMClassifierEstimator,
    TunedLightGBMClassifierEstimator,
)
from skrec.estimator.classification.logreg_classifier import (
    LogisticRegressionClassifierEstimator,
    TunedLogisticRegressionClassifierEstimator,
)
from skrec.estimator.classification.multioutput_classifier import (
    MultiOutputClassifierEstimator,
    TunedMultiOutputClassifierEstimator,
)
from skrec.estimator.classification.sklearn_universal_classifier import (
    SklearnUniversalClassifierEstimator,
    TunedSklearnUniversalClassifierEstimator,
)
from skrec.estimator.classification.xgb_classifier import (
    BatchXGBClassifierEstimator,
    TunedXGBClassifierEstimator,
    WeightedXGBClassifierEstimator,
    XGBClassifierEstimator,
)
from skrec.estimator.datatypes import HPOType
from skrec.estimator.explainer import Explainer
from skrec.estimator.tuned_estimator import TunedEstimator
from skrec.scorer.universal import UniversalScorer
from skrec.util.config_loader import load_config
from skrec.util.logger import get_logger
from tests.utils import parse_config

logger = get_logger(__name__)


@pytest.fixture
def setup_fixture():
    test = {}
    files_path = Path.cwd() / "skrec/examples/estimators/"

    test["multi_class_df"] = pd.read_csv(files_path / "multi_class_data.csv")
    test["reward_model_df"] = pd.read_csv(files_path / "reward_model_data_classification.csv")
    test["personalized_df"] = pd.read_csv(files_path / "generated_personalized_data.csv")
    test["estimator_config"] = load_config(files_path / "estimator_hyperparameters.yaml")

    test["reward_model_y"] = test["reward_model_df"]["y"].to_numpy()
    test["reward_model_x"] = test["reward_model_df"].drop(columns=["y"])

    test["multioutput_x"] = pd.DataFrame(
        {
            "Age": [28, 49, 35, 30, 22, 41, 55, 33, 27, 46, 38, 31],
            "Gender": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    test["multioutput_y"] = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 1],
        ]
    )
    return test


TORCH_INSTALLED = True
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    TORCH_INSTALLED = False

skip_if_torch_not_installed = pytest.mark.skipif(
    not TORCH_INSTALLED, reason="Skipping test because optional dependency torch is not installed"
)


def test_wrong_method_called_for_WeightedXGBClassifierEstimator(setup_fixture):
    X = setup_fixture["personalized_df"].drop("reward", axis=1)
    y = setup_fixture["personalized_df"]["reward"].to_numpy()
    with pytest.raises(Exception):
        params = {"colsample_bynode": 0.1, "max_depth": 2, "importance_type": "weight"}
        estimator = WeightedXGBClassifierEstimator(params=params, action_weight=100)
        estimator._fit_model(X, y)


def test_multioutput_xgboost_reward_model(setup_fixture):
    hpo_params, hpo_method, optimizer_params, first_params = parse_config(
        setup_fixture["estimator_config"], "MultiOutputXGBoostClassifier"
    )
    try:
        # No HPO
        estimator = MultiOutputClassifierEstimator(XGBClassifier, first_params)
        estimator.fit(setup_fixture["multioutput_x"], setup_fixture["multioutput_y"])
        estimator.predict_proba(setup_fixture["multioutput_x"])

        # With HPO
        estimator = TunedMultiOutputClassifierEstimator(XGBClassifier, hpo_method, hpo_params, optimizer_params)
        estimator.fit(setup_fixture["multioutput_x"], setup_fixture["multioutput_y"])
        estimator.predict_proba(setup_fixture["multioutput_x"])

    except Exception:
        pytest.fail("Fit-Predict MultiOutput XGB Reward Model Failed")


def test_multioutput_random_forest_reward_model(setup_fixture):
    hpo_params, hpo_method, optimizer_params, first_params = parse_config(
        setup_fixture["estimator_config"], "MultiOutputRandomForestClassifier"
    )
    try:
        # No HPO
        estimator = MultiOutputClassifierEstimator(RandomForestClassifier, first_params)
        estimator.fit(setup_fixture["multioutput_x"], setup_fixture["multioutput_y"])
        estimator.predict_proba(setup_fixture["multioutput_x"])

        # With HPO
        estimator = TunedMultiOutputClassifierEstimator(
            RandomForestClassifier, hpo_method, hpo_params, optimizer_params
        )
        estimator.fit(setup_fixture["multioutput_x"], setup_fixture["multioutput_y"])
        estimator.predict_proba(setup_fixture["multioutput_x"])

    except Exception:
        pytest.fail("Fit-Predict MultiOutput Random Forest Reward Model Failed")


def test_xgb_reward_model(setup_fixture):
    xgb_params, hpo_method, optimizer_params, first_params = parse_config(
        setup_fixture["estimator_config"], "XGBoostClassifier"
    )
    try:
        # No HPO
        estimator = XGBClassifierEstimator(first_params)
        estimator.fit(setup_fixture["reward_model_x"], setup_fixture["reward_model_y"])
        estimator.predict_proba(setup_fixture["reward_model_x"])

        try:
            explainer = Explainer(estimator)
            explanation = explainer.get_explanation(setup_fixture["reward_model_x"])
            assert explanation.values.shape == setup_fixture["reward_model_x"].shape
        except ImportError:
            pass  # shap not installed — skip explainer assertions

        # With HPO
        estimator = TunedXGBClassifierEstimator(hpo_method, xgb_params, optimizer_params)
        estimator.fit(setup_fixture["reward_model_x"], setup_fixture["reward_model_y"])
        estimator.predict_proba(setup_fixture["reward_model_x"])

        try:
            explainer = Explainer(estimator)
            explanation = explainer.get_explanation(setup_fixture["reward_model_x"])
            assert explanation.values.shape == setup_fixture["reward_model_x"].shape
        except ImportError:
            pass  # shap not installed — skip explainer assertions

    except Exception:
        pytest.fail("Fit Predict XGB Reward Model Failed")


def test_sklearn_universal_reward_model(setup_fixture):
    dummy_params, hpo_method, optimizer_params, first_params = parse_config(
        setup_fixture["estimator_config"], "SklearnDummyClassifier"
    )
    try:
        # No HPO
        estimator = SklearnUniversalClassifierEstimator(DummyClassifier, first_params)
        estimator.fit(setup_fixture["reward_model_x"], setup_fixture["reward_model_y"])
        prediction = estimator.predict_proba(setup_fixture["reward_model_x"])
        assert_array_equal(prediction, [[0.5, 0.5]] * 100)

        # With HPO
        estimator = TunedSklearnUniversalClassifierEstimator(
            DummyClassifier, hpo_method, dummy_params, optimizer_params
        )
        estimator.fit(setup_fixture["reward_model_x"], setup_fixture["reward_model_y"])
        prediction = estimator.predict_proba(setup_fixture["reward_model_x"])
        assert_array_equal(prediction, [[0.5, 0.5]] * 100)

    except Exception:
        pytest.fail("Fit-Predict Dummy Reward Model Failed")


def test_classification_with_one_hot_encoding():
    df = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "feature1": ["A", "B", "A", "C", "B"], "target": [0, 1, 0, 1, 1]})

    schema_with_vocab = {
        "columns": [{"name": "user_id", "type": "int"}, {"name": "feature1", "type": "str", "vocab": ["A", "B", "C"]}]
    }

    schema = DatasetSchema(schema_with_vocab)
    X = schema.apply(df.drop(columns=["target"]))
    y = df["target"]

    try:
        classifier = LogisticRegression()
        classifier.fit(X, y)
        classifier.predict(X)
    except Exception:
        pytest.fail("Fit-Predict Dummy Reward Model Failed")


def test_logreg_reward_model(setup_fixture):
    logreg_params, hpo_method, optimizer_params, first_params = parse_config(
        setup_fixture["estimator_config"], "LogisticRegression"
    )
    try:
        # No HPO
        estimator = LogisticRegressionClassifierEstimator(first_params)
        estimator.fit(setup_fixture["reward_model_x"], setup_fixture["reward_model_y"])
        estimator.predict_proba(setup_fixture["reward_model_x"])

        # With HPO
        estimator = TunedLogisticRegressionClassifierEstimator(hpo_method, logreg_params, optimizer_params)
        estimator.fit(setup_fixture["reward_model_x"], setup_fixture["reward_model_y"])
        estimator.predict_proba(setup_fixture["reward_model_x"])

    except Exception:
        pytest.fail("Fit-Predict LogReg Reward Model Failed")


def test_illformed_x_y(setup_fixture):
    xgb_params = setup_fixture["estimator_config"]["XGBoostClassifier"]

    expected_error = "Ill formatted x and y: Rows Mismatch"
    with pytest.raises(ValueError, match=expected_error):
        wrong_x = setup_fixture["reward_model_x"].iloc[:-2, :]
        my_xgb_classifier = XGBClassifierEstimator(params=xgb_params)
        my_xgb_classifier.fit(X=wrong_x, y=setup_fixture["reward_model_y"])

    expected_error = "Invalid classes inferred from unique values of `y`."
    with pytest.raises(ValueError, match=expected_error):
        wrong_y = setup_fixture["reward_model_x"].iloc[:, :-3]
        my_xgb_classifier = XGBClassifierEstimator(params=xgb_params)
        my_xgb_classifier.fit(X=setup_fixture["reward_model_x"], y=wrong_y)


def test_fit_predict_1_feature_xgb(setup_fixture):
    one_feature_x = pd.DataFrame([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    _, _, _, xgb_params = parse_config(setup_fixture["estimator_config"], "XGBoostClassifier")
    my_xgb_classifier = XGBClassifierEstimator(params=xgb_params)
    my_xgb_classifier.fit(X=one_feature_x, y=y)

    assert my_xgb_classifier.X_has_only_1_feature

    one_feature_x_1D_array_for_predict = pd.DataFrame([1, 2, 3, 4, 5])
    result = my_xgb_classifier._process_for_predict(one_feature_x_1D_array_for_predict)
    expected_reshaped_X = np.array([[1], [2], [3], [4], [5]])
    assert_array_equal(result, expected_reshaped_X)

    try:
        my_xgb_classifier.predict_proba(one_feature_x_1D_array_for_predict)
        logger.info("Finished Testing One-Feature Predict for Multiple Rows")
        my_xgb_classifier.predict_proba(pd.DataFrame([1]))
        logger.info("Finished Testing One-Feature Predict for One Row")
    except Exception:
        pytest.fail("Test Fit-Predict 1 Feature Predict-Proba Failed")


def test_multi_class(setup_fixture):
    y = setup_fixture["multi_class_df"]["y"].to_numpy()
    le = LabelEncoder()
    y_train = le.fit_transform(y)

    x = setup_fixture["multi_class_df"].drop(columns=["y"])

    xgb_params, hpo_method, optimizer_params, first_params = parse_config(
        setup_fixture["estimator_config"], "XGBoostClassifier"
    )
    try:
        # No HPO
        estimator = XGBClassifierEstimator(first_params)
        estimator.fit(x, y_train)
        estimator.predict_proba(x)

        # With HPO
        estimator = TunedXGBClassifierEstimator(hpo_method, xgb_params, optimizer_params)
        estimator.fit(x, y_train)
        estimator.predict_proba(x)

    except Exception:
        pytest.fail("Multi Class XGB Fit-Predict Failed")

    logreg_params, hpo_method, optimizer_params, first_params = parse_config(
        setup_fixture["estimator_config"], "SklearnDummyClassifier"
    )

    try:
        # No HPO
        estimator = SklearnUniversalClassifierEstimator(DummyClassifier, first_params)
        estimator.fit(x, y_train)
        estimator.predict_proba(x)

        # With HPO
        estimator = TunedSklearnUniversalClassifierEstimator(
            DummyClassifier, hpo_method, logreg_params, optimizer_params
        )
        estimator.fit(x, y_train)
        estimator.predict_proba(x)

    except Exception:
        pytest.fail("Multi Class Sklearn Fit-Predict Failed")

    logreg_params, hpo_method, optimizer_params, first_params = parse_config(
        setup_fixture["estimator_config"], "LogisticRegression"
    )

    try:
        # No HPO
        estimator = LogisticRegressionClassifierEstimator(first_params)
        estimator.fit(x, y_train)
        estimator.predict_proba(x)

        # With HPO
        estimator = TunedLogisticRegressionClassifierEstimator(hpo_method, logreg_params, optimizer_params)
        estimator.fit(x, y_train)
        estimator.predict_proba(x)

    except Exception:
        pytest.fail("Multi Class LogReg Fit-Predict Failed")


@skip_if_torch_not_installed
def test_fit_predict_1_feature_deep_fm():
    from skrec.estimator.classification.deep_fm_classifier import DeepFMClassifier

    one_feature_x = pd.DataFrame([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    params = {
        "batch_size": one_feature_x.shape[0] // 2,  # error if batch size is larger than training examples
    }

    # can initialize without params
    estimator = DeepFMClassifier()

    estimator = DeepFMClassifier(params)

    estimator = DeepFMClassifier(params)
    estimator.fit(one_feature_x, y)
    estimator.predict_proba(one_feature_x)


@skip_if_torch_not_installed
@pytest.mark.parametrize("l1_reg", [0.0, 0.1])
def test_multi_class_deep_fm(setup_fixture, l1_reg):
    from skrec.estimator.classification.deep_fm_classifier import DeepFMClassifier

    X = setup_fixture["multi_class_df"]
    y = X["y"]
    le = LabelEncoder()
    y_train = le.fit_transform(y)
    X = X.drop(columns=["y"])

    params = {
        "batch_size": X.shape[0] // 2,  # error if batch size is larger than training examples
        "l1_reg": l1_reg,
    }

    estimator = DeepFMClassifier(params)
    estimator.fit(X, y_train)
    estimator.predict_proba(X)


@skip_if_torch_not_installed
def test_deep_fm_with_validation_set(setup_fixture):
    from skrec.estimator.classification.deep_fm_classifier import DeepFMClassifier

    X = setup_fixture["reward_model_x"]
    y = setup_fixture["reward_model_y"]
    X_valid = X[1:]
    y_valid = y[1:]

    params = {
        "batch_size": X.shape[0] // 2,  # error if batch size is larger than training examples
    }

    estimator = DeepFMClassifier(params)

    with pytest.raises(AttributeError, match="Estimator did not store column names during training"):
        _ = estimator.predict_proba(X)

    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    estimator.predict_proba(X)

    # test if y and y_valid are Series
    y = pd.Series(y, index=X.index)
    y_valid = pd.Series(y_valid, index=X_valid.index)
    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)


@skip_if_torch_not_installed
@pytest.mark.parametrize("l1_reg", [0.0, 0.1])
def test_deep_fm__eval_loss(setup_fixture, l1_reg):
    from skrec.estimator.classification.deep_fm_classifier import DeepFMClassifier

    X = setup_fixture["reward_model_x"]
    y = setup_fixture["reward_model_y"]
    X_valid = X[1:]
    y_valid = y[1:]

    params = {
        "batch_size": X.shape[0] // 2,  # error if batch size is larger than training examples
    }

    dtype = torch.float32
    criterion = torch.nn.BCEWithLogitsLoss()

    # init classifier but don't fit
    estimator = DeepFMClassifier(params)

    with pytest.raises(RuntimeError, match="Model is not trained yet"):
        _ = estimator._eval_loss(
            X_val=torch.from_numpy(X_valid.values).to(device="cpu", dtype=dtype),
            y_val=torch.from_numpy(y_valid).to(device="cpu", dtype=dtype).squeeze(-1),
            criterion=criterion,
            batch_size=params["batch_size"],
            l1_reg=l1_reg,
        )

    # after fitting, eval_loss should run
    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    device = next(estimator._model.parameters()).device
    loss = estimator._eval_loss(
        X_val=torch.from_numpy(X_valid.values).to(device=device, dtype=dtype),
        y_val=torch.from_numpy(y_valid).to(device=device, dtype=dtype).squeeze(-1),
        criterion=criterion,
        batch_size=params["batch_size"],
        l1_reg=l1_reg,
    )
    assert loss > 0.0


def test_weighted_xgb_estimator(setup_fixture, caplog):
    caplog.set_level(logging.WARNING, logger="skrec.estimator.classification.xgb_classifier")
    with caplog.at_level(logging.WARNING, logger="skrec.estimator.classification.xgb_classifier"):
        estimator = WeightedXGBClassifierEstimator()
    assert ("WARNING  skrec.estimator.classification.xgb_classifier:xgb_classifier.py") in caplog.text

    assert ("No custom weights are being used, so this will act like a normal XGBClassifierEstimator\n") in caplog.text

    with pytest.raises(ValueError) as cm:
        estimator = WeightedXGBClassifierEstimator(action_weight=20)
    assert "Action weighting requires colsample_bynode < 1" == str(cm.value)

    X = setup_fixture["personalized_df"].drop("reward", axis=1)
    y = setup_fixture["personalized_df"]["reward"].to_numpy()

    with pytest.raises(ValueError) as cm:
        with patch("skrec.estimator.classification.xgb_classifier.ITEM_ID_NAME", "FOO"):
            estimator = WeightedXGBClassifierEstimator(params={"colsample_bynode": 0.1}, action_weight=100)
            estimator.fit(X, y)
    assert "No action columns found matching item id name FOO" == str(cm.value)

    def get_top_feats(estimator, top_k=10):
        feature_importances = estimator._model.feature_importances_
        feature_names = estimator.feature_names
        feature_importance_df = pd.DataFrame({"feature_name": feature_names, "importance": feature_importances})
        feature_importance_df.sort_values(by="importance", ascending=False, inplace=True)
        top_feats = list(feature_importance_df["feature_name"])[0:top_k]
        return top_feats

    estimator = WeightedXGBClassifierEstimator(params={"importance_type": "weight"})
    estimator.fit(X, y)
    top_feats = get_top_feats(estimator)
    assert len(top_feats) == 10

    params = {"colsample_bynode": 0.1, "max_depth": 2, "importance_type": "weight"}
    estimator = WeightedXGBClassifierEstimator(params=params, action_weight=100)
    estimator.fit(X, y)
    top_feats = get_top_feats(estimator)
    expected = np.array(
        [
            "context",
            "ITEM_ID=item_2",
            "ITEM_ID=item_1",
            "user=likes_all",
            "noise_66",
            "ITEM_ID=item_3",
            "noise_62",
            "noise_18",
            "noise_86",
            "noise_65",
        ]
    )
    assert len(top_feats) == 10

    with pytest.raises(ValueError) as cm:
        estimator = WeightedXGBClassifierEstimator(
            params={"importance_type": "weight"}, item_sample_weights={"foo": 100}
        )
        estimator.fit(X, y)
    assert "No column found with item named foo" == str(cm.value)

    estimator = WeightedXGBClassifierEstimator(
        params={"importance_type": "weight"}, item_sample_weights={"item_2": 100}
    )
    estimator.fit(X, y)
    top_feats = get_top_feats(estimator, top_k=1)
    expected = np.array(["ITEM_ID=item_2"])
    assert_array_equal(top_feats, expected)


def test_estimators_with_validation_set(setup_fixture):
    X = setup_fixture["reward_model_x"]
    y = setup_fixture["reward_model_y"]
    X_valid = X[1:]
    y_valid = y[1:]
    try:
        estimator = XGBClassifierEstimator()
        estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
        estimator.predict_proba(X)
    except Exception:
        pytest.fail("XGB classification with validation set and no early stopping Failed")

    try:
        estimator = XGBClassifierEstimator({"early_stopping_rounds": 1})
        estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
        estimator.predict_proba(X)
    except Exception:
        pytest.fail("XGB classification with validation set and early stopping Failed")

    xgb_params, hpo_method, optimizer_params, _ = parse_config(setup_fixture["estimator_config"], "XGBoostClassifier")
    try:
        estimator = TunedXGBClassifierEstimator(hpo_method, xgb_params, optimizer_params)
        estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
        estimator.predict_proba(X)

    except Exception:
        pytest.fail("TunedXGB classification with validation set and early stopping Failed")

    X = setup_fixture["personalized_df"].drop("reward", axis=1)
    y = setup_fixture["personalized_df"]["reward"].to_numpy()
    X_valid = X[1:]
    y_valid = y[1:]
    try:
        estimator = WeightedXGBClassifierEstimator(params={"importance_type": "weight"})
        estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
        estimator.predict_proba(X)
    except Exception:
        pytest.fail("XGB classification with validation set Failed")

    estimator = LogisticRegressionClassifierEstimator({})
    with pytest.warns(UserWarning, match="does not support early stopping"):
        estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    estimator.predict_proba(X)


def test_batch_training(setup_small_datasets):
    interactions_dataset = setup_small_datasets["interactions_dataset"]
    users_dataset = setup_small_datasets["users_dataset"]
    items_dataset = setup_small_datasets["items_dataset"]

    estimator = BatchXGBClassifierEstimator()
    scorer = UniversalScorer(estimator=estimator)

    training_iterator = BatchTrainingDataset(
        scorer=scorer,
        interactions_dataset=interactions_dataset,
        users_dataset=users_dataset,
        items_dataset=items_dataset,
    )
    estimator._batch_fit_model(training_iterator)

    training_iterator.reset()
    X, y = training_iterator.read_data()
    scores = estimator.predict_proba(X)

    assert scores.shape[1] == 2
    assert np.max(scores) <= 1.0
    assert np.min(scores) >= 0.0
    assert np.all(np.isfinite(scores))

    # test with validation set
    validation_iterator = BatchTrainingDataset(
        scorer=scorer,
        interactions_dataset=interactions_dataset,
        users_dataset=users_dataset,
        items_dataset=items_dataset,
    )

    estimator._batch_fit_model(training_iterator, validation_iterator)

    X, y = training_iterator.read_data()
    scores = estimator.predict_proba(X)

    assert scores.shape[1] == 2
    assert np.max(scores) <= 1.0
    assert np.min(scores) >= 0.0
    assert np.all(np.isfinite(scores))

    # test with more than one epoch
    xgb_params = {"early_stopping_rounds": 5, "num_boost_round": 3}
    estimator = BatchXGBClassifierEstimator(xgb_params)
    scorer = UniversalScorer(estimator=estimator)

    validation_iterator = BatchTrainingDataset(
        scorer=scorer,
        interactions_dataset=interactions_dataset,
        users_dataset=users_dataset,
        items_dataset=items_dataset,
    )

    estimator._batch_fit_model(training_iterator, validation_iterator)
    X, y = training_iterator.read_data()
    scores = estimator.predict_proba(X)

    assert scores.shape[1] == 2
    assert np.max(scores) <= 1.0
    assert np.min(scores) >= 0.0
    assert np.all(np.isfinite(scores))


def test_lightgbm_classifier_estimator(setup_fixture):
    X = setup_fixture["reward_model_x"]
    y = setup_fixture["reward_model_y"]
    X_valid = X[1:100]
    y_valid = y[1:100]

    estimator = LightGBMClassifierEstimator()

    # test fit with eval set
    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    estimator.feature_names = X.columns
    ans = estimator.predict_proba(X)
    assert ans.shape == (X.shape[0], 2)

    # test fit without eval set
    estimator.fit(X, y)
    estimator.feature_names = X.columns
    ans = estimator.predict_proba(X)
    assert ans.shape == (X.shape[0], 2)

    # test fit with training params and model params
    model_params = {"num_boost_round": 3, "objective": "binary", "early_stopping_rounds": 50}
    training_params = {
        "eval_set": [(X_valid, y_valid)],
        "callbacks": [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=1)],
    }
    estimator = LightGBMClassifierEstimator(model_params)
    estimator.set_training_params(training_params)
    estimator.fit(X, y)
    estimator.feature_names = X.columns
    ans = estimator.predict_proba(X)
    assert ans.shape == (X.shape[0], 2)


@skip_if_torch_not_installed
def test_deep_fm_parent_validation(setup_fixture):
    """DeepFMClassifier must delegate to parent template methods now that the
    redundant fit() / predict_proba() overrides have been removed.

    Verifies:
    - predict_proba before fit raises AttributeError via BaseClassifier._process_for_predict
    - fit with mismatched X/y row count raises ValueError via BaseEstimator._validate_for_fit
    """
    from skrec.estimator.classification.deep_fm_classifier import DeepFMClassifier

    X = setup_fixture["reward_model_x"]
    y = setup_fixture["reward_model_y"]
    estimator = DeepFMClassifier({"batch_size": X.shape[0] // 2})

    with pytest.raises(AttributeError, match="Estimator did not store column names during training"):
        estimator.predict_proba(X)

    with pytest.raises(ValueError, match="Ill formatted x and y: Rows Mismatch"):
        estimator.fit(X.iloc[:-2], y)


def test_tuned_estimator_standalone_classifier():
    """TunedEstimator can be instantiated and used directly — not just as a mixin.

    This was previously broken: TunedEstimator inherited BaseEstimator but did not
    implement the abstract _fit_model, so instantiation raised TypeError.
    """
    X = pd.DataFrame({"f1": [0, 1, 0, 1, 0, 1], "f2": [1, 0, 1, 0, 1, 0]})
    y = np.array([0, 1, 0, 1, 0, 1])

    estimator = TunedEstimator(DummyClassifier, HPOType.GRID_SEARCH_CV, {}, {"cv": 2})
    estimator.fit(X, y)

    proba = estimator.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_tuned_lightgbm_classifier_estimator(setup_fixture):
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
    estimator = TunedLightGBMClassifierEstimator(hp_method, hp_params, optimizer_params)
    estimator.fit(X, y)
    estimator.feature_names = X.columns
    ans = estimator.predict_proba(X)
    assert ans.shape == (X.shape[0], 2)

    # test fit without validation set, with training params
    training_params = {
        "callbacks": [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=1)],
    }
    estimator = TunedLightGBMClassifierEstimator(hp_method, hp_params, optimizer_params)
    estimator.set_training_params(training_params)

    # test fit with validation set, no training params
    estimator = TunedLightGBMClassifierEstimator(hp_method, hp_params, optimizer_params)
    estimator.fit(X, y, X_valid=X_valid, y_valid=y_valid)
    estimator.feature_names = X.columns
    ans = estimator.predict_proba(X)
    assert ans.shape == (X.shape[0], 2)

    # test fit with validation set, with training params
    estimator = TunedLightGBMClassifierEstimator(hp_method, hp_params, optimizer_params)
    estimator.set_training_params(training_params)
    training_params = {
        "eval_set": [(X_valid, y_valid)],
        "callbacks": [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=1)],
    }
    estimator.set_training_params(training_params)
    estimator.fit(X, y)
    estimator.feature_names = X.columns
    ans = estimator.predict_proba(X)
    assert ans.shape == (X.shape[0], 2)
