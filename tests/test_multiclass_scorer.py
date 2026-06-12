import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from skrec.dataset.interactions_dataset import InteractionMultiClassDataset
from skrec.examples.datasets import sample_multi_class_interactions
from skrec.scorer.multiclass import MulticlassScorer
from tests.utils import MockClassifier


@pytest.fixture
def setup_fixture(setup_small_datasets):
    setup_small_datasets["multiclass_interactions_dataset"] = InteractionMultiClassDataset(
        data_location=setup_small_datasets["dst"] / setup_small_datasets["multiclass_interactions_data"]
    )

    setup_small_datasets["multiclass_interactions_df"] = setup_small_datasets[
        "multiclass_interactions_dataset"
    ].fetch_data()
    setup_small_datasets["multiclass_scorer"] = MulticlassScorer(estimator=MockClassifier())
    setup_small_datasets["multiclass_scorer"].item_names = np.array(["mock1", "mock2"])
    return setup_small_datasets


def test_outcome_not_allowed(setup_fixture, caplog):
    interactions = setup_fixture["multiclass_interactions_df"]
    interactions["OUTCOME"] = 1

    expected_msg1 = "OUTCOME field not allowed in Interactions Dataset for MulticlassScorer"
    with pytest.raises(ValueError, match=expected_msg1):
        X, y = setup_fixture["multiclass_scorer"].process_datasets(
            users_df=None, items_df=None, interactions_df=interactions
        )


def test_users_not_allowed(setup_fixture):
    meaningless_df = pd.DataFrame(
        [["John", 3, 0.2], ["Doe", 4, 0.1]], columns=["RandomName", "RandomNum1", "RandomNum2"]
    )

    expected_msg1 = "Users Dataset will not be used in MulticlassScorer"
    with pytest.raises(ValueError, match=expected_msg1):
        X, y = setup_fixture["multiclass_scorer"].process_datasets(
            users_df=meaningless_df, items_df=None, interactions_df=setup_fixture["multiclass_interactions_df"]
        )


def test_items_not_allowed(setup_fixture):
    meaningless_df = pd.DataFrame(
        [["John", 3, 0.2], ["Doe", 4, 0.1]], columns=["RandomName", "RandomNum1", "RandomNum2"]
    )

    expected_msg1 = "Items Dataset will not be used in MulticlassScorer"
    with pytest.raises(ValueError, match=expected_msg1):
        X, y = setup_fixture["multiclass_scorer"].process_datasets(
            users_df=None, items_df=meaningless_df, interactions_df=setup_fixture["multiclass_interactions_df"]
        )


def test_process_datasets(setup_fixture, caplog):
    X_correct = pd.DataFrame({"Age": [30, 28, 49, 35], "Gender": [0, 1, 0, 1]})
    X_correct = X_correct.values

    y_correct = np.array([1, 0, 2, 1])
    correct_item_names = np.array(["Item1", "Item2", "Item3"])

    X, y = setup_fixture["multiclass_scorer"].process_datasets(
        users_df=None, items_df=None, interactions_df=setup_fixture["multiclass_interactions_df"]
    )

    assert_array_equal(X, X_correct)
    assert_array_equal(y, y_correct)
    assert_array_equal(setup_fixture["multiclass_scorer"].item_names, correct_item_names)
    assert setup_fixture["multiclass_scorer"].items_df is None


def test_score_items(setup_fixture):
    interactions = pd.DataFrame({"USER_ID": [11, 12], "a": [1, 4], "b": [2, 5], "c": [3.3, 6.6]})

    setup_fixture["multiclass_scorer"].estimator.feature_names = ["a", "b", "c"]
    result = setup_fixture["multiclass_scorer"].score_items(interactions)
    expected = np.array([[6.3, 2.1], [15.6, 5.2]])
    assert_array_equal(result, expected)

    expected_msg = "Multiclass Scorer cannot accept Users Dataframe, set it to None!"
    with pytest.raises(ValueError, match=expected_msg):
        setup_fixture["multiclass_scorer"].score_items(interactions=interactions, users=interactions)


def test_process_dataset_large(setup_fixture):
    X, y = setup_fixture["multiclass_scorer"].process_datasets(
        users_df=None, items_df=None, interactions_df=sample_multi_class_interactions.fetch_data()
    )

    assert X.shape == (5000, 5)
    assert y.shape == (5000,)


def test_score_items_with_item_subset(setup_fixture):
    interactions = pd.DataFrame({"USER_ID": [11, 12], "a": [1, 4], "b": [2, 5], "c": [3.3, 6.6]})

    scorer = setup_fixture["multiclass_scorer"]
    scorer.estimator.feature_names = ["a", "b", "c"]
    item_subset = ["mock2"]
    scorer.set_item_subset(item_subset)
    result = scorer.score_items(interactions)
    expected = np.array([[2.1], [5.2]])
    assert_array_equal(result, expected)


def _imbalanced_multiclass_interactions(n=900, seed=0):
    """Wide multiclass interactions: USER_ID + ITEM_ID (3 imbalanced classes) + features.

    Reproduces the firmographics-industry shape: a single multiclass target with a
    dominant majority class and rare minority classes that class-balancing targets.
    """
    import skrec.constants as C

    rng = np.random.RandomState(seed)
    feats = rng.rand(n, 4)
    classes = rng.choice(["A", "B", "C"], size=n, p=[0.7, 0.2, 0.1])  # ~[0.7, 0.2, 0.1]
    feats[:, 0] += np.select([classes == "A", classes == "B", classes == "C"], [0.0, 0.6, 1.2]) + rng.rand(n) * 0.5
    df = pd.DataFrame(feats, columns=[f"f{i}" for i in range(4)])
    df[C.USER_ID_NAME] = np.arange(n).astype(str)
    df[C.ITEM_ID_NAME] = classes
    return df


def test_multiclass_balanced_weights_shift_minority_scores():
    """firmographics-industry recipe: multi:softprob + sample_weight='balanced'
    trains end-to-end through MulticlassScorer and raises the score mass on the
    rare minority class relative to unweighted training."""
    from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator

    df = _imbalanced_multiclass_interactions()

    def _mean_minority_score(sample_weight):
        scorer = MulticlassScorer(estimator=XGBClassifierEstimator({"n_estimators": 40}, sample_weight=sample_weight))
        X, y = scorer.process_datasets(users_df=None, items_df=None, interactions_df=df)
        scorer.train_model(X, y)
        scores = scorer.score_items(interactions=df)  # (n, n_classes)
        c_idx = list(scorer.item_names).index("C")  # rare class column, via scorer encoding
        return np.asarray(scores)[:, c_idx].mean()

    unweighted = _mean_minority_score(None)
    balanced = _mean_minority_score("balanced")
    assert balanced > unweighted  # balancing up-weights the rare class
