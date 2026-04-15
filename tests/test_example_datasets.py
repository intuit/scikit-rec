import pandas as pd
import pytest

from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_items,
    sample_binary_reward_users,
    sample_continuous_reward_interactions,
    sample_continuous_reward_items,
    sample_continuous_reward_users,
    sample_multi_class_interactions,
    sample_multi_outcome_interactions,
    sample_multi_output_interactions,
)
from skrec.util.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture
def setup_fixture():
    test_data = {}
    test_data["req_df"] = True
    test_data["binary_items"] = ["ITEM_1", "ITEM_2", "ITEM_3"]
    test_data["multi_output_items"] = [
        "ITEM_200",
        "ITEM_600",
        "ITEM_800",
        "ITEM_850",
        "ITEM_910",
        "ITEM_920",
        "ITEM_930",
        "ITEM_940",
        "ITEM_965",
        "ITEM_970",
        "ITEM_975",
    ]
    test_data["multi_class_items"] = [
        "item_0",
        "item_1",
        "item_2",
        "item_3",
    ]
    test_data["continuous_items"] = [
        "item_0",
        "item_1",
        "item_2",
        "item_3",
    ]
    return test_data


def test_fetch_binary_reward_datasets(setup_fixture):
    interactions_df = sample_binary_reward_interactions.fetch_data()
    users_df = sample_binary_reward_users.fetch_data()
    items_df = sample_binary_reward_items.fetch_data()

    item_actual = items_df["ITEM_ID"].sample(n=1).iloc[0]

    assert len(interactions_df) == len(users_df)
    assert item_actual in setup_fixture["binary_items"]
    pd.testing.assert_frame_equal(
        users_df[["USER_ID"]],
        interactions_df[["USER_ID"]],
        "user id does not match for binary reward sample dataset",
    )


def test_fetch_continuous_reward_datasets(setup_fixture):
    interactions_df = sample_continuous_reward_interactions.fetch_data()
    users_df = sample_continuous_reward_users.fetch_data()
    items_df = sample_continuous_reward_items.fetch_data()

    item_actual = items_df["ITEM_ID"].sample(n=1).iloc[0]

    assert len(interactions_df) == len(users_df)
    assert item_actual in setup_fixture["continuous_items"]
    pd.testing.assert_frame_equal(
        users_df[["USER_ID"]], interactions_df[["USER_ID"]], "user id does not match for continuous reward dataset"
    )


def test_fetch_multi_class_datasets(setup_fixture):
    interactions_df = sample_multi_class_interactions.fetch_data()
    items_actual = set(interactions_df["ITEM_ID"])
    assert items_actual == set(setup_fixture["multi_class_items"])


def test_fetch_multi_output_datasets(setup_fixture):
    interactions_df = sample_multi_output_interactions.fetch_data()

    items_actual = interactions_df.columns.tolist()
    items_actual = [i for i in items_actual if "ITEM_" in i]

    assert items_actual == setup_fixture["multi_output_items"]


def test_fetch_multi_outcome_datasets(setup_fixture):
    interactions_df = sample_multi_outcome_interactions.fetch_data()

    outcomes_actual = interactions_df.columns.tolist()
    outcomes_actual = [i for i in outcomes_actual if "OUTCOME" in i]

    assert len(outcomes_actual) == 2
