import numpy as np
import pandas as pd
import pytest

from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.util.logger import get_logger

logger = get_logger(__name__)


def test_users_dataset(setup_small_datasets):
    """
    type: one of three values
    """
    truth_df = pd.DataFrame(
        [["John", 30, 1], ["Doe", 35, 0], ["Amy", 28, 1], ["Bill", 49, 0]], columns=["USER_ID", "Age", "Gender"]
    )
    users_dataset = setup_small_datasets["users_dataset"]

    pd.testing.assert_frame_equal(truth_df, users_dataset.fetch_data())


def test_items_dataset(setup_small_datasets):
    """
    type: one of three values
    """
    truth_df = pd.DataFrame(
        [["Item1", 1, 2], ["Item2", 0, 1], ["Item3", 2, 4]], columns=["ITEM_ID", "ItemFeature1", "ItemFeature2"]
    )
    truth_df[["ItemFeature1", "ItemFeature2"]] = truth_df[["ItemFeature1", "ItemFeature2"]].astype("float32")
    items_dataset = setup_small_datasets["items_dataset"]

    pd.testing.assert_frame_equal(truth_df, items_dataset.fetch_data())


def test_interactions_dataset(setup_small_datasets):
    """
    type: one of three values
    """
    truth_df = pd.DataFrame(
        [
            ["John", "Item2", 0, 1, 0.1],
            ["Amy", "Item1", 1, 2, 0.2],
            ["Bill", "Item1", 0, 3, 0.3],
            ["Amy", "Item2", 1, 4, 0.4],
        ],
        columns=["USER_ID", "ITEM_ID", "OUTCOME", "Context1", "Context2"],
    )
    truth_df[["OUTCOME"]] = truth_df[["OUTCOME"]].astype("float32")
    truth_df[["Context1"]] = truth_df[["Context1"]].astype("float32")
    truth_df[["Context2"]] = truth_df[["Context2"]].astype("float32")

    interactions_dataset = setup_small_datasets["interactions_dataset"]
    pd.testing.assert_frame_equal(truth_df, interactions_dataset.fetch_data())


def test_interactions_dataset_inference(setup_small_datasets):
    """
    type: one of three values
    """
    truth_df = pd.DataFrame([["John", 3, 0.2], ["Doe", 4, 0.1]], columns=["USER_ID", "Context1", "Context2"])
    truth_df[["Context1"]] = truth_df[["Context1"]].astype("float32")
    truth_df[["Context2"]] = truth_df[["Context2"]].astype("float32")

    data_path = setup_small_datasets["dst"] / setup_small_datasets["interactions_filename_inference"]
    schema_path = setup_small_datasets["dst"] / setup_small_datasets["interactions_client_schema_filename"]

    # When you send data without label for training, it will throw an error
    expected_error_msg = "not found in dataset"
    with pytest.raises(Exception, match=expected_error_msg):
        interactions_dataset = InteractionsDataset(client_schema_path=schema_path, data_location=data_path)
        interactions_dataset.fetch_data()

    interactions_dataset = InteractionsDataset(
        client_schema_path=schema_path, data_location=data_path, is_training=False
    )

    pd.testing.assert_frame_equal(truth_df, interactions_dataset.fetch_data())


def test_interactions_dataset_with_time(setup_small_datasets):
    """
    type: one of three values
    """
    truth_df = pd.DataFrame(
        [["John", "Item1", 0, 1000], ["Doe", "Item2", 1, 2000]],
        columns=["USER_ID", "ITEM_ID", "OUTCOME", "TIMESTAMP"],
    )
    truth_df[["OUTCOME"]] = truth_df[["OUTCOME"]].astype("float32")
    data_path = setup_small_datasets["dst"] / setup_small_datasets["interactions_filename_with_timestamp"]
    schema_path = (
        setup_small_datasets["dst"] / setup_small_datasets["interactions_client_schema_with_timestamp_filename"]
    )
    addl_schema_path = setup_small_datasets["dst"] / setup_small_datasets["interactions_req_addl_schema_filename"]
    interactions_dataset = InteractionsDataset(
        client_schema_path=schema_path, extra_required_schema_path=addl_schema_path, data_location=data_path
    )

    pd.testing.assert_frame_equal(truth_df, interactions_dataset.fetch_data())


def test_schema_check_failure(setup_small_datasets):
    data_path = setup_small_datasets["dst"] / setup_small_datasets["interactions_filename"]
    schema_path = setup_small_datasets["dst"] / setup_small_datasets["interactions_client_schema_filename"]
    addl_schema_path = setup_small_datasets["dst"] / setup_small_datasets["interactions_req_addl_schema_filename"]

    expected_error = "Client Schema does not conform to Required Schema"
    with pytest.raises(Exception, match=expected_error):
        InteractionsDataset(
            client_schema_path=schema_path, extra_required_schema_path=addl_schema_path, data_location=data_path
        )


# ---------------------------------------------------------------------------
# Tests for required-schema type enforcement (ID columns always str)
# ---------------------------------------------------------------------------


def test_items_dataset_item_id_coerced_to_str_without_schema(tmp_path):
    """ItemsDataset always returns ITEM_ID as str, even when the CSV contains int ITEM_IDs."""
    pd.DataFrame({"ITEM_ID": [1, 10, 100]}).to_csv(tmp_path / "items.csv", index=False)
    ds = ItemsDataset(data_location=str(tmp_path / "items.csv"))
    result = ds.fetch_data()
    assert result["ITEM_ID"].dtype == object  # str
    assert result["ITEM_ID"].tolist() == ["1", "10", "100"]


def test_interactions_dataset_ids_coerced_to_str_without_schema(tmp_path):
    """InteractionsDataset always returns USER_ID and ITEM_ID as str without a client schema."""
    pd.DataFrame(
        {
            "USER_ID": [1, 2, 3],
            "ITEM_ID": [10, 20, 30],
            "OUTCOME": [1.0, 0.0, 1.0],
        }
    ).to_csv(tmp_path / "interactions.csv", index=False)
    ds = InteractionsDataset(data_location=str(tmp_path / "interactions.csv"))
    result = ds.fetch_data()
    assert result["USER_ID"].dtype == object  # str
    assert result["ITEM_ID"].dtype == object  # str
    assert result["ITEM_ID"].tolist() == ["10", "20", "30"]
    assert result["OUTCOME"].dtype == np.float32


def test_items_dataset_missing_item_id_raises(tmp_path):
    """ItemsDataset raises RuntimeError when ITEM_ID is absent from the CSV."""
    pd.DataFrame({"some_col": [1, 2, 3]}).to_csv(tmp_path / "items.csv", index=False)
    ds = ItemsDataset(data_location=str(tmp_path / "items.csv"))
    with pytest.raises(RuntimeError, match="ITEM_ID"):
        ds.fetch_data()
