"""
Location for your shared test fixtures
"""

import os

# Set threading environment variables before importing PyTorch
# to avoid segmentation faults from OpenMP/MKL threading conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

# ---------------------------------------------------------------------------
# Skip torch-dependent test files when torch is not installed.
# Using collect_ignore instead of pytest.importorskip so that VS Code's
# test explorer doesn't surface module-level Skipped as discovery errors.
# ---------------------------------------------------------------------------
_TORCH_TEST_FILES = [
    "test_embedding_estimators.py",
    "test_hrnn_estimator.py",
    "test_sasrec_estimator.py",
    "test_ncf_estimator.py",
    "test_ncf_integration.py",
    "test_sequential_recommender.py",
    "test_ranking_recommender.py",
    "test_deep_fm_network.py",
]

try:
    import torch  # noqa: F401
except ImportError:
    collect_ignore = _TORCH_TEST_FILES

from skrec.dataset.interactions_dataset import InteractionsDataset  # noqa: E402
from skrec.dataset.items_dataset import ItemsDataset  # noqa: E402
from skrec.dataset.users_dataset import UsersDataset  # noqa: E402


@pytest.fixture
def setup_small_datasets(tmp_path_factory):
    test_data = {}
    test_data["dst"] = tmp_path_factory.mktemp("temp_data")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    test_data["users_filename"] = "sample_users_data.csv"
    users_df = pd.DataFrame(
        [["John", 30, 1], ["Doe", 35, 0], ["Amy", 28, 1], ["Bill", 49, 0]], columns=["USER_ID", "Age", "Gender"]
    )
    users_df.to_csv(test_data["dst"] / test_data["users_filename"], index=False)

    # ------------------------------------------------------------------------------------------------------------------

    test_data["items_filename"] = "sample_items_data.csv"
    items_df = pd.DataFrame(
        [["Item1", 1, 2], ["Item2", 0, 1], ["Item3", 2, 4]], columns=["ITEM_ID", "ItemFeature1", "ItemFeature2"]
    )
    items_df.to_csv(test_data["dst"] / test_data["items_filename"], index=False)

    # ------------------------------------------------------------------------------------------------------------------

    test_data["interactions_filename"] = "sample_interactions_data.csv"
    interactions_df = pd.DataFrame(
        [
            ["John", "Item2", 0, 1, 0.1],
            ["Amy", "Item1", 1, 2, 0.2],
            ["Bill", "Item1", 0, 3, 0.3],
            ["Amy", "Item2", 1, 4, 0.4],
        ],
        columns=["USER_ID", "ITEM_ID", "OUTCOME", "Context1", "Context2"],
    )
    interactions_df.to_csv(test_data["dst"] / test_data["interactions_filename"], index=False)

    # ------------------------------------------------------------------------------------------------------------------

    test_data["interactions_filename_inference"] = "sample_interactions_inference_data.csv"
    interactions_df = pd.DataFrame([["John", 3, 0.2], ["Doe", 4, 0.1]], columns=["USER_ID", "Context1", "Context2"])
    interactions_df.to_csv(test_data["dst"] / test_data["interactions_filename_inference"], index=False)

    # ------------------------------------------------------------------------------------------------------------------

    test_data["interactions_filename_with_timestamp"] = "sample_interactions_timestamp_data.csv"
    interactions_df = pd.DataFrame(
        [["John", "Item1", 0, 1000], ["Doe", "Item2", 1, 2000]],
        columns=["USER_ID", "ITEM_ID", "OUTCOME", "TIMESTAMP"],
    )
    interactions_df.to_csv(test_data["dst"] / test_data["interactions_filename_with_timestamp"], index=False)

    # ------------------------------------------------------------------------------------------------------------------

    test_data["multioutput_interactions_data"] = "multioutput_interactions_data.csv"
    interactions_df = pd.DataFrame(
        [
            ["John", 25, 50000, 1, 1, 0],
            ["Amy", 30, 60000, 1, 0, 0],
            ["Bill", 40, 80000, 0, 0, 1],
            ["Doe", 21, 40000, 0, 1, 0],
        ],
        columns=["USER_ID", "age", "Income", "ITEM_1", "ITEM_2", "ITEM_3"],
    )
    interactions_df.to_csv(test_data["dst"] / test_data["multioutput_interactions_data"], index=False)
    # ------------------------------------------------------------------------------------------------------------------

    test_data["multioutcome_interactions_data"] = "multioutcome_interactions_data.csv"

    interactions_df = pd.DataFrame(
        [
            ["John", "Item2", 0, 1, 0.1, 1.5, 1.5],
            ["Amy", "Item1", 1, 2, 0.2, 1.5, 1.5],
            ["Bill", "Item1", 0, 3, 0.3, 1.5, 1.5],
            ["Amy", "Item2", 1, 4, 0.4, 1.5, 1.5],
        ],
        columns=["USER_ID", "ITEM_ID", "OUTCOME", "Context1", "Context2", "OUTCOME_1", "OUTCOME_2"],
    )
    interactions_df.to_csv(test_data["dst"] / test_data["multioutcome_interactions_data"], index=False)

    # ------------------------------------------------------------------------------------------------------------------

    test_data["multiclass_interactions_data"] = "multiclass_interactions_data.csv"

    interactions_df = pd.DataFrame(
        [
            ["John", "Item2", 30, 0],
            ["Amy", "Item1", 28, 1],
            ["Bill", "Item3", 49, 0],
            ["Amy", "Item2", 35, 1],
        ],
        columns=["USER_ID", "ITEM_ID", "Age", "Gender"],
    )
    interactions_df.to_csv(test_data["dst"] / test_data["multiclass_interactions_data"], index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    test_data["users_client_schema_filename"] = "sample_users_schema.yaml"
    users_schema_dict = {
        "columns": [
            {"name": "USER_ID", "type": "str"},
            {"name": "Age", "type": "int"},
            {"name": "Gender", "type": "int"},
        ]
    }
    dest_loc = test_data["dst"] / test_data["users_client_schema_filename"]
    yaml.dump(users_schema_dict, open(dest_loc, "w"), default_flow_style=False)

    # ------------------------------------------------------------------------------------------------------------------

    test_data["items_client_schema_filename"] = "sample_items_schema.yaml"
    items_schema_dict = {
        "columns": [
            {"name": "ITEM_ID", "type": "str"},
            {"name": "ItemFeature1", "type": "float"},
            {"name": "ItemFeature2", "type": "float"},
        ]
    }
    dest_loc = test_data["dst"] / test_data["items_client_schema_filename"]
    yaml.dump(items_schema_dict, open(dest_loc, "w"), default_flow_style=False)

    # ------------------------------------------------------------------------------------------------------------------

    test_data["interactions_client_schema_filename"] = "sample_interactions_schema.yaml"
    interactions_schema_dict = {
        "columns": [
            {"name": "USER_ID", "type": "str"},
            {"name": "ITEM_ID", "type": "str"},
            {"name": "OUTCOME", "type": "float"},
            {"name": "Context1", "type": "float"},
            {"name": "Context2", "type": "float"},
        ]
    }
    dest_loc = test_data["dst"] / test_data["interactions_client_schema_filename"]
    yaml.dump(interactions_schema_dict, open(dest_loc, "w"), default_flow_style=False)

    # ------------------------------------------------------------------------------------------------------------------

    test_data["interactions_client_schema_with_timestamp_filename"] = "sample_interactions_with_timestamp_schema.yaml"
    interactions_schema_dict = {
        "columns": [
            {"name": "USER_ID", "type": "str"},
            {"name": "ITEM_ID", "type": "str"},
            {"name": "OUTCOME", "type": "float"},
            {"name": "TIMESTAMP", "type": "int"},
        ]
    }
    dest_loc = test_data["dst"] / test_data["interactions_client_schema_with_timestamp_filename"]
    yaml.dump(interactions_schema_dict, open(dest_loc, "w"), default_flow_style=False)

    # ------------------------------------------------------------------------------------------------------------------

    test_data["interactions_req_addl_schema_filename"] = "sample_interactions_addl_schema_timestamp.yaml"
    interactions_schema_dict = {"columns": [{"name": "TIMESTAMP", "type": "int"}]}
    dest_loc = test_data["dst"] / test_data["interactions_req_addl_schema_filename"]
    yaml.dump(interactions_schema_dict, open(dest_loc, "w"), default_flow_style=False)
    # ------------------------------------------------------------------------------------------------------------------

    data_path = test_data["dst"] / test_data["interactions_filename"]
    interactions_schema_path = test_data["dst"] / test_data["interactions_client_schema_filename"]
    test_data["interactions_dataset"] = InteractionsDataset(
        client_schema_path=interactions_schema_path, data_location=data_path
    )

    data_path = test_data["dst"] / test_data["users_filename"]
    users_schema_path = test_data["dst"] / test_data["users_client_schema_filename"]
    test_data["users_dataset"] = UsersDataset(client_schema_path=users_schema_path, data_location=data_path)

    data_path = test_data["dst"] / test_data["items_filename"]
    items_schema_path = test_data["dst"] / test_data["items_client_schema_filename"]
    test_data["items_dataset"] = ItemsDataset(client_schema_path=items_schema_path, data_location=data_path)

    return test_data
