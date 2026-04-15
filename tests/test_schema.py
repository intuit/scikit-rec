import os
import tempfile

import boto3
import numpy as np
import pandas as pd
import pytest
import yaml
from boto3.s3.transfer import S3Transfer
from moto import mock_aws

from skrec.dataset.schema import DatasetSchema
from skrec.util.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture
def setup_fixture(tmp_path_factory):
    test_schema = {}
    test_schema["original_schema"] = {"columns": [{"name": "USER_ID", "type": "str"}]}
    test_schema["column_names"] = ["USER_ID"]
    test_schema["column_types"] = {"USER_ID": "str"}

    interactions_schema_dict = {"columns": [{"name": "USER_ID", "type": "str"}]}
    src_schema = tmp_path_factory.mktemp("temp_data")
    dest_loc = src_schema / "dataset_required_schema.yaml"
    yaml.dump(interactions_schema_dict, open(dest_loc, "w"), default_flow_style=False)

    test_schema["required_schema_path"] = dest_loc
    return test_schema


def test_correct_schema(setup_fixture):
    created_schema = DatasetSchema.create(setup_fixture["required_schema_path"])
    assert created_schema.raw_schema == setup_fixture["original_schema"]


def test_incorrect_schema():
    expected_error = "Schema location invalid"
    with pytest.raises(Exception, match=expected_error):
        DatasetSchema.create("sample_invalid_location")


def test_column_names(setup_fixture):
    created_schema = DatasetSchema.create(setup_fixture["required_schema_path"])
    assert created_schema.columns == setup_fixture["column_names"]


def test_column_types(setup_fixture):
    created_schema = DatasetSchema.create(setup_fixture["required_schema_path"])
    assert created_schema.column_types == setup_fixture["column_types"]


def test_invalid_column_type():
    with tempfile.TemporaryDirectory() as tempdir:
        schema = {
            "columns": [
                {"name": "a", "type": "int"},
                {"name": "b", "type": "test_type"},
                {"name": "c", "type": "str"},
            ]
        }
        schema_filename = os.path.join(tempdir, "schema.yaml")
        yaml.dump(schema, open(schema_filename, "w"))
        expected_error = "Invalid type: test_type for column: b"
        with pytest.raises(RuntimeError, match=expected_error):
            DatasetSchema.create(schema_filename)


def test_apply_schema_with_vocab():
    schema_with_vocab = {
        "columns": [
            {"name": "USER_ID", "type": "int"},
            {"name": "feature1", "type": "str", "vocab": ["a", "b", "c"]},
        ]
    }
    df = pd.DataFrame({"USER_ID": [1, 2, 3], "feature1": ["a", "b", "a"]})

    expected_df = pd.DataFrame(
        {
            "USER_ID": [1, 2, 3],
            "feature1_0": [1, 0, 1],
            "feature1_1": [0, 1, 0],
            "feature1_2": [0, 0, 0],
            "feature1_unknown": [0, 0, 0],
        }
    ).astype("int32")

    schema = DatasetSchema(schema_with_vocab)
    transformed_df = schema.apply(df).astype("int32")
    transformed_df = transformed_df.reindex(sorted(transformed_df.columns), axis=1)
    pd.testing.assert_frame_equal(transformed_df, expected_df)


def test_apply_schema_with_nans():
    schema_with_vocab = {
        "columns": [
            {"name": "USER_ID", "type": "int"},
            {"name": "feature1", "type": "str", "vocab": ["a", "b", "c"]},
        ]
    }

    df = pd.DataFrame({"USER_ID": [1, 2, 3, 4], "feature1": ["a", "b", None, "c"]})

    expected_df = pd.DataFrame(
        {
            "USER_ID": [1, 2, 3, 4],
            "feature1_0": [1, 0, 0, 0],
            "feature1_1": [0, 1, 0, 0],
            "feature1_2": [0, 0, 0, 1],
            "feature1_unknown": [0, 0, 1, 0],
        }
    ).astype("int32")

    schema = DatasetSchema(schema_with_vocab)
    transformed_df = schema.apply(df).astype("int32")
    transformed_df = transformed_df.reindex(sorted(transformed_df.columns), axis=1)
    pd.testing.assert_frame_equal(transformed_df, expected_df)


def test_apply_schema_with_out_of_vocab():
    schema_with_vocab = {
        "columns": [
            {"name": "USER_ID", "type": "int"},
            {"name": "feature1", "type": "str", "vocab": ["a", "b", "c"]},
        ]
    }

    df = pd.DataFrame({"USER_ID": [1, 2, 3, 4], "feature1": ["a", "b", "d", "c"]})

    expected_df = pd.DataFrame(
        {
            "USER_ID": [1, 2, 3, 4],
            "feature1_0": [1, 0, 0, 0],
            "feature1_1": [0, 1, 0, 0],
            "feature1_2": [0, 0, 0, 1],
            "feature1_unknown": [0, 0, 1, 0],
        }
    ).astype("int32")

    schema = DatasetSchema(schema_with_vocab)
    transformed_df = schema.apply(df).astype("int32")
    transformed_df = transformed_df.reindex(sorted(transformed_df.columns), axis=1)
    pd.testing.assert_frame_equal(transformed_df, expected_df)


def test_apply_schema_with_hash_buckets():
    schema_with_hash_buckets = {
        "columns": [
            {"name": "USER_ID", "type": "int"},
            {"name": "feature1", "type": "str", "hash_buckets": 3},
        ]
    }

    df = pd.DataFrame({"USER_ID": [1, 2, 3], "feature1": ["a", "j", "z"]})
    schema = DatasetSchema(schema_with_hash_buckets)
    transformed_df = schema.apply(df).astype("int32")
    transformed_df = transformed_df.reindex(sorted(transformed_df.columns), axis=1)
    expected_df = pd.DataFrame(
        {
            "USER_ID": [1, 2, 3],
            "feature1_0": [1, 1, 0],
            "feature1_1": [0, 0, 1],
            "feature1_2": [0, 0, 0],
        }
    ).astype("int32")
    pd.testing.assert_frame_equal(transformed_df, expected_df)


@mock_aws
def test_s3_schema(setup_fixture):
    s3 = boto3.client("s3", region_name="us-west-2")
    s3.create_bucket(Bucket="test", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    local_schema = setup_fixture["required_schema_path"]
    S3Transfer(s3).upload_file(filename=local_schema, bucket="test", key="dataset_schema/schema.yaml")
    s3_schema_path = "s3://test/dataset_schema/schema.yaml"
    created_schema = DatasetSchema.create(s3_schema_path)
    assert created_schema.raw_schema == setup_fixture["original_schema"]


@mock_aws
def test_s3_schema_fail():
    s3_schema_path = "s3://test/dataset_schema/not_exist_schema.yaml"
    with pytest.raises(Exception) as context:
        DatasetSchema.create(s3_schema_path)
        assert "Unable to load config" in str(context.value)


def test_apply_default_schema():
    df = pd.DataFrame(
        {
            "USER_ID": [1, 2, 3],
            "feature1": [1.0, 2.0, 3.0],
            "feature2": ["a", "b", "c"],
            "feature3": [True, False, True],
        }
    )
    expected_dtype = {"USER_ID": "int64", "feature1": "float32", "feature2": "object", "feature3": "int64"}
    auto_schema = DatasetSchema.apply_default_schema(df)
    auto_schema_dtypes = {k: str(v) for k, v in auto_schema.dtypes.to_dict().items()}
    assert auto_schema_dtypes == expected_dtype
    assert auto_schema["feature3"].isin([0, 1]).all()


# ---------------------------------------------------------------------------
# Tests for apply_type_coercions
# ---------------------------------------------------------------------------


def test_apply_type_coercions_casts_to_declared_types():
    """Columns present in the schema are coerced to their declared type."""
    schema = DatasetSchema(
        {
            "columns": [
                {"name": "ITEM_ID", "type": "str"},
                {"name": "score", "type": "float"},
            ]
        }
    )
    df = pd.DataFrame({"ITEM_ID": [1, 10, 100], "score": ["1.5", "2.5", "3.5"]})
    result = schema.apply_type_coercions(df)
    assert result["ITEM_ID"].dtype == object  # coerced to str
    assert result["ITEM_ID"].tolist() == ["1", "10", "100"]
    assert result["score"].dtype == np.float32  # coerced to float32


def test_apply_type_coercions_preserves_extra_columns():
    """Columns not declared in the schema are kept unchanged."""
    schema = DatasetSchema({"columns": [{"name": "ITEM_ID", "type": "str"}]})
    df = pd.DataFrame({"ITEM_ID": [1, 2], "extra": ["a", "b"], "another": [1.0, 2.0]})
    result = schema.apply_type_coercions(df)
    assert set(result.columns) == {"ITEM_ID", "extra", "another"}
    assert result["ITEM_ID"].dtype == object  # coerced


def test_apply_type_coercions_raises_when_required_column_missing():
    """RuntimeError is raised when a column declared in the schema is absent from df."""
    schema = DatasetSchema({"columns": [{"name": "ITEM_ID", "type": "str"}]})
    df = pd.DataFrame({"USER_ID": ["u1", "u2"]})  # ITEM_ID missing
    with pytest.raises(RuntimeError, match="ITEM_ID"):
        schema.apply_type_coercions(df)
