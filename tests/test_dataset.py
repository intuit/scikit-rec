import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import yaml
from moto import mock_aws

from skrec.dataset.dataset import Dataset
from skrec.dataset.local_data_reader import LocalDataReader
from skrec.dataset.s3_data_reader import S3DataReader
from skrec.dataset.schema import DatasetSchema
from skrec.util.logger import get_logger

logger = get_logger(__name__)


@mock_aws
class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data = pd.DataFrame([["John", 30, 1], ["Doe", 35, 0]], columns=["USER_ID", "Age", "Gender"])
        self.test_data[["Age", "Gender"]] = self.test_data[["Age", "Gender"]].astype("int64")

        self.temp_folder = Path.cwd() / "tests/test_dataset_temp/"
        if os.path.isdir(self.temp_folder):
            shutil.rmtree(self.temp_folder)
        os.makedirs(str(self.temp_folder))

        self.csv_data_file_name = "sample_test_data.csv"
        self.csv_data_file_location = self.temp_folder / self.csv_data_file_name
        self.test_data.to_csv(self.csv_data_file_location, index=False)

        self.parquet_data_file_name = "sample_test_data.parquet"
        self.test_data.to_csv(self.temp_folder / self.csv_data_file_name, index=False)

        self.client_schema_path = self.temp_folder / "dataset_client_schema.yaml"
        schema_dict = {
            "columns": [
                {"name": "USER_ID", "type": "str"},
                {"name": "Age", "type": "int"},
                {"name": "Gender", "type": "int"},
            ]
        }
        dest_loc = self.temp_folder / self.client_schema_path
        yaml.dump(schema_dict, open(dest_loc, "w"), default_flow_style=False)

        self.required_schema_path = self.temp_folder / "dataset_required_schema.yaml"
        schema_dict = {"columns": [{"name": "USER_ID", "type": "str"}]}
        dest_loc = self.temp_folder / self.required_schema_path
        yaml.dump(schema_dict, open(dest_loc, "w"), default_flow_style=False)

        self.sample_client_schema = DatasetSchema.create(self.client_schema_path)
        self.sample_required_schema = DatasetSchema.create(self.required_schema_path)

        bucket_name = "testbucket"
        key = "testkey/test_data.csv"
        s3 = boto3.resource("s3", region_name="us-west-2")
        s3.Bucket(bucket_name).create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        with self._use_bytestream(s3, bucket_name, key) as bytestream:
            self.test_data.to_csv(bytestream, index=False)

        self.parquet_mutifile_location = self.temp_folder / "data"
        os.makedirs(self.parquet_mutifile_location, exist_ok=True)
        self.test_data.iloc[:1].to_parquet(self.parquet_mutifile_location / "1.parquet", index=False)
        self.test_data.iloc[1:].to_parquet(self.parquet_mutifile_location / "2.parquet", index=False)
        s3.meta.client.upload_file(
            Filename=str(self.parquet_mutifile_location / "1.parquet"), Bucket=bucket_name, Key="test-parquet/1.parquet"
        )
        s3.meta.client.upload_file(
            Filename=str(self.parquet_mutifile_location / "2.parquet"), Bucket=bucket_name, Key="test-parquet/2.parquet"
        )

    @contextmanager
    def _use_bytestream(self, s3, bucket_name, test_key):
        object = s3.Object(bucket_name, test_key)
        bytestream = BytesIO()

        yield bytestream

        bytestream.seek(0)
        object.put(Body=bytestream)

    def test_dataset_local_csv(self):
        dataset = Dataset(
            self.sample_client_schema, self.sample_required_schema, data_location=self.csv_data_file_location
        )
        df = dataset.fetch_data()
        pd.testing.assert_frame_equal(df, self.test_data)

    def test_dataset_local_parquet(self):
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test.parquet")
            self.test_data.to_parquet(filename)
            dataset = Dataset(self.sample_client_schema, self.sample_required_schema, data_location=filename)
            df = dataset.fetch_data()
            pd.testing.assert_frame_equal(df, self.test_data)

    def test_dataset_s3_csv(self):
        dataset = Dataset(
            self.sample_client_schema,
            self.sample_required_schema,
            data_location="s3://testbucket/testkey/test_data.csv",
        )
        df = dataset.fetch_data()
        pd.testing.assert_frame_equal(df, self.test_data)

    def test_dataset_s3_parquet(self):
        """
        The test for moto reading multiple S3 parquet files will fail without extra patch and extra depenencies.
        The issue is documented here: https://github.com/aio-libs/aiobotocore/issues/755
        """
        dataset = Dataset(
            self.sample_client_schema,
            self.sample_required_schema,
            data_location="s3://testbucket/test-parquet/",
        )
        df = dataset.fetch_data()
        pd.testing.assert_frame_equal(df, self.test_data)

    def test_dataset_fetch_failure(self):
        dataset = Dataset(
            self.sample_client_schema,
            self.sample_required_schema,
            data_location="",
        )
        expected_error_msg = "Unknown data source provided"
        with self.assertRaises(FileNotFoundError) as context:
            dataset.fetch_data()
        self.assertTrue(str(context.exception), expected_error_msg)

    def test_s3_read_failure(self):
        dataset = Dataset(
            self.sample_client_schema,
            self.sample_required_schema,
            data_location="s3:////test_data.parquet",
        )
        expected_error_msg = "Unable to read from S3 due to missing bucket and key"
        with self.assertRaises(Exception) as context:
            dataset.fetch_data()
        self.assertTrue(str(context.exception), expected_error_msg)

    def test_data_format_failure(self):
        dataset = Dataset(
            self.sample_client_schema,
            self.sample_required_schema,
            data_location="s3://testbucket/testkey/test_data.pdf",
        )
        expected_error_msg = "Unknown data file format"
        with self.assertRaises(ValueError) as context:
            dataset.fetch_data()
        self.assertTrue(str(context.exception), expected_error_msg)

    def test_get_data_filenames_local(self):
        with tempfile.TemporaryDirectory() as tempdir:
            data_location = Path(tempdir) / "data.csv"
            data_location.touch()
            reader = LocalDataReader(data_location=data_location, file_extension=data_location.suffix)
            self.assertEqual(reader.get_data_filenames(), {str(data_location)})

        with tempfile.TemporaryDirectory() as tempdir:
            f1 = Path(tempdir) / "data1.parquet"
            f1.touch()
            f2 = Path(tempdir) / "data2.parquet"
            f2.touch()
            reader = LocalDataReader(data_location=tempdir, file_extension="")
            self.assertEqual(reader.get_data_filenames(), {str(f1), str(f2)})

    def test_get_data_filenames_s3(self):
        s3 = boto3.resource("s3", region_name="us-west-2")
        s3.Bucket("test").create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        f1 = "s3://test/data-parquet/data1.parquet"
        f2 = "s3://test/data-parquet/data2.parquet"

        def touch_s3(s3_uri):
            bucket, key = s3_uri.replace("s3://", "").split("/", 1)
            s3.Object(bucket, key).put(Body=b"")

        touch_s3(f1)
        touch_s3(f2)

        reader = S3DataReader(file_extension="parquet", data_location=f1)
        self.assertEqual(reader.get_data_filenames(), {f1})

        reader = S3DataReader(file_extension="", data_location="s3://test/data-parquet/")
        self.assertEqual(reader.get_data_filenames(), {f1, f2})

    def test_large_number_of_files_get_data_filenames_s3(self):
        # This test is to check if the S3DataReader can handle a large number of files
        # Boto3.client used to have limit for 1000 files per bucket, but boto3.resource does not
        s3 = boto3.resource("s3", region_name="us-west-2")
        s3.Bucket("test").create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        def touch_s3(s3_uri):
            bucket, key = s3_uri.replace("s3://", "").split("/", 1)
            s3.Object(bucket, key).put(Body=b"")

        for i in range(2000):
            test_file = f"s3://test/data-parquet/data{i}.parquet"
            touch_s3(test_file)

        reader = S3DataReader(file_extension="", data_location="s3://test/data-parquet/")
        self.assertEqual(len(reader.get_data_filenames()), 2000)

    def test_data_missing_columns(self):
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test.parquet")
            data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            data.to_parquet(filename)
            schema = {
                "columns": [
                    {"name": "a", "type": "int"},
                    {"name": "b", "type": "int"},
                ]
            }
            schema_filename = os.path.join(tempdir, "schema.yaml")
            yaml.dump(schema, open(schema_filename, "w"))
            dataset_schema = DatasetSchema.create(schema_filename)

            dataset = Dataset(
                dataset_schema,
                dataset_schema,
                data_location=filename,
            )
            # successful fetch
            dataset.fetch_data()
            schema["columns"].append({"name": "c", "type": "int"})
            yaml.dump(schema, open(schema_filename, "w"))
            dataset_schema = DatasetSchema.create(schema_filename)

            with self.assertRaises(Exception):
                dataset = Dataset(
                    dataset_schema,
                    dataset_schema,
                    data_location=filename,
                )
                dataset.fetch_data()

    def test_apply_data_type_from_schema(self):
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test.parquet")
            data = pd.DataFrame({"a": ["1", "2", "3"], "b": [4, 5, 6], "c": [1, 2, 3]})
            data.to_parquet(filename)
            schema = {
                "columns": [
                    {"name": "a", "type": "int"},
                    {"name": "b", "type": "float"},
                    {"name": "c", "type": "str"},
                ]
            }
            schema_filename = os.path.join(tempdir, "schema.yaml")
            yaml.dump(schema, open(schema_filename, "w"))
            dataset_schema = DatasetSchema.create(schema_filename)

            dataset = Dataset(
                dataset_schema,
                dataset_schema,
                data_location=filename,
            )
            data_df = dataset.fetch_data()
            assert data_df["a"].dtype == np.int64
            assert data_df["b"].dtype == np.float32
            assert data_df["c"].dtype == object

    def test_no_schema(self):
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test.parquet")
            data = pd.DataFrame({"a": ["1", "2", "3"], "b": [4, 5, 6], "c": [1.0, 2.0, 3.0]})
            data.to_parquet(filename)
            required_schema = {
                "columns": [
                    {"name": "a", "type": "int"},
                ]
            }
            schema_filename = os.path.join(tempdir, "required-schema.yaml")
            yaml.dump(required_schema, open(schema_filename, "w"))
            required_schema = DatasetSchema.create(schema_filename)

            dataset = Dataset(
                client_schema=None,
                required_schema=required_schema,
                data_location=filename,
            )
            data_df = dataset.fetch_data()
            # required_schema declares "a" as int, so apply_type_coercions coerces it
            assert data_df["a"].dtype == np.int64
            assert data_df["b"].dtype == np.int64
            assert data_df["c"].dtype == np.float32

    def test_required_schema_coerces_int_ids_to_str(self):
        """Without a client schema, required_schema types are still enforced on fetch_data()."""
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test.csv")
            data = pd.DataFrame({"USER_ID": [1, 2, 3], "ITEM_ID": [10, 20, 30], "OUTCOME": [1.0, 0.0, 1.0]})
            data.to_csv(filename, index=False)

            schema_dict = {
                "columns": [
                    {"name": "USER_ID", "type": "str"},
                    {"name": "ITEM_ID", "type": "str"},
                    {"name": "OUTCOME", "type": "float"},
                ]
            }
            schema_filename = os.path.join(tempdir, "schema.yaml")
            yaml.dump(schema_dict, open(schema_filename, "w"))
            required_schema = DatasetSchema.create(schema_filename)

            dataset = Dataset(client_schema=None, required_schema=required_schema, data_location=filename)
            data_df = dataset.fetch_data()

            assert data_df["USER_ID"].dtype == object  # str
            assert data_df["ITEM_ID"].dtype == object  # str
            assert data_df["ITEM_ID"].tolist() == ["10", "20", "30"]
            assert data_df["OUTCOME"].dtype == np.float32

    def tearDown(self):
        shutil.rmtree(self.temp_folder)
