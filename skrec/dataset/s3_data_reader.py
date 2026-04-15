import os
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urlparse

import pandas as pd

from skrec.dataset.datatypes import DataFileFormat

REGION = None

# Design decision: We do not want to add capability to write dataset to S3


def _import_boto3():
    try:
        import boto3

        return boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for reading data from S3 but is not installed. "
            "Install the [aws] extra to use S3 features."
        ) from None


class S3DataReader:
    def __init__(self, file_extension: str, data_location: str, region: Optional[str] = REGION):
        if isinstance(data_location, Path):
            data_location = str(data_location)
        self.file_extension = file_extension
        self.s3_path = data_location
        self.region = region

    def read(self):
        boto3 = _import_boto3()
        if not self.s3_path:
            raise ValueError("Unable to read from S3 due to missing s3_path")
        bucket, key = self.extract_key_from_url()
        if not bucket or not key:
            raise ValueError("Unable to read from S3 due to missing bucket and key")
        resource = boto3.resource("s3", region_name=self.region)
        obj = resource.Object(bucket, key)
        return self.check_format_and_read(obj, resource=resource)

    def extract_key_from_url(self):
        parsed_result = urlparse(self.s3_path)
        bucket = parsed_result.netloc
        key = parsed_result.path
        # Remove first slash
        key = key[1:]
        return bucket, key

    def check_format_and_read(self, obj, resource=None):
        if self.file_extension == DataFileFormat.CSV:
            body = obj.get()["Body"]
            csv_string = body.read().decode("utf-8")
            df = pd.read_csv(StringIO(csv_string))
        elif self.file_extension == DataFileFormat.PARQUET:
            body = obj.get()["Body"]
            read_body = body.read()
            bytes_body = BytesIO(read_body)
            df = pd.read_parquet(bytes_body)
        elif self.file_extension == "":
            # partitioned parquet directory: reuse the boto3 resource created in read()
            # to avoid creating a redundant client for this path.
            if resource is None:
                resource = _import_boto3().resource("s3", region_name=self.region)
            dfs = []
            for file_path in sorted(self.get_data_filenames()):
                parsed = urlparse(file_path)
                file_obj = resource.Object(parsed.netloc, parsed.path.lstrip("/"))
                dfs.append(pd.read_parquet(BytesIO(file_obj.get()["Body"].read())))
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError("Unknown data file format")
        return df

    def get_data_filenames(self) -> Set[str]:
        """
        List all dataset files.
            - If the dataset is a single file, return a list with a single element
            - If the dataset is a directory, return a list of all data files in the directory
        """
        if self.file_extension:
            # this is a CSV or Parquet file, not a directory
            return set([self.s3_path])

        bucket_name, key = self.extract_key_from_url()
        bucket = _import_boto3().resource("s3", region_name=self.region).Bucket(bucket_name)
        data_filenames = set()

        for item in bucket.objects.filter(Prefix=key):
            file_name = item.key.replace(key, "").lstrip("/")
            full_file_name = os.path.join(self.s3_path, file_name)

            if full_file_name.endswith(".parquet"):
                data_filenames.add(full_file_name)

        return data_filenames
