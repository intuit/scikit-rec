import os

import boto3
from moto import mock_aws

from skrec.util.s3 import get_s3_stream

TEST_BUCKET = "test-bucket"


@mock_aws
def test_get_s3_stream():
    # moto's default region is us-east-1; omit LocationConstraint so the
    # bucket creation matches the mocked endpoint region.
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    s3 = boto3.resource("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=TEST_BUCKET)
    obj = s3.Object(TEST_BUCKET, "test.txt")
    obj.put(Body=b"test content")

    stream = get_s3_stream(f"s3://{TEST_BUCKET}/test.txt")

    assert stream == b"test content"
