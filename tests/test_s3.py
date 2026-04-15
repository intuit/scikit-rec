import boto3
from moto import mock_aws

from skrec.util.s3 import get_s3_stream

TEST_BUCKET = "test-bucket"


@mock_aws
def test_get_s3_stream():
    s3 = boto3.resource("s3")
    s3.create_bucket(Bucket=TEST_BUCKET, CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    obj = s3.Object(TEST_BUCKET, "test.txt")
    obj.put(Body=b"test content")

    stream = get_s3_stream(f"s3://{TEST_BUCKET}/test.txt")

    assert stream == b"test content"
