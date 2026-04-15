from typing import Tuple
from urllib.parse import urlparse


def get_s3_bucket_key_from_url(s3_full_path: str) -> Tuple[str, str]:
    parsed_result = urlparse(s3_full_path)
    bucket = parsed_result.netloc
    key = parsed_result.path
    # Remove first slash
    key = key[1:]
    return bucket, key


def get_s3_stream(s3_path: str) -> bytes:
    import boto3

    s3_resource = boto3.resource("s3")
    bucket, key = get_s3_bucket_key_from_url(s3_path)
    obj = s3_resource.Object(bucket, key)
    return bytes(obj.get()["Body"].read())
