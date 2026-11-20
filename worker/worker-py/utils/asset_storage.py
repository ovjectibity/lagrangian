"""Transfer workload files to and from S3 using the default AWS credential chain."""

from pathlib import Path
from urllib.parse import urlsplit

import boto3


def upload_file(local_path: str | Path, s3_path: str) -> str:
    """Upload a file to an s3://bucket/key URI and return it after success.

    This call blocks; async callers should use asyncio.to_thread(). AWS errors
    propagate to the caller, and the local file is retained after uploading.
    """
    destination = urlsplit(s3_path)
    key = destination.path.removeprefix("/")
    if (
        destination.scheme != "s3"
        or not destination.netloc
        or not key
        or destination.query
        or destination.fragment
    ):
        raise ValueError("s3_path must be an s3://bucket/key URI without a query or fragment")

    source = Path(local_path)
    if not source.is_file():
        raise FileNotFoundError(f"Local upload file does not exist or is not a file: {source}")

    client = boto3.client("s3")
    client.upload_file(str(source), destination.netloc, key)
    return s3_path


def download_file(s3_path: str, local_path: str | Path) -> Path:
    """Download an s3://bucket/key URI and return the local path after success.

    Creates parent directories and overwrites an existing destination file.
    This call blocks; async callers should use asyncio.to_thread(). AWS errors
    propagate to the caller.
    """
    source = urlsplit(s3_path)
    key = source.path.removeprefix("/")
    if (
        source.scheme != "s3"
        or not source.netloc
        or not key
        or source.query
        or source.fragment
    ):
        raise ValueError("s3_path must be an s3://bucket/key URI without a query or fragment")

    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    client = boto3.client("s3")
    client.download_file(source.netloc, key, str(destination))
    return destination
