"""Hive-style partition directories (``key=value/``) in dataset file paths."""

import os
from typing import Dict, List, Optional
from urllib.parse import unquote

# Spark and Hive write null partition values as this directory name
HIVE_NULL_PARTITION = "__HIVE_DEFAULT_PARTITION__"


def _relative_dirs(file_path: str, root: str) -> List[str]:
    if os.sep != "/":
        # local paths on Windows; S3 keys always use "/"
        file_path, root = file_path.replace(os.sep, "/"), root.replace(os.sep, "/")
    root = root.rstrip("/") + "/"
    if not file_path.startswith(root):
        return []
    return file_path[len(root) :].split("/")[:-1]


def hive_partition_values(file_path: str, root: str) -> Dict[str, Optional[str]]:
    """``key=value`` directory names between ``root`` and ``file_path``.

    For example ``<root>/ITEM_ID=foo/part-0.parquet`` gives ``{"ITEM_ID": "foo"}``.
    Values are URL-decoded, as Spark and pyarrow escape special characters in
    partition directory names; the Hive null partition gives ``None``.
    """
    values: Dict[str, Optional[str]] = {}
    for part in _relative_dirs(file_path, root):
        if "=" in part:
            key, value = part.split("=", 1)
            values[unquote(key)] = None if value == HIVE_NULL_PARTITION else unquote(value)
    return values


def is_hidden_path(file_path: str, root: str) -> bool:
    """Whether a directory between ``root`` and ``file_path`` starts with ``_`` or ``.``.

    Spark/Hadoop use these for temporary and metadata directories (e.g. ``_temporary``),
    and pyarrow skips them when reading a directory.
    """
    return any(part.startswith(("_", ".")) for part in _relative_dirs(file_path, root))
