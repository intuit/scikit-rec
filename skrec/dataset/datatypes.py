from enum import Enum


class DataFileFormat(str, Enum):
    CSV = ".csv"
    PARQUET = ".parquet"


class DataSource(str, Enum):
    LOCAL = "LOCAL"
    S3 = "S3"


class ColumnDataType:
    dtype_map = {
        "int": "int64",
        "float": "float32",
        "str": "str",
    }
