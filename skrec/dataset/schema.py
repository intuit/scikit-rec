import os
from collections import defaultdict
from typing import List, Optional

import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

from skrec.dataset.datatypes import ColumnDataType
from skrec.util.config_loader import load_config
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class DatasetSchema:
    def __init__(self, schema: dict) -> None:
        self.raw_schema: dict = schema
        self.columns: Optional[List[str]] = None
        self.column_types: Optional[dict] = None
        self.precomputed_columns: dict = {}

        self._parse_schema()
        self._validate_schema()

    @classmethod
    def create(cls, path: str):
        """Load a ``DatasetSchema`` from a YAML file on the local filesystem or S3.

        Args:
            path: Absolute local file path or ``s3://bucket/key`` URI pointing
                to a YAML schema file.

        Returns:
            A new ``DatasetSchema`` instance parsed from the file.

        Raises:
            FileNotFoundError: If the path does not exist or the URI is not recognised.
        """
        if os.path.exists(path) or str(path).startswith("s3://"):
            return cls(load_config(path=path))
        else:
            raise FileNotFoundError(f"Schema location invalid: {path}")

    def _parse_schema(self) -> None:
        self.columns = []
        self.column_types = {}

        if "columns" in self.raw_schema:
            list_of_cols = self.raw_schema["columns"]
            for col_info in list_of_cols:
                column_name = col_info["name"]
                self.column_types[column_name] = col_info["type"]

                if "vocab" in col_info:
                    vocab = col_info["vocab"]
                    encoded_columns = [f"{column_name}_unknown"]
                    encoded_columns.extend([f"{column_name}_{i}" for i in range(len(vocab))])
                    self.columns.extend(encoded_columns)
                    self.precomputed_columns[column_name] = encoded_columns
                elif "hash_buckets" in col_info:
                    num_buckets = col_info["hash_buckets"]
                    encoded_columns = [f"{column_name}_{i}" for i in range(num_buckets)]
                    self.columns.extend(encoded_columns)
                    self.precomputed_columns[column_name] = encoded_columns
                else:
                    self.columns.append(column_name)

    def _validate_schema(self) -> None:
        for name, dtype in self.column_types.items():
            if dtype not in ColumnDataType.dtype_map:
                raise RuntimeError(f"Invalid type: {dtype} for column: {name}")

    def remove_column(self, column_name: str) -> None:
        """Remove a column declaration from the schema in place.

        If the column is not present the call is a no-op (logged at INFO level).

        Args:
            column_name: Name of the column to remove.
        """
        original_len = len(self.raw_schema["columns"])
        self.raw_schema["columns"] = [e for e in self.raw_schema["columns"] if e["name"] != column_name]

        if len(self.raw_schema["columns"]) < original_len:
            del self.column_types[column_name]
            # Columns with vocab/hash_buckets expand to multiple entries in self.columns
            if column_name in self.precomputed_columns:
                for expanded_col in self.precomputed_columns.pop(column_name):
                    self.columns.remove(expanded_col)
            else:
                self.columns.remove(column_name)
            logger.info(f"Removed {column_name} from schema successfully")
        else:
            logger.info(f"Did not find {column_name} in schema")

    def merge(self, add_schema: "DatasetSchema") -> None:
        """Merge another schema's column declarations into this schema in place.

        All columns from ``add_schema`` are appended to this schema's column
        list.  After merging, ``_parse_schema`` is re-run to update
        ``self.columns`` and ``self.column_types``.

        Args:
            add_schema: The ``DatasetSchema`` whose columns will be added.
        """
        add_schema_dict = add_schema.raw_schema

        self.raw_schema = defaultdict(list, self.raw_schema)

        for key, value in add_schema_dict.items():
            self.raw_schema[key].extend(value)

        self._parse_schema()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply schema transformations to a DataFrame and return the result.

        For each column declared in the schema:
        - Columns with a ``vocab`` are one-hot encoded into
          ``<name>_<value>`` columns (unknown values map to
          ``<name>_unknown``).
        - Columns with ``hash_buckets`` are hashed into ``<name>_<i>``
          indicator columns.
        - All other columns are cast to the declared type.

        Columns not declared in the schema are silently dropped with a
        warning.

        Args:
            df: Input DataFrame.  Must contain all columns declared in the
                schema.

        Returns:
            Transformed DataFrame containing exactly the schema's columns in
            the schema-declared order.

        Raises:
            RuntimeError: If a required column is absent from ``df``.
        """
        unknown_cols = [i for i in df.columns if i not in self.columns]
        if len(unknown_cols) > 0:
            logger.warning(f"Will ignore columns {unknown_cols} in the DataFrame, they are unknown to the schema")

        for column_info in self.raw_schema["columns"]:
            name = column_info["name"]
            schema_type = column_info["type"]

            if name not in df.columns:
                raise RuntimeError(f"Column '{name}' not found in dataset")

            if "vocab" in column_info:
                all_vocabs = column_info["vocab"]
                df[name] = pd.Categorical(df[name], categories=all_vocabs, ordered=False)
                encoded_df = pd.get_dummies(df[name].cat.codes, prefix=name).astype("int32")
                encoded_df.rename(columns={f"{name}_-1": f"{name}_unknown"}, inplace=True)
                encoded_df = encoded_df.reindex(columns=self.precomputed_columns[name], fill_value=0).astype("int32")
                df = pd.concat([df.drop(columns=[name]), encoded_df], axis=1)
            elif "hash_buckets" in column_info:
                num_buckets = column_info["hash_buckets"]
                hash_bucket_column = pd.util.hash_array(df[name].fillna("nan").values) % num_buckets
                encoded_df = pd.get_dummies(hash_bucket_column, prefix=name)
                encoded_df = encoded_df.reindex(columns=self.precomputed_columns[name], fill_value=0).astype("int32")
                df = pd.concat([df.drop(columns=[name]), encoded_df], axis=1)
            else:
                dtype = ColumnDataType.dtype_map[schema_type]
                df[name] = df[name].astype(dtype)

        # drop columns not in schema
        df = df[self.columns]

        return df

    def apply_type_coercions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce the types declared in this schema on df.

        Raises RuntimeError if a required column is absent. Used after the
        client/default schema has been applied to ensure required columns always
        carry the contracted types regardless of whether a client schema was provided.
        """
        for column_info in self.raw_schema.get("columns", []):
            name = column_info["name"]
            if name not in df.columns:
                raise RuntimeError(
                    f"Required column '{name}' not found in dataset. Available columns: {list(df.columns)}"
                )
            dtype = ColumnDataType.dtype_map[column_info["type"]]
            df[name] = df[name].astype(dtype)
        return df

    @classmethod
    def apply_default_schema(cls, df: pd.DataFrame) -> pd.DataFrame:
        """in cases where no schema is provided, apply default data type mapping:
        - all int become int64
        - all float become float32
        - all bool become int64, True/False become 1/0
        - everything else becomes string
        """
        for column in df.columns:
            feature = df[column]
            if is_integer_dtype(feature):
                df[column] = feature.astype("int64")
            elif is_float_dtype(feature):
                df[column] = feature.astype("float32")
            elif is_bool_dtype(feature):
                df[column] = feature.astype("int64")
            else:
                df[column] = feature.astype("str")
        return df
