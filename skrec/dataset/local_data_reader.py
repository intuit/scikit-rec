import glob
import os
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import pyarrow.parquet as pq

from skrec.dataset.datatypes import DataFileFormat

# Design decision: We do not want to add capability to write dataset to local location


class LocalDataReader:
    def __init__(self, file_extension: str, data_location: str):
        if isinstance(data_location, Path):
            data_location = str(data_location)
        self.file_extension = file_extension
        self.local_path = data_location

    def read(self, columns: Optional[List[str]] = None):
        """Read the dataset into a DataFrame.

        Args:
            columns: Optional list of source column names to read. When provided,
                the read is projected to just these columns (parquet reads only
                the selected column chunks off disk; CSV parses only these
                columns). When ``None`` (default) every column is read, preserving
                the original full-read behavior.
        """
        if self.file_extension == DataFileFormat.CSV:
            df = pd.read_csv(self.local_path, usecols=columns)
        elif self.file_extension in {DataFileFormat.PARQUET, ""}:
            df = pd.read_parquet(self.local_path, columns=columns)
        else:
            raise ValueError("Unknown data file format")
        return df

    def available_columns(self) -> List[str]:
        """Return the column names present in the source data, cheaply.

        Reads only file metadata, not column data: the parquet footer schema for
        parquet, or a zero-row header parse for CSV. For a partitioned parquet
        directory the schema of a single file is used, assuming a uniform
        partition schema. Used to intersect a schema-derived projection against
        the columns that actually exist, so a genuinely-absent declared column is
        left for ``DatasetSchema.apply`` to reject with its usual ``RuntimeError``
        rather than surfacing a reader-level error with different semantics.
        """
        if self.file_extension == DataFileFormat.CSV:
            return list(pd.read_csv(self.local_path, nrows=0).columns)

        if self.file_extension in {DataFileFormat.PARQUET, ""}:
            if self.file_extension == "":
                # partitioned directory: read one file's footer
                filenames = sorted(self.get_data_filenames())
                if not filenames:
                    return []
                path = filenames[0]
            else:
                path = self.local_path
            return list(pq.ParquetFile(path).schema.names)

        raise ValueError("Unknown data file format")

    def get_data_filenames(self) -> Set[str]:
        """
        List all dataset files.
            - If the dataset is a single file, return a list with a single element
            - If the dataset is a directory, return a list of all data files in the directory
        """
        if self.file_extension:
            # this is a CSV or Parquet file, not a directory
            return set([self.local_path])
        else:
            parquet_expression = os.path.join(self.local_path, "*.parquet")
            filenames = glob.glob(parquet_expression, recursive=True)
            return set(filenames)
