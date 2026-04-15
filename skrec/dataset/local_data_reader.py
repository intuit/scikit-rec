import glob
import os
from pathlib import Path
from typing import Set

import pandas as pd

from skrec.dataset.datatypes import DataFileFormat

# Design decision: We do not want to add capability to write dataset to local location


class LocalDataReader:
    def __init__(self, file_extension: str, data_location: str):
        if isinstance(data_location, Path):
            data_location = str(data_location)
        self.file_extension = file_extension
        self.local_path = data_location

    def read(self):
        if self.file_extension == DataFileFormat.CSV:
            df = pd.read_csv(self.local_path)
        elif self.file_extension in {DataFileFormat.PARQUET, ""}:
            df = pd.read_parquet(self.local_path)
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
            return set([self.local_path])
        else:
            parquet_expression = os.path.join(self.local_path, "*.parquet")
            filenames = glob.glob(parquet_expression, recursive=True)
            return set(filenames)
