import pathlib
from functools import lru_cache
from typing import Any, List, Optional, Union

import pandas as pd

from skrec.dataset.datatypes import DataSource
from skrec.dataset.local_data_reader import LocalDataReader
from skrec.dataset.s3_data_reader import S3DataReader
from skrec.dataset.schema import DatasetSchema

S3SUBSTRING = "s3://"


class Dataset:
    def __init__(
        self,
        client_schema: Optional[DatasetSchema],
        required_schema: DatasetSchema,
        data_location: str,
        region: Optional[str] = None,
    ) -> None:
        self.client_schema = client_schema
        self.required_schema = required_schema
        self.data_location = data_location
        self.region = region
        self._validate_schema()

    def _validate_schema(self) -> None:
        if self.client_schema is None:
            return
        if not self._is_subset(self.required_schema.raw_schema, self.client_schema.raw_schema):
            raise ValueError("Client Schema does not conform to Required Schema")

    def _get_reader(self, data_partition: Optional[str] = None) -> Union[S3DataReader, LocalDataReader]:
        data_partition = self.data_location if data_partition is None else data_partition
        file_extension = pathlib.Path(self.data_location).suffix
        data_src = self.get_data_src(data_partition)

        if data_src == DataSource.S3:
            return S3DataReader(file_extension, data_partition, self.region)

        if data_src == DataSource.LOCAL:
            return LocalDataReader(file_extension, data_partition)

        raise ValueError("Unknown data source provided")

    @lru_cache(maxsize=None)
    def data_filenames(self) -> List[str]:
        """Return the list of data file paths for this dataset.

        For partitioned datasets (multiple files in a directory or S3 prefix)
        this returns all partition file paths.  For single-file datasets it
        returns a one-element list.  Results are cached after the first call.

        Returns:
            List of file path strings accessible by the configured data reader.
        """
        reader = self._get_reader()
        return reader.get_data_filenames()

    @lru_cache(maxsize=1)
    def fetch_data(self, data_partition: Optional[str] = None) -> pd.DataFrame:
        """Load and return the dataset as a DataFrame.

        Applies type coercions and schema validation.  When a ``client_schema``
        is provided, only the columns declared in that schema are returned and
        type conversions are applied as declared.  Without a schema, a default
        type-mapping is applied (int→int64, float→float32, other→str).

        Results are cached after the first call — pass a different
        ``data_partition`` to bypass the cache and load an alternate partition.

        Args:
            data_partition: Optional path override.  When ``None``, loads from
                ``self.data_location``.

        Returns:
            Validated and type-coerced pandas DataFrame.
        """
        reader = self._get_reader(data_partition)
        data = reader.read()

        if self.client_schema is None:
            # converts the data types to standard data types.
            # floats becomes float32, ints becomes int64, everything else becomes str
            data_with_schema = DatasetSchema.apply_default_schema(data)
            # client_schema.apply() handles type enforcement when a schema is provided.
            # Without one, apply required_schema types explicitly (e.g. ITEM_ID → str)
            # and raise if a required column is missing entirely.
            data_with_schema = self.required_schema.apply_type_coercions(data_with_schema)
        else:
            data_with_schema = self.client_schema.apply(data)

        self.column_names = data_with_schema.columns.tolist()
        return data_with_schema

    def _is_subset(self, subset: Any, superset: Any) -> bool:
        if isinstance(subset, dict):
            return all(key in superset and self._is_subset(val, superset[key]) for key, val in subset.items())
        if isinstance(subset, list) or isinstance(subset, set):
            return all(any(self._is_subset(subitem, superitem) for superitem in superset) for subitem in subset)
        # assume that subset is a plain value if none of the above match
        return subset == superset

    def get_data_src(self, data_location: Union[str, pathlib.Path]) -> DataSource:
        """Determine whether a data location points to S3 or the local filesystem.

        Args:
            data_location: Path string (or path-like object).  Paths beginning
                with ``s3://`` are treated as S3; all others as local.

        Returns:
            ``DataSource.S3`` or ``DataSource.LOCAL``.
        """
        if type(data_location) is not str:
            data_location = str(data_location)

        if data_location.find(S3SUBSTRING) != -1:
            return DataSource.S3
        else:
            return DataSource.LOCAL
