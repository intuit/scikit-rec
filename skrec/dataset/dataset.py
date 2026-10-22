import pathlib
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from skrec.dataset.datatypes import DataSource
from skrec.dataset.local_data_reader import LocalDataReader
from skrec.dataset.partitioning import hive_partition_values
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
        # The partition's own suffix: a partition file of a directory dataset is a file, not a directory
        file_extension = pathlib.Path(data_partition).suffix
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

        # Hive partition columns (``<data_location>/ITEM_ID=<value>/...``) are in the path, not the file
        partition_columns = self._partition_columns(data_partition)

        # Push a column projection down into the read when a client schema is
        # present: only the schema's declared source columns are materialized,
        # instead of reading every column and dropping the undeclared ones in
        # client_schema.apply(). This is a pure I/O/memory optimization — the
        # output DataFrame is identical, since apply() narrows to schema columns
        # anyway. The projection is intersected against the columns that actually
        # exist so a genuinely-absent declared column is left for apply() to
        # reject with its usual RuntimeError (rather than a reader-level error).
        projection: Optional[List[str]] = None
        if self.client_schema is not None:
            available = set(reader.available_columns())
            if data_partition is None:
                # whole-directory reads add the partition columns themselves
                available |= set(partition_columns)
            projection = [c for c in self.client_schema.source_columns() if c in available]

        data = reader.read(columns=projection)
        if data_partition is not None:
            for column, value in partition_columns.items():
                if column not in data.columns:
                    data[column] = value
        else:
            # Whole-directory reads add every partition key (pyarrow does for local data); keep declared ones only
            undeclared = [c for c in self._partition_keys() if c not in partition_columns and c in data.columns]
            data = data.drop(columns=undeclared)

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

    def peek_columns(self) -> List[str]:
        """Source column names of the dataset, from file metadata instead of reading the data.

        These are the columns :meth:`fetch_data` reads, before any vocab or hash-bucket
        expansion: the declared columns that exist when a client schema is set,
        otherwise all of them, plus the declared Hive partition columns.
        """
        filenames = sorted(self.data_filenames())
        partition = filenames[0] if len(filenames) > 1 else None
        available = list(self._get_reader(partition).available_columns())
        available += [c for c in self._partition_columns(partition) if c not in available]
        if self.client_schema is None:
            return available
        return [c for c in self.client_schema.source_columns() if c in set(available)]

    def _declared_columns(self) -> set:
        declared = set(self.required_schema.source_columns())
        if self.client_schema is not None:
            declared |= set(self.client_schema.source_columns())
        return declared

    def _partition_keys(self) -> List[str]:
        """Hive partition keys of the dataset, from its first file."""
        if pathlib.Path(self.data_location).suffix:
            return []
        filenames = sorted(self.data_filenames())
        return list(hive_partition_values(filenames[0], str(self.data_location))) if filenames else []

    def _partition_columns(self, data_partition: Optional[str]) -> Dict[str, Optional[str]]:
        """Declared Hive partition columns: the values of one partition file, or the keys of the whole directory.

        Partition keys are often for data management (``month=``, ``dt=``), so only those
        declared by the required or client schema (e.g. ``ITEM_ID``) become columns.
        """
        declared = self._declared_columns()
        if data_partition is not None:
            values = hive_partition_values(str(data_partition), str(self.data_location))
        else:
            values = {key: None for key in self._partition_keys()}
        return {k: v for k, v in values.items() if k in declared}

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
