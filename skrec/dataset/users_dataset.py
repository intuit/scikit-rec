import os
from typing import Optional

from skrec.dataset.dataset import Dataset
from skrec.dataset.schema import DatasetSchema


class UsersDataset(Dataset):
    dirname = os.path.dirname(__file__)
    REQUIRED_SCHEMA_PATH = os.path.join(dirname, "required_schemas/users_schema.yaml")

    def __init__(
        self,
        data_location: str,
        client_schema_path: Optional[str] = None,
        extra_required_schema_path: Optional[str] = None,
        is_partitioned: bool = False,
    ):
        """
        Keyword arguments:
            is_partitioned:
                Indicates the users dataset is partitioned into the same files names as interactions dataset.
                In addition to that, all USER_IDs found in a file from the interactions dataset should be found
                    in the same filename from the users dataset.
        """
        self.is_partitioned = is_partitioned
        required_schema = DatasetSchema.create(self.REQUIRED_SCHEMA_PATH)
        if client_schema_path is None:
            client_schema = None
        else:
            client_schema = DatasetSchema.create(client_schema_path)

        if extra_required_schema_path is not None:
            extra_schema = DatasetSchema.create(extra_required_schema_path)
            required_schema.merge(extra_schema)

        super().__init__(client_schema, required_schema, data_location)
