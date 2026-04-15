import os
from typing import Optional

from skrec.constants import ITEM_ID_NAME, LABEL_NAME
from skrec.dataset.dataset import Dataset
from skrec.dataset.schema import DatasetSchema


class InteractionsDataset(Dataset):
    dirname = os.path.dirname(__file__)
    REQUIRED_SCHEMA_PATH_TRAINING = os.path.join(dirname, "required_schemas/interactions_schema_training.yaml")
    REQUIRED_SCHEMA_PATH_INFERENCE = os.path.join(dirname, "required_schemas/interactions_schema_inference.yaml")

    def __init__(
        self,
        data_location: str,
        client_schema_path: Optional[str] = None,
        extra_required_schema_path: Optional[str] = None,
        is_training: bool = True,
    ):
        if is_training:
            required_schema = DatasetSchema.create(self.REQUIRED_SCHEMA_PATH_TRAINING)
        else:
            required_schema = DatasetSchema.create(self.REQUIRED_SCHEMA_PATH_INFERENCE)

        if client_schema_path is None:
            client_schema = None
        else:
            client_schema = DatasetSchema.create(client_schema_path)

        # We expect the client to use the same schema for training and inference
        # But the client may not have label columns in the inference data, so we modify the inference schema here
        if not is_training and client_schema_path is not None:
            columns_to_remove = [ITEM_ID_NAME, LABEL_NAME]
            for col in columns_to_remove:
                client_schema.remove_column(col)

        if extra_required_schema_path is not None:
            extra_schema = DatasetSchema.create(extra_required_schema_path)
            required_schema.merge(extra_schema)

        super().__init__(client_schema, required_schema, data_location)


class InteractionMultiOutputDataset(InteractionsDataset):
    dirname = os.path.dirname(__file__)
    REQUIRED_SCHEMA_PATH_TRAINING = os.path.join(
        dirname, "required_schemas/interaction_multioutput_schema_training.yaml"
    )

    def __init__(
        self,
        data_location: str,
        client_schema_path: Optional[str] = None,
        extra_required_schema_path: Optional[str] = None,
        is_training: bool = True,
    ):
        super().__init__(data_location, client_schema_path, extra_required_schema_path, is_training)


class InteractionMultiClassDataset(InteractionsDataset):
    dirname = os.path.dirname(__file__)
    REQUIRED_SCHEMA_PATH_TRAINING = os.path.join(
        dirname, "required_schemas/interaction_multiclass_schema_training.yaml"
    )

    def __init__(
        self,
        data_location: str,
        client_schema_path: Optional[str] = None,
        extra_required_schema_path: Optional[str] = None,
        is_training: bool = True,
    ):
        super().__init__(data_location, client_schema_path, extra_required_schema_path, is_training)
        if self.client_schema:
            if "OUTCOME" in self.client_schema.columns:
                raise ValueError("For MultiClass Dataset, field OUTCOME should not be included")
