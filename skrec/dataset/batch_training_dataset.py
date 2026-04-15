import os
import tempfile
from pathlib import Path
from random import shuffle
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost

from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class BatchTrainingDataset(xgboost.DataIter):
    """
    This dataset should only be used for XGB incremental training.
    """

    def __init__(
        self,
        scorer,
        interactions_dataset: InteractionsDataset,
        items_dataset: ItemsDataset,
        users_dataset: Optional[UsersDataset] = None,
    ):
        self.interactions_dataset = interactions_dataset
        self.items_dataset = items_dataset
        self.users_dataset = users_dataset
        self.scorer = scorer

        self.reset()
        self.cache_dir = tempfile.TemporaryDirectory()
        super().__init__(cache_prefix=self.cache_dir.name)

    def __del__(self):
        self.cache_dir.cleanup()

    def read_data(self, relative_filename=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read one data file from all datasets and join them into one DataFrame.
        If filename is None, read the whole datasets
        """
        if relative_filename is not None:
            filename = os.path.join(self.interactions_dataset.data_location, relative_filename)
        else:
            filename = None

        interactions_df = self.interactions_dataset.fetch_data(filename)
        interactions_df.reset_index(drop=True, inplace=True)

        self.scorer._validate_interactions(interactions_df)

        if self.users_dataset is not None:
            users_df = self.users_dataset.fetch_data(data_partition=None)
            users_df.reset_index(drop=True, inplace=True)
        else:
            users_df = None

        if not hasattr(self.scorer, "items_df"):
            items_df = self.items_dataset.fetch_data()
            items_df.reset_index(drop=True, inplace=True)

            is_partitioned = len(self.interactions_dataset.data_filenames()) > 1
            self.scorer.item_names, self.scorer.items_df = self.scorer._process_items(
                items_df=items_df, interactions_df=None, is_partitioned=is_partitioned
            )
        joined_data = self.scorer._join_data_train(
            users_df=users_df, items_df=self.scorer.items_df, interactions_df=interactions_df
        )
        if not len(joined_data):
            raise RuntimeError("Batch not joined!")
        X, y = self.scorer._process_X_y(joined_data)
        return X, y

    def has_next(self) -> bool:
        return bool(self.filenames)

    def next(self, input_data: Callable) -> int:
        if not self.has_next():
            return 0
        filename = self.filenames.pop()
        X, y = self.read_data(filename)
        X = pd.DataFrame(X)
        input_data(data=X, label=y)
        return 1

    def _relative_data_filenames(self, dataset) -> set:
        base_path = dataset.data_location
        filenames = dataset.data_filenames()
        relative_filenames = set()
        for filename in filenames:
            relative_filenames.add(str(Path(filename).relative_to(base_path)))
        return relative_filenames

    def reset(self) -> None:
        # set filenames
        interaction_filenames = self.interactions_dataset.data_filenames()
        if len(interaction_filenames) > 1:
            # the dataset is partitioned
            interaction_filenames = self._relative_data_filenames(self.interactions_dataset)

            filenames = list(interaction_filenames)
            shuffle(filenames)
            self.filenames = filenames

            if self.users_dataset is not None and self.users_dataset.is_partitioned:
                logger.warning("The users dataset is partitioned, but we load the whole dataset")
        else:
            # the dataset has only one file
            # if filename is None, read the whole dataset
            self.filenames = [None]
