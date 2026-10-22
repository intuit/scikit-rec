import os
import tempfile
from pathlib import Path
from random import shuffle
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost

from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.util.logger import get_logger

logger = get_logger(__name__)


def fetch_users(users_dataset: Optional[UsersDataset]) -> Optional[pd.DataFrame]:
    """The whole users dataset with a fresh index, or None without one."""
    if users_dataset is None:
        return None
    return users_dataset.fetch_data(data_partition=None).reset_index(drop=True)


def fetch_interactions(
    interactions_dataset: InteractionsDataset,
    partition: Optional[str] = None,
    process_interactions: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """One interactions partition (all of them if None), with the recommender's outcome-column handling applied."""
    interactions_df = interactions_dataset.fetch_data(partition).reset_index(drop=True)
    return process_interactions(interactions_df) if process_interactions is not None else interactions_df


class FileBatchIterator(xgboost.DataIter):
    """
    XGBoost external-memory iterator yielding one (X, y) batch per file, in shuffled order each pass.

    Subclasses set their attributes before calling ``__init__`` (which calls ``reset``) and
    implement ``_batch_files`` and ``read_data``.
    """

    def __init__(self):
        self.reset()
        self.cache_dir = tempfile.TemporaryDirectory()
        super().__init__(cache_prefix=self.cache_dir.name)

    def __del__(self):
        cache_dir = getattr(self, "cache_dir", None)
        if cache_dir is not None:
            cache_dir.cleanup()

    def _batch_files(self) -> List[Optional[str]]:
        raise NotImplementedError

    def read_data(self, filename: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError

    def has_next(self) -> bool:
        return bool(self.filenames)

    def next(self, input_data: Callable) -> int:
        if not self.has_next():
            return 0
        X, y = self.read_data(self.filenames.pop())
        input_data(data=pd.DataFrame(X), label=y)
        return 1

    def reset(self) -> None:
        filenames = list(self._batch_files())
        shuffle(filenames)
        self.filenames = filenames


class BatchTrainingDataset(FileBatchIterator):
    """
    This dataset should only be used for XGB incremental training.
    """

    def __init__(
        self,
        scorer,
        interactions_dataset: InteractionsDataset,
        items_dataset: ItemsDataset,
        users_dataset: Optional[UsersDataset] = None,
        process_interactions: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ):
        self.interactions_dataset = interactions_dataset
        self.process_interactions = process_interactions
        self.items_dataset = items_dataset
        self.users_dataset = users_dataset
        self.scorer = scorer
        # Fetched once: Dataset.fetch_data's single-slot cache is evicted by every partition read
        self.users_df = fetch_users(users_dataset)
        super().__init__()

    def read_data(self, relative_filename=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read one data file from all datasets and join them into one DataFrame.
        If filename is None, read the whole datasets
        """
        interactions_df, users_df = self._fetch_partition(relative_filename)

        self.scorer._validate_interactions(interactions_df)

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

    def _fetch_partition(self, relative_filename=None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Fetch one interactions file (all of them if None) and the whole users dataset."""
        if relative_filename is not None:
            filename = os.path.join(self.interactions_dataset.data_location, relative_filename)
        else:
            filename = None
        interactions_df = fetch_interactions(self.interactions_dataset, filename, self.process_interactions)
        return interactions_df, self.users_df

    def _relative_data_filenames(self, dataset) -> set:
        base_path = dataset.data_location
        filenames = dataset.data_filenames()
        relative_filenames = set()
        for filename in filenames:
            relative_filenames.add(str(Path(filename).relative_to(base_path)))
        return relative_filenames

    def _batch_files(self) -> List[Optional[str]]:
        interaction_filenames = self.interactions_dataset.data_filenames()
        if len(interaction_filenames) > 1:
            # the dataset is partitioned
            if self.users_dataset is not None and self.users_dataset.is_partitioned:
                logger.warning("The users dataset is partitioned, but we load the whole dataset")
            return list(self._relative_data_filenames(self.interactions_dataset))
        # the dataset has only one file
        # if filename is None, read the whole dataset
        return [None]


class ItemBatchTrainingDataset(FileBatchIterator):
    """
    One item's training rows, for IndependentScorer batch training.

    Reads the item's files with ``load_chunk``: the per-item parquet chunks the scorer
    split out of the interactions, or the dataset's own ``ITEM_ID=<value>/`` partitions.
    Applies the item-specific features and joins users. ``users_df`` is fetched once by
    the caller and shared across items. A validation iterator passes its training
    iterator as ``reference`` so a column mismatch raises a clear error.
    """

    def __init__(
        self,
        scorer,
        item: str,
        chunk_files: List[str],
        users_df: Optional[pd.DataFrame] = None,
        load_chunk: Callable[[str], pd.DataFrame] = pd.read_parquet,
        reference: Optional["ItemBatchTrainingDataset"] = None,
    ):
        self.scorer = scorer
        self.item = item
        self.chunk_files = list(chunk_files)
        self.users_df = users_df
        self.load_chunk = load_chunk
        self.reference = reference
        self.feature_names: Optional[List[str]] = None
        super().__init__()

    def _batch_files(self) -> List[Optional[str]]:
        return list(self.chunk_files)

    def read_data(self, chunk_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        interactions_df = self.load_chunk(chunk_file)
        users_df, interactions_df = self.scorer._apply_item_specific_features(self.item, self.users_df, interactions_df)
        joined_data = self.scorer._join_data_train(users_df=users_df, items_df=None, interactions_df=interactions_df)
        X, y = self.scorer._generate_X_y(joined_data)

        columns = X.columns.tolist()
        if self.feature_names is None:
            self.feature_names = columns
        elif columns != self.feature_names:
            raise RuntimeError(f"Partitions have different feature columns for item {self.item}!")
        # The training data is fully read before validation, so its columns are known by now
        if self.reference is not None and columns != self.reference.feature_names:
            raise ValueError(f"Training and validation data have different column names for item {self.item}")
        return X, y
