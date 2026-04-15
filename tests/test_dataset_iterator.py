import os

import pandas as pd

from skrec.dataset.batch_training_dataset import BatchTrainingDataset
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.classification.xgb_classifier import BatchXGBClassifierEstimator
from skrec.scorer.universal import UniversalScorer


class IteratorDataCollector:
    def __init__(self):
        self.x = []
        self.y = []

    def grab_x_y(self, data, label):
        self.x += data.values.tolist()
        self.y += label.tolist()


def test_single_file_dataset(setup_small_datasets):
    interactions_dataset = setup_small_datasets["interactions_dataset"]
    users_dataset = setup_small_datasets["users_dataset"]
    items_dataset = setup_small_datasets["items_dataset"]

    estimator = BatchXGBClassifierEstimator()
    scorer = UniversalScorer(estimator=estimator)

    batch_iterator = BatchTrainingDataset(
        scorer=scorer,
        interactions_dataset=interactions_dataset,
        users_dataset=users_dataset,
        items_dataset=items_dataset,
    )

    collector = IteratorDataCollector()

    assert batch_iterator.next(collector.grab_x_y) == 1
    assert batch_iterator.next(collector.grab_x_y) == 0
    assert len(collector.x) == len(collector.y) == 4
    assert sum(collector.y) == 2


def test_partitioned_interactions_dataset(setup_small_datasets):
    data_dir = setup_small_datasets["dst"]
    interactions_data = pd.read_csv(setup_small_datasets["dst"] / setup_small_datasets["interactions_filename"])
    interactions_dir = os.path.join(data_dir, "interactions")
    os.makedirs(interactions_dir)
    interactions_data.to_parquet(os.path.join(interactions_dir, "part-00000.parquet"))
    interactions_data["USER_ID"] = interactions_data["USER_ID"] + "f2"
    interactions_data.to_parquet(os.path.join(interactions_dir, "part-00001.parquet"))

    users_data_batch1 = pd.read_csv(setup_small_datasets["dst"] / setup_small_datasets["users_filename"])
    users_data_batch2 = users_data_batch1.copy()
    users_data_batch2["USER_ID"] = users_data_batch2["USER_ID"] + "f2"
    users_data = pd.concat([users_data_batch1, users_data_batch2], axis=0)

    users_dir = os.path.join(data_dir, "users")
    os.makedirs(users_dir)
    users_data.to_parquet(os.path.join(users_dir, "part-00000.parquet"))

    interactions_dataset = InteractionsDataset(
        data_location=interactions_dir,
    )

    users_dataset = UsersDataset(data_location=users_dir)

    items_dataset = ItemsDataset(
        data_location=os.path.join(setup_small_datasets["dst"] / setup_small_datasets["items_filename"]),
    )

    estimator = BatchXGBClassifierEstimator()
    scorer = UniversalScorer(estimator=estimator)

    batch_iterator = BatchTrainingDataset(
        scorer=scorer,
        interactions_dataset=interactions_dataset,
        users_dataset=users_dataset,
        items_dataset=items_dataset,
    )

    collector = IteratorDataCollector()

    assert batch_iterator.next(collector.grab_x_y) == 1
    assert batch_iterator.next(collector.grab_x_y) == 1
    assert batch_iterator.next(collector.grab_x_y) == 0
    assert len(collector.x) == len(interactions_data) * 2 == len(collector.y)

    users_data_batch1.to_parquet(os.path.join(users_dir, "part-00000.parquet"))
    users_data_batch2.to_parquet(os.path.join(users_dir, "part-00001.parquet"))

    users_dataset = UsersDataset(data_location=users_dir, is_partitioned=True)

    estimator = BatchXGBClassifierEstimator()
    scorer = UniversalScorer(estimator=estimator)

    batch_iterator = BatchTrainingDataset(
        scorer=scorer,
        interactions_dataset=interactions_dataset,
        users_dataset=users_dataset,
        items_dataset=items_dataset,
    )

    collector = IteratorDataCollector()

    assert batch_iterator.next(collector.grab_x_y) == 1
    assert batch_iterator.next(collector.grab_x_y) == 1
    assert batch_iterator.next(collector.grab_x_y) == 0
    assert len(collector.x) == len(interactions_data) * 2 == len(collector.y)


def test_partitioned_users_dataset(setup_small_datasets):
    data_dir = setup_small_datasets["dst"]
    interactions_data = pd.read_csv(setup_small_datasets["dst"] / setup_small_datasets["interactions_filename"])
    interactions_dir = os.path.join(data_dir, "interactions")
    os.makedirs(interactions_dir)
    interactions_data.to_parquet(os.path.join(interactions_dir, "a.parquet"))
    interactions_data.to_parquet(os.path.join(interactions_dir, "b.parquet"))

    users_data = pd.read_csv(setup_small_datasets["dst"] / setup_small_datasets["users_filename"])
    users_dir = os.path.join(data_dir, "users")
    os.makedirs(users_dir)
    users_data.to_parquet(os.path.join(users_dir, "c.parquet"))
    users_data.to_parquet(os.path.join(users_dir, "d.parquet"))

    interactions_dataset = InteractionsDataset(
        client_schema_path=os.path.join(data_dir, "sample_interactions_schema.yaml"),
        data_location=interactions_dir,
    )

    users_dataset = UsersDataset(
        client_schema_path=os.path.join(data_dir, "sample_users_schema.yaml"),
        data_location=users_dir,
        is_partitioned=True,
    )

    items_dataset = ItemsDataset(
        client_schema_path=os.path.join(data_dir, "sample_items_schema.yaml"),
        data_location=setup_small_datasets["dst"] / setup_small_datasets["items_filename"],
    )

    estimator = BatchXGBClassifierEstimator()
    scorer = UniversalScorer(estimator=estimator)
    BatchTrainingDataset(
        scorer=scorer,
        interactions_dataset=interactions_dataset,
        users_dataset=users_dataset,
        items_dataset=items_dataset,
    )
