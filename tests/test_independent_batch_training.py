import inspect
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from skrec.dataset.batch_training_dataset import ItemBatchTrainingDataset
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.classification.xgb_classifier import BatchXGBClassifierEstimator
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_items,
    sample_binary_reward_users,
)
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.base_scorer import BaseScorer
from skrec.scorer.independent import IndependentScorer

BATCH_PARAMS = {"objective": "binary:logistic", "num_boost_round": 5, "tree_method": "hist"}


@pytest.fixture
def partitioned_datasets(tmp_path):
    """Two interaction partitions, with three things the batch path has to handle:

    - ITEM_A appears only in part-0, so its iterator must skip part-1.
    - ITEM_B has one label per partition but both labels overall, so the label
      check has to look at the whole dataset, not each partition.
    - ITEM_C is in the items dataset but has no interactions, so it is dropped.
    """
    users = pd.DataFrame({"USER_ID": [f"U{i}" for i in range(8)], "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    part0 = pd.DataFrame(
        {
            "USER_ID": ["U0", "U1", "U2", "U3"],
            "ITEM_ID": ["ITEM_A", "ITEM_A", "ITEM_B", "ITEM_B"],
            "OUTCOME": [0, 1, 0, 0],
        }
    )
    part1 = pd.DataFrame(
        {"USER_ID": ["U4", "U5", "U6"], "ITEM_ID": ["ITEM_B", "ITEM_B", "ITEM_B"], "OUTCOME": [1, 1, 1]}
    )
    items = pd.DataFrame({"ITEM_ID": ["ITEM_A", "ITEM_B", "ITEM_C"]})

    interactions_dir = tmp_path / "interactions"
    os.makedirs(interactions_dir)
    part0.to_parquet(interactions_dir / "part-0.parquet")
    part1.to_parquet(interactions_dir / "part-1.parquet")
    users.to_csv(tmp_path / "users.csv", index=False)
    items.to_csv(tmp_path / "items.csv", index=False)

    return {
        "users": UsersDataset(data_location=str(tmp_path / "users.csv")),
        "items": ItemsDataset(data_location=str(tmp_path / "items.csv")),
        "interactions": InteractionsDataset(data_location=str(interactions_dir)),
        "part0": part0,
        "part1": part1,
    }


def _recommend(recommender, users_df):
    interactions = pd.DataFrame({"USER_ID": users_df["USER_ID"]})
    return recommender.recommend(interactions, users_df, len(recommender.scorer.item_names))


def test_support_batch_training():
    assert IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)).support_batch_training() is True
    assert IndependentScorer({"ITEM_1": BatchXGBClassifierEstimator(BATCH_PARAMS)}).support_batch_training() is True
    assert IndependentScorer({}).support_batch_training() is False


@pytest.mark.parametrize("as_dict", [False, True], ids=["single", "dict"])
def test_batch_train_single_file(as_dict):
    if as_dict:
        estimator = {item: BatchXGBClassifierEstimator(BATCH_PARAMS) for item in ["ITEM_1", "ITEM_2", "ITEM_3"]}
    else:
        estimator = BatchXGBClassifierEstimator(BATCH_PARAMS)
    recommender = RankingRecommender(IndependentScorer(estimator))
    recommender.train(sample_binary_reward_users, sample_binary_reward_items, sample_binary_reward_interactions)

    trained = recommender.scorer.estimator
    assert_array_equal(np.array(list(trained.keys())), np.array(["ITEM_1", "ITEM_2", "ITEM_3"]))
    for est in trained.values():
        assert isinstance(est, BatchXGBClassifierEstimator)
        assert est.feature_names == ["feat1", "feat2"]
    if as_dict:
        for item, est in estimator.items():
            assert trained[item] is est

    users = pd.DataFrame({"USER_ID": ["Emma", "Jasper"], "feat1": [18, 65], "feat2": [0, 1]})
    assert _recommend(recommender, users).shape == (2, 3)


def test_batch_train_partitioned(partitioned_datasets):
    ds = partitioned_datasets
    recommender = RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    recommender.train(users_ds=ds["users"], items_ds=ds["items"], interactions_ds=ds["interactions"])

    # ITEM_C has no interactions and is dropped, as in the in-memory path
    assert_array_equal(recommender.scorer.item_names, np.array(["ITEM_A", "ITEM_B"]))
    for item in ["ITEM_A", "ITEM_B"]:
        assert recommender.scorer.estimator[item].feature_names == ["feat1"]

    users = pd.DataFrame({"USER_ID": ["U0", "U5"], "feat1": [1.0, 6.0]})
    scores = recommender.scorer.score_items(pd.DataFrame({"USER_ID": users["USER_ID"]}), users)
    assert scores.shape == (2, 2)
    assert np.all((scores.values >= 0) & (scores.values <= 1))


@pytest.mark.parametrize("buffer_rows, item_b_chunks", [(1_000_000, 1), (1, 2)], ids=["buffered", "flush_each"])
def test_split_partitions_by_item_and_iterate(partitioned_datasets, tmp_path, buffer_rows, item_b_chunks):
    ds = partitioned_datasets
    scorer = IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS))
    scorer.item_names = np.array(["ITEM_A", "ITEM_B", "ITEM_C"])
    scorer.batch_chunk_buffer_rows = buffer_rows
    item_labels, chunks = scorer._split_partitions_by_item(ds["interactions"], str(tmp_path / "chunks"))

    # ITEM_A is only in part-0, ITEM_B in both, ITEM_C in neither. Buffered rows are written
    # one file per item per flush, so ITEM_B's two partitions become one chunk unless flushed in between.
    assert {item: len(files) for item, files in chunks.items()} == {"ITEM_A": 1, "ITEM_B": item_b_chunks}
    assert sorted(map(tuple, item_labels.values.tolist())) == [
        ("ITEM_A", 0),
        ("ITEM_A", 1),
        ("ITEM_B", 0),
        ("ITEM_B", 1),
    ]

    def collect(item):
        rows, labels = [], []
        it = ItemBatchTrainingDataset(scorer, item, chunks[item], ds["users"].fetch_data())

        def grab(data, label):
            rows.extend(data["feat1"].tolist())
            labels.extend(label.tolist())

        batches = 0
        while it.next(grab):
            batches += 1
        return batches, sorted(rows), sorted(labels), it.feature_names

    assert collect("ITEM_A") == (1, [1.0, 2.0], [0, 1], ["feat1"])
    assert collect("ITEM_B") == (item_b_chunks, [3.0, 4.0, 5.0, 6.0, 7.0], [0, 0, 1, 1, 1], ["feat1"])


def test_batch_train_label_check_spans_partitions(partitioned_datasets, tmp_path):
    ds = partitioned_datasets
    # Same data, but ITEM_B is all-zero overall: the whole-dataset label check must reject it
    part1 = ds["part1"].assign(OUTCOME=0)
    bad_dir = tmp_path / "bad_interactions"
    os.makedirs(bad_dir)
    ds["part0"].to_parquet(bad_dir / "part-0.parquet")
    part1.to_parquet(bad_dir / "part-1.parquet")

    recommender = RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    with pytest.raises(ValueError, match="number of unique values of the target label must be 2"):
        recommender.train(
            users_ds=ds["users"], items_ds=ds["items"], interactions_ds=InteractionsDataset(data_location=str(bad_dir))
        )


def test_batch_train_with_validation_missing_an_item(partitioned_datasets, caplog):
    ds = partitioned_datasets
    # Validation has ITEM_B only, so ITEM_A trains without an eval set
    valid_dir = ds["interactions"].data_location + "_valid"
    os.makedirs(valid_dir)
    ds["part1"].assign(OUTCOME=[0, 1, 0]).to_parquet(os.path.join(valid_dir, "part-0.parquet"))
    ds["part1"].assign(OUTCOME=[1, 0, 1]).to_parquet(os.path.join(valid_dir, "part-1.parquet"))

    recommender = RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    recommender.train(
        users_ds=ds["users"],
        items_ds=ds["items"],
        interactions_ds=ds["interactions"],
        valid_users_ds=ds["users"],
        valid_interactions_ds=InteractionsDataset(data_location=valid_dir),
    )
    assert "Item ITEM_A has no validation interactions" in caplog.text
    assert_array_equal(recommender.scorer.item_names, np.array(["ITEM_A", "ITEM_B"]))


def test_batch_train_validation_requires_valid_users(partitioned_datasets):
    ds = partitioned_datasets
    recommender = RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    with pytest.raises(RuntimeError, match="Validation users dataset not found"):
        recommender.train(
            users_ds=ds["users"],
            items_ds=ds["items"],
            interactions_ds=ds["interactions"],
            valid_interactions_ds=ds["interactions"],
        )


def test_batch_train_with_item_specific_features(tmp_path):
    users = pd.DataFrame({"USER_ID": ["U0", "U1", "U2", "U3"], "f1": [1.0, 2.0, 3.0, 4.0], "f2": [4.0, 3.0, 2.0, 1.0]})
    interactions = pd.DataFrame(
        {
            "USER_ID": ["U0", "U1", "U2", "U3"] * 2,
            "ITEM_ID": ["ITEM_A"] * 4 + ["ITEM_B"] * 4,
            "OUTCOME": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    users.to_csv(tmp_path / "users.csv", index=False)
    interactions.to_csv(tmp_path / "interactions.csv", index=False)
    pd.DataFrame({"ITEM_ID": ["ITEM_A", "ITEM_B"]}).to_csv(tmp_path / "items.csv", index=False)

    recommender = RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    recommender.set_item_specific_features(
        item_specific_features_users={"ITEM_A": ["f1"]}, item_specific_features_interactions=None
    )
    recommender.train(
        users_ds=UsersDataset(data_location=str(tmp_path / "users.csv")),
        items_ds=ItemsDataset(data_location=str(tmp_path / "items.csv")),
        interactions_ds=InteractionsDataset(data_location=str(tmp_path / "interactions.csv")),
    )
    assert recommender.scorer.estimator["ITEM_A"].feature_names == ["f1"]
    assert recommender.scorer.estimator["ITEM_B"].feature_names == ["f1", "f2"]
    assert _recommend(recommender, users).shape == (4, 2)


def test_batch_train_drops_auxiliary_outcome_columns(partitioned_datasets, tmp_path):
    # OUTCOME_* columns are auxiliary labels, not features (the in-memory path drops them too)
    ds = partitioned_datasets
    aux_dir = tmp_path / "aux_interactions"
    os.makedirs(aux_dir)
    ds["part0"].assign(OUTCOME_click=1).to_parquet(aux_dir / "part-0.parquet")
    ds["part1"].assign(OUTCOME_click=0).to_parquet(aux_dir / "part-1.parquet")

    recommender = RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    recommender.train(
        users_ds=ds["users"], items_ds=ds["items"], interactions_ds=InteractionsDataset(data_location=str(aux_dir))
    )
    for item in ["ITEM_A", "ITEM_B"]:
        assert recommender.scorer.estimator[item].feature_names == ["feat1"]
    # inference input has no OUTCOME_click column, which must not break recommend()
    users = pd.DataFrame({"USER_ID": ["U0", "U5"], "feat1": [1.0, 6.0]})
    assert _recommend(recommender, users).shape == (2, 2)


def test_batch_train_label_check_matches_in_memory_for_dicts(partitioned_datasets, tmp_path):
    # The in-memory path runs the two-label check only for a single classifier, not a dict
    ds = partitioned_datasets
    bad_dir = tmp_path / "bad_interactions"
    os.makedirs(bad_dir)
    ds["part0"].to_parquet(bad_dir / "part-0.parquet")
    ds["part1"].assign(OUTCOME=0).to_parquet(bad_dir / "part-1.parquet")

    estimators = {item: BatchXGBClassifierEstimator(BATCH_PARAMS) for item in ["ITEM_A", "ITEM_B", "ITEM_C"]}
    recommender = RankingRecommender(IndependentScorer(estimators))
    recommender.train(
        users_ds=ds["users"], items_ds=ds["items"], interactions_ds=InteractionsDataset(data_location=str(bad_dir))
    )
    assert_array_equal(recommender.scorer.item_names, np.array(["ITEM_A", "ITEM_B"]))


def test_batch_train_model_signature_matches_base():
    base = list(inspect.signature(BaseScorer.batch_train_model).parameters)
    assert list(inspect.signature(IndependentScorer.batch_train_model).parameters) == base


def test_batch_train_fetches_users_once(partitioned_datasets):
    ds = partitioned_datasets
    recommender = RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    original = UsersDataset.fetch_data
    calls = []

    def counting_fetch(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    with patch.object(UsersDataset, "fetch_data", counting_fetch):
        recommender.train(users_ds=ds["users"], items_ds=ds["items"], interactions_ds=ds["interactions"])
    # once, for batch training: train() skips it in batch mode, and not per item or partition
    assert len(calls) == 1


def _counting(cls):
    """Patch cls.fetch_data to record the data_partition of every call."""
    original = cls.fetch_data
    calls = []

    def counting_fetch(self, data_partition=None):
        calls.append(data_partition)
        return original(self, data_partition)

    return patch.object(cls, "fetch_data", counting_fetch), calls


def test_batch_train_reads_each_partition_once_and_never_whole(partitioned_datasets):
    # One pass for all items, and BaseRecommender.train no longer loads the whole dataset
    ds = partitioned_datasets
    recommender = RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    patcher, calls = _counting(InteractionsDataset)
    with patcher:
        recommender.train(users_ds=ds["users"], items_ds=ds["items"], interactions_ds=ds["interactions"])
    assert None not in calls
    # train() finds the outcome column names from parquet metadata, so each partition is read exactly once
    assert sorted(calls) == sorted(ds["interactions"].data_filenames())


def test_batch_train_validation_column_mismatch(partitioned_datasets, tmp_path):
    ds = partitioned_datasets
    valid_users = pd.DataFrame({"USER_ID": [f"U{i}" for i in range(8)], "feat2": [1.0] * 8})
    valid_users.to_csv(tmp_path / "valid_users.csv", index=False)

    recommender = RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    with pytest.raises(ValueError, match="Training and validation data have different column names"):
        recommender.train(
            users_ds=ds["users"],
            items_ds=ds["items"],
            interactions_ds=ds["interactions"],
            valid_users_ds=UsersDataset(data_location=str(tmp_path / "valid_users.csv")),
            valid_interactions_ds=ds["interactions"],
        )


def test_batch_train_with_popularity_retriever(partitioned_datasets):
    # Batch training skips the eager interactions load; the retriever loads them on demand
    from skrec.retriever.popularity_retriever import PopularityRetriever

    ds = partitioned_datasets
    recommender = RankingRecommender(
        IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)), retriever=PopularityRetriever(top_k=2)
    )
    recommender.train(users_ds=ds["users"], items_ds=ds["items"], interactions_ds=ds["interactions"])
    assert len(recommender._train_interactions_df) == len(ds["part0"]) + len(ds["part1"])
    users = pd.DataFrame({"USER_ID": ["U0", "U5"], "feat1": [1.0, 6.0]})
    assert _recommend(recommender, users).shape == (2, 2)


def test_labels_collected_only_when_checked(partitioned_datasets, monkeypatch):
    # Unique (item, label) pairs of a continuous target are nearly every row, so collect labels only when checked
    ds = partitioned_datasets
    calls = []
    original = IndependentScorer._split_partitions_by_item

    def spy(self, interactions_ds, out_dir, process_interactions=None, collect_labels=True):
        seen, chunks = original(self, interactions_ds, out_dir, process_interactions, collect_labels)
        calls.append((collect_labels, list(seen.columns)))
        return seen, chunks

    monkeypatch.setattr(IndependentScorer, "_split_partitions_by_item", spy)
    single = RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    single.train(
        users_ds=ds["users"],
        items_ds=ds["items"],
        interactions_ds=ds["interactions"],
        valid_users_ds=ds["users"],
        valid_interactions_ds=ds["interactions"],
    )
    per_item = {item: BatchXGBClassifierEstimator(BATCH_PARAMS) for item in ["ITEM_A", "ITEM_B", "ITEM_C"]}
    RankingRecommender(IndependentScorer(per_item)).train(
        users_ds=ds["users"], items_ds=ds["items"], interactions_ds=ds["interactions"]
    )
    # single classifier: labels for training only (validation is never label-checked); dict: never
    assert calls == [(True, ["ITEM_ID", "OUTCOME"]), (False, ["ITEM_ID"]), (False, ["ITEM_ID"])]


def test_batch_train_scorer_pickled_before_batch_support(partitioned_datasets):
    # Older pickles lack attributes added to __init__ for batch training
    ds = partitioned_datasets
    scorer = IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS))
    del scorer._check_binary_labels
    del scorer.batch_chunk_dir
    recommender = RankingRecommender(scorer)
    recommender.train(users_ds=ds["users"], items_ds=ds["items"], interactions_ds=ds["interactions"])
    assert_array_equal(recommender.scorer.item_names, np.array(["ITEM_A", "ITEM_B"]))


def test_universal_batch_fetches_users_once(partitioned_datasets):
    from skrec.scorer.universal import UniversalScorer

    ds = partitioned_datasets
    original = UsersDataset.fetch_data
    calls = []

    def counting_fetch(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    recommender = RankingRecommender(UniversalScorer(BatchXGBClassifierEstimator(BATCH_PARAMS)))
    with patch.object(UsersDataset, "fetch_data", counting_fetch):
        recommender.train(users_ds=ds["users"], items_ds=ds["items"], interactions_ds=ds["interactions"])
    # once per iterator, not once per partition
    assert len(calls) == 1


def test_in_memory_retrain_still_checks_labels(tmp_path):
    # After a first train self.estimator is the per-item dict; the label check must still apply on retraining
    from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator

    users = pd.DataFrame({"USER_ID": ["U0", "U1", "U2", "U3"], "feat1": [1.0, 2.0, 3.0, 4.0]})
    good = pd.DataFrame({"USER_ID": ["U0", "U1", "U2", "U3"], "ITEM_ID": ["A", "A", "B", "B"], "OUTCOME": [0, 1, 0, 1]})
    users.to_csv(tmp_path / "users.csv", index=False)
    good.to_csv(tmp_path / "good.csv", index=False)
    good.assign(OUTCOME=1).to_csv(tmp_path / "bad.csv", index=False)
    users_ds = UsersDataset(data_location=str(tmp_path / "users.csv"))

    recommender = RankingRecommender(IndependentScorer(XGBClassifierEstimator({})))
    recommender.train(users_ds=users_ds, interactions_ds=InteractionsDataset(data_location=str(tmp_path / "good.csv")))
    with pytest.raises(ValueError, match="number of unique values of the target label must be 2"):
        recommender.train(
            users_ds=users_ds, interactions_ds=InteractionsDataset(data_location=str(tmp_path / "bad.csv"))
        )
