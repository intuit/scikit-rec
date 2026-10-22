import os
from urllib.parse import quote

import boto3
import numpy as np
import pandas as pd
import pytest
import yaml
from moto import mock_aws
from numpy.testing import assert_array_almost_equal, assert_array_equal

from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.partitioning import hive_partition_values, is_hidden_path
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.classification.xgb_classifier import BatchXGBClassifierEstimator
from skrec.orchestrator.factory import create_scorer
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.independent import IndependentScorer

BATCH_PARAMS = {"objective": "binary:logistic", "num_boost_round": 5, "tree_method": "hist", "nthread": 1}

# ITEM_A's labels are split across its two files; "ITEM B/x" needs URL-escaping in a path
ROWS = {
    "ITEM_A": [
        pd.DataFrame({"USER_ID": ["U0", "U1"], "OUTCOME": [0, 0]}),
        pd.DataFrame({"USER_ID": ["U2", "U3"], "OUTCOME": [1, 1]}),
    ],
    "ITEM B/x": [pd.DataFrame({"USER_ID": ["U4", "U5", "U6"], "OUTCOME": [1, 0, 1]})],
}
USERS = pd.DataFrame({"USER_ID": [f"U{i}" for i in range(7)], "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})


def write_hive(root, rows=ROWS, extra_columns=None):
    """Write rows as <root>/ITEM_ID=<item>/part-<i>.parquet, without an ITEM_ID column in the files."""
    for item, parts in rows.items():
        item_dir = os.path.join(root, f"ITEM_ID={quote(item, safe='')}")
        os.makedirs(item_dir)
        for i, part in enumerate(parts):
            part.assign(**(extra_columns or {})).to_parquet(os.path.join(item_dir, f"part-{i}.parquet"))
    return str(root)


def write_flat(root, rows=ROWS):
    """The same rows as flat partitions with an ITEM_ID column, part-<i> holding the i-th file of every item."""
    os.makedirs(root)
    for i in range(max(len(parts) for parts in rows.values())):
        frames = [parts[i].assign(ITEM_ID=item) for item, parts in rows.items() if i < len(parts)]
        pd.concat(frames, ignore_index=True).to_parquet(os.path.join(root, f"part-{i}.parquet"))
    return str(root)


@pytest.fixture
def datasets(tmp_path):
    USERS.to_csv(tmp_path / "users.csv", index=False)
    pd.DataFrame({"ITEM_ID": ["ITEM_A", "ITEM B/x", "ITEM_C"]}).to_csv(tmp_path / "items.csv", index=False)
    return {
        "users": UsersDataset(data_location=str(tmp_path / "users.csv")),
        "items": ItemsDataset(data_location=str(tmp_path / "items.csv")),
        "hive": InteractionsDataset(data_location=write_hive(tmp_path / "hive")),
        "flat": InteractionsDataset(data_location=write_flat(tmp_path / "flat")),
    }


def _batch_recommender(**scorer_kwargs):
    return RankingRecommender(IndependentScorer(BatchXGBClassifierEstimator(BATCH_PARAMS), **scorer_kwargs))


# --- dataset layer ---------------------------------------------------------------------------------------------------


def test_hive_partition_values():
    root = "s3://bucket/data/"
    assert hive_partition_values("s3://bucket/data/ITEM_ID=a%2Fb/day=1/part-0.parquet", root) == {
        "ITEM_ID": "a/b",
        "day": "1",
    }
    assert hive_partition_values("/data/ITEM_ID=__HIVE_DEFAULT_PARTITION__/p.parquet", "/data") == {"ITEM_ID": None}
    assert hive_partition_values("/data/plain/p.parquet", "/data") == {}
    assert hive_partition_values("/other/ITEM_ID=a/p.parquet", "/data") == {}
    assert is_hidden_path("/data/_temporary/0/p.parquet", "/data")
    assert not is_hidden_path("/data/ITEM_ID=_a/p.parquet", "/data")


def test_local_listing_is_recursive_and_skips_hidden_dirs(tmp_path):
    root = write_hive(tmp_path / "hive")
    os.makedirs(tmp_path / "hive" / "_temporary")
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "hive" / "_temporary" / "junk.parquet")
    files = InteractionsDataset(data_location=root).data_filenames()
    assert sorted(os.path.relpath(f, root) for f in files) == [
        "ITEM_ID=ITEM%20B%2Fx/part-0.parquet",
        "ITEM_ID=ITEM_A/part-0.parquet",
        "ITEM_ID=ITEM_A/part-1.parquet",
    ]


@pytest.mark.parametrize("with_schema", [False, True], ids=["no_schema", "schema"])
def test_partition_and_whole_reads_add_partition_column(tmp_path, with_schema):
    schema_path = None
    if with_schema:
        schema_path = str(tmp_path / "schema.yaml")
        columns = [{"name": "USER_ID", "type": "str"}, {"name": "ITEM_ID", "type": "str"}]
        columns.append({"name": "OUTCOME", "type": "float"})
        with open(schema_path, "w") as f:
            yaml.safe_dump({"columns": columns}, f)
    ds = InteractionsDataset(data_location=write_hive(tmp_path / "hive"), client_schema_path=schema_path)

    parts = [ds.fetch_data(f) for f in sorted(ds.data_filenames())]
    assert [sorted(set(p["ITEM_ID"])) for p in parts] == [["ITEM B/x"], ["ITEM_A"], ["ITEM_A"]]
    whole = ds.fetch_data()
    assert sorted(whole["ITEM_ID"]) == sorted(pd.concat(parts)["ITEM_ID"])
    assert whole["ITEM_ID"].dtype == parts[0]["ITEM_ID"].dtype


@mock_aws
def test_s3_whole_read_adds_partition_column(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    s3 = boto3.resource("s3", region_name="us-west-2")
    s3.Bucket("test").create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    for item, parts in ROWS.items():
        for i, part in enumerate(parts):
            key = f"hive/ITEM_ID={quote(item, safe='')}/part-{i}.parquet"
            s3.Object("test", key).put(Body=part.to_parquet())

    whole = InteractionsDataset(data_location="s3://test/hive/").fetch_data()
    assert sorted(whole["ITEM_ID"]) == ["ITEM B/x"] * 3 + ["ITEM_A"] * 4


# --- IndependentScorer batch training on item-partitioned data -------------------------------------------------------


def test_item_partitioned_training_makes_no_copy(datasets, monkeypatch):
    def no_split(*args, **kwargs):
        raise AssertionError("item-partitioned data must not be split into chunks")

    monkeypatch.setattr(IndependentScorer, "_split_partitions_by_item", no_split)
    original = InteractionsDataset.fetch_data
    reads = []

    def counting_fetch(self, data_partition=None):
        reads.append(data_partition)
        return original(self, data_partition)

    monkeypatch.setattr(InteractionsDataset, "fetch_data", counting_fetch)
    recommender = _batch_recommender()
    recommender.train(users_ds=datasets["users"], items_ds=datasets["items"], interactions_ds=datasets["hive"])

    # ITEM_C has no partition directory, so it is dropped
    assert_array_equal(recommender.scorer.item_names, np.array(["ITEM B/x", "ITEM_A"]))
    files = sorted(datasets["hive"].data_filenames())
    # never the whole dataset; each file at most twice (validate, train)
    assert None not in reads
    assert all(reads.count(f) <= 2 for f in files)
    for item in recommender.scorer.item_names:
        assert recommender.scorer.estimator[item].feature_names == ["feat1"]


def test_item_partitioned_matches_flat_layout(datasets):
    hive = _batch_recommender()
    hive.train(users_ds=datasets["users"], items_ds=datasets["items"], interactions_ds=datasets["hive"])
    flat = _batch_recommender()
    flat.train(users_ds=datasets["users"], items_ds=datasets["items"], interactions_ds=datasets["flat"])

    assert_array_equal(hive.scorer.item_names, flat.scorer.item_names)
    users = USERS.head(4)
    interactions = pd.DataFrame({"USER_ID": users["USER_ID"]})
    assert_array_almost_equal(
        hive.scorer.score_items(interactions, users).values, flat.scorer.score_items(interactions, users).values
    )


def test_item_partitioned_label_check(datasets, tmp_path):
    rows = {"ITEM_A": [pd.DataFrame({"USER_ID": ["U0", "U1"], "OUTCOME": [1, 1]})]}
    bad = InteractionsDataset(data_location=write_hive(tmp_path / "bad", rows))
    with pytest.raises(ValueError, match="number of unique values of the target label must be 2"):
        _batch_recommender().train(users_ds=datasets["users"], items_ds=datasets["items"], interactions_ds=bad)


def test_item_partitioned_drops_auxiliary_outcome_columns(datasets, tmp_path):
    aux = InteractionsDataset(data_location=write_hive(tmp_path / "aux", extra_columns={"OUTCOME_click": 1}))
    recommender = _batch_recommender()
    recommender.train(users_ds=datasets["users"], items_ds=datasets["items"], interactions_ds=aux)
    assert recommender.outcome_cols == ["OUTCOME_click"]
    for item in recommender.scorer.item_names:
        assert recommender.scorer.estimator[item].feature_names == ["feat1"]


def test_item_partitioned_with_validation(datasets, tmp_path):
    valid = InteractionsDataset(data_location=write_hive(tmp_path / "valid", {"ITEM_A": ROWS["ITEM_A"]}))
    recommender = _batch_recommender()
    recommender.train(
        users_ds=datasets["users"],
        items_ds=datasets["items"],
        interactions_ds=datasets["hive"],
        valid_users_ds=datasets["users"],
        valid_interactions_ds=valid,
    )
    assert_array_equal(recommender.scorer.item_names, np.array(["ITEM B/x", "ITEM_A"]))


def test_partly_item_partitioned_falls_back_to_split(datasets, tmp_path, monkeypatch):
    root = write_hive(tmp_path / "mixed")
    ROWS["ITEM_A"][0].assign(ITEM_ID="ITEM_A").to_parquet(os.path.join(root, "loose.parquet"))
    calls = []
    original = IndependentScorer._split_partitions_by_item

    def spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IndependentScorer, "_split_partitions_by_item", spy)
    recommender = _batch_recommender()
    recommender.train(
        users_ds=datasets["users"],
        items_ds=datasets["items"],
        interactions_ds=InteractionsDataset(data_location=root),
    )
    assert len(calls) == 1
    assert_array_equal(recommender.scorer.item_names, np.array(["ITEM B/x", "ITEM_A"]))


# --- batch_chunk_dir -------------------------------------------------------------------------------------------------


def test_batch_chunk_dir_holds_chunks_and_is_cleaned_up(datasets, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    os.makedirs(scratch)
    out_dirs = []
    original = IndependentScorer._split_partitions_by_item

    def spy(self, interactions_ds, out_dir, *args, **kwargs):
        out_dirs.append(out_dir)
        result = original(self, interactions_ds, out_dir, *args, **kwargs)
        assert all(f.startswith(str(scratch)) for files in result[1].values() for f in files)
        return result

    monkeypatch.setattr(IndependentScorer, "_split_partitions_by_item", spy)
    recommender = _batch_recommender(batch_chunk_dir=str(scratch))
    recommender.train(users_ds=datasets["users"], items_ds=datasets["items"], interactions_ds=datasets["flat"])

    assert len(out_dirs) == 1 and out_dirs[0].startswith(str(scratch))
    assert os.listdir(scratch) == []


def test_batch_chunk_dir_from_scorer_config(tmp_path):
    scorer = create_scorer(
        BatchXGBClassifierEstimator(BATCH_PARAMS),
        {"scorer_type": "independent", "scorer_config": {"batch_chunk_dir": str(tmp_path)}},
    )
    assert isinstance(scorer, IndependentScorer)
    assert scorer.batch_chunk_dir == str(tmp_path)


# --- S3 --------------------------------------------------------------------------------------------------------------


def _put_s3(s3, key, df):
    s3.Object("test", key).put(Body=df.to_parquet())


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    with mock_aws():
        resource = boto3.resource("s3", region_name="us-west-2")
        resource.Bucket("test").create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        yield resource


def test_s3_single_partition_read_and_hidden_dirs(s3):
    for item, parts in ROWS.items():
        for i, part in enumerate(parts):
            _put_s3(s3, f"hive/ITEM_ID={quote(item, safe='')}/part-{i}.parquet", part)
    # leftovers of a failed Spark write must not be listed or read
    _put_s3(s3, "hive/_temporary/0/ITEM_ID=ITEM_A/part-9.parquet", ROWS["ITEM_A"][0])

    ds = InteractionsDataset(data_location="s3://test/hive/")
    files = sorted(ds.data_filenames())
    assert [f.split("hive/", 1)[1] for f in files] == [
        "ITEM_ID=ITEM%20B%2Fx/part-0.parquet",
        "ITEM_ID=ITEM_A/part-0.parquet",
        "ITEM_ID=ITEM_A/part-1.parquet",
    ]
    part = ds.fetch_data(files[1])
    assert part["USER_ID"].tolist() == ["U0", "U1"] and set(part["ITEM_ID"]) == {"ITEM_A"}
    assert sorted(ds.peek_columns()) == ["ITEM_ID", "OUTCOME", "USER_ID"]


@pytest.mark.parametrize("layout", ["hive", "flat"])
def test_s3_batch_training(s3, datasets, layout):
    if layout == "hive":
        for item, parts in ROWS.items():
            for i, part in enumerate(parts):
                _put_s3(s3, f"data/ITEM_ID={quote(item, safe='')}/part-{i}.parquet", part)
    else:
        for i in range(2):
            frames = [parts[i].assign(ITEM_ID=item) for item, parts in ROWS.items() if i < len(parts)]
            _put_s3(s3, f"data/part-{i}.parquet", pd.concat(frames, ignore_index=True))

    recommender = _batch_recommender()
    recommender.train(
        users_ds=datasets["users"],
        items_ds=datasets["items"],
        interactions_ds=InteractionsDataset(data_location="s3://test/data/"),
    )
    assert_array_equal(recommender.scorer.item_names, np.array(["ITEM B/x", "ITEM_A"]))
    for item in recommender.scorer.item_names:
        assert recommender.scorer.estimator[item].feature_names == ["feat1"]


# --- undeclared partition keys ---------------------------------------------------------------------------------------


def _write_month_layout(root):
    for month in (1, 2):
        item_dir = os.path.join(root, f"month={month}", "ITEM_ID=ITEM_A")
        os.makedirs(item_dir)
        pd.DataFrame({"USER_ID": ["U0", "U1"], "OUTCOME": [0, 1]}).to_parquet(os.path.join(item_dir, "p.parquet"))
    return str(root)


def _schema(tmp_path, extra=()):
    path = str(tmp_path / "schema.yaml")
    columns = [{"name": "USER_ID", "type": "str"}, {"name": "ITEM_ID", "type": "str"}]
    columns += [{"name": "OUTCOME", "type": "float"}, *extra]
    with open(path, "w") as f:
        yaml.safe_dump({"columns": columns}, f)
    return path


def test_undeclared_partition_keys_are_not_columns(tmp_path):
    # month= is for data management; it must not turn into a feature implicitly
    ds = InteractionsDataset(data_location=_write_month_layout(tmp_path / "data"))
    for df in [ds.fetch_data(), *(ds.fetch_data(f) for f in sorted(ds.data_filenames()))]:
        assert "month" not in df.columns and set(df["ITEM_ID"]) == {"ITEM_A"}
    assert "month" not in ds.peek_columns()


def test_declared_partition_keys_are_columns(tmp_path):
    schema = _schema(tmp_path, extra=[{"name": "month", "type": "int"}])
    ds = InteractionsDataset(data_location=_write_month_layout(tmp_path / "data"), client_schema_path=schema)
    whole = ds.fetch_data()
    assert sorted(whole["month"]) == [1, 1, 2, 2] and whole["month"].dtype == "int64"
    parts = [ds.fetch_data(f) for f in sorted(ds.data_filenames())]
    assert [p["month"].tolist() for p in parts] == [[1, 1], [2, 2]] and parts[0]["month"].dtype == "int64"
    assert "month" in ds.peek_columns()


def test_month_layout_batch_trains_without_month_feature(tmp_path, datasets):
    recommender = _batch_recommender()
    recommender.train(
        users_ds=datasets["users"],
        items_ds=datasets["items"],
        interactions_ds=InteractionsDataset(data_location=_write_month_layout(tmp_path / "data")),
    )
    assert recommender.scorer.estimator["ITEM_A"].feature_names == ["feat1"]


# --- peek_columns ----------------------------------------------------------------------------------------------------


def test_peek_columns_reads_no_data(tmp_path, monkeypatch):
    root = write_flat(tmp_path / "flat", {k: [p.assign(OUTCOME_click=1) for p in v] for k, v in ROWS.items()})

    def no_read(*args, **kwargs):
        raise AssertionError("peek_columns must read metadata only")

    monkeypatch.setattr("skrec.dataset.local_data_reader.LocalDataReader.read", no_read)
    assert sorted(InteractionsDataset(data_location=root).peek_columns()) == sorted(
        ["USER_ID", "OUTCOME", "OUTCOME_click", "ITEM_ID"]
    )
    schema = _schema(tmp_path)
    assert InteractionsDataset(data_location=root, client_schema_path=schema).peek_columns() == [
        "USER_ID",
        "ITEM_ID",
        "OUTCOME",
    ]


# --- batch_chunk_dir only when needed --------------------------------------------------------------------------------


def test_item_partitioned_ignores_missing_batch_chunk_dir(datasets, tmp_path):
    recommender = _batch_recommender(batch_chunk_dir=str(tmp_path / "does-not-exist"))
    recommender.train(users_ds=datasets["users"], items_ds=datasets["items"], interactions_ds=datasets["hive"])
    assert_array_equal(recommender.scorer.item_names, np.array(["ITEM B/x", "ITEM_A"]))


# --- third review round ----------------------------------------------------------------------------------------------


def test_item_dir_with_conflicting_item_column_raises(datasets, tmp_path):
    root = tmp_path / "conflict"
    os.makedirs(root / "ITEM_ID=ITEM_A")
    pd.DataFrame({"USER_ID": ["U0", "U1"], "ITEM_ID": ["ITEM_A", "ITEM_C"], "OUTCOME": [0, 1]}).to_parquet(
        root / "ITEM_ID=ITEM_A" / "p.parquet"
    )
    with pytest.raises(ValueError, match=r"is under ITEM_ID=ITEM_A/ but holds rows for \['ITEM_C'\]"):
        _batch_recommender().train(
            users_ds=datasets["users"],
            items_ds=datasets["items"],
            interactions_ds=InteractionsDataset(data_location=str(root)),
        )


def test_item_partitioned_without_label_check_reads_catalogue_files_once(datasets, tmp_path, monkeypatch):
    # A dict of estimators skips the label check, so files are read only to train, and only for catalogue items
    rows = {**ROWS, "ITEM_Z": [pd.DataFrame({"USER_ID": ["U0", "U1"], "OUTCOME": [0, 1]})]}
    ds = InteractionsDataset(data_location=write_hive(tmp_path / "hive_z", rows))
    original = InteractionsDataset.fetch_data
    reads = []

    def counting_fetch(self, data_partition=None):
        reads.append(data_partition)
        return original(self, data_partition)

    monkeypatch.setattr(InteractionsDataset, "fetch_data", counting_fetch)
    estimators = {item: BatchXGBClassifierEstimator(BATCH_PARAMS) for item in ["ITEM_A", "ITEM B/x", "ITEM_C"]}
    recommender = RankingRecommender(IndependentScorer(estimators))
    recommender.train(users_ds=datasets["users"], items_ds=datasets["items"], interactions_ds=ds)

    assert_array_equal(recommender.scorer.item_names, np.array(["ITEM B/x", "ITEM_A"]))
    catalogue_files = [f for f in ds.data_filenames() if "ITEM_Z" not in f]
    assert sorted(reads) == sorted(catalogue_files)


def test_windows_separators(monkeypatch):
    import skrec.dataset.partitioning as partitioning

    monkeypatch.setattr(partitioning.os, "sep", "\\")
    root = "C:\\data\\train"
    assert hive_partition_values(root + "\\ITEM_ID=a\\part-0.parquet", root) == {"ITEM_ID": "a"}
    assert is_hidden_path(root + "\\_temporary\\part-0.parquet", root)


def test_s3_whole_read_prefers_file_column(s3):
    # The file's own ITEM_ID wins over the directory's, on whole and per-file reads alike
    _put_s3(s3, "both/ITEM_ID=1.0/p.parquet", pd.DataFrame({"USER_ID": ["U0"], "ITEM_ID": ["1"], "OUTCOME": [1]}))
    ds = InteractionsDataset(data_location="s3://test/both/")
    assert ds.fetch_data()["ITEM_ID"].tolist() == ["1"]
    assert ds.fetch_data(sorted(ds.data_filenames())[0])["ITEM_ID"].tolist() == ["1"]


@pytest.mark.parametrize("per_item", [False, True], ids=["label_scan", "validate_while_training"])
def test_hive_null_partition_rejected(datasets, tmp_path, per_item):
    # Spark writes rows with a null ITEM_ID under ITEM_ID=__HIVE_DEFAULT_PARTITION__/
    root = write_hive(tmp_path / "hive_null")
    null_dir = os.path.join(root, "ITEM_ID=__HIVE_DEFAULT_PARTITION__")
    os.makedirs(null_dir)
    pd.DataFrame({"USER_ID": ["U0", "U1"], "OUTCOME": [0, 1]}).to_parquet(os.path.join(null_dir, "p.parquet"))

    estimator = BatchXGBClassifierEstimator(BATCH_PARAMS)
    if per_item:
        estimator = {item: BatchXGBClassifierEstimator(BATCH_PARAMS) for item in ["ITEM_A", "ITEM B/x", "ITEM_C"]}
    with pytest.raises(ValueError, match="'ITEM_ID' column in interactions contains 2 null value"):
        RankingRecommender(IndependentScorer(estimator)).train(
            users_ds=datasets["users"],
            items_ds=datasets["items"],
            interactions_ds=InteractionsDataset(data_location=root),
        )
