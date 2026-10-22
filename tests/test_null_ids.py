import pandas as pd
import pytest
import yaml

from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.universal import UniversalScorer

USERS = pd.DataFrame({"USER_ID": ["U0", "U1", "U2", "U3"], "feat1": [1.0, 2.0, 3.0, 4.0]})
ITEMS = pd.DataFrame({"ITEM_ID": ["A", "B"]})
INTERACTIONS = pd.DataFrame(
    {"USER_ID": ["U0", "U1", "U2", "U3"], "ITEM_ID": ["A", "A", "B", "B"], "OUTCOME": [0, 1, 0, 1]}
)


def _recommender():
    return RankingRecommender(UniversalScorer(XGBClassifierEstimator({"n_estimators": 2})))


def _write(tmp_path, name, df, fmt):
    path = tmp_path / f"{name}.{fmt}"
    df.to_csv(path, index=False) if fmt == "csv" else df.to_parquet(path)
    return str(path)


def _schema(tmp_path):
    path = tmp_path / "schema.yaml"
    columns = [{"name": "USER_ID", "type": "str"}, {"name": "ITEM_ID", "type": "str"}]
    with open(path, "w") as f:
        yaml.safe_dump({"columns": columns + [{"name": "OUTCOME", "type": "float"}]}, f)
    return str(path)


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
@pytest.mark.parametrize("with_schema", [False, True], ids=["no_schema", "schema"])
def test_null_ids_stay_null(tmp_path, fmt, with_schema):
    # CSV blanks used to become "nan" and parquet nulls "None", hiding them from validation
    interactions = INTERACTIONS.astype(object)
    interactions.loc[1, "USER_ID"] = None
    interactions.loc[2, "ITEM_ID"] = None
    ds = InteractionsDataset(
        data_location=_write(tmp_path, "interactions", interactions, fmt),
        client_schema_path=_schema(tmp_path) if with_schema else None,
    )
    df = ds.fetch_data()
    assert df["USER_ID"].isnull().tolist() == [False, True, False, False]
    assert df["ITEM_ID"].isnull().tolist() == [False, False, True, False]
    assert df.loc[0, "USER_ID"] == "U0" and df.loc[0, "ITEM_ID"] == "A"


def test_training_rejects_null_interaction_ids(tmp_path):
    interactions = INTERACTIONS.astype(object)
    interactions.loc[3, "ITEM_ID"] = None
    with pytest.raises(
        ValueError, match=r"'ITEM_ID' column in interactions contains 1 null value\(s\) \(first rows: \[3\]\)"
    ):
        _recommender().train(
            users_ds=UsersDataset(data_location=_write(tmp_path, "users", USERS, "csv")),
            interactions_ds=InteractionsDataset(data_location=_write(tmp_path, "interactions", interactions, "csv")),
        )


def test_training_rejects_null_user_ids_in_users(tmp_path):
    users = USERS.astype(object)
    users.loc[0, "USER_ID"] = None
    with pytest.raises(ValueError, match="'USER_ID' column in users contains 1 null value"):
        _recommender().train(
            users_ds=UsersDataset(data_location=_write(tmp_path, "users", users, "parquet")),
            interactions_ds=InteractionsDataset(data_location=_write(tmp_path, "interactions", INTERACTIONS, "csv")),
        )


def test_training_rejects_null_item_ids_in_items(tmp_path):
    items = pd.DataFrame({"ITEM_ID": ["A", None, "B"], "item_feat": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="'ITEM_ID' column in items contains 1 null value"):
        _recommender().train(
            users_ds=UsersDataset(data_location=_write(tmp_path, "users", USERS, "csv")),
            items_ds=ItemsDataset(data_location=_write(tmp_path, "items", items, "parquet")),
            interactions_ds=InteractionsDataset(data_location=_write(tmp_path, "interactions", INTERACTIONS, "csv")),
        )


def test_recommend_rejects_null_user_ids(tmp_path):
    recommender = _recommender()
    recommender.train(
        users_ds=UsersDataset(data_location=_write(tmp_path, "users", USERS, "csv")),
        items_ds=ItemsDataset(data_location=_write(tmp_path, "items", ITEMS, "csv")),
        interactions_ds=InteractionsDataset(data_location=_write(tmp_path, "interactions", INTERACTIONS, "csv")),
    )
    users = pd.DataFrame({"USER_ID": ["U0", None], "feat1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="'USER_ID' column in users contains 1 null value"):
        recommender.recommend(pd.DataFrame({"USER_ID": ["U0", "U1"]}), users, top_k=1)
    with pytest.raises(ValueError, match="'USER_ID' column in interactions contains 1 null value"):
        recommender.recommend(pd.DataFrame({"USER_ID": ["U0", None]}), USERS.head(2), top_k=1)


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_non_id_text_columns_unchanged(tmp_path, fmt):
    # IDs only: other text columns keep today's behaviour, where a missing value reads as text
    users = USERS.assign(segment=["a", None, "b", "c"])
    df = UsersDataset(data_location=_write(tmp_path, "users", users, fmt)).fetch_data()
    assert df["segment"].tolist() == ["a", "nan" if fmt == "csv" else "None", "b", "c"]


@pytest.mark.parametrize("table", ["interactions", "users"])
def test_training_rejects_null_ids_in_validation_data(tmp_path, table):
    # validation data goes through the same checks as training data
    valid_interactions, valid_users = INTERACTIONS.astype(object), USERS.astype(object)
    if table == "interactions":
        valid_interactions.loc[0, "USER_ID"] = None
    else:
        valid_users.loc[2, "USER_ID"] = None
    with pytest.raises(ValueError, match=f"'USER_ID' column in {table} contains 1 null value"):
        _recommender().train(
            users_ds=UsersDataset(data_location=_write(tmp_path, "users", USERS, "csv")),
            interactions_ds=InteractionsDataset(data_location=_write(tmp_path, "interactions", INTERACTIONS, "csv")),
            valid_users_ds=UsersDataset(data_location=_write(tmp_path, "valid_users", valid_users, "parquet")),
            valid_interactions_ds=InteractionsDataset(
                data_location=_write(tmp_path, "valid_interactions", valid_interactions, "parquet")
            ),
        )


def test_item_subset_rejects_null_item_ids(tmp_path):
    recommender = _recommender()
    recommender.train(
        users_ds=UsersDataset(data_location=_write(tmp_path, "users", USERS, "csv")),
        items_ds=ItemsDataset(data_location=_write(tmp_path, "items", ITEMS, "csv")),
        interactions_ds=InteractionsDataset(data_location=_write(tmp_path, "interactions", INTERACTIONS, "csv")),
    )
    with pytest.raises(ValueError, match="item_subset contains null item IDs"):
        recommender.set_item_subset(["A", None])
