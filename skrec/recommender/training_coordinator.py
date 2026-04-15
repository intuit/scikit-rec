"""Training dispatch for :class:`~skrec.recommender.base_recommender.BaseRecommender`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pandas import DataFrame

from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator

if TYPE_CHECKING:
    from skrec.recommender.base_recommender import BaseRecommender


def coordinate_training(
    recommender: BaseRecommender,
    *,
    users_df: Optional[DataFrame],
    items_df: Optional[DataFrame],
    interactions_df: Optional[DataFrame],
    valid_users_df: Optional[DataFrame],
    valid_interactions_df: Optional[DataFrame],
    users_ds: Optional[UsersDataset],
    items_ds: Optional[ItemsDataset],
    interactions_ds: Optional[InteractionsDataset],
    valid_users_ds: Optional[UsersDataset],
    valid_interactions_ds: Optional[InteractionsDataset],
) -> None:
    """Run batch, tabular, or embedding training on ``recommender.scorer``."""
    scorer = recommender.scorer

    if scorer.estimator.support_batch_training():
        if interactions_ds is None:
            raise ValueError("interactions_ds is required for batch training.")
        is_partitioned = len(interactions_ds.data_filenames()) > 1
        scorer.item_names, scorer.items_df = scorer._process_items(items_df, None, is_partitioned=is_partitioned)

        scorer.batch_train_model(
            users_ds=users_ds,
            items_ds=items_ds,
            interactions_ds=interactions_ds,
            valid_interactions_ds=valid_interactions_ds,
            valid_users_ds=valid_users_ds,
        )
        return

    if not isinstance(scorer.estimator, BaseEmbeddingEstimator):
        X, y = scorer.process_datasets(users_df, items_df, interactions_df)
        if valid_interactions_df is not None:
            if users_df is not None and valid_users_df is None:
                raise RuntimeError(
                    "valid_users_ds must be provided when users_ds is provided and "
                    "valid_interactions_ds is used for validation."
                )
            X_valid, y_valid = scorer.process_datasets(
                valid_users_df, items_df, valid_interactions_df, is_training=False
            )
            scorer.train_model(X, y, X_valid, y_valid)
        else:
            scorer.train_model(X, y)
        return

    if interactions_df is None:
        raise ValueError("InteractionsDataset must be provided for training embedding models.")

    train_users, train_items, train_interactions = scorer.process_factorized_datasets(
        users_df=users_df, items_df=items_df, interactions_df=interactions_df, is_training=True
    )

    val_users, val_interactions = None, None
    if valid_interactions_df is not None:
        val_users, _, val_interactions = scorer.process_factorized_datasets(
            users_df=valid_users_df,
            items_df=items_df,
            interactions_df=valid_interactions_df,
            is_training=False,
        )

    scorer.train_embedding_model(
        users=train_users,
        items=train_items,
        interactions=train_interactions,
        valid_users=val_users,
        valid_interactions=val_interactions,
    )
