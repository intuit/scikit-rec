import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, USER_ID_NAME
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.estimator.regression.base_regressor import BaseRegressor
from skrec.scorer.base_scorer import BaseScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class IndependentScorer(BaseScorer):
    """Per-item scorer: trains and runs a separate estimator for each item.

    .. warning::
        **Not thread-safe.**  Shared mutable state (``item_subset``,
        ``item_names``, ``item_specific_features``, and the thread-pool
        executor) is not protected by locks.  Do not call ``score_items``,
        ``set_parallel_inference``, or ``set_item_specific_features``
        concurrently on the same instance.  Create separate scorer instances
        for concurrent use (e.g. one per serving thread).
    """

    def __init__(self, estimator: Union[BaseClassifier, Dict[str, BaseClassifier]]) -> None:
        if isinstance(estimator, BaseEmbeddingEstimator):
            raise TypeError(
                "IndependentScorer does not support BaseEmbeddingEstimator. "
                "Use UniversalScorer for embedding estimators (e.g. MatrixFactorizationEstimator, NCFEstimator)."
            )
        if isinstance(estimator, dict):
            for _item, est in estimator.items():
                if isinstance(est, BaseEmbeddingEstimator):
                    raise TypeError(
                        "IndependentScorer does not support BaseEmbeddingEstimator. "
                        "Use UniversalScorer for embedding estimators "
                        "(e.g. MatrixFactorizationEstimator, NCFEstimator)."
                    )
        self.estimator = estimator
        self.parallel_inference_status = False
        self.num_cores = 1
        self.item_specific_features = None
        self._executor: Optional[ThreadPoolExecutor] = None

    def set_item_specific_features(
        self,
        item_specific_features_users: Optional[Dict[str, List[str]]] = None,
        item_specific_features_interactions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Configure per-item feature subsets for the ``IndependentScorer``.

        When set, each item's estimator is trained and scored using only the
        specified subset of user/interaction columns rather than the full
        feature set.  This is useful when different items have meaningfully
        different predictive signals.

        Args:
            item_specific_features_users: Mapping from item ID to the list of
                user feature columns to use for that item.  ``USER_ID_NAME``
                must not appear in these lists (it is automatically handled).
            item_specific_features_interactions: Mapping from item ID to the
                list of interaction feature columns to use for that item.
                ``USER_ID_NAME`` must not appear in these lists.

        Raises:
            ValueError: If ``USER_ID_NAME`` appears in any feature list.
        """
        self.item_specific_features = {}
        self.item_specific_features["users"] = {}
        self.item_specific_features["interactions"] = {}

        if item_specific_features_users is not None:
            for k, v in item_specific_features_users.items():
                if USER_ID_NAME in v:
                    raise ValueError(f"USER_ID must not be in item_specific_features_users for item {k}")

        if item_specific_features_interactions is not None:
            for k, v in item_specific_features_interactions.items():
                if USER_ID_NAME in v:
                    raise ValueError(f"USER_ID must not be in item_specific_features_interactions for item {k}")

        self.item_specific_features["users"] = item_specific_features_users
        self.item_specific_features["interactions"] = item_specific_features_interactions

    def process_datasets(
        self,
        users_df: Optional[DataFrame] = None,
        items_df: Optional[DataFrame] = None,
        interactions_df: Optional[DataFrame] = None,
        is_training: Optional[bool] = True,
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """
        Overloaded process_datasets method where item-level filtering happens first,
        then join occurs.
        """

        # Validate input
        self._validate_interactions(interactions_df)
        if users_df is not None:
            self._validate_users(users_df)
        if items_df is not None:
            self._validate_items(items_df)
        if is_training:
            self.item_names, self.items_df = self._process_items(items_df, interactions_df)
            # Filter items to only include those present in interactions_df
            self._filter_items_by_interactions(interactions_df)

        # Process X and y first (item-level filtering)
        X, y = self._process_X_y_join_and_filter(interactions_df, users_df)

        return X, y

    def _filter_items_by_interactions(self, interactions_df: DataFrame) -> None:
        """
        Filter item_names and items_df to only include items present in interactions_df.
        """

        new_item_names = np.array([item for item in self.item_names if item in interactions_df[ITEM_ID_NAME].unique()])
        if len(new_item_names) != len(self.item_names):
            num_removed_items = len(self.item_names) - len(new_item_names)
            logger.info(f"{num_removed_items} items were removed because they are not present in interactions_df!")
            self.item_names = new_item_names

        if self.items_df is not None:
            self.items_df = self.items_df[self.items_df[ITEM_ID_NAME].isin(self.item_names)]
            self.items_df = self.items_df.reset_index(drop=True)

    def _apply_item_specific_features(
        self,
        item: str,
        users_df: Optional[DataFrame],
        interactions_df: DataFrame,
        one_row: bool = False,
    ) -> Tuple[Optional[DataFrame], DataFrame]:
        # the 1row operation deletes the USER_ID_NAME column in a previous step, so we cannot read it from the dataframe
        if self.item_specific_features is not None:
            if self.item_specific_features["users"] and item in self.item_specific_features["users"]:
                cols = self.item_specific_features["users"][item]
                cols = [USER_ID_NAME] + cols if not one_row else cols
                users_df = users_df[cols]
            if self.item_specific_features["interactions"] and item in self.item_specific_features["interactions"]:
                cols = self.item_specific_features["interactions"][item]
                cols = [USER_ID_NAME] + cols if not one_row else cols
                interactions_df = interactions_df[cols]
        return users_df, interactions_df

    def _process_X_y_join_and_filter(
        self,
        interactions_df: DataFrame,
        users_df: Optional[DataFrame] = None,
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """
        Process X and y with item-level filtering first, then join data.
        """
        # Split interactions by item first
        interactions_by_item = self._split_by_item(interactions_df)

        X = {}
        y = {}

        for item in self.item_names:
            item_interactions = interactions_by_item[item]

            # Join data for this specific item
            users_temp, interactions_temp = self._apply_item_specific_features(item, users_df, item_interactions)
            joined_data = self._join_data_train(users_df=users_temp, items_df=None, interactions_df=interactions_temp)

            # Generate X and y for this item
            X[item], y[item] = self._generate_X_y(joined_data)

        return X, y

    def _process_items(
        self, items_df: DataFrame, interactions_df: DataFrame
    ) -> Tuple[NDArray, Dict[str, BaseClassifier]]:
        item_names, items_df = super()._process_items(items_df, interactions_df)

        # Leave this warning here, because we don't want to set items_df to None earlier
        # Batched/Partitioned mode requires items_df
        if items_df is not None:
            logger.warning("Item Dataset will not be used in IndependentScorer.")
            items_df = None

        self._process_estimators_per_item(item_names)
        return item_names, items_df

    def _validate_interactions(self, interactions_df: DataFrame) -> None:
        super()._validate_interactions(interactions_df)

        if isinstance(self.estimator, BaseClassifier):
            logger.info("Checking if the target label has only two unique values for classification")
            nunique_interactions = interactions_df.groupby(ITEM_ID_NAME)[LABEL_NAME].nunique()
            items_to_check = nunique_interactions[nunique_interactions != 2]
            if len(items_to_check) > 0:
                raise ValueError(
                    "The number of unique values of the target label must be 2!!"
                    "Some items have either only one unique target value or more than two unqiue target values."
                    f"See details: \n {items_to_check}!!"
                )
            else:
                logger.info("Check Successful!")

    def _process_estimators_per_item(self, item_names: NDArray) -> None:
        if type(self.estimator) is dict:
            item_no_estimator = list(set(item_names) - set(self.estimator.keys()))
            if len(item_no_estimator) > 0:
                raise ValueError(
                    f"Estimator missing for the item {item_no_estimator}."
                    "When multiple binary classification models are used,"
                    "the key names in the dictionary of estimators and the item names must be the same!!"
                )
        else:
            duplicate_estimators = {}
            for item in item_names:
                duplicate_estimators[item] = deepcopy(self.estimator)
            self.estimator = duplicate_estimators

    def _split_by_item(self, joined_data: DataFrame) -> Dict[str, DataFrame]:
        joined_data_by_item = {}
        for item in self.item_names:
            joined_data_by_item[item] = joined_data[joined_data[ITEM_ID_NAME] == item]
        return joined_data_by_item

    def train_model(
        self,
        X: Dict[str, NDArray],
        y: Dict[str, NDArray],
        X_valid: Optional[Dict[str, NDArray]] = None,
        y_valid: Optional[Dict[str, NDArray]] = None,
    ) -> None:
        for item, estimator in self.estimator.items():
            if X_valid is not None:
                self._fit_estimator(estimator, X[item], y[item], X_valid[item], y_valid[item])
            else:
                self._fit_estimator(estimator, X[item], y[item])

    def _calculate_scores(self, joined: NDArray) -> NDArray:
        relevant_items = self.item_subset if self.item_subset else self.item_names

        # Snapshot the executor reference so a concurrent set_parallel_inference()
        # call cannot swap or shutdown the executor mid-scoring.
        executor = self._executor
        if self.parallel_inference_status and executor is not None:
            scores = list(executor.map(lambda item: self._calculate_scores_by_item(item, joined), relevant_items))
        else:
            scores = [self._calculate_scores_by_item(item, joined) for item in relevant_items]

        return np.column_stack(scores)

    def _calculate_scores_by_item(self, item: str, joined: Union[NDArray, Dict[str, NDArray]]) -> NDArray:
        # Convert joined (NDArray) back to DataFrame with feature names
        # for estimator.predict/predict_proba method which expects DataFrame
        estimator_instance = self.estimator[item]
        if estimator_instance.feature_names is None:
            raise ValueError(
                f"Estimator for item {item} does not have feature_names set. "
                "Ensure the estimator is fitted before scoring."
            )

        if isinstance(joined, Dict):
            joined_input = joined[item]
        else:
            joined_input = joined

        # Select only the columns the model was trained on.  This is a no-op when all
        # features are shared across items, but is critical when set_item_specific_features()
        # has been called: each item's model has a different feature_names subset, and passing
        # the full merged DataFrame (as score_fast does) would otherwise feed wrong columns.
        if isinstance(joined_input, DataFrame) and estimator_instance.feature_names is not None:
            joined_input = joined_input[list(estimator_instance.feature_names)]

        if isinstance(estimator_instance, BaseRegressor):
            score_by_item = estimator_instance.predict(joined_input)
        else:
            score_by_item = estimator_instance.predict_proba(joined_input)[:, 1]
        return score_by_item

    def score_fast(self, features: DataFrame) -> DataFrame:
        if features.shape[0] != 1:
            raise ValueError(
                f"score_fast() expects exactly 1 row, got {features.shape[0]}. Use score_items() for batch scoring."
            )
        drop_cols = [c for c in [USER_ID_NAME, ITEM_ID_NAME, LABEL_NAME] if c in features.columns]
        if drop_cols:
            features = features.drop(columns=drop_cols)
        scores = self._calculate_scores(features)
        return self._create_df_from_scores(scores)

    def shutdown(self) -> None:
        """Shut down the thread-pool executor, if any.

        Call this when the scorer is no longer needed to release OS threads
        immediately.  Safe to call multiple times.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self) -> None:
        # Best-effort cleanup if the caller forgets to call shutdown().
        # Swallow errors: interpreter may be tearing down imports/handlers.
        try:
            self.shutdown()
        except Exception:
            pass

    def set_parallel_inference(self, parallel_inference_status: bool = False, num_cores: Optional[int] = None) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self.parallel_inference_status = parallel_inference_status
        if parallel_inference_status:
            self.num_cores = num_cores if num_cores is not None else os.cpu_count()
            self._executor = ThreadPoolExecutor(max_workers=self.num_cores)
            logger.info(
                "Parallel inference enabled with %d threads. GIL-releasing estimators "
                "(LightGBM, XGBoost) achieve true CPU parallelism; pure-Python estimators "
                "will be serialized by the GIL. For maximum speedup, initialize each "
                "estimator with n_jobs=1 to avoid CPU oversubscription across item threads.",
                self.num_cores,
            )
        else:
            self.num_cores = 1
            logger.info("Parallel inference disabled. Scoring will run sequentially.")
