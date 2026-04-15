from enum import Enum
from typing import Callable, Dict, Optional, Type

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from skrec.constants import ITEM_ID_NAME
from skrec.estimator.base_estimator import BaseEstimator
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.regression.base_regressor import BaseRegressor
from skrec.scorer.base_scorer import BaseScorer


class UpliftMode(str, Enum):
    T_LEARNER = "t_learner"
    S_LEARNER = "s_learner"
    X_LEARNER = "x_learner"


class UpliftScorerAdapter(BaseScorer):
    """Adapter that implements the ``BaseScorer`` interface for uplift scoring.

    Manages one or more internal scorers (one per treatment for T/X-Learner,
    or a single shared scorer for S-Learner) and routes ``process_datasets``,
    ``train_model``, and ``score_items`` calls to them.

    This is intentionally a ``BaseScorer`` subclass so that ``BaseRecommender``
    can drive training and scoring through its standard ``self.scorer.*`` calls
    without uplift-specific overrides.  Methods that have no meaning in the
    uplift context (``score_fast``, ``_calculate_scores``) raise
    ``NotImplementedError``.
    """

    def __init__(
        self,
        scorer_class: Type[BaseScorer],
        mode: UpliftMode,
        control_item_id: str,
        estimator_factory: Optional[Callable[[], BaseEstimator]] = None,
        estimator_dict: Optional[Dict[str, BaseEstimator]] = None,
        **scorer_kwargs,
    ):
        # Resolve the placeholder estimator *before* calling super().__init__
        # so that BaseScorer always receives a real estimator, not None.
        placeholder_estimator: Optional[BaseEstimator] = None
        if estimator_factory:
            placeholder_estimator = estimator_factory()
        elif estimator_dict:
            if control_item_id not in estimator_dict:
                raise ValueError(
                    f"When using T-Learner with estimator_dict, the dictionary must "
                    f"contain the control_item_id: '{control_item_id}'"
                )
            placeholder_estimator = estimator_dict[control_item_id]

        if placeholder_estimator is None:
            raise ValueError("UpliftScorerAdapter requires either an estimator_factory or an estimator_dict.")

        super().__init__(placeholder_estimator)

        self.scorer_class = scorer_class
        self.mode = mode
        self.control_item_id = control_item_id
        self.estimator_factory = estimator_factory
        self.estimator_dict = estimator_dict or {}
        self.scorer_kwargs = scorer_kwargs

        self.treatment_item_ids: list[str] = []
        self.scorers: Dict[str, BaseScorer] = {}
        self.shared_scorer: Optional[BaseScorer] = None

    def process_datasets(
        self,
        users_df: Optional[DataFrame],
        items_df: Optional[DataFrame],
        interactions_df: DataFrame,
        is_training: bool = True,
    ) -> tuple:
        all_item_ids = sorted(interactions_df[ITEM_ID_NAME].unique())

        if is_training:
            self.item_names = all_item_ids
            self.treatment_item_ids = [item for item in all_item_ids if item != self.control_item_id]

        if self.mode in (UpliftMode.T_LEARNER, UpliftMode.X_LEARNER):
            X_dict, y_dict = {}, {}

            if is_training:
                self.scorers = {}

            for item in all_item_ids:
                if is_training:
                    estimator = self.estimator_dict.get(item) or (self.estimator_factory and self.estimator_factory())
                    if not estimator:
                        raise ValueError(
                            f"T-Learner requires an estimator for item '{item}' via estimator_dict or estimator_factory"
                        )
                    scorer = self.scorer_class(estimator=estimator, **self.scorer_kwargs)
                    self.scorers[item] = scorer
                else:
                    if item not in self.scorers:
                        continue
                    scorer = self.scorers[item]

                item_interactions_df = interactions_df[interactions_df[ITEM_ID_NAME] == item]

                if item_interactions_df.empty:
                    if is_training:
                        raise ValueError(f"No interaction data found for item '{item}' during training.")
                    continue

                X, y = scorer.process_datasets(users_df, items_df, item_interactions_df, is_training)
                X_dict[item], y_dict[item] = X, y

            return X_dict, y_dict

        elif self.mode == UpliftMode.S_LEARNER:
            if is_training:
                if not self.estimator_factory:
                    raise ValueError("S-Learner requires a single estimator from estimator_factory.")
                self.shared_scorer = self.scorer_class(estimator=self.estimator_factory(), **self.scorer_kwargs)
                self.estimator = self.shared_scorer.estimator

            return self.shared_scorer.process_datasets(users_df, items_df, interactions_df, is_training)

    def _get_outcome_prediction(self, scorer: BaseScorer, X) -> np.ndarray:
        """Get outcome predictions from a trained scorer's estimator."""
        estimator = scorer.estimator
        # IndependentScorer stores estimators as a dict; unwrap the single entry
        if isinstance(estimator, dict):
            estimator = next(iter(estimator.values()))
        if isinstance(estimator, BaseClassifier):
            return estimator.predict_proba(X)[:, 1]
        elif isinstance(estimator, BaseRegressor):
            return estimator.predict(X)
        else:
            raise TypeError(f"Estimator of type {type(estimator)} is not supported for cross-prediction.")

    def train_model(self, X, y, X_valid: Optional = None, y_valid: Optional = None) -> None:
        if self.mode in (UpliftMode.T_LEARNER, UpliftMode.X_LEARNER):
            # Stage 1: Train outcome models per item (same as T-learner)
            for item, scorer in self.scorers.items():
                scorer.train_model(
                    X[item], y[item], X_valid.get(item) if X_valid else None, y_valid.get(item) if y_valid else None
                )

            if self.mode == UpliftMode.X_LEARNER:
                self._train_x_learner_stages(X, y)
        else:
            self.shared_scorer.train_model(X, y, X_valid, y_valid)

    @staticmethod
    def _unwrap_item_dict(d: dict, item: str):
        """Unwrap nested dict from IndependentScorer: {item: {item: value}} → value."""
        inner = d[item]
        if isinstance(inner, dict):
            return next(iter(inner.values()))
        return inner

    def _train_x_learner_stages(self, X: dict, y: dict) -> None:
        """Stages 2-4 of the X-learner: impute effects, train CATE models, train propensity."""
        from sklearn.linear_model import LogisticRegression, Ridge

        self.cate_treatment_models = {}
        self.cate_control_models = {}
        self.propensity_models = {}

        X_control = self._unwrap_item_dict(X, self.control_item_id)
        y_control = np.asarray(self._unwrap_item_dict(y, self.control_item_id), dtype=np.float64)

        for treatment in self.treatment_item_ids:
            X_treat = self._unwrap_item_dict(X, treatment)
            y_treat = np.asarray(self._unwrap_item_dict(y, treatment), dtype=np.float64)

            # Stage 2: Cross-predict to get imputed treatment effects
            mu_0_on_treated = self._get_outcome_prediction(self.scorers[self.control_item_id], X_treat)
            mu_t_on_control = self._get_outcome_prediction(self.scorers[treatment], X_control)

            D_treatment = y_treat - mu_0_on_treated  # imputed effect for treated users
            D_control = mu_t_on_control - y_control  # imputed effect for control users

            # Stage 3: Train CATE regressors on imputed effects
            X_treat_np = X_treat.values if hasattr(X_treat, "values") else X_treat
            X_control_np = X_control.values if hasattr(X_control, "values") else X_control

            tau_1 = Ridge(alpha=1.0)
            tau_1.fit(X_treat_np, D_treatment)
            self.cate_treatment_models[treatment] = tau_1

            tau_0 = Ridge(alpha=1.0)
            tau_0.fit(X_control_np, D_control)
            self.cate_control_models[treatment] = tau_0

            # Stage 4: Train propensity model (P(treatment | X))
            X_combined = np.vstack([X_treat_np, X_control_np])
            y_propensity = np.concatenate([np.ones(len(X_treat_np)), np.zeros(len(X_control_np))])
            propensity = LogisticRegression(max_iter=1000)
            propensity.fit(X_combined, y_propensity)
            self.propensity_models[treatment] = propensity

    def score_items(
        self,
        interactions: Optional[DataFrame] = None,
        users: Optional[DataFrame] = None,
    ) -> DataFrame:
        if self.item_subset is not None:
            treatments_to_score = [t for t in self.treatment_item_ids if t in self.item_subset]
        else:
            treatments_to_score = self.treatment_item_ids

        if self.mode == UpliftMode.T_LEARNER:
            control_scores = self.scorers[self.control_item_id].score_items(interactions=interactions, users=users)
            uplift_outputs = {}
            for treatment in treatments_to_score:
                treatment_scores = self.scorers[treatment].score_items(interactions=interactions, users=users)
                uplift_outputs[treatment] = treatment_scores.values - control_scores.values

            return pd.DataFrame({k: v.squeeze() for k, v in uplift_outputs.items()}, index=control_scores.index)

        elif self.mode == UpliftMode.X_LEARNER:
            # Get user features via any scorer's inference path
            user_interactions_df = self.scorers[self.control_item_id]._get_user_interactions_df(interactions, users)
            X_score = user_interactions_df.values

            uplift_outputs = {}
            for treatment in treatments_to_score:
                tau_1_pred = self.cate_treatment_models[treatment].predict(X_score)
                tau_0_pred = self.cate_control_models[treatment].predict(X_score)
                g = self.propensity_models[treatment].predict_proba(X_score)[:, 1]

                # τ(x) = g(x)·τ₀(x) + (1 − g(x))·τ₁(x)
                uplift_outputs[treatment] = g * tau_0_pred + (1 - g) * tau_1_pred

            return pd.DataFrame(uplift_outputs, index=user_interactions_df.index)

        else:  # S-Learner

            def _get_prediction(model, data):
                if isinstance(model, BaseRegressor):
                    return model.predict(data)
                elif isinstance(model, BaseClassifier):
                    return model.predict_proba(data)[:, 1]
                else:
                    raise TypeError(f"Estimator of type {type(model)} is not supported for S-Learner uplift scoring.")

            user_interactions_df = self.shared_scorer._get_user_interactions_df(interactions, users)
            n_users = user_interactions_df.shape[0]

            items_df_indexed = self.shared_scorer.items_df.set_index(ITEM_ID_NAME)
            estimator = self.shared_scorer.estimator

            user_cols = user_interactions_df.columns.tolist()
            item_cols = items_df_indexed.columns.tolist()
            all_cols = user_cols + item_cols

            def _build_scoring_df(item_id: str) -> DataFrame:
                item_features = items_df_indexed.loc[[item_id]].values
                combined = np.concatenate([user_interactions_df.values, np.tile(item_features, (n_users, 1))], axis=1)
                return pd.DataFrame(combined, columns=all_cols, index=user_interactions_df.index, dtype=np.float64)

            control_scores = _get_prediction(estimator, _build_scoring_df(self.control_item_id))

            uplift_outputs = {}
            for treatment in treatments_to_score:
                treatment_scores = _get_prediction(estimator, _build_scoring_df(treatment))
                uplift_outputs[treatment] = treatment_scores - control_scores

            return pd.DataFrame(uplift_outputs, index=user_interactions_df.index)

    def score_fast(self, features: DataFrame) -> DataFrame:
        raise NotImplementedError(
            "recommend_online() is not supported for uplift recommenders. "
            "Uplift scoring requires comparing treatment vs. control scores across all items — "
            "use recommend() instead."
        )

    def _calculate_scores(self, joined: DataFrame) -> NDArray:
        raise NotImplementedError("UpliftScorerAdapter delegates scoring to its internal scorers.")
