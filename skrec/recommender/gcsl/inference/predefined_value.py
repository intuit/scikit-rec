from typing import Dict, List

from pandas import DataFrame

from skrec.recommender.gcsl.inference.base_inference import BaseInference


class PredefinedValue(BaseInference):
    """Goal-conditioned inference that uses fixed, user-specified goal values.

    Goal values are supplied at construction time and do not depend on training
    data. ``fit()`` fetches training min/max statistics solely to support
    out-of-distribution warnings at ``transform`` time.

    A ``UserWarning`` is emitted at ``transform`` time if any goal falls outside
    the observed training range.

    Args:
        goal_values: Mapping from outcome column name to target value.
            E.g. ``{"revenue": 5.0, "clicks": 1.0}``.

    Example::

        inference = PredefinedValue({"revenue": 5.0, "clicks": 1.0})
        recommender = GcslRecommender(scorer, inference_method=inference)
        recommender.train(users_ds, items_ds, interactions_ds)
        recommender.recommend(interactions, users, top_k=10)
    """

    def __init__(self, goal_values: Dict[str, float]) -> None:
        super().__init__()
        self.goal_values = goal_values
        self.outcome_min_: Dict[str, float] = {}
        self.outcome_max_: Dict[str, float] = {}

    def fit(self, interactions_df: DataFrame, outcome_cols: List[str]) -> "PredefinedValue":
        """Validate goal keys and record the training range for OOD warnings.

        Args:
            interactions_df: Training interactions DataFrame including outcome columns.
            outcome_cols: Outcome column names to condition on.

        Returns:
            self

        Raises:
            ValueError: If a goal value is not provided for every outcome column.
        """
        for outcome in outcome_cols:
            if outcome not in self.goal_values:
                raise ValueError(
                    f"No goal value provided for outcome column '{outcome}'. "
                    "Provide a value for each outcome column in 'goal_values'."
                )
            col_data = interactions_df[outcome]
            self.outcome_min_[outcome] = float(col_data.min())
            self.outcome_max_[outcome] = float(col_data.max())
        self.outcome_cols_ = outcome_cols
        self._fitted = True
        return self

    def transform(self, interactions: DataFrame) -> DataFrame:
        """Inject predefined goal values into a copy of interactions.

        Args:
            interactions: Inference-time interactions DataFrame.

        Returns:
            Copy of interactions with outcome columns set to predefined goals.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        self._check_fitted()
        interactions = interactions.copy()
        for outcome in self.outcome_cols_:
            goal = self.goal_values[outcome]
            self._warn_if_ood(goal, outcome, self.outcome_min_[outcome], self.outcome_max_[outcome])
            interactions[outcome] = goal
        return interactions
