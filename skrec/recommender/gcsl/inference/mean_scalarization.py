from typing import Dict, List

from pandas import DataFrame

from skrec.recommender.gcsl.inference.base_inference import BaseInference


class MeanScalarization(BaseInference):
    """Goal-conditioned inference that targets a scaled multiple of the training mean.

    At ``fit`` time, computes the per-outcome training mean and multiplies by a
    user-supplied scalar to derive the inference goal. A scalar of 1.0 targets
    the average training outcome; scalars above 1.0 push toward above-average
    outcomes.

    A ``UserWarning`` is emitted at ``transform`` time if a computed goal falls
    outside the observed training range — the model has no signal for values it
    never saw during training.

    Args:
        scalars: Mapping from outcome column name to scalar multiplier applied
            to the training mean. E.g. ``{"revenue": 1.5, "clicks": 1.2}``
            targets 1.5× mean revenue and 1.2× mean clicks.

    Example::

        inference = MeanScalarization({"revenue": 1.5, "clicks": 1.0})
        recommender = GcslRecommender(scorer, inference_method=inference)
        recommender.train(users_ds, items_ds, interactions_ds)
        recommender.recommend(interactions, users, top_k=10)
    """

    def __init__(self, scalars: Dict[str, float]) -> None:
        super().__init__()
        self.scalars = scalars
        self.goal_values_: Dict[str, float] = {}
        self.outcome_min_: Dict[str, float] = {}
        self.outcome_max_: Dict[str, float] = {}

    def fit(self, interactions_df: DataFrame, outcome_cols: List[str]) -> "MeanScalarization":
        """Compute scaled-mean goals from training data.

        Args:
            interactions_df: Training interactions DataFrame including outcome columns.
            outcome_cols: Outcome column names to condition on.

        Returns:
            self

        Raises:
            ValueError: If a scalar multiplier is not provided for every outcome column.
        """
        for outcome in outcome_cols:
            if outcome not in self.scalars:
                raise ValueError(
                    f"No scalar provided for outcome column '{outcome}'. "
                    "Provide a scalar multiplier for each outcome column in 'scalars'."
                )
            col_data = interactions_df[outcome]
            self.goal_values_[outcome] = float(col_data.mean() * self.scalars[outcome])
            self.outcome_min_[outcome] = float(col_data.min())
            self.outcome_max_[outcome] = float(col_data.max())
        self.outcome_cols_ = outcome_cols
        self._fitted = True
        return self

    def transform(self, interactions: DataFrame) -> DataFrame:
        """Inject scaled-mean goal values into a copy of interactions.

        Args:
            interactions: Inference-time interactions DataFrame.

        Returns:
            Copy of interactions with outcome columns set to scaled-mean goals.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        self._check_fitted()
        interactions = interactions.copy()
        for outcome in self.outcome_cols_:
            goal = self.goal_values_[outcome]
            self._warn_if_ood(goal, outcome, self.outcome_min_[outcome], self.outcome_max_[outcome])
            interactions[outcome] = goal
        return interactions
