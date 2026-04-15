from typing import Dict, List

import numpy as np
from pandas import DataFrame

from skrec.recommender.gcsl.inference.base_inference import BaseInference


class PercentileValue(BaseInference):
    """Goal-conditioned inference that targets a specific percentile of training outcomes.

    At ``fit`` time, computes the per-outcome value at the requested percentile
    of the training distribution. Goals derived this way are **always within the
    observed training range** — making this the safest way to specify goals.

    No out-of-distribution warning is emitted because percentile-based goals are
    bounded by construction.

    Args:
        percentiles: Mapping from outcome column name to target percentile (0–100).
            Percentile 50 = median outcome; 80 = top-20% outcome; 95 = elite outcome.
            E.g. ``{"revenue": 80, "clicks": 75}`` targets 80th-percentile revenue
            and 75th-percentile clicks.

    Raises:
        ValueError: At construction time if any percentile is outside [0, 100].

    Example::

        inference = PercentileValue({"revenue": 80, "clicks": 75})
        recommender = GcslRecommender(scorer, inference_method=inference)
        recommender.train(users_ds, items_ds, interactions_ds)
        recommender.recommend(interactions, users, top_k=10)
    """

    def __init__(self, percentiles: Dict[str, float]) -> None:
        super().__init__()
        for col, pct in percentiles.items():
            if not 0 <= pct <= 100:
                raise ValueError(f"Percentile for '{col}' must be between 0 and 100, got {pct}.")
        self.percentiles = percentiles
        self.goal_values_: Dict[str, float] = {}

    def fit(self, interactions_df: DataFrame, outcome_cols: List[str]) -> "PercentileValue":
        """Compute percentile-based goals from training data.

        Args:
            interactions_df: Training interactions DataFrame including outcome columns.
            outcome_cols: Outcome column names to condition on.

        Returns:
            self

        Raises:
            ValueError: If a percentile is not provided for every outcome column.
        """
        for outcome in outcome_cols:
            if outcome not in self.percentiles:
                raise ValueError(
                    f"No percentile provided for outcome column '{outcome}'. "
                    "Provide a percentile (0–100) for each outcome column in 'percentiles'."
                )
            self.goal_values_[outcome] = float(np.percentile(interactions_df[outcome], self.percentiles[outcome]))
        self.outcome_cols_ = outcome_cols
        self._fitted = True
        return self

    def transform(self, interactions: DataFrame) -> DataFrame:
        """Inject percentile-derived goal values into a copy of interactions.

        Goals are always within the training distribution; no OOD warning is emitted.

        Args:
            interactions: Inference-time interactions DataFrame.

        Returns:
            Copy of interactions with outcome columns set to percentile goals.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        self._check_fitted()
        interactions = interactions.copy()
        for outcome in self.outcome_cols_:
            interactions[outcome] = self.goal_values_[outcome]
        return interactions
