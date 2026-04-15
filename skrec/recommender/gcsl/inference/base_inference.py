import warnings
from abc import ABC, abstractmethod
from typing import List, Optional

from pandas import DataFrame


class NotFittedError(ValueError):
    """Raised when transform() is called before fit()."""


class BaseInference(ABC):
    """Base class for GCSL goal-injection inference methods.

    The lifecycle has two steps:

    1. ``fit(interactions_df, outcome_cols)`` — called once during training to
       compute or validate goal statistics from the training DataFrame.
    2. ``transform(interactions)`` — called at inference to inject goal values
       into a copy of the interactions DataFrame before scoring.

    All ``transform`` implementations must return a **copy** and never mutate
    the input.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self.outcome_cols_: Optional[List[str]] = None

    @abstractmethod
    def fit(self, interactions_df: DataFrame, outcome_cols: List[str]) -> "BaseInference":
        """Compute goal statistics from the training interactions DataFrame.

        Args:
            interactions_df: The already-fetched training interactions DataFrame,
                including outcome columns.
            outcome_cols: Names of the outcome columns to condition on.

        Returns:
            self, for method chaining.
        """
        ...

    @abstractmethod
    def transform(self, interactions: DataFrame) -> DataFrame:
        """Inject goal values into a copy of the interactions DataFrame.

        Args:
            interactions: Inference-time interactions DataFrame containing
                outcome columns (with stale or placeholder values).

        Returns:
            A copy of ``interactions`` with outcome columns overwritten by
            goal values.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        ...

    def _check_fitted(self) -> None:
        """Raise NotFittedError if fit() has not been called."""
        if not self._fitted:
            raise NotFittedError(f"{type(self).__name__} is not fitted. Call fit() before transform().")

    def _warn_if_ood(self, goal: float, col: str, col_min: float, col_max: float) -> None:
        """Emit a UserWarning if goal falls outside the observed training range."""
        if goal < col_min or goal > col_max:
            warnings.warn(
                f"Goal for '{col}' ({goal:.4g}) is outside the training range "
                f"[{col_min:.4g}, {col_max:.4g}]. Model behavior is undefined "
                "outside the training distribution.",
                UserWarning,
                stacklevel=2,
            )
