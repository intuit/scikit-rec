from enum import Enum


class HPOType(str, Enum):
    GRID_SEARCH_CV = "grid_search_cv"
    RANDOMIZED_SEARCH_CV = "randomized_search_cv"


class MFAlgorithm(str, Enum):
    """Algorithm for native matrix factorization (collaborative filtering) estimator."""

    ALS = "als"
    SGD = "sgd"


class MFOutcomeType(str, Enum):
    """
    Outcome type for matrix factorization (reward / label semantics).

    - CONTINUOUS: Real-valued outcomes (e.g. watch time, revenue). MSE loss; raw score.
    - BINARY: Binary 0/1 (e.g. click, like). BCE with SGD, sigmoid at predict → [0, 1].
    - ORDINAL: Ordered discrete levels (e.g. 1–5 star ratings). MSE on numeric levels;
      predictions can be clamped to an optional [min, max] scale.
    """

    CONTINUOUS = "continuous"
    BINARY = "binary"
    ORDINAL = "ordinal"
