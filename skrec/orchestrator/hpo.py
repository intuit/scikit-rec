import collections.abc
import copy
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import ParameterSampler

from skrec.constants import ITEM_ID_NAME, LABEL_NAME, OUTCOME_PREFIX
from skrec.dataset.interactions_dataset import InteractionsDataset
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.utils import parse_metric_name
from skrec.orchestrator.factory import RecommenderConfig, create_recommender_pipeline
from skrec.recommender.base_recommender import BaseRecommender
from skrec.util.logger import get_logger

logger = get_logger(__name__)


# Helper function for deep dictionary updates with dot notation support
def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping, handling dot notation in keys.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        keys = key.split(".")
        d = source
        for k in keys[:-1]:  # Navigate to the parent dictionary
            if k not in d or not isinstance(d[k], collections.abc.MutableMapping):
                d[k] = {}  # Create nested dict if it doesn't exist or isn't a dict
            d = d[k]

        final_key = keys[-1]
        if isinstance(value, collections.abc.Mapping) and value:
            # If the value is a dictionary, recurse only if the target exists and is also a dict
            if final_key in d and isinstance(d[final_key], collections.abc.MutableMapping):
                deep_update(d[final_key], value)
            else:  # Otherwise, overwrite or set the value
                d[final_key] = value
        else:
            d[final_key] = value  # Set the final value
    return source


# --- Search Space Definition Types ---

# Each entry maps a dotted parameter name to a dict describing the dimension.
# Supported dimension types:
#   {"type": "int", "low": 10, "high": 300}
#   {"type": "int", "low": 10, "high": 300, "step": 10}
#   {"type": "int", "low": 10, "high": 300, "log": True}
#   {"type": "float", "low": 0.001, "high": 0.5}
#   {"type": "float", "low": 0.001, "high": 0.5, "log": True}
#   {"type": "float", "low": 0.001, "high": 0.5, "step": 0.01}
#   {"type": "categorical", "choices": ["adam", "sgd"]}
SearchSpace = Dict[str, Dict[str, Any]]


def _suggest_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    """Suggest a single hyperparameter from an optuna Trial based on a spec dict."""
    dim_type = spec["type"]
    if dim_type == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1), log=spec.get("log", False))
    elif dim_type == "float":
        kwargs: Dict[str, Any] = {}
        if "step" in spec:
            kwargs["step"] = spec["step"]
        if "log" in spec:
            kwargs["log"] = spec["log"]
        return trial.suggest_float(name, spec["low"], spec["high"], **kwargs)
    elif dim_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"Unknown search space type '{dim_type}' for parameter '{name}'")


# --- Sampler helpers ---

# Mapping of string names to optuna sampler classes for convenience.
_SAMPLER_REGISTRY: Dict[str, type] = {
    "tpe": optuna.samplers.TPESampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "random": optuna.samplers.RandomSampler,
    "grid": optuna.samplers.GridSampler,
    "qmc": optuna.samplers.QMCSampler,
}

# Extend with optional GP sampler (optuna >= 4.0)
try:
    _SAMPLER_REGISTRY["gp"] = optuna.samplers.GPSampler
except AttributeError:
    pass

SamplerLike = Union[str, optuna.samplers.BaseSampler]


def _resolve_sampler(
    sampler: SamplerLike, sampler_kwargs: Optional[Dict[str, Any]] = None
) -> optuna.samplers.BaseSampler:
    """Return an optuna BaseSampler instance from a string name or pass-through an existing instance."""
    if isinstance(sampler, optuna.samplers.BaseSampler):
        return sampler
    key = sampler.lower()
    if key not in _SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler '{sampler}'. Available: {sorted(_SAMPLER_REGISTRY.keys())}")
    try:
        return _SAMPLER_REGISTRY[key](**(sampler_kwargs or {}))
    except TypeError as e:
        raise TypeError(
            f"Sampler '{sampler}' requires additional arguments via sampler_kwargs "
            f"(e.g., GridSampler needs 'search_space'). Pass them in sampler_kwargs "
            f"or provide a pre-built sampler instance. Original error: {e}"
        ) from e


class HyperparameterOptimizer:
    """
    Manages Hyperparameter Optimization (HPO) for recommender pipelines.

    Supports Random Search and Bayesian Optimization via Optuna.
    Any optuna sampler (TPE, CMA-ES, GP, QMC, Grid, Random, or a custom
    ``optuna.samplers.BaseSampler`` instance) can be used.
    Handles incremental results persistence to local disk or S3.
    """

    def __init__(
        self,
        base_config: RecommenderConfig,
        search_space: SearchSpace,
        metric_definitions: List[str],
        training_interactions_ds: InteractionsDataset,
        validation_interactions_ds: InteractionsDataset,
        training_users_ds: Optional[UsersDataset] = None,
        training_items_ds: Optional[ItemsDataset] = None,
        validation_users_ds: Optional[UsersDataset] = None,
        evaluator_type: str = "simple",
        persistence_path: Optional[str] = None,
    ):
        """
        Initializes the HyperparameterOptimizer.

        Args:
            base_config: Base configuration for the recommender pipeline.
                         Hyperparameters defined in search_space will override these.
            search_space: Dictionary defining the hyperparameter search space.
                          Keys are dot-notation parameter paths, values are dicts
                          describing the dimension (see ``SearchSpace`` type alias).
            metric_definitions: List of metric names (e.g., "NDCG@10") to compute
                                on the validation set.
            training_interactions_ds: Training interactions dataset.
            validation_interactions_ds: Validation interactions dataset.
            training_users_ds: Optional training users dataset.
            training_items_ds: Optional training items dataset.
            validation_users_ds: Optional validation users dataset.
            evaluator_type: Type of evaluator to use (e.g., "simple", "replay_match").
            persistence_path: Optional local path or S3 URI (s3://...) to save/load
                              HPO results (Parquet format).
        """
        self.base_config = base_config
        self.search_space = search_space
        self.metric_definitions = metric_definitions
        self.training_interactions_ds = training_interactions_ds
        self.validation_interactions_ds = validation_interactions_ds
        self.training_users_ds = training_users_ds
        self.training_items_ds = training_items_ds
        self.validation_users_ds = validation_users_ds
        self.evaluator_type = evaluator_type
        self.persistence_path = persistence_path
        self.results_df = pd.DataFrame()  # Initialize empty results

        # Load previous results if path provided
        if self.persistence_path:
            self.load_results()

        logger.info(f"HyperparameterOptimizer initialized. Persistence path: {self.persistence_path}")
        if not self.results_df.empty:
            logger.info(f"Loaded {len(self.results_df)} previous trial results.")

    # --------------------------------------------------------------------- #
    #  Core: run a single trial given a param dict                           #
    # --------------------------------------------------------------------- #

    def _run_trial(self, params_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Train a pipeline with *params_dict*, evaluate it, persist results,
        and return the metric scores.
        """
        # Coerce values to their declared types — pandas/parquet round-trips
        # can silently convert int columns to float.
        coerced = {}
        for name, value in params_dict.items():
            spec = self.search_space.get(name)
            if spec and spec["type"] == "int":
                value = int(value)
            elif spec and spec["type"] == "float":
                value = float(value)
            coerced[name] = value
        params_dict = coerced

        logger.info(f"Starting trial with params: {params_dict}")
        trial_start_time = time.time()

        current_config = deep_update(copy.deepcopy(self.base_config), params_dict)

        metric_scores: Dict[str, float] = {}
        try:
            pipeline = create_recommender_pipeline(current_config)

            logger.info("Training model...")
            pipeline.train(
                users_ds=self.training_users_ds,
                items_ds=self.training_items_ds,
                interactions_ds=self.training_interactions_ds,
                valid_interactions_ds=self.validation_interactions_ds,
                valid_users_ds=self.validation_users_ds,
            )
            logger.info("Training complete.")

            # Prepare evaluation data
            valid_interactions_df = self.validation_interactions_ds.fetch_data()
            valid_users_df = self.validation_users_ds.fetch_data() if self.validation_users_ds else None
            score_items_kwargs = {"interactions": valid_interactions_df, "users": valid_users_df}

            if ITEM_ID_NAME not in valid_interactions_df.columns:
                raise ValueError(f"Required column '{ITEM_ID_NAME}' not found in validation interactions DataFrame.")

            reward_col = None
            if LABEL_NAME in valid_interactions_df.columns:
                reward_col = LABEL_NAME
            else:
                outcome_cols = [col for col in valid_interactions_df.columns if col.startswith(OUTCOME_PREFIX)]
                if outcome_cols:
                    reward_col = outcome_cols[0]
                else:
                    raise ValueError(
                        f"Cannot find '{LABEL_NAME}' or '{OUTCOME_PREFIX}*' column in "
                        "validation interactions DataFrame for reward."
                    )

            logged_items_np = valid_interactions_df[ITEM_ID_NAME].to_numpy().reshape(-1, 1)
            logged_rewards_np = valid_interactions_df[reward_col].to_numpy().reshape(-1, 1)

            eval_data = {
                "logged_items": logged_items_np,
                "logged_rewards": logged_rewards_np,
            }

            metric_scores = self._evaluate_metrics(pipeline, eval_data, score_items_kwargs)

        except Exception as e:
            logger.error(f"Trial failed during training or evaluation: {e}", exc_info=True)
            for metric_name in self.metric_definitions:
                metric_scores[metric_name] = float("nan")

        trial_duration = time.time() - trial_start_time
        logger.info(f"Trial duration: {trial_duration:.2f} seconds")

        # Persist
        trial_results = {**params_dict, **metric_scores, "trial_duration": trial_duration}
        self.results_df = pd.concat([self.results_df, pd.DataFrame([trial_results])], ignore_index=True)

        if self.persistence_path:
            self._save_results()

        return metric_scores

    # --------------------------------------------------------------------- #
    #  Random Search                                                         #
    # --------------------------------------------------------------------- #

    def run_random_search(self, n_trials: int):
        """
        Performs Random Search HPO for a specified number of trials.

        Args:
            n_trials: The number of random parameter combinations to try.
        """
        logger.info(f"Starting Random Search for {n_trials} trials...")
        param_sampler = ParameterSampler(self.search_space, n_iter=n_trials, random_state=None)

        trial_count = 0
        for params_dict in param_sampler:
            trial_count += 1
            logger.info(f"--- Random Search Trial {trial_count}/{n_trials} ---")
            self._run_trial(params_dict)

        logger.info("Random Search finished.")
        return self.results_df

    # --------------------------------------------------------------------- #
    #  Optuna-based optimization                                             #
    # --------------------------------------------------------------------- #

    def run_optimization(
        self,
        n_trials: int,
        objective_metric: str,
        sampler: SamplerLike = "tpe",
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        direction: str = "maximize",
        study_name: Optional[str] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        optuna_callbacks: Optional[List] = None,
    ) -> tuple[pd.DataFrame, optuna.Study]:
        """
        Run HPO using any Optuna sampler.

        Args:
            n_trials: Number of optimization trials to run.
            objective_metric: The metric in ``metric_definitions`` to optimize.
            sampler: Sampler name (``"tpe"``, ``"gp"``, ``"cmaes"``, ``"random"``,
                     ``"qmc"``, ``"grid"``) or an ``optuna.samplers.BaseSampler``
                     instance for full control.
            sampler_kwargs: Extra keyword arguments forwarded to the sampler
                            constructor when *sampler* is a string.
            direction: ``"maximize"`` or ``"minimize"``.
            study_name: Optional name for the Optuna study.
            pruner: Optional Optuna pruner (e.g. ``MedianPruner``).
            optuna_callbacks: Optional list of Optuna callbacks.

        Returns:
            Tuple of (results_df, optuna.Study).
        """
        if objective_metric not in self.metric_definitions:
            raise ValueError(
                f"Objective metric '{objective_metric}' not found in metric_definitions: {self.metric_definitions}"
            )

        resolved_sampler = _resolve_sampler(sampler, sampler_kwargs)
        logger.info(
            f"Starting Optuna optimization: {n_trials} trials, "
            f"sampler={type(resolved_sampler).__name__}, metric={objective_metric}, direction={direction}"
        )

        study = optuna.create_study(
            study_name=study_name or f"hpo_{objective_metric}",
            direction=direction,
            sampler=resolved_sampler,
            pruner=pruner,
        )

        # Warm-start from previous results
        self._enqueue_previous_trials(study, objective_metric, direction)

        def objective(trial: optuna.Trial) -> float:
            params_dict = {name: _suggest_param(trial, name, spec) for name, spec in self.search_space.items()}
            metric_scores = self._run_trial(params_dict)
            score = metric_scores.get(objective_metric, float("nan"))
            if pd.isna(score):
                raise optuna.TrialPruned(f"Metric '{objective_metric}' is NaN")
            return score

        study.optimize(objective, n_trials=n_trials, callbacks=optuna_callbacks)

        logger.info(f"Optimization finished. Best value: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        return self.results_df, study

    # Keep a convenience alias that matches the old method name
    def run_bayesian_optimization(
        self,
        n_trials: int,
        objective_metric: str,
        sampler: SamplerLike = "tpe",
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        direction: str = "maximize",
        **kwargs,
    ) -> tuple[pd.DataFrame, optuna.Study]:
        """Convenience wrapper around ``run_optimization`` (backwards-compatible name)."""
        return self.run_optimization(
            n_trials=n_trials,
            objective_metric=objective_metric,
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
            direction=direction,
            **kwargs,
        )

    # --------------------------------------------------------------------- #
    #  Warm-start helpers                                                    #
    # --------------------------------------------------------------------- #

    def _enqueue_previous_trials(self, study: optuna.Study, objective_metric: str, direction: str):
        """Add completed trials from ``results_df`` into the study for warm-starting."""
        if self.results_df.empty:
            return

        param_cols = list(self.search_space.keys())
        required_cols = param_cols + [objective_metric]

        if not all(c in self.results_df.columns for c in required_cols):
            logger.warning("Previous results missing required columns for warm-start. Skipping.")
            return

        valid = self.results_df.dropna(subset=required_cols)
        if valid.empty:
            return

        # Filter to rows within current search space bounds
        mask = pd.Series(True, index=valid.index)
        for param_name, spec in self.search_space.items():
            values = valid[param_name]
            if spec["type"] in ("int", "float"):
                mask &= (values >= spec["low"]) & (values <= spec["high"])
            elif spec["type"] == "categorical":
                mask &= values.isin(spec["choices"])
        valid = valid[mask]

        if valid.empty:
            logger.warning("No previous results within current search space bounds.")
            return

        for _, row in valid.iterrows():
            params = {p: row[p] for p in param_cols}
            value = row[objective_metric]
            dist = _build_distributions(self.search_space)
            study.add_trial(
                optuna.trial.create_trial(
                    params=params,
                    distributions=dist,
                    values=[value],
                    state=optuna.trial.TrialState.COMPLETE,
                )
            )

        logger.info(f"Warm-started study with {len(valid)} previous trials.")

    # --------------------------------------------------------------------- #
    #  Evaluation                                                            #
    # --------------------------------------------------------------------- #

    def _evaluate_metrics(
        self, pipeline: BaseRecommender, eval_data: Dict[str, np.ndarray], score_items_kwargs
    ) -> Dict[str, float]:
        """Evaluate the trained pipeline against all defined metrics."""
        logger.info("Evaluating metrics...")
        metric_scores = {}
        try:
            eval_type_enum = RecommenderEvaluatorType[self.evaluator_type.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid evaluator_type: {self.evaluator_type}. "
                f"Valid types: {[e.name for e in RecommenderEvaluatorType]}"
            )

        for metric_name_full in self.metric_definitions:
            try:
                metric_type_enum, eval_k = parse_metric_name(metric_name_full)
                score = pipeline.evaluate(
                    eval_type=eval_type_enum,
                    metric_type=metric_type_enum,
                    eval_top_k=eval_k,
                    eval_kwargs=eval_data,
                    score_items_kwargs=score_items_kwargs,
                )
                metric_scores[metric_name_full] = score
                logger.info(f"Metric {metric_name_full}: {score}")
            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name_full}: {e}", exc_info=True)
                metric_scores[metric_name_full] = float("nan")

        logger.info("Evaluation complete.")
        return metric_scores

    # --------------------------------------------------------------------- #
    #  Persistence                                                           #
    # --------------------------------------------------------------------- #

    def _save_results(self):
        """Save current results_df to persistence_path."""
        logger.info(f"Saving {len(self.results_df)} results to {self.persistence_path}")
        try:
            if not self.persistence_path.startswith("s3://"):
                import os

                dir_path = os.path.dirname(self.persistence_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
            self.results_df.to_parquet(self.persistence_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save results to {self.persistence_path}: {e}")

    def load_results(self):
        """Loads previous HPO results from the persistence_path."""
        if not self.persistence_path:
            logger.warning("No persistence path specified, cannot load results.")
            return

        logger.info(f"Attempting to load previous results from {self.persistence_path}...")
        try:
            self.results_df = pd.read_parquet(self.persistence_path)
            logger.info(f"Successfully loaded {len(self.results_df)} results.")
        except FileNotFoundError:
            logger.warning(f"Persistence file not found at {self.persistence_path}. Starting fresh.")
            self.results_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load results from {self.persistence_path}: {e}. Starting fresh.")
            self.results_df = pd.DataFrame()


# --------------------------------------------------------------------- #
#  Module-level helpers                                                  #
# --------------------------------------------------------------------- #


def _build_distributions(search_space: SearchSpace) -> Dict[str, optuna.distributions.BaseDistribution]:
    """Convert a SearchSpace dict to optuna distributions (needed for add_trial)."""
    dists: Dict[str, optuna.distributions.BaseDistribution] = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "int":
            dists[name] = optuna.distributions.IntDistribution(
                low=spec["low"], high=spec["high"], step=spec.get("step", 1), log=spec.get("log", False)
            )
        elif t == "float":
            kwargs: Dict[str, Any] = {}
            if "step" in spec:
                kwargs["step"] = spec["step"]
            if "log" in spec:
                kwargs["log"] = spec["log"]
            dists[name] = optuna.distributions.FloatDistribution(low=spec["low"], high=spec["high"], **kwargs)
        elif t == "categorical":
            dists[name] = optuna.distributions.CategoricalDistribution(choices=spec["choices"])
        else:
            raise ValueError(f"Unknown type '{t}' for parameter '{name}'")
    return dists
