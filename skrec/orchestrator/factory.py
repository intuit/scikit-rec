from typing import Any, Dict, Optional, TypedDict

from skrec.estimator.base_estimator import BaseEstimator
from skrec.estimator.classification.multioutput_classifier import (
    MultiOutputClassifierEstimator,
    TunedMultiOutputClassifierEstimator,
)
from skrec.estimator.classification.xgb_classifier import (
    TunedXGBClassifierEstimator,
    WeightedXGBClassifierEstimator,
    XGBClassifier,
    XGBClassifierEstimator,
)
from skrec.estimator.datatypes import HPOType
from skrec.estimator.regression.xgb_regressor import (
    TunedXGBRegressorEstimator,
    XGBRegressorEstimator,
)
from skrec.recommender.bandits.contextual_bandits import ContextualBanditsRecommender
from skrec.recommender.base_recommender import BaseRecommender
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.scorer.base_scorer import BaseScorer
from skrec.scorer.independent import IndependentScorer
from skrec.scorer.multiclass import MulticlassScorer
from skrec.scorer.multioutput import MultioutputScorer
from skrec.scorer.universal import UniversalScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)

# --- Type Definitions ---

ParamSpace = Dict[str, Any]
OptimizerParams = Dict[str, Any]


class XGBConfig(TypedDict, total=False):
    pass  # Define specific XGBoost keys if known


class HPOConfig(TypedDict, total=False):
    hpo_method: HPOType
    param_space: ParamSpace
    optimizer_params: OptimizerParams


class WeightsConfig(TypedDict, total=False):
    action_weight: float
    item_sample_weights: Optional[Dict[Any, float]]


class EstimatorConfig(TypedDict, total=False):
    ml_task: str
    xgboost: XGBConfig
    hpo: HPOConfig
    weights: WeightsConfig


class RecommenderConfig(TypedDict, total=False):
    recommender_type: str
    scorer_type: str
    estimator_config: EstimatorConfig


# --- Factory Functions ---
def create_estimator(estimator_config: EstimatorConfig, scorer_type: Optional[str] = None) -> BaseEstimator:
    """
    Factory function to create an estimator instance based on its specific configuration.

    Args:
        estimator_config: Dictionary containing configuration specific to the estimator.
                          Keys like 'ml_task', 'xgboost', 'hpo', 'weights'.
        scorer_type: Optional string indicating the scorer type, used to select
                     specialized estimators like MultiOutputClassifierEstimator.

    Returns:
        An instance of a BaseEstimator subclass.

    Raises:
        NotImplementedError: If the ml_task is not supported.
        ValueError: If configuration is inconsistent.
    """
    # Extract configurations from estimator_config
    ml_task = estimator_config.get("ml_task", "classification")
    xgb_config = estimator_config.get("xgboost", {})
    hpo_config = estimator_config.get("hpo", {})
    weights_config = estimator_config.get("weights", {})

    is_tuned_mode = bool(
        hpo_config.get("hpo_method") or hpo_config.get("param_space") or hpo_config.get("optimizer_params")
    )

    logger.info(f"Creating estimator. ML Task: {ml_task}, Scorer Type Hint: {scorer_type}, Tuned Mode: {is_tuned_mode}")

    if ml_task not in {"classification", "regression"}:
        raise NotImplementedError(f"ML task {ml_task} not implemented.")

    estimator: BaseEstimator

    if is_tuned_mode:
        # Ensure required HPO keys are present if is_tuned_mode is True
        if not all(k in hpo_config for k in ["hpo_method", "param_space", "optimizer_params"]):
            raise ValueError(
                "Missing required HPO configuration keys (hpo_method, param_space, optimizer_params) for tuned mode."
            )
        hpo_method = hpo_config["hpo_method"]
        param_space = hpo_config["param_space"]
        optimizer_params = hpo_config["optimizer_params"]

        if ml_task == "classification":
            if scorer_type == "multioutput":
                logger.info("Creating TunedMultiOutputClassifierEstimator")
                estimator = TunedMultiOutputClassifierEstimator(
                    base_estimator=XGBClassifier,
                    hpo_method=hpo_method,
                    param_space=param_space,
                    optimizer_params=optimizer_params,
                )
            else:
                logger.info("Creating TunedXGBClassifierEstimator")
                estimator = TunedXGBClassifierEstimator(
                    hpo_method=hpo_method,
                    param_space=param_space,
                    optimizer_params=optimizer_params,
                )
        else:
            logger.info("Creating TunedXGBRegressorEstimator")
            estimator = TunedXGBRegressorEstimator(
                hpo_method=hpo_method,
                param_space=param_space,
                optimizer_params=optimizer_params,
            )
    else:
        if ml_task == "classification":
            action_weight = weights_config.get("action_weight", 1)
            item_sample_weights = weights_config.get("item_sample_weights")

            if scorer_type == "multioutput":
                logger.info("Creating MultiOutputClassifierEstimator with XGBClassifier")
                # Pass base model class and its params separately
                estimator = MultiOutputClassifierEstimator(XGBClassifier, xgb_config)
            elif action_weight != 1 or item_sample_weights is not None:
                logger.info("Creating WeightedXGBClassifierEstimator")
                estimator = WeightedXGBClassifierEstimator(
                    params=xgb_config,
                    action_weight=action_weight,
                    item_sample_weights=item_sample_weights,
                )
            else:
                logger.info("Creating XGBClassifierEstimator")
                estimator = XGBClassifierEstimator(xgb_config)
        else:  # regression
            logger.info("Creating XGBRegressorEstimator")
            estimator = XGBRegressorEstimator(xgb_config)

    return estimator


def create_scorer(estimator: BaseEstimator, config: RecommenderConfig) -> BaseScorer:
    """
    Factory function to create a scorer instance based on the overall recommender configuration.

    Args:
        estimator: The estimator instance to be used by the scorer.
        config: The main recommender configuration dictionary.
                Expected key: 'scorer_type'.

    Returns:
        An instance of a BaseScorer subclass.

    Raises:
        NotImplementedError: If the scorer_type is not supported.
        ValueError: If scorer_type is missing.
    """
    scorer_type = config.get("scorer_type")
    if not scorer_type:
        raise ValueError("'scorer_type' must be specified in the configuration.")

    logger.info(f"Creating scorer of type: {scorer_type}")

    scorer: BaseScorer

    if scorer_type == "multioutput":
        scorer = MultioutputScorer(estimator=estimator)
    elif scorer_type == "multiclass":
        scorer = MulticlassScorer(estimator=estimator)
    elif scorer_type == "independent":
        scorer = IndependentScorer(estimator=estimator)
    elif scorer_type == "universal":
        scorer = UniversalScorer(estimator=estimator)
    else:
        raise NotImplementedError(f"Scorer type '{scorer_type}' not supported.")

    return scorer


def create_recommender(scorer: BaseScorer, config: RecommenderConfig) -> BaseRecommender:
    """
    Factory function to create a recommender instance based on the overall recommender configuration.

    Args:
        scorer: The scorer instance to be used by the recommender.
        config: The main recommender configuration dictionary.
                Expected key: 'recommender_type'.

    Returns:
        An instance of a BaseRecommender subclass.
    """
    # Defaulting to ranking, align with trainer.py logic
    recommender_type = config.get("recommender_type", "ranking")
    logger.info(f"Creating recommender of type: {recommender_type}")

    recommender: BaseRecommender

    if recommender_type == "bandits":
        recommender = ContextualBanditsRecommender(scorer=scorer)
    elif recommender_type == "ranking":
        recommender = RankingRecommender(scorer=scorer)
    else:
        # Assuming RankingRecommender is the default/fallback
        logger.warning(f"Unsupported recommender_type '{recommender_type}'. Defaulting to RankingRecommender.")
        recommender = RankingRecommender(scorer=scorer)

    return recommender


def create_recommender_pipeline(config: RecommenderConfig) -> BaseRecommender:
    """
    Factory function to create a complete recommender pipeline (Estimator -> Scorer -> Recommender)
    from the main recommender configuration dictionary.

    Args:
        config: The main recommender configuration dictionary containing nested
                'estimator_config' and top-level 'scorer_type', 'recommender_type'.

    Returns:
        A fully assembled BaseRecommender instance.
    """
    logger.info("Creating recommender pipeline from config...")

    estimator_config = config.get("estimator_config", {})
    scorer_type = config.get("scorer_type")

    if not estimator_config:
        logger.warning("estimator_config not found in main config. Attempting to proceed with empty estimator config.")

    # Create components using their respective factory functions
    estimator = create_estimator(estimator_config, scorer_type=scorer_type)
    scorer = create_scorer(estimator, config)
    recommender = create_recommender(scorer, config)

    logger.info("Recommender pipeline created successfully.")
    return recommender
