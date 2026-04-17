import importlib
from typing import Any, Dict, Optional, Tuple, Type, TypedDict, Union

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
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.estimator.regression.xgb_regressor import (
    TunedXGBRegressorEstimator,
    XGBRegressorEstimator,
)
from skrec.estimator.sequential.base_sequential_estimator import SequentialEstimator
from skrec.recommender.bandits.contextual_bandits import ContextualBanditsRecommender
from skrec.recommender.base_recommender import BaseRecommender
from skrec.recommender.gcsl.gcsl_recommender import GcslRecommender
from skrec.recommender.gcsl.inference.base_inference import BaseInference
from skrec.recommender.gcsl.inference.mean_scalarization import MeanScalarization
from skrec.recommender.gcsl.inference.percentile_value import PercentileValue
from skrec.recommender.gcsl.inference.predefined_value import PredefinedValue
from skrec.recommender.ranking.ranking_recommender import RankingRecommender
from skrec.recommender.sequential.hierarchical_recommender import HierarchicalSequentialRecommender
from skrec.recommender.sequential.sequential_recommender import SequentialRecommender
from skrec.recommender.uplift_model.uplift_recommender import UpliftRecommender
from skrec.retriever.base_retriever import BaseCandidateRetriever
from skrec.retriever.content_based_retriever import ContentBasedRetriever
from skrec.retriever.embedding_retriever import EmbeddingRetriever
from skrec.retriever.popularity_retriever import PopularityRetriever
from skrec.scorer.base_scorer import BaseScorer
from skrec.scorer.hierarchical import HierarchicalScorer
from skrec.scorer.independent import IndependentScorer
from skrec.scorer.multiclass import MulticlassScorer
from skrec.scorer.multioutput import MultioutputScorer
from skrec.scorer.sequential import SequentialScorer
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


class EmbeddingConfig(TypedDict, total=False):
    model_type: str  # "matrix_factorization", "ncf", "two_tower", "deep_cross_network", "neural_factorization"
    params: Dict[str, Any]


class SequentialConfig(TypedDict, total=False):
    model_type: str  # "sasrec_classifier", "sasrec_regressor", "hrnn_classifier", "hrnn_regressor"
    params: Dict[str, Any]


class EstimatorConfig(TypedDict, total=False):
    estimator_type: str  # "tabular" (default), "embedding", "sequential"
    # --- tabular (existing) ---
    ml_task: str
    xgboost: XGBConfig
    hpo: HPOConfig
    weights: WeightsConfig
    # --- embedding ---
    embedding: EmbeddingConfig
    # --- sequential ---
    sequential: SequentialConfig


class InferenceMethodConfig(TypedDict, total=False):
    type: str  # "mean_scalarization", "percentile_value", "predefined_value"
    params: Dict[str, Any]


class RetrieverConfig(TypedDict, total=False):
    type: str  # "popularity", "content_based", "embedding"
    params: Dict[str, Any]


class RecommenderParams(TypedDict, total=False):
    """Per-recommender constructor parameters.

    Not all keys apply to every recommender type. The mapping:

    - ``ranking``: ``retriever`` (optional)
    - ``sequential``: ``max_len``
    - ``hierarchical_sequential``: ``max_sessions``, ``max_session_len``,
      ``session_timeout_minutes``
    - ``uplift``: ``control_item_id`` (**required**), ``mode``
    - ``gcsl``: ``inference_method``, ``retriever``
    - ``bandits``: (none)

    Keys irrelevant to the chosen recommender_type are silently ignored.
    """

    # --- sequential ---
    max_len: int
    # --- hierarchical_sequential ---
    max_sessions: int
    max_session_len: int
    session_timeout_minutes: float
    # --- uplift ---
    control_item_id: str  # required for uplift
    mode: str  # optional; auto-detects from scorer type
    # --- gcsl ---
    inference_method: InferenceMethodConfig
    # --- ranking / gcsl ---
    retriever: RetrieverConfig


class RecommenderConfig(TypedDict, total=False):
    recommender_type: str  # "ranking", "bandits", "sequential", "hierarchical_sequential", "uplift", "gcsl"
    scorer_type: str  # "multioutput", "multiclass", "independent", "universal", "sequential", "hierarchical"
    estimator_config: EstimatorConfig
    recommender_params: RecommenderParams


# --- Class Maps ---
# Embedding and sequential estimator maps use lazy imports to avoid pulling in
# PyTorch at module load time. Each entry is (module_path, class_name).

_EMB_MOD = "skrec.estimator.embedding"
_EMBEDDING_ESTIMATOR_MAP: Dict[str, tuple] = {
    "matrix_factorization": (f"{_EMB_MOD}.matrix_factorization_estimator", "MatrixFactorizationEstimator"),
    "ncf": (f"{_EMB_MOD}.ncf_estimator", "NCFEstimator"),
    "two_tower": (f"{_EMB_MOD}.contextualized_two_tower_estimator", "ContextualizedTwoTowerEstimator"),
    "deep_cross_network": (f"{_EMB_MOD}.deep_cross_network_estimator", "DeepCrossNetworkEstimator"),
    "neural_factorization": (f"{_EMB_MOD}.neural_factorization_estimator", "NeuralFactorizationEstimator"),
}

_SEQUENTIAL_ESTIMATOR_MAP: Dict[str, tuple] = {
    "sasrec_classifier": ("skrec.estimator.sequential.sasrec_estimator", "SASRecClassifierEstimator"),
    "sasrec_regressor": ("skrec.estimator.sequential.sasrec_estimator", "SASRecRegressorEstimator"),
    "hrnn_classifier": ("skrec.estimator.sequential.hrnn_estimator", "HRNNClassifierEstimator"),
    "hrnn_regressor": ("skrec.estimator.sequential.hrnn_estimator", "HRNNRegressorEstimator"),
}


def _resolve_lazy(registry: Dict[str, Tuple[str, str]], key: str) -> Type:
    """Resolve a lazy (module_path, class_name) entry to an actual class."""
    module_path, class_name = registry[key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

_INFERENCE_METHOD_MAP = {
    "mean_scalarization": MeanScalarization,
    "percentile_value": PercentileValue,
    "predefined_value": PredefinedValue,
}

_RETRIEVER_MAP = {
    "popularity": PopularityRetriever,
    "content_based": ContentBasedRetriever,
    "embedding": EmbeddingRetriever,
}

_NON_TABULAR_KEYS = {"embedding", "sequential"}
_TABULAR_SCORER_TYPES = {"multioutput", "multiclass", "independent", "universal"}
_EMBEDDING_INCOMPATIBLE_SCORERS = {"multioutput", "multiclass", "independent"}


# --- Private Helpers ---


def _create_embedding_estimator(embedding_config: EmbeddingConfig) -> BaseEstimator:
    """Create an embedding-based estimator from config."""
    model_type = embedding_config.get("model_type")
    if not model_type:
        raise ValueError("'model_type' is required in embedding config.")

    if model_type not in _EMBEDDING_ESTIMATOR_MAP:
        raise NotImplementedError(
            f"Embedding model type '{model_type}' not supported. "
            f"Supported types: {list(_EMBEDDING_ESTIMATOR_MAP.keys())}"
        )

    cls = _resolve_lazy(_EMBEDDING_ESTIMATOR_MAP, model_type)
    params = embedding_config.get("params", {})
    logger.info(f"Creating {cls.__name__} with params: {params}")
    return cls(**params)


def _create_sequential_estimator(sequential_config: SequentialConfig) -> SequentialEstimator:
    """Create a sequential estimator from config."""
    model_type = sequential_config.get("model_type")
    if not model_type:
        raise ValueError("'model_type' is required in sequential config.")

    if model_type not in _SEQUENTIAL_ESTIMATOR_MAP:
        raise NotImplementedError(
            f"Sequential model type '{model_type}' not supported. "
            f"Supported types: {list(_SEQUENTIAL_ESTIMATOR_MAP.keys())}"
        )

    cls = _resolve_lazy(_SEQUENTIAL_ESTIMATOR_MAP, model_type)
    params = sequential_config.get("params", {})
    logger.info(f"Creating {cls.__name__} with params: {params}")
    return cls(**params)


def _create_inference_method(config: InferenceMethodConfig) -> BaseInference:
    """Create a GCSL inference method from config."""
    method_type = config.get("type")
    if not method_type:
        raise ValueError("'type' is required in inference_method config.")

    cls = _INFERENCE_METHOD_MAP.get(method_type)
    if cls is None:
        raise NotImplementedError(
            f"Inference method type '{method_type}' not supported. "
            f"Supported types: {list(_INFERENCE_METHOD_MAP.keys())}"
        )

    params = config.get("params", {})
    logger.info(f"Creating inference method {cls.__name__} with params: {params}")
    return cls(**params)


def _create_retriever(config: RetrieverConfig) -> BaseCandidateRetriever:
    """Create a candidate retriever from config."""
    retriever_type = config.get("type")
    if not retriever_type:
        raise ValueError("'type' is required in retriever config.")

    cls = _RETRIEVER_MAP.get(retriever_type)
    if cls is None:
        raise NotImplementedError(
            f"Retriever type '{retriever_type}' not supported. "
            f"Supported types: {list(_RETRIEVER_MAP.keys())}"
        )

    params = config.get("params", {})
    logger.info(f"Creating retriever {cls.__name__} with params: {params}")
    return cls(**params)


# --- Factory Functions ---
def create_estimator(
    estimator_config: EstimatorConfig, scorer_type: Optional[str] = None
) -> Union[BaseEstimator, SequentialEstimator]:
    """
    Factory function to create an estimator instance based on its specific configuration.

    Args:
        estimator_config: Dictionary containing configuration specific to the estimator.
                          Keys like 'estimator_type', 'ml_task', 'xgboost', 'hpo',
                          'weights', 'embedding', 'sequential'.
        scorer_type: Optional string indicating the scorer type, used to select
                     specialized estimators like MultiOutputClassifierEstimator.

    Returns:
        An instance of BaseEstimator (tabular/embedding) or SequentialEstimator (sequential).

    Raises:
        NotImplementedError: If the estimator_type or ml_task is not supported.
        ValueError: If configuration is inconsistent.
    """
    estimator_type = estimator_config.get("estimator_type") or "tabular"
    logger.info(f"Creating estimator. Estimator type: {estimator_type}")

    if estimator_type == "embedding":
        embedding_config = estimator_config.get("embedding")
        if embedding_config is None:
            raise ValueError("'embedding' key is required in estimator_config when estimator_type is 'embedding'.")
        if not embedding_config:
            raise ValueError("'embedding' config is empty. It must contain at least 'model_type'.")
        return _create_embedding_estimator(embedding_config)

    if estimator_type == "sequential":
        sequential_config = estimator_config.get("sequential")
        if sequential_config is None:
            raise ValueError("'sequential' key is required in estimator_config when estimator_type is 'sequential'.")
        if not sequential_config:
            raise ValueError("'sequential' config is empty. It must contain at least 'model_type'.")
        return _create_sequential_estimator(sequential_config)

    if estimator_type != "tabular":
        raise NotImplementedError(
            f"Estimator type '{estimator_type}' not supported. "
            f"Supported types: 'tabular', 'embedding', 'sequential'."
        )

    # --- Tabular estimator path (existing logic) ---
    # Warn if config contains keys meant for other estimator types
    unexpected = _NON_TABULAR_KEYS & set(estimator_config.keys())
    if unexpected:
        logger.warning(
            f"estimator_type is 'tabular' but config contains keys {unexpected} "
            f"which will be ignored. Did you mean to set estimator_type='embedding' or 'sequential'?"
        )

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


def create_scorer(
    estimator: Union[BaseEstimator, SequentialEstimator], config: RecommenderConfig
) -> BaseScorer:
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
        TypeError: If estimator type is incompatible with scorer type.
    """
    scorer_type = config.get("scorer_type")
    if not scorer_type:
        raise ValueError("'scorer_type' must be specified in the configuration.")

    logger.info(f"Creating scorer of type: {scorer_type}")

    # Guard: tabular scorers require BaseEstimator, not SequentialEstimator
    if scorer_type in _TABULAR_SCORER_TYPES and isinstance(estimator, SequentialEstimator):
        raise TypeError(
            f"Scorer type '{scorer_type}' requires a BaseEstimator, "
            f"got {type(estimator).__name__}. Use scorer_type='sequential' or 'hierarchical' "
            f"with sequential estimators."
        )

    # Guard: multioutput/multiclass/independent scorers reject embedding estimators
    if scorer_type in _EMBEDDING_INCOMPATIBLE_SCORERS and isinstance(estimator, BaseEmbeddingEstimator):
        raise TypeError(
            f"Scorer type '{scorer_type}' does not support embedding estimators "
            f"(got {type(estimator).__name__}). Use scorer_type='universal' with embedding estimators."
        )

    scorer: BaseScorer

    if scorer_type == "multioutput":
        scorer = MultioutputScorer(estimator=estimator)
    elif scorer_type == "multiclass":
        scorer = MulticlassScorer(estimator=estimator)
    elif scorer_type == "independent":
        scorer = IndependentScorer(estimator=estimator)
    elif scorer_type == "universal":
        scorer = UniversalScorer(estimator=estimator)
    elif scorer_type == "sequential":
        if not isinstance(estimator, SequentialEstimator):
            raise TypeError(
                f"Sequential scorer requires a SequentialEstimator, got {type(estimator).__name__}."
            )
        scorer = SequentialScorer(estimator=estimator)
    elif scorer_type == "hierarchical":
        if not isinstance(estimator, SequentialEstimator):
            raise TypeError(
                f"Hierarchical scorer requires a SequentialEstimator, got {type(estimator).__name__}."
            )
        scorer = HierarchicalScorer(estimator=estimator)
    else:
        raise NotImplementedError(f"Scorer type '{scorer_type}' not supported.")

    return scorer


def create_recommender(scorer: BaseScorer, config: RecommenderConfig) -> BaseRecommender:
    """
    Factory function to create a recommender instance based on the overall recommender configuration.

    Args:
        scorer: The scorer instance to be used by the recommender.
        config: The main recommender configuration dictionary.
                Expected keys: 'recommender_type', 'recommender_params'.

    Returns:
        An instance of a BaseRecommender subclass.
    """
    # Use `or` so explicit None is treated as "use default", not as a distinct value
    recommender_type = config.get("recommender_type") or "ranking"
    recommender_params = config.get("recommender_params", {})
    logger.info(f"Creating recommender of type: {recommender_type}")

    recommender: BaseRecommender

    if recommender_type == "bandits":
        recommender = ContextualBanditsRecommender(scorer=scorer)
    elif recommender_type == "ranking":
        retriever = _create_retriever(recommender_params["retriever"]) if recommender_params.get("retriever") else None
        recommender = RankingRecommender(scorer=scorer, retriever=retriever)
    elif recommender_type == "sequential":
        if not isinstance(scorer, SequentialScorer):
            raise TypeError(
                f"SequentialRecommender requires a SequentialScorer, got {type(scorer).__name__}."
            )
        recommender = SequentialRecommender(
            scorer=scorer,
            max_len=recommender_params.get("max_len", 50),
        )
    elif recommender_type == "hierarchical_sequential":
        if not isinstance(scorer, HierarchicalScorer):
            raise TypeError(
                f"HierarchicalSequentialRecommender requires a HierarchicalScorer, "
                f"got {type(scorer).__name__}."
            )
        recommender = HierarchicalSequentialRecommender(
            scorer=scorer,
            max_sessions=recommender_params.get("max_sessions", 10),
            max_session_len=recommender_params.get("max_session_len", 20),
            session_timeout_minutes=recommender_params.get("session_timeout_minutes", 30.0),
        )
    elif recommender_type == "uplift":
        control_item_id = recommender_params.get("control_item_id")
        if control_item_id is None:
            raise ValueError("'control_item_id' is required in recommender_params for uplift recommender.")
        recommender = UpliftRecommender(
            scorer=scorer,
            control_item_id=control_item_id,
            mode=recommender_params.get("mode"),
        )
    elif recommender_type == "gcsl":
        inference_config = recommender_params.get("inference_method")
        inference_method = _create_inference_method(inference_config) if inference_config else None
        retriever = _create_retriever(recommender_params["retriever"]) if recommender_params.get("retriever") else None
        recommender = GcslRecommender(
            scorer=scorer,
            inference_method=inference_method,
            retriever=retriever,
        )
    else:
        raise NotImplementedError(
            f"Recommender type '{recommender_type}' not supported. "
            f"Supported types: 'ranking', 'bandits', 'sequential', "
            f"'hierarchical_sequential', 'uplift', 'gcsl'."
        )

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
    recommender_type = config.get("recommender_type") or "ranking"
    estimator_type = estimator_config.get("estimator_type") or "tabular"

    if not estimator_config:
        logger.warning("estimator_config not found in main config. Attempting to proceed with empty estimator config.")

    # Cross-cutting validation: catch mismatches early
    if recommender_type in ("sequential", "hierarchical_sequential"):
        if estimator_type != "sequential":
            raise ValueError(
                f"recommender_type '{recommender_type}' requires estimator_type 'sequential', "
                f"got '{estimator_type}'."
            )
    if recommender_type == "sequential" and scorer_type != "sequential":
        raise ValueError(
            f"recommender_type 'sequential' requires scorer_type 'sequential', got '{scorer_type}'."
        )
    if recommender_type == "hierarchical_sequential" and scorer_type != "hierarchical":
        raise ValueError(
            f"recommender_type 'hierarchical_sequential' requires scorer_type 'hierarchical', "
            f"got '{scorer_type}'."
        )
    if scorer_type in ("sequential", "hierarchical") and estimator_type != "sequential":
        raise ValueError(
            f"scorer_type '{scorer_type}' requires estimator_type 'sequential', got '{estimator_type}'."
        )
    if estimator_type == "embedding" and scorer_type in ("multioutput", "multiclass", "independent"):
        raise ValueError(
            f"scorer_type '{scorer_type}' does not support embedding estimators. "
            f"Use scorer_type='universal' with embedding estimators."
        )
    if recommender_type == "uplift" and scorer_type not in ("independent", "universal"):
        raise ValueError(
            f"recommender_type 'uplift' requires scorer_type 'independent' or 'universal', "
            f"got '{scorer_type}'."
        )

    # Create components using their respective factory functions
    estimator = create_estimator(estimator_config, scorer_type=scorer_type)
    scorer = create_scorer(estimator, config)
    recommender = create_recommender(scorer, config)

    logger.info("Recommender pipeline created successfully.")
    return recommender
