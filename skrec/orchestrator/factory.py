import importlib
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, Union

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from skrec.estimator.base_estimator import BaseEstimator
from skrec.estimator.classification.joint_xgb_multioutput import (
    JointXGBMultiOutputClassifierEstimator,
)
from skrec.estimator.classification.lightgbm_classifier import (
    LightGBMClassifierEstimator,
    TunedLightGBMClassifierEstimator,
)
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
from skrec.estimator.regression.joint_xgb_multioutput import (
    JointXGBMultiOutputRegressorEstimator,
)
from skrec.estimator.regression.lightgbm_regressor import (
    LightGBMRegressorEstimator,
    TunedLightGBMRegressorEstimator,
)
from skrec.estimator.regression.multioutput_regressor import (
    MultiOutputRegressorEstimator,
    TunedMultiOutputRegressorEstimator,
)
from skrec.estimator.regression.xgb_regressor import (
    TunedXGBRegressorEstimator,
    XGBRegressor,
    XGBRegressorEstimator,
)
from skrec.estimator.sequential.base_sequential_estimator import SequentialEstimator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.metrics.datatypes import RecommenderMetricType
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
from skrec.scorer.mixed_type_multi_target import (
    TARGET_TYPE_TO_METRICS,
    MixedTypeMultiTargetScorer,
    TargetType,
)
from skrec.scorer.multiclass import MulticlassScorer
from skrec.scorer.multioutput import DegenerateTargetPolicy, MultioutputScorer
from skrec.scorer.sequential import SequentialScorer
from skrec.scorer.universal import UniversalScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)

# Authoritative, single-source-of-truth enums — these back both the factory's
# upfront validation (inside `create_recommender_pipeline`) and any external
# consumer (e.g., a system-prompt builder) that needs to introspect
# scikit-rec's capabilities.
RECOMMENDER_TYPES: Tuple[str, ...] = (
    "ranking",
    "bandits",
    "sequential",
    "hierarchical_sequential",
    "uplift",
    "gcsl",
)
SCORER_TYPES: Tuple[str, ...] = (
    "universal",
    "independent",
    "multiclass",
    "multioutput",
    "mixed_type_multi_target",
    "sequential",
    "hierarchical",
)
ESTIMATOR_TYPES: Tuple[str, ...] = ("tabular", "embedding", "sequential")

# --- Type Definitions ---

ParamSpace = Dict[str, Any]
OptimizerParams = Dict[str, Any]


class XGBConfig(TypedDict, total=False):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    colsample_bynode: float
    objective: str
    eval_metric: str
    n_jobs: int
    random_state: int


class LGBMConfig(TypedDict, total=False):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    num_leaves: int
    min_child_samples: int
    n_jobs: int
    random_state: int


class MultiTargetIndependentTypeConfig(TypedDict, total=False):
    """Single (TargetType → sub-estimator) selection inside ``independent.defaults``.

    Maps one declared target type (e.g. ``"binary"``) to the chosen
    sub-estimator type name and its params.
    """

    estimator_type: str  # "xgboost", "lightgbm", "logreg", "sklearn"
    params: Dict[str, Any]


class MultiTargetIndependentConfig(TypedDict, total=False):
    """``independent`` sub-config for ``mode="independent"``.

    Resolved per fanned-out target: first a name-keyed override in
    ``per_target`` is checked; if absent, the ``TargetType``-keyed default
    in ``defaults`` is used. Both maps may be omitted only when the other
    covers every declared target.
    """

    defaults: Dict[str, MultiTargetIndependentTypeConfig]  # keyed by TargetType.value
    per_target: Dict[str, MultiTargetIndependentTypeConfig]  # keyed by target name


class MultiTargetConfig(TypedDict, total=False):
    """``multi_target`` sub-config under ``estimator_config``.

    Selects one of the three v2 multi-target estimator families and supplies
    its hyperparameters. For ``mode="independent"`` the ``independent``
    block holds per-target sub-estimator specs; the joint modes use ``params``
    directly.
    """

    mode: str  # "joint_mlp" | "joint_transformer" | "independent"
    params: Dict[str, Any]  # joint_mlp / joint_transformer hyperparameters
    independent: MultiTargetIndependentConfig
    # Top-level random seed used by ``_create_multi_target_estimator`` to
    # propagate ``random_state`` into independent-mode sub-estimator
    # ``params`` dicts that don't already set one. Joint modes read
    # ``params["seed"]`` directly via their estimator constructors. None →
    # no auto-injection (each sub-estimator picks its own default, which
    # may be unseeded — runs are non-reproducible).
    random_state: Optional[int]


class DeepFMConfig(TypedDict, total=False):
    embedding_dim: int
    hidden_dim1: int
    hidden_dim2: int
    hidden_dim3: int
    batch_size: int
    epochs: int
    lr: float
    l1_reg: float
    l2_reg: float
    dropout: float
    cosine_tmax: int
    device: Optional[str]
    use_cross_layer: bool
    num_cross_layers: int
    use_batch_norm: bool
    bn_momentum: float


class HPOConfig(TypedDict, total=False):
    hpo_method: HPOType
    param_space: ParamSpace
    optimizer_params: OptimizerParams


class WeightsConfig(TypedDict, total=False):
    action_weight: float
    item_sample_weights: Optional[Dict[Any, float]]
    # Generic fit-time passthrough (sklearn-API estimators only). `sample_weight`
    # is a row-weight strategy: 'balanced', a callable fn(y)->weights, or an
    # explicit array (None = uniform). `fit_params` is a dict of static kwargs
    # forwarded verbatim to the wrapped model's fit (feature_weights, callbacks,
    # base_margin, custom objective, ...). See skrec.estimator._fit_params_mixin.
    sample_weight: Any
    fit_params: Dict[str, Any]


class EmbeddingConfig(TypedDict, total=False):
    model_type: str  # "matrix_factorization", "ncf", "two_tower", "deep_cross_network", "neural_factorization"
    params: Dict[str, Any]


class SequentialConfig(TypedDict, total=False):
    model_type: str  # "sasrec_classifier", "sasrec_regressor", "hrnn_classifier", "hrnn_regressor"
    params: Dict[str, Any]


class EstimatorConfig(TypedDict, total=False):
    estimator_type: str  # "tabular" (default), "embedding", "sequential"
    # --- tabular (sklearn-compatible, no torch required) ---
    ml_task: str
    xgboost: XGBConfig
    lightgbm: LGBMConfig
    hpo: HPOConfig
    weights: WeightsConfig
    # --- multioutput estimator structure (scorer_type="multioutput" only) ---
    # "per_label" (default): N independent boosters via MultiOutputClassifier/
    # Regressor. "joint": a single joint XGBoost booster
    # (JointXGBMultiOutput{Classifier,Regressor}Estimator). XGBoost-only; for the
    # joint case set ``xgboost.multi_strategy`` to "one_output_per_tree"
    # (GPU; per-label-equivalent) or "multi_output_tree" (cross-label; CPU-only).
    multioutput_strategy: str
    # --- deep tabular (tabular input, PyTorch training; requires scikit-rec[torch]) ---
    deepfm: DeepFMConfig
    # --- embedding ---
    embedding: EmbeddingConfig
    # --- sequential ---
    sequential: SequentialConfig
    # --- multi_target (ml_task="multi_target") ---
    multi_target: MultiTargetConfig


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


class ScorerConfig(TypedDict, total=False):
    """Per-scorer constructor kwargs.

    Not every scorer accepts every key. The accepted-keys whitelist lives in
    ``_SCORER_CONFIG_ALLOWED`` (and is mirrored in ``capability_matrix()``
    under ``"scorer_config_keys"`` for external introspection). Passing a key
    that the chosen ``scorer_type`` does not accept raises ``ValueError``
    upfront in :func:`create_scorer` — same defensive posture as the
    ``_NON_TABULAR_KEYS`` warning on ``estimator_config``.

    Current mapping:

    - ``multioutput``: ``on_degenerate_target``
    - ``multiclass``, ``independent``, ``universal``, ``sequential``,
      ``hierarchical``: (none yet — included in the whitelist as empty sets)
    """

    # --- multioutput ---
    on_degenerate_target: Union[DegenerateTargetPolicy, str]
    # --- mixed_type_multi_target ---
    target_specs: Dict[str, Any]  # dict[str, TargetType | TargetGroupSpec]


class RecommenderConfig(TypedDict, total=False):
    recommender_type: str  # "ranking", "bandits", "sequential", "hierarchical_sequential", "uplift", "gcsl"
    scorer_type: str  # "multioutput", "multiclass", "independent", "universal", "sequential", "hierarchical"
    estimator_config: EstimatorConfig
    scorer_config: ScorerConfig
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

# Deep tabular models: tabular input (flat feature row) but PyTorch training.
# Lazy-imported so torch is not pulled in unless actually requested.
_DEEP_TABULAR_ESTIMATOR_MAP: Dict[str, tuple] = {
    "deepfm": ("skrec.estimator.classification.deep_fm_classifier", "DeepFMClassifier"),
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

TABULAR_MODEL_TYPES: Tuple[str, ...] = ("xgboost", "lightgbm", "deepfm")
MULTI_TARGET_MODEL_TYPES: Tuple[str, ...] = (
    "joint_mlp",
    "joint_transformer",
    "independent",
    # v3: conditional joint families accept OBSERVED_* columns at inference
    # via the ConditionalMultiTargetEstimator Protocol. Independent +
    # conditional is NOT supported (v3 locked decision #1).
    "conditional_joint_mlp",
    "conditional_joint_transformer",
)

_NON_TABULAR_KEYS = {"embedding", "sequential"}
_TABULAR_SCORER_TYPES = {
    "multioutput",
    "multiclass",
    "independent",
    "universal",
    "mixed_type_multi_target",
}
_EMBEDDING_INCOMPATIBLE_SCORERS = {
    "multioutput",
    "multiclass",
    "independent",
    "mixed_type_multi_target",
}

# Whitelist of scorer_config keys accepted per scorer_type. Extend when a new
# scorer-level constructor kwarg lands. Empty sets are not redundant — they
# pin the contract that the scorer takes no kwargs today, so passing one
# raises rather than silently dropping the value.
_SCORER_CONFIG_ALLOWED: Dict[str, frozenset] = {
    "multioutput": frozenset({"on_degenerate_target"}),
    "multiclass": frozenset(),
    "independent": frozenset(),
    "universal": frozenset(),
    "mixed_type_multi_target": frozenset({"target_specs"}),
    "sequential": frozenset(),
    "hierarchical": frozenset(),
}

# scorer_type → concrete scorer class. Used by ``capability_matrix()`` to
# derive capability flags (e.g., ``supports_observed_conditioning``) from
# the scorer class attributes themselves rather than maintaining a parallel
# hand-edited table that drifts. Kept aligned with the ``create_scorer``
# dispatch chain below — a unit test pins them in lockstep.
_SCORER_TYPE_TO_CLASS: Dict[str, type] = {
    "multioutput": MultioutputScorer,
    "multiclass": MulticlassScorer,
    "independent": IndependentScorer,
    "universal": UniversalScorer,
    "mixed_type_multi_target": MixedTypeMultiTargetScorer,
    "sequential": SequentialScorer,
    "hierarchical": HierarchicalScorer,
}

# Curated sub-estimator types per declared target type for
# ``IndependentMultiTargetEstimator``. Drives factory-time composition AND
# the ``capability_matrix()["independent_target_compat"]`` table that
# scikit-rec-agent reads for pre-flight validation.
#
# Notable omission: ``xgboost`` for MULTICLASS. XGBClassifierEstimator's
# inplace_predict path returns (n, 2K) instead of (n, K) on multiclass
# targets — see the defensive shape guard in IndependentMultiTargetEstimator.
# Routing multiclass → XGB at factory time would hit that guard at predict
# time; better to keep XGB out of the multiclass-compat list so the factory
# rejects the misconfiguration up-front with a clear list of alternatives.
_INDEPENDENT_TARGET_COMPAT: Dict[TargetType, Tuple[str, ...]] = {
    TargetType.BINARY: ("xgboost", "lightgbm", "logreg", "sklearn"),
    TargetType.REGRESSION: ("xgboost", "lightgbm", "sklearn"),
    TargetType.MULTICLASS: ("lightgbm", "logreg"),
    # MULTILABEL row is NOT consulted by the factory compose loop
    # below — ``_fanned_out_targets_with_types`` flattens each multilabel
    # member into a BINARY entry, so ``_create_independent_sub_estimator``
    # only ever sees BINARY at the multilabel call site. The MULTILABEL
    # entry exists for the agent-side pre-flight validator that reads
    # ``capability_matrix()["independent_target_compat"]`` and wants to
    # display the group-level capability (which estimator types the agent
    # may suggest to a user writing a per-target spec keyed by group
    # type). Do NOT drop this row without also updating the agent
    # surface — that's how dropping it would silently break a UX
    # downstream rather than producing a clean compose-time error here.
    TargetType.MULTILABEL: ("xgboost", "lightgbm", "logreg", "sklearn"),
}


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


def _create_independent_sub_estimator(
    target_type: TargetType, estimator_type: str, params: Optional[Dict[str, Any]]
) -> BaseEstimator:
    """Construct one sub-estimator for ``IndependentMultiTargetEstimator``.

    Lookup is keyed by (target_type, estimator_type). Out-of-compat
    combinations are rejected before construction. Lazy imports keep this
    factory free of import-time dependency on every estimator class.

    Args:
        target_type: Declared target type — drives compatibility check + class choice.
        estimator_type: One of ``_INDEPENDENT_TARGET_COMPAT[target_type]``.
        params: Hyperparameters forwarded to the sub-estimator constructor.

    Returns:
        A ``BaseClassifier`` for binary/multiclass/multilabel targets, a
        ``BaseRegressor`` for regression targets.
    """
    compat = _INDEPENDENT_TARGET_COMPAT[target_type]
    if estimator_type not in compat:
        raise ValueError(
            f"estimator_type {estimator_type!r} not compatible with target type "
            f"{target_type.value!r}. Compatible types: {sorted(compat)}."
        )
    params = params or {}

    # Lazy imports: pulling these eagerly would balloon factory import time.
    if target_type in (
        TargetType.BINARY,
        TargetType.MULTICLASS,
        TargetType.MULTILABEL,
    ):
        if estimator_type == "xgboost":
            from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator

            return XGBClassifierEstimator(params=params)
        if estimator_type == "lightgbm":
            from skrec.estimator.classification.lightgbm_classifier import (
                LightGBMClassifierEstimator,
            )

            # Silence LGBM's stdout chatter unless caller explicitly overrides.
            lgbm_params = {"verbose": -1, **params}
            return LightGBMClassifierEstimator(params=lgbm_params)
        if estimator_type == "logreg":
            from skrec.estimator.classification.logreg_classifier import (
                LogisticRegressionClassifierEstimator,
            )

            return LogisticRegressionClassifierEstimator(params=params)
        if estimator_type == "sklearn":
            from sklearn.linear_model import LogisticRegression

            from skrec.estimator.classification.sklearn_universal_classifier import (
                SklearnUniversalClassifierEstimator,
            )

            return SklearnUniversalClassifierEstimator(LogisticRegression, params)
    elif target_type == TargetType.REGRESSION:
        if estimator_type == "xgboost":
            from skrec.estimator.regression.xgb_regressor import XGBRegressorEstimator

            return XGBRegressorEstimator(params=params)
        if estimator_type == "lightgbm":
            from skrec.estimator.regression.lightgbm_regressor import (
                LightGBMRegressorEstimator,
            )

            lgbm_params = {"verbose": -1, **params}
            return LightGBMRegressorEstimator(params=lgbm_params)
        if estimator_type == "sklearn":
            from sklearn.linear_model import Ridge

            from skrec.estimator.regression.sklearn_universal_regressor import (
                SklearnUniversalRegressorEstimator,
            )

            return SklearnUniversalRegressorEstimator(Ridge, params)

    # Should be unreachable — compat table covered every case above.
    raise NotImplementedError(
        f"No sub-estimator mapping for target_type={target_type.value!r}, estimator_type={estimator_type!r}."
    )


def _create_multi_target_estimator(
    multi_target_config: "MultiTargetConfig",
    target_specs: Dict[str, Any],
) -> BaseEstimator:
    """Compose one of the three v2 multi-target estimator families from config.

    Lazy-imports the concrete classes (torch-heavy for joint families) so
    importing the factory itself does not require PyTorch.
    """
    mode = multi_target_config.get("mode")
    if mode not in MULTI_TARGET_MODEL_TYPES:
        raise ValueError(f"multi_target.mode must be one of {MULTI_TARGET_MODEL_TYPES}; got {mode!r}.")

    if mode == "joint_mlp":
        from skrec.estimator.classification.joint_multi_target_mlp import (
            JointMultiTargetMLPEstimator,
        )

        params = multi_target_config.get("params", {})
        logger.info("Creating JointMultiTargetMLPEstimator")
        return JointMultiTargetMLPEstimator(target_specs=target_specs, params=params)

    if mode == "joint_transformer":
        from skrec.estimator.classification.joint_multi_target_transformer import (
            JointMultiTargetTransformerEstimator,
        )

        params = multi_target_config.get("params", {})
        logger.info("Creating JointMultiTargetTransformerEstimator")
        return JointMultiTargetTransformerEstimator(target_specs=target_specs, params=params)

    if mode == "conditional_joint_mlp":
        # v3: same MLP encoder + per-target heads as joint_mlp, plus a label
        # encoder fed by per-(row, target) Bernoulli masking. Accepts
        # OBSERVED_* at inference via predict_with_observed.
        from skrec.estimator.classification.conditional_joint_multi_target_mlp import (
            ConditionalJointMultiTargetMLPEstimator,
        )

        params = multi_target_config.get("params", {})
        logger.info("Creating ConditionalJointMultiTargetMLPEstimator")
        return ConditionalJointMultiTargetMLPEstimator(target_specs=target_specs, params=params)

    if mode == "conditional_joint_transformer":
        # v3: FT-Transformer-style feature tokenizer + label tokenizer for
        # per-target observed-label inputs.
        from skrec.estimator.classification.conditional_joint_multi_target_transformer import (
            ConditionalJointMultiTargetTransformerEstimator,
        )

        params = multi_target_config.get("params", {})
        logger.info("Creating ConditionalJointMultiTargetTransformerEstimator")
        return ConditionalJointMultiTargetTransformerEstimator(target_specs=target_specs, params=params)

    # mode == "independent"
    from skrec.estimator.classification.independent_multi_target import (
        IndependentMultiTargetEstimator,
    )

    independent_config = multi_target_config.get("independent") or {}
    defaults: Dict[str, Any] = independent_config.get("defaults", {}) or {}
    per_target: Dict[str, Any] = independent_config.get("per_target", {}) or {}
    fanned_out = _fanned_out_targets_with_types(target_specs)

    # Top-level random_state → per-sub-estimator ``random_state`` (only when
    # the sub-estimator's params don't already specify one). Without this
    # plumbing, every independent-mode run is non-reproducible by default
    # because xgb/lgbm/sklearn estimators each pick their own internal RNG
    # at construction. We never overwrite a caller-supplied value — that's
    # an explicit override and must take precedence.
    top_seed = multi_target_config.get("random_state")

    # Upfront coverage validation (v2 plan asks for this explicitly).
    # We check value presence (``is None``) rather than key membership
    # so a caller who sets ``defaults["binary"] = None`` doesn't slip
    # past the upfront check only to crash later on the post-lookup
    # assert. Either path means "no resolvable spec for this target."
    missing_default_keys = sorted(
        {
            defaults_key
            for fanned_name, _, defaults_key in fanned_out
            if per_target.get(fanned_name) is None and defaults.get(defaults_key) is None
        }
    )
    if missing_default_keys:
        raise ValueError(
            f"Independent mode missing coverage. The following declared target "
            f"type(s) have no defaults entry and no per_target override: "
            f"{missing_default_keys}. Either add a 'defaults' entry for each "
            f"missing type, or add a 'per_target' override for the specific "
            f"target column(s). Declared targets: "
            f"{[name for name, _, _ in fanned_out]}."
        )

    # Compose sub-estimators per fanned-out target. Lookup precedence:
    # name-keyed override in per_target beats group-keyed default in defaults.
    # For multilabel members the defaults key is "multilabel" (not "binary")
    # so the user-facing config table stays consistent with the v2 plan.
    estimators: Dict[str, BaseEstimator] = {}
    for fanned_name, sub_target_type, defaults_key in fanned_out:
        spec_block = per_target.get(fanned_name)
        if spec_block is None:
            spec_block = defaults.get(defaults_key)
        # Unreachable after the upfront ``is None`` coverage check above
        # — kept as a defence-in-depth assertion for future refactors.
        assert spec_block is not None, (
            f"Internal: missing spec for {fanned_name!r} after upfront "
            f"coverage validation (should have been caught earlier)."
        )
        estimator_type = spec_block.get("estimator_type")
        if not estimator_type:
            raise ValueError(f"Spec for target {fanned_name!r} missing 'estimator_type'.")
        params = dict(spec_block.get("params") or {})
        if top_seed is not None and "random_state" not in params:
            params["random_state"] = int(top_seed)
        estimators[fanned_name] = _create_independent_sub_estimator(
            target_type=sub_target_type,
            estimator_type=estimator_type,
            params=params,
        )

    logger.info(
        "Creating IndependentMultiTargetEstimator with %d sub-estimators",
        len(estimators),
    )
    return IndependentMultiTargetEstimator(target_specs=target_specs, estimators=estimators)


def _fanned_out_targets_with_types(
    target_specs: Dict[str, Any],
) -> List[Tuple[str, TargetType, str]]:
    """Flatten target_specs into ``(fanned_out_name, sub_estimator_target_type, defaults_lookup_key)``.

    Three slots:
      - ``fanned_out_name``: the column name (simple target name or
        multilabel member column).
      - ``sub_estimator_target_type``: the ``TargetType`` used for compat-
        checking the sub-estimator (multilabel members → BINARY at the
        sub-estimator level because each is its own binary classifier).
      - ``defaults_lookup_key``: which key in ``independent.defaults`` to
        fall back on when no ``per_target`` override exists. Multilabel
        members look up ``"multilabel"`` (the group-level default), NOT
        ``"binary"`` — keeps the user-facing config table honest.
    """
    out: List[Tuple[str, TargetType, str]] = []
    for key, spec in target_specs.items():
        if isinstance(spec, TargetType):
            out.append((key, spec, spec.value))
        elif isinstance(spec, dict):
            group_type = spec.get("type")
            if isinstance(group_type, str):
                group_type = TargetType(group_type)
            for col in spec.get("columns", []):
                out.append((col, TargetType.BINARY, group_type.value))
        else:
            raise ValueError(f"target_specs[{key!r}] must be a TargetType or dict; got {type(spec).__name__}.")
    return out


def _create_retriever(config: RetrieverConfig) -> BaseCandidateRetriever:
    """Create a candidate retriever from config."""
    retriever_type = config.get("type")
    if not retriever_type:
        raise ValueError("'type' is required in retriever config.")

    cls = _RETRIEVER_MAP.get(retriever_type)
    if cls is None:
        raise NotImplementedError(
            f"Retriever type '{retriever_type}' not supported. Supported types: {list(_RETRIEVER_MAP.keys())}"
        )

    params = config.get("params", {})
    logger.info(f"Creating retriever {cls.__name__} with params: {params}")
    return cls(**params)


# --- Factory Functions ---
def create_estimator(
    estimator_config: EstimatorConfig,
    scorer_type: Optional[str] = None,
    target_specs: Optional[Dict[str, Any]] = None,
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
            f"Estimator type '{estimator_type}' not supported. Supported types: 'tabular', 'embedding', 'sequential'."
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

    # Surface the "config has multi_target but ml_task isn't multi_target"
    # mismatch as a hard error rather than silently dropping the
    # multi_target block. Before this guard, a user who set up
    # ``multi_target=...`` but forgot to set ``ml_task="multi_target"``
    # would silently route through the xgb/lgbm/deepfm path and produce
    # a single-output estimator — the per-target schema would never reach
    # the model. Symmetric to the existing _NON_TABULAR_KEYS warning.
    if ml_task != "multi_target" and "multi_target" in estimator_config:
        raise ValueError(
            f"estimator_config has a 'multi_target' block but ml_task="
            f"{ml_task!r} — multi_target requires ml_task='multi_target'. "
            f"Either set ml_task='multi_target' (and provide target_specs "
            f"via scorer_config) or remove the 'multi_target' block."
        )

    # Multi-target ml_task is handled before the xgb/lgbm/deepfm fork:
    # composition happens off the `multi_target` sub-config, not the per-
    # backend tabular keys, and the resulting estimator is an instance of
    # one of the v2 MultiTargetEstimator implementations.
    if ml_task == "multi_target":
        if target_specs is None or not target_specs:
            raise ValueError(
                "ml_task='multi_target' requires target_specs (non-empty). "
                "Pass it via scorer_config['target_specs']; "
                "create_recommender_pipeline threads it through automatically."
            )
        mt_config: MultiTargetConfig = estimator_config.get("multi_target") or {}
        if not mt_config:
            raise ValueError(
                "ml_task='multi_target' requires estimator_config['multi_target'] "
                f"(must include 'mode' from {MULTI_TARGET_MODEL_TYPES})."
            )
        return _create_multi_target_estimator(mt_config, target_specs)

    xgb_config = estimator_config.get("xgboost", {})
    lgbm_config = estimator_config.get("lightgbm")
    deepfm_config = estimator_config.get("deepfm")
    hpo_config = estimator_config.get("hpo", {})
    weights_config = estimator_config.get("weights", {})

    tabular_model_keys = [k for k in ("xgboost", "lightgbm", "deepfm") if estimator_config.get(k) is not None]
    if len(tabular_model_keys) > 1:
        raise ValueError(
            f"Specify only one tabular model key in estimator_config, got: {tabular_model_keys}. Remove all but one."
        )

    # DeepFM: tabular input shape but PyTorch training. Lazy-imported so torch
    # is not pulled at module load — only when deepfm is actually requested.
    if deepfm_config is not None:
        try:
            cls = _resolve_lazy(_DEEP_TABULAR_ESTIMATOR_MAP, "deepfm")
        except ImportError as e:
            raise ImportError(
                "DeepFMClassifier requires PyTorch. Install it with: pip install scikit-rec[torch]"
            ) from e
        if ml_task != "classification":
            raise ValueError(
                "DeepFMClassifier only supports ml_task='classification'. For regression use 'xgboost' or 'lightgbm'."
            )
        logger.info(f"Creating DeepFMClassifier with params: {deepfm_config}")
        return cls(params=deepfm_config)

    use_lightgbm = lgbm_config is not None
    model_params = lgbm_config if use_lightgbm else xgb_config

    is_tuned_mode = bool(
        hpo_config.get("hpo_method") or hpo_config.get("param_space") or hpo_config.get("optimizer_params")
    )

    logger.info(
        f"Creating estimator. ML Task: {ml_task}, Model: {'lightgbm' if use_lightgbm else 'xgboost'}, "
        f"Scorer Type Hint: {scorer_type}, Tuned Mode: {is_tuned_mode}"
    )

    if ml_task not in {"classification", "regression"}:
        raise NotImplementedError(
            f"ML task {ml_task!r} not implemented for tabular path. Valid: "
            f"'classification', 'regression', 'multi_target'."
        )

    estimator: BaseEstimator

    if is_tuned_mode:
        if not all(k in hpo_config for k in ["hpo_method", "param_space", "optimizer_params"]):
            raise ValueError(
                "Missing required HPO configuration keys (hpo_method, param_space, optimizer_params) for tuned mode."
            )
        hpo_method = hpo_config["hpo_method"]
        param_space = hpo_config["param_space"]
        optimizer_params = hpo_config["optimizer_params"]

        if estimator_config.get("multioutput_strategy") == "joint":
            raise ValueError(
                "multioutput_strategy='joint' is not supported in tuned/HPO mode — the joint "
                "XGBoost estimators have no GridSearchCV/RandomizedSearchCV wrapper. Use non-tuned "
                "mode for a joint booster, or multioutput_strategy='per_label' (default) with HPO."
            )

        if ml_task == "classification":
            if scorer_type == "multioutput":
                base_cls = LGBMClassifier if use_lightgbm else XGBClassifier
                logger.info(f"Creating TunedMultiOutputClassifierEstimator with {base_cls.__name__}")
                estimator = TunedMultiOutputClassifierEstimator(
                    base_estimator=base_cls,
                    hpo_method=hpo_method,
                    param_space=param_space,
                    optimizer_params=optimizer_params,
                )
            elif use_lightgbm:
                logger.info("Creating TunedLightGBMClassifierEstimator")
                estimator = TunedLightGBMClassifierEstimator(
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
        else:  # regression
            if scorer_type == "multioutput":
                base_cls = LGBMRegressor if use_lightgbm else XGBRegressor
                logger.info(f"Creating TunedMultiOutputRegressorEstimator with {base_cls.__name__}")
                estimator = TunedMultiOutputRegressorEstimator(
                    base_estimator=base_cls,
                    hpo_method=hpo_method,
                    param_space=param_space,
                    optimizer_params=optimizer_params,
                )
            elif use_lightgbm:
                logger.info("Creating TunedLightGBMRegressorEstimator")
                estimator = TunedLightGBMRegressorEstimator(
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
        # Generic fit-time passthrough (sklearn-API estimators). Resolved per-row
        # at fit time inside the estimator; see skrec.estimator._fit_params_mixin.
        sample_weight = weights_config.get("sample_weight")
        fit_params = weights_config.get("fit_params")
        # Multioutput estimator structure: "per_label" (N independent boosters)
        # or "joint" (one joint XGBoost booster). XGBoost-only.
        multioutput_strategy = estimator_config.get("multioutput_strategy", "per_label")

        if ml_task == "classification":
            action_weight = weights_config.get("action_weight", 1)
            item_sample_weights = weights_config.get("item_sample_weights")

            if scorer_type == "multioutput" and multioutput_strategy == "joint":
                if use_lightgbm:
                    raise ValueError(
                        "multioutput_strategy='joint' is XGBoost-only (a single joint booster); "
                        "LightGBM has no native joint multi-output. Use multioutput_strategy='per_label', "
                        "or drop the lightgbm config to use XGBoost."
                    )
                logger.info("Creating JointXGBMultiOutputClassifierEstimator (single joint booster)")
                estimator = JointXGBMultiOutputClassifierEstimator(
                    xgb_config, fit_params=fit_params, sample_weight=sample_weight
                )
            elif scorer_type == "multioutput":
                base_cls = LGBMClassifier if use_lightgbm else XGBClassifier
                logger.info(f"Creating MultiOutputClassifierEstimator with {base_cls.__name__}")
                estimator = MultiOutputClassifierEstimator(
                    base_cls, model_params, fit_params=fit_params, sample_weight=sample_weight
                )
            elif use_lightgbm:
                logger.info("Creating LightGBMClassifierEstimator")
                estimator = LightGBMClassifierEstimator(
                    model_params, fit_params=fit_params, sample_weight=sample_weight
                )
            elif action_weight != 1 or item_sample_weights is not None:
                logger.info("Creating WeightedXGBClassifierEstimator")
                estimator = WeightedXGBClassifierEstimator(
                    params=xgb_config,
                    action_weight=action_weight,
                    item_sample_weights=item_sample_weights,
                    fit_params=fit_params,
                    sample_weight=sample_weight,
                )
            else:
                logger.info("Creating XGBClassifierEstimator")
                estimator = XGBClassifierEstimator(xgb_config, fit_params=fit_params, sample_weight=sample_weight)
        else:  # regression
            if scorer_type == "multioutput" and multioutput_strategy == "joint":
                if use_lightgbm:
                    raise ValueError(
                        "multioutput_strategy='joint' is XGBoost-only (a single joint booster); "
                        "LightGBM has no native joint multi-output. Use multioutput_strategy='per_label', "
                        "or drop the lightgbm config to use XGBoost."
                    )
                logger.info("Creating JointXGBMultiOutputRegressorEstimator (single joint booster)")
                estimator = JointXGBMultiOutputRegressorEstimator(
                    xgb_config, fit_params=fit_params, sample_weight=sample_weight
                )
            elif scorer_type == "multioutput":
                base_cls = LGBMRegressor if use_lightgbm else XGBRegressor
                logger.info(f"Creating MultiOutputRegressorEstimator with {base_cls.__name__}")
                estimator = MultiOutputRegressorEstimator(
                    base_cls, model_params, fit_params=fit_params, sample_weight=sample_weight
                )
            elif use_lightgbm:
                logger.info("Creating LightGBMRegressorEstimator")
                estimator = LightGBMRegressorEstimator(model_params, fit_params=fit_params, sample_weight=sample_weight)
            else:
                logger.info("Creating XGBRegressorEstimator")
                estimator = XGBRegressorEstimator(xgb_config, fit_params=fit_params, sample_weight=sample_weight)

    return estimator


def create_scorer(
    estimator: Union[BaseEstimator, SequentialEstimator],
    config: RecommenderConfig,
    scorer_config: Optional[Dict[str, Any]] = None,
) -> BaseScorer:
    """
    Factory function to create a scorer instance based on the overall recommender configuration.

    Args:
        estimator: The estimator instance to be used by the scorer.
        config: The main recommender configuration dictionary.
                Expected key: 'scorer_type'.
        scorer_config: Optional pre-extracted scorer_config dict. When
            supplied (typical from ``create_recommender_pipeline``),
            takes precedence over ``config.get("scorer_config")`` —
            avoids mutating the caller's ``config`` dict to thread the
            single-read snapshot.

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

    # Guard: sequential/hierarchical scorers require a SequentialEstimator.
    # Hoisted above the scorer_config validation block so estimator/scorer
    # type-compat errors surface before kwarg-shape errors — a caller who
    # passed both a tabular estimator AND a bad scorer_config key should see
    # the more fundamental type mismatch first.
    if scorer_type in ("sequential", "hierarchical") and not isinstance(estimator, SequentialEstimator):
        raise TypeError(
            f"{scorer_type.capitalize()} scorer requires a SequentialEstimator, got {type(estimator).__name__}."
        )

    # Validate scorer_config against the per-scorer whitelist. Gated on
    # ``scorer_type in _SCORER_CONFIG_ALLOWED`` so unknown scorer_types fall
    # through to the NotImplementedError below — otherwise a non-empty config
    # plus an unknown scorer_type would surface as "scorer_type='zzz' does
    # not accept scorer_config keys: [...]", which misleads (implying zzz is
    # valid with restricted keys) and hides the real unsupported-scorer error.
    #
    # Prefer the explicit scorer_config argument when supplied (the
    # pipeline factory threads its single-read snapshot through this way
    # so we don't have to mutate the caller's config dict). Fall back to
    # reading from config for standalone create_scorer callers.
    if scorer_config is None:
        scorer_config = dict(config.get("scorer_config") or {})
    else:
        scorer_config = dict(scorer_config)
    if scorer_type in _SCORER_CONFIG_ALLOWED:
        allowed = _SCORER_CONFIG_ALLOWED[scorer_type]
        unknown = set(scorer_config) - allowed
        if unknown:
            raise ValueError(
                f"scorer_type={scorer_type!r} does not accept scorer_config keys: "
                f"{sorted(unknown)}. Accepted keys: {sorted(allowed) or '(none)'}."
            )

    # Every branch spreads ``**scorer_config`` so adding a future scorer-level
    # kwarg is a single-site edit (the whitelist) — the construction site no
    # longer has to be touched in tandem. Today the whitelist enforces empty
    # kwargs for every scorer except multioutput, so the spread is a no-op
    # for the others.
    scorer: BaseScorer

    if scorer_type == "multioutput":
        scorer = MultioutputScorer(estimator=estimator, **scorer_config)
    elif scorer_type == "multiclass":
        scorer = MulticlassScorer(estimator=estimator, **scorer_config)
    elif scorer_type == "independent":
        scorer = IndependentScorer(estimator=estimator, **scorer_config)
    elif scorer_type == "universal":
        scorer = UniversalScorer(estimator=estimator, **scorer_config)
    elif scorer_type == "mixed_type_multi_target":
        # target_specs is required and validated upstream in
        # create_recommender_pipeline; defensive check here in case create_scorer
        # is called standalone.
        if not scorer_config.get("target_specs"):
            raise ValueError(
                "scorer_type='mixed_type_multi_target' requires scorer_config['target_specs'] (non-empty)."
            )
        scorer = MixedTypeMultiTargetScorer(estimator=estimator, **scorer_config)
    elif scorer_type == "sequential":
        scorer = SequentialScorer(estimator=estimator, **scorer_config)
    elif scorer_type == "hierarchical":
        scorer = HierarchicalScorer(estimator=estimator, **scorer_config)
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
    recommender_type = config.get("recommender_type")
    if not recommender_type:
        raise ValueError("'recommender_type' must be specified in the configuration.")
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
            raise TypeError(f"SequentialRecommender requires a SequentialScorer, got {type(scorer).__name__}.")
        recommender = SequentialRecommender(
            scorer=scorer,
            max_len=recommender_params.get("max_len", 50),
        )
    elif recommender_type == "hierarchical_sequential":
        if not isinstance(scorer, HierarchicalScorer):
            raise TypeError(
                f"HierarchicalSequentialRecommender requires a HierarchicalScorer, got {type(scorer).__name__}."
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
    recommender_type = config.get("recommender_type")
    if not recommender_type:
        raise ValueError("'recommender_type' must be specified in the configuration.")
    estimator_type = estimator_config.get("estimator_type") or "tabular"

    if not estimator_config:
        logger.warning("estimator_config not found in main config. Attempting to proceed with empty estimator config.")

    # Upfront validation against the authoritative enum tuples. Makes the
    # tuples the explicit contract for top-level config consumers, so that
    # `capability_matrix()` stays in lockstep with what the factory accepts.
    if recommender_type not in RECOMMENDER_TYPES:
        raise ValueError(f"Unknown recommender_type '{recommender_type}'. Valid: {RECOMMENDER_TYPES}")
    if scorer_type is not None and scorer_type not in SCORER_TYPES:
        raise ValueError(f"Unknown scorer_type '{scorer_type}'. Valid: {SCORER_TYPES}")
    if estimator_type not in ESTIMATOR_TYPES:
        raise ValueError(f"Unknown estimator_type '{estimator_type}'. Valid: {ESTIMATOR_TYPES}")

    # Cross-cutting validation: catch mismatches early
    if recommender_type in ("sequential", "hierarchical_sequential"):
        if estimator_type != "sequential":
            raise ValueError(
                f"recommender_type '{recommender_type}' requires estimator_type 'sequential', got '{estimator_type}'."
            )
    if recommender_type == "sequential" and scorer_type != "sequential":
        raise ValueError(f"recommender_type 'sequential' requires scorer_type 'sequential', got '{scorer_type}'.")
    if recommender_type == "hierarchical_sequential" and scorer_type != "hierarchical":
        raise ValueError(
            f"recommender_type 'hierarchical_sequential' requires scorer_type 'hierarchical', got '{scorer_type}'."
        )
    if scorer_type in ("sequential", "hierarchical") and estimator_type != "sequential":
        raise ValueError(f"scorer_type '{scorer_type}' requires estimator_type 'sequential', got '{estimator_type}'.")
    if estimator_type == "embedding" and scorer_type in ("multioutput", "multiclass", "independent"):
        raise ValueError(
            f"scorer_type '{scorer_type}' does not support embedding estimators. "
            f"Use scorer_type='universal' with embedding estimators."
        )
    if recommender_type == "uplift" and scorer_type not in ("independent", "universal"):
        raise ValueError(
            f"recommender_type 'uplift' requires scorer_type 'independent' or 'universal', got '{scorer_type}'."
        )

    # Mixed-type multi-target cross-cutting checks:
    #   - scorer_type='mixed_type_multi_target' requires non-empty target_specs
    #     in scorer_config (target_specs is the scorer's required kwarg, but
    #     it also drives estimator construction in ml_task='multi_target').
    #   - ml_task='multi_target' is only valid with the matching scorer_type.
    #
    # Single-read invariant: snapshot ``scorer_config`` once here and
    # pass the snapshot through to ``create_scorer`` via its
    # ``scorer_config`` keyword. We deliberately do NOT write the
    # snapshot back into ``config`` — that mutated the caller's dict
    # and broke "same config → two pipeline builds" idempotency. The
    # ``scorer_config`` parameter on ``create_scorer`` is the explicit
    # plumbing for the single read.
    scorer_config_block = dict(config.get("scorer_config") or {})
    target_specs = scorer_config_block.get("target_specs")
    estimator_ml_task = estimator_config.get("ml_task")
    if scorer_type == "mixed_type_multi_target":
        if not target_specs:
            raise ValueError(
                "scorer_type='mixed_type_multi_target' requires scorer_config['target_specs'] (non-empty)."
            )
        if estimator_ml_task != "multi_target":
            raise ValueError(
                f"scorer_type='mixed_type_multi_target' requires "
                f"estimator_config['ml_task']='multi_target' (got {estimator_ml_task!r})."
            )
    if estimator_ml_task == "multi_target" and scorer_type != "mixed_type_multi_target":
        raise ValueError(
            f"estimator_config['ml_task']='multi_target' requires "
            f"scorer_type='mixed_type_multi_target' (got {scorer_type!r})."
        )

    # Create components using their respective factory functions
    estimator = create_estimator(estimator_config, scorer_type=scorer_type, target_specs=target_specs)
    scorer = create_scorer(estimator, config, scorer_config=scorer_config_block)
    recommender = create_recommender(scorer, config)

    logger.info("Recommender pipeline created successfully.")
    return recommender


def capability_matrix() -> Dict[str, Union[Tuple[str, ...], Dict[str, Tuple[str, ...]]]]:
    """Authoritative enum tuples for every factory-recognized dimension.

    Callers (e.g., a system-prompt builder or a validator) can use this to
    stay in lockstep with scikit-rec's capabilities without hardcoding enum
    values or reaching into private registry maps.

    The ``"scorer_config_keys"`` entry maps each scorer_type to the tuple of
    ``scorer_config`` keys it accepts — empty tuple when the scorer takes no
    scorer-level kwargs today. External consumers (e.g. the agent layer's
    train_model schema) can use this to surface per-scorer knobs without
    grepping source.

    The ``"evaluator_types"`` and ``"metric_types"`` entries enumerate the
    valid values for ``evaluate()`` — agents can use these directly without
    having to import and introspect the enum classes separately.
    """
    return {
        "recommender_types": RECOMMENDER_TYPES,
        "scorer_types": SCORER_TYPES,
        "estimator_types": ESTIMATOR_TYPES,
        "tabular_model_types": TABULAR_MODEL_TYPES,
        "embedding_model_types": tuple(_EMBEDDING_ESTIMATOR_MAP.keys()),
        "sequential_model_types": tuple(_SEQUENTIAL_ESTIMATOR_MAP.keys()),
        "multi_target_model_types": MULTI_TARGET_MODEL_TYPES,
        "inference_method_types": tuple(_INFERENCE_METHOD_MAP.keys()),
        "retriever_types": tuple(_RETRIEVER_MAP.keys()),
        "scorer_config_keys": {k: tuple(sorted(v)) for k, v in _SCORER_CONFIG_ALLOWED.items()},
        # Keys accepted under estimator_config["weights"] (sklearn-API estimators):
        # item/action weighting plus the generic fit-time passthrough
        # (sample_weight strategy + static fit_params). Sourced from WeightsConfig
        # so it can't drift from the TypedDict.
        "weights_config_keys": tuple(WeightsConfig.__annotations__.keys()),
        # estimator_config["multioutput_strategy"] values (scorer_type="multioutput"):
        # "per_label" = N independent boosters; "joint" = one joint XGBoost booster.
        "multioutput_strategy_types": ("per_label", "joint"),
        "evaluator_types": tuple(e.value for e in RecommenderEvaluatorType),
        "metric_types": tuple(m.value for m in RecommenderMetricType),
        # Multi-target capabilities — read by scikit-rec-agent for pre-flight
        # validation. Sources of truth: TARGET_TYPE_TO_METRICS in
        # skrec.scorer.mixed_type_multi_target; _INDEPENDENT_TARGET_COMPAT here.
        "target_types": tuple(t.value for t in TargetType),
        "target_type_metric_compat": {t.value: TARGET_TYPE_TO_METRICS[t] for t in TargetType},
        "independent_target_compat": {t.value: tuple(sorted(_INDEPENDENT_TARGET_COMPAT[t])) for t in TargetType},
        # v3: MixedTypeMultiTargetScorer supports OBSERVED_* conditioning when
        # paired with a ConditionalMultiTargetEstimator. Vanilla estimators
        # still reject OBSERVED_* at the scorer's inference validator.
        #
        # Derived from the per-scorer ``supports_observed_conditioning``
        # class attribute (BaseScorer default False; MixedTypeMultiTarget
        # overrides to True). Adding a new scorer that opts in is then a
        # one-line attribute set on the subclass — no manual sync of this
        # table required, no drift risk.
        "scorer_supports_observed_conditioning": tuple(
            sorted(
                stype
                for stype, scls in _SCORER_TYPE_TO_CLASS.items()
                if getattr(scls, "supports_observed_conditioning", False)
            )
        ),
    }


def contract_from_dataframe(
    df: pd.DataFrame,
    target_specs: Optional[Dict[str, Any]] = None,
) -> str:
    """Detect the scikit-rec dataset contract from a DataFrame's shape.

    Returns one of:
        - ``"long_interactions"`` — ``(USER_ID, ITEM_ID, OUTCOME)`` triples
        - ``"long_with_timestamp"`` — long + ``TIMESTAMP``
        - ``"wide_multioutput"`` — one row per user + ≥2 ``ITEM_*`` columns
          (all-binary contract; pairs with ``MultioutputScorer``)
        - ``"wide_mixed_type_multi_target"`` — wide format with heterogeneous
          ``TargetType`` declarations (pairs with ``MixedTypeMultiTargetScorer``).
          **Requires** ``target_specs`` to disambiguate from wide_multioutput.
        - ``"multiclass"`` — one row per user, ``ITEM_ID`` IS the class
        - ``"prebuilt_sequences"`` — list-typed columns present
        - ``"sessions"`` — ``SESSION_SEQUENCES`` column present

    The detection rules:
        - List-dtype columns → ``prebuilt_sequences`` / ``sessions``
        - ``USER_ID`` + ``ITEM_ID`` + ``OUTCOME`` → long-format
        - ``USER_ID`` + ≥2 ``ITEM_*`` columns + ``target_specs`` containing
          any non-BINARY type or any ``TargetGroupSpec`` →
          ``wide_mixed_type_multi_target``; otherwise ``wide_multioutput``
        - ``USER_ID`` + ``ITEM_ID`` only (no OUTCOME) → ``multiclass``

    Co-located in ``skrec.orchestrator`` so scikit-rec-agent and other
    external callers share one source of contract detection — preventing
    silent misroute of mixed-type data to ``MultioutputScorer``.

    Args:
        df: Source DataFrame to classify.
        target_specs: Optional declared per-target schema. Required when
            distinguishing ``wide_mixed_type_multi_target`` from
            ``wide_multioutput`` — without it, the heuristic defaults to
            ``wide_multioutput`` (legacy behavior).

    Returns:
        Contract identifier string.

    Raises:
        ValueError: If no contract matches.
    """
    cols = set(df.columns)

    # Session / sequence detection: list-dtype columns are the signal.
    list_cols = [c for c in df.columns if df[c].dtype == object and df[c].apply(lambda v: isinstance(v, list)).any()]
    if list_cols:
        if "SESSION_SEQUENCES" in cols:
            return "sessions"
        return "prebuilt_sequences"

    has_user = "USER_ID" in cols
    has_item = "ITEM_ID" in cols
    has_outcome = "OUTCOME" in cols
    has_timestamp = "TIMESTAMP" in cols

    # Long-format triples.
    if has_user and has_item and has_outcome:
        return "long_with_timestamp" if has_timestamp else "long_interactions"

    # Wide-format detection: USER_ID + ≥2 ITEM_* columns (excluding ITEM_ID).
    item_prefix_cols = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
    if has_user and len(item_prefix_cols) >= 2:
        # Intent signal: if the caller passed target_specs at all, they have
        # explicitly opted into the per-target-typed scorer family. Honor
        # that intent even when every declared type is BINARY — the user
        # presumably wants the per-target output contract (one column per
        # declared target, deterministic order) that
        # MixedTypeMultiTargetScorer provides, not MultioutputScorer's
        # implicit "all binary, all columns" contract. Without this branch,
        # all-BINARY target_specs would silently fall through to
        # wide_multioutput and the caller's intent (e.g., subset of columns,
        # multilabel group declaration with all-binary members) would be
        # ignored.
        if target_specs:
            return "wide_mixed_type_multi_target"
        return "wide_multioutput"

    # Multiclass: user + item_id only.
    if has_user and has_item and not has_outcome:
        return "multiclass"

    raise ValueError(
        f"Cannot detect scikit-rec contract from columns: {sorted(cols)}. "
        f"Expected one of the known shapes (long_interactions, wide_multioutput, "
        f"wide_mixed_type_multi_target, multiclass, prebuilt_sequences, sessions)."
    )
