# Tests for M5: factory + orchestrator surfaces for the multi-target scorer.
#
# Covers:
#   - create_estimator dispatch for ml_task='multi_target' across all 3 modes
#   - create_scorer dispatch for scorer_type='mixed_type_multi_target'
#   - create_recommender_pipeline end-to-end (config → fitted-able recommender)
#   - cross-cutting validation: target_specs required; ml_task/scorer_type pairing
#   - capability_matrix new keys (target_types, target_type_metric_compat,
#     independent_target_compat, scorer_supports_observed_conditioning,
#     scorer_config_keys[mixed_type_multi_target])
#   - contract_from_dataframe heuristic

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # joint families need torch

from skrec.constants import USER_ID_NAME  # noqa: E402, F401  # used by moved tests
from skrec.estimator.classification import (  # noqa: E402
    ConditionalJointMultiTargetMLPEstimator,
    ConditionalJointMultiTargetTransformerEstimator,
    IndependentMultiTargetEstimator,
    JointMultiTargetMLPEstimator,
    JointMultiTargetTransformerEstimator,
)
from skrec.estimator.classification.lightgbm_classifier import (  # noqa: E402, F401
    LightGBMClassifierEstimator,
)
from skrec.estimator.regression.lightgbm_regressor import (  # noqa: E402, F401
    LightGBMRegressorEstimator,
)
from skrec.orchestrator import (  # noqa: E402
    MULTI_TARGET_MODEL_TYPES,
    TargetGroupSpec,
    TargetType,
    capability_matrix,
    contract_from_dataframe,
    create_estimator,
    create_recommender_pipeline,
    create_scorer,
)
from skrec.orchestrator.factory import _INDEPENDENT_TARGET_COMPAT  # noqa: E402
from skrec.recommender.ranking.ranking_recommender import RankingRecommender  # noqa: E402
from skrec.scorer.mixed_type_multi_target import (  # noqa: E402
    TARGET_TYPE_TO_METRICS,
    MixedTypeMultiTargetScorer,
)

# ---------------------------------------------------------------------- #
# create_estimator: each multi_target mode produces the right class.
# ---------------------------------------------------------------------- #


def test_create_estimator_joint_mlp():
    target_specs = {"ITEM_a": TargetType.BINARY}
    est = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "joint_mlp",
                "params": {"epochs": 1, "hidden_dim": 16, "num_layers": 1},
            },
        },
        target_specs=target_specs,
    )
    assert isinstance(est, JointMultiTargetMLPEstimator)
    assert est.target_specs == target_specs


def test_create_estimator_joint_transformer():
    target_specs = {"ITEM_a": TargetType.BINARY}
    est = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "joint_transformer",
                "params": {
                    "epochs": 1,
                    "d_model": 16,
                    "n_heads": 2,
                    "n_layers": 1,
                    "ffn_dim": 32,
                },
            },
        },
        target_specs=target_specs,
    )
    assert isinstance(est, JointMultiTargetTransformerEstimator)


def test_create_estimator_independent_from_defaults():
    target_specs = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    est = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "independent",
                "independent": {
                    "defaults": {
                        "binary": {"estimator_type": "xgboost", "params": {"n_estimators": 5}},
                        "regression": {
                            "estimator_type": "lightgbm",
                            "params": {"n_estimators": 5},
                        },
                    },
                },
            },
        },
        target_specs=target_specs,
    )
    assert isinstance(est, IndependentMultiTargetEstimator)
    # Sub-estimators composed in the right slots.
    assert "ITEM_clicked" in est.estimators
    assert "ITEM_revenue" in est.estimators


def test_create_estimator_independent_per_target_override():
    target_specs = {
        "ITEM_a": TargetType.BINARY,
        "ITEM_b": TargetType.BINARY,
    }
    est = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "independent",
                "independent": {
                    "defaults": {
                        "binary": {"estimator_type": "xgboost", "params": {"n_estimators": 5}},
                    },
                    "per_target": {
                        "ITEM_b": {
                            "estimator_type": "logreg",
                            "params": {"max_iter": 50},
                        },
                    },
                },
            },
        },
        target_specs=target_specs,
    )
    from skrec.estimator.classification.logreg_classifier import (
        LogisticRegressionClassifierEstimator,
    )
    from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator

    assert isinstance(est.estimators["ITEM_a"], XGBClassifierEstimator)
    assert isinstance(est.estimators["ITEM_b"], LogisticRegressionClassifierEstimator)


def test_create_estimator_independent_multilabel_member_fanout():
    target_specs = {
        "g": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_a", "ITEM_b", "ITEM_c"]),
    }
    est = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "independent",
                "independent": {
                    "defaults": {
                        "multilabel": {
                            "estimator_type": "xgboost",
                            "params": {"n_estimators": 5},
                        },
                    },
                },
            },
        },
        target_specs=target_specs,
    )
    # Members fanned out; group key NOT in estimators.
    assert set(est.estimators.keys()) == {"ITEM_a", "ITEM_b", "ITEM_c"}


def test_create_estimator_independent_rejects_incompatible_type():
    target_specs = {"ITEM_revenue": TargetType.REGRESSION}
    with pytest.raises(ValueError, match="not compatible"):
        create_estimator(
            {
                "estimator_type": "tabular",
                "ml_task": "multi_target",
                "multi_target": {
                    "mode": "independent",
                    "independent": {
                        "defaults": {
                            "regression": {"estimator_type": "logreg", "params": {}},
                        },
                    },
                },
            },
            target_specs=target_specs,
        )


def test_create_estimator_independent_rejects_multiclass_xgb():
    """Curated _INDEPENDENT_TARGET_COMPAT excludes xgboost from multiclass."""
    target_specs = {"ITEM_action": TargetType.MULTICLASS}
    with pytest.raises(ValueError, match="not compatible"):
        create_estimator(
            {
                "estimator_type": "tabular",
                "ml_task": "multi_target",
                "multi_target": {
                    "mode": "independent",
                    "independent": {
                        "defaults": {
                            "multiclass": {"estimator_type": "xgboost", "params": {}},
                        },
                    },
                },
            },
            target_specs=target_specs,
        )


def test_create_estimator_conditional_joint_mlp_empty_params_yields_documented_defaults():
    """B2 follow-up: the factory passes raw params straight through, with
    defaults living on the conditional estimator's ``DEFAULT_PARAMS`` /
    ``__init__`` validation. Pin that ``mode=conditional_joint_mlp`` with
    NO params reaches the documented ``mask_prob=0.5`` /
    ``label_embedding_dim=8`` defaults — catches a future change that
    silently shifts the defaults or drops the factory's pass-through.
    """
    target_specs = {"ITEM_a": TargetType.BINARY}
    est = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {"mode": "conditional_joint_mlp"},  # no params block
        },
        target_specs=target_specs,
    )
    from skrec.estimator.classification import (
        ConditionalJointMultiTargetMLPEstimator,
    )

    assert isinstance(est, ConditionalJointMultiTargetMLPEstimator)
    # Documented defaults reach the estimator unchanged.
    assert est.params["mask_prob"] == 0.5
    assert est.params["label_embedding_dim"] == 8
    # Inherited base defaults too.
    assert est.params["epochs"] == 10
    assert est.params["batch_size"] == 1024


def test_create_estimator_conditional_joint_transformer_empty_params_yields_documented_defaults():
    """Same B2 contract for the conditional Transformer mode."""
    target_specs = {"ITEM_a": TargetType.BINARY}
    est = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {"mode": "conditional_joint_transformer"},
        },
        target_specs=target_specs,
    )
    from skrec.estimator.classification import (
        ConditionalJointMultiTargetTransformerEstimator,
    )

    assert isinstance(est, ConditionalJointMultiTargetTransformerEstimator)
    assert est.params["mask_prob"] == 0.5
    assert est.params["label_embedding_dim"] == 8
    assert est.params["d_model"] == 64
    assert est.params["n_heads"] == 4


def test_create_estimator_multi_target_invalid_mode():
    target_specs = {"ITEM_a": TargetType.BINARY}
    with pytest.raises(ValueError, match="mode must be one of"):
        create_estimator(
            {
                "estimator_type": "tabular",
                "ml_task": "multi_target",
                "multi_target": {"mode": "joint_resnet"},
            },
            target_specs=target_specs,
        )


def test_create_estimator_multi_target_missing_target_specs():
    with pytest.raises(ValueError, match="target_specs"):
        create_estimator(
            {
                "estimator_type": "tabular",
                "ml_task": "multi_target",
                "multi_target": {"mode": "joint_mlp"},
            },
            target_specs=None,
        )


def test_create_estimator_multi_target_missing_multi_target_block():
    target_specs = {"ITEM_a": TargetType.BINARY}
    with pytest.raises(ValueError, match="multi_target"):
        create_estimator(
            {"estimator_type": "tabular", "ml_task": "multi_target"},
            target_specs=target_specs,
        )


# ---------------------------------------------------------------------- #
# create_scorer dispatch for mixed_type_multi_target
# ---------------------------------------------------------------------- #


def test_create_scorer_mixed_type_multi_target():
    target_specs = {"ITEM_a": TargetType.BINARY}
    estimator = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1})
    scorer = create_scorer(
        estimator=estimator,
        config={
            "scorer_type": "mixed_type_multi_target",
            "scorer_config": {"target_specs": target_specs},
        },
    )
    assert isinstance(scorer, MixedTypeMultiTargetScorer)
    assert scorer.target_specs == target_specs


def test_create_scorer_mixed_type_missing_target_specs():
    target_specs = {"ITEM_a": TargetType.BINARY}
    estimator = JointMultiTargetMLPEstimator(target_specs=target_specs, params={"epochs": 1})
    with pytest.raises(ValueError, match="target_specs"):
        create_scorer(
            estimator=estimator,
            config={"scorer_type": "mixed_type_multi_target", "scorer_config": {}},
        )


# ---------------------------------------------------------------------- #
# create_recommender_pipeline end-to-end
# ---------------------------------------------------------------------- #


def test_create_recommender_pipeline_multi_target_joint_mlp():
    target_specs = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    config = {
        "recommender_type": "ranking",
        "scorer_type": "mixed_type_multi_target",
        "scorer_config": {"target_specs": target_specs},
        "estimator_config": {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "joint_mlp",
                "params": {"epochs": 1, "hidden_dim": 8, "num_layers": 1},
            },
        },
    }
    recommender = create_recommender_pipeline(config)
    assert isinstance(recommender.scorer, MixedTypeMultiTargetScorer)
    assert isinstance(recommender.scorer.estimator, JointMultiTargetMLPEstimator)


def test_create_recommender_pipeline_multi_target_independent():
    target_specs = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    config = {
        "recommender_type": "ranking",
        "scorer_type": "mixed_type_multi_target",
        "scorer_config": {"target_specs": target_specs},
        "estimator_config": {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "independent",
                "independent": {
                    "defaults": {
                        "binary": {"estimator_type": "xgboost", "params": {"n_estimators": 5}},
                        "regression": {
                            "estimator_type": "lightgbm",
                            "params": {"n_estimators": 5},
                        },
                    },
                },
            },
        },
    }
    recommender = create_recommender_pipeline(config)
    assert isinstance(recommender.scorer.estimator, IndependentMultiTargetEstimator)


def test_create_recommender_pipeline_requires_target_specs():
    config = {
        "recommender_type": "ranking",
        "scorer_type": "mixed_type_multi_target",
        "scorer_config": {},
        "estimator_config": {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {"mode": "joint_mlp"},
        },
    }
    with pytest.raises(ValueError, match="target_specs"):
        create_recommender_pipeline(config)


def test_create_recommender_pipeline_ml_task_scorer_pairing():
    target_specs = {"ITEM_a": TargetType.BINARY}
    # ml_task='multi_target' with wrong scorer_type
    with pytest.raises(ValueError, match="scorer_type='mixed_type_multi_target'"):
        create_recommender_pipeline(
            {
                "recommender_type": "ranking",
                "scorer_type": "universal",
                "scorer_config": {"target_specs": target_specs},
                "estimator_config": {
                    "estimator_type": "tabular",
                    "ml_task": "multi_target",
                    "multi_target": {"mode": "joint_mlp"},
                },
            }
        )

    # mixed_type_multi_target scorer with wrong ml_task
    with pytest.raises(ValueError, match="ml_task.*multi_target"):
        create_recommender_pipeline(
            {
                "recommender_type": "ranking",
                "scorer_type": "mixed_type_multi_target",
                "scorer_config": {"target_specs": target_specs},
                "estimator_config": {
                    "estimator_type": "tabular",
                    "ml_task": "classification",
                },
            }
        )


# ---------------------------------------------------------------------- #
# capability_matrix new keys
# ---------------------------------------------------------------------- #


def test_capability_matrix_publishes_mixed_type_scorer():
    cm = capability_matrix()
    assert "mixed_type_multi_target" in cm["scorer_types"]
    assert cm["scorer_config_keys"]["mixed_type_multi_target"] == ("target_specs",)


def test_capability_matrix_publishes_multi_target_keys():
    cm = capability_matrix()
    # v3 added the two conditional modes alongside the three v2 modes.
    assert cm["multi_target_model_types"] == (
        "joint_mlp",
        "joint_transformer",
        "independent",
        "conditional_joint_mlp",
        "conditional_joint_transformer",
    )
    assert cm["target_types"] == ("binary", "regression", "multiclass", "multilabel")
    # target_type_metric_compat mirrors the in-code constant.
    for tt, expected in TARGET_TYPE_TO_METRICS.items():
        assert cm["target_type_metric_compat"][tt.value] == expected
    # independent_target_compat mirrors _INDEPENDENT_TARGET_COMPAT.
    for tt, expected in _INDEPENDENT_TARGET_COMPAT.items():
        assert cm["independent_target_compat"][tt.value] == tuple(sorted(expected))
    # forward-compat v3 hook is empty in v2.
    # v3: MixedTypeMultiTargetScorer supports OBSERVED_* conditioning with
    # a ConditionalMultiTargetEstimator.
    assert cm["scorer_supports_observed_conditioning"] == ("mixed_type_multi_target",)


def test_capability_matrix_json_serializable():
    """Gate 8 sanity: every capability_matrix key must be JSON-serializable
    so scikit-rec-agent can stream it through tool envelopes."""
    import json

    cm = capability_matrix()
    # tuple → list conversion is handled by default JSON encoder via list(...)
    cm_json = json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in cm.items()},
        default=lambda o: list(o) if isinstance(o, tuple) else str(o),
    )
    assert "mixed_type_multi_target" in cm_json


# ---------------------------------------------------------------------- #
# contract_from_dataframe
# ---------------------------------------------------------------------- #


def test_contract_long_interactions():
    df = pd.DataFrame({"USER_ID": ["u1"], "ITEM_ID": ["i1"], "OUTCOME": [1.0]})
    assert contract_from_dataframe(df) == "long_interactions"


def test_contract_long_with_timestamp():
    df = pd.DataFrame(
        {
            "USER_ID": ["u1"],
            "ITEM_ID": ["i1"],
            "OUTCOME": [1.0],
            "TIMESTAMP": [1234567890],
        }
    )
    assert contract_from_dataframe(df) == "long_with_timestamp"


def test_contract_wide_multioutput_default():
    """Without target_specs, wide-format defaults to wide_multioutput
    (legacy heuristic)."""
    df = pd.DataFrame(
        {
            "USER_ID": ["u1", "u2"],
            "ITEM_a": [0, 1],
            "ITEM_b": [1, 0],
        }
    )
    assert contract_from_dataframe(df) == "wide_multioutput"


def test_contract_wide_mixed_type_with_target_specs():
    df = pd.DataFrame(
        {
            "USER_ID": ["u1", "u2"],
            "ITEM_clicked": [0, 1],
            "ITEM_revenue": [10.0, 20.0],
        }
    )
    target_specs = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
    }
    assert contract_from_dataframe(df, target_specs=target_specs) == "wide_mixed_type_multi_target"


def test_contract_wide_with_multilabel_group():
    df = pd.DataFrame(
        {
            "USER_ID": ["u1"],
            "ITEM_a": [0],
            "ITEM_b": [1],
        }
    )
    target_specs = {
        "g": TargetGroupSpec(type=TargetType.MULTILABEL, columns=["ITEM_a", "ITEM_b"]),
    }
    assert contract_from_dataframe(df, target_specs=target_specs) == "wide_mixed_type_multi_target"


def test_contract_multiclass():
    df = pd.DataFrame({"USER_ID": ["u1"], "ITEM_ID": ["class_A"]})
    assert contract_from_dataframe(df) == "multiclass"


def test_contract_unrecognized_raises():
    df = pd.DataFrame({"x": [1], "y": [2]})
    with pytest.raises(ValueError, match="Cannot detect"):
        contract_from_dataframe(df)


# ====================================================================== #
# Factory v2-list #3: defaults + per_target gap-filling
# ====================================================================== #


def test_factory_3_defaults_and_per_target_gap_fill():
    """defaults covers some types; per_target covers specific names;
    union must cover every fanned-out target. The factory must compose
    them correctly (per_target overrides where present, defaults fill
    gaps)."""
    ts = {
        "ITEM_a": TargetType.BINARY,
        "ITEM_b": TargetType.REGRESSION,
        "ITEM_c": TargetType.BINARY,
    }
    est = create_estimator(
        estimator_config={
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "independent",
                "independent": {
                    "defaults": {
                        # Cover BINARY in defaults; REGRESSION via per_target.
                        "binary": {"estimator_type": "lightgbm", "params": {"n_estimators": 5, "verbose": -1}},
                    },
                    "per_target": {
                        # Specific name override for the regression target.
                        "ITEM_b": {"estimator_type": "lightgbm", "params": {"n_estimators": 5, "verbose": -1}},
                        # Specific override of one binary target (beats default).
                        "ITEM_c": {"estimator_type": "logreg", "params": {}},
                    },
                },
            },
        },
        scorer_type="mixed_type_multi_target",
        target_specs=ts,
    )
    assert isinstance(est.estimators["ITEM_a"], LightGBMClassifierEstimator)
    assert isinstance(est.estimators["ITEM_b"], LightGBMRegressorEstimator)
    # ITEM_c got the per_target logreg override, not the binary default.
    assert "Logistic" in type(est.estimators["ITEM_c"]).__name__


# ====================================================================== #
# Factory v2-list #5: multilabel member per_target override beats group default
# ====================================================================== #


def test_factory_5_multilabel_member_per_target_overrides_group_default():
    """A multilabel member listed in per_target overrides the multilabel
    group default — name-key precedence holds even for fanned-out
    members."""
    ts = {
        "g": {"type": TargetType.MULTILABEL, "columns": ["ITEM_mem_a", "ITEM_mem_b"]},
    }
    est = create_estimator(
        estimator_config={
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "independent",
                "independent": {
                    "defaults": {
                        "multilabel": {"estimator_type": "lightgbm", "params": {"n_estimators": 5, "verbose": -1}},
                    },
                    "per_target": {
                        "ITEM_mem_a": {"estimator_type": "logreg", "params": {}},
                    },
                },
            },
        },
        scorer_type="mixed_type_multi_target",
        target_specs=ts,
    )
    # ITEM_mem_a got logreg via per_target; ITEM_mem_b got lightgbm via defaults.
    assert "Logistic" in type(est.estimators["ITEM_mem_a"]).__name__
    assert isinstance(est.estimators["ITEM_mem_b"], LightGBMClassifierEstimator)


# ====================================================================== #
# Factory v2-list #7: exhaustive rejection matrix
# ====================================================================== #


@pytest.mark.parametrize(
    "target_type,estimator_type,should_fail",
    [
        (TargetType.MULTICLASS, "xgboost", True),  # excluded — see plan
        (TargetType.MULTICLASS, "lightgbm", False),
        (TargetType.REGRESSION, "logreg", True),  # logreg is a classifier
        (TargetType.BINARY, "lightgbm", False),
    ],
)
def test_factory_7_independent_compat_matrix(target_type, estimator_type, should_fail):
    """Walk through the independent_target_compat table and verify the
    rejection matrix: every (target_type, estimator_type) ∉ table must
    raise, and every ∈ table must construct."""
    from skrec.orchestrator.factory import _create_independent_sub_estimator

    if should_fail:
        with pytest.raises((ValueError, KeyError, TypeError, NotImplementedError)):
            _create_independent_sub_estimator(
                target_type=target_type,
                estimator_type=estimator_type,
                params={},
            )
    else:
        sub = _create_independent_sub_estimator(
            target_type=target_type,
            estimator_type=estimator_type,
            params={"verbose": -1} if estimator_type == "lightgbm" else {},
        )
        assert sub is not None


# ====================================================================== #
# Factory v2-list #10: target_specs is the SAME object passed to both factories
# ====================================================================== #


def test_factory_10_target_specs_object_identity_through_pipeline():
    """The target_specs dict the user passes via scorer_config must reach
    BOTH the scorer and the estimator without being mutated or copied.
    Identity equality (``is``) over the inner reference confirms no
    silent reshape happens between the two factory calls."""
    from skrec.orchestrator.factory import create_recommender_pipeline

    ts = {"ITEM_a": TargetType.BINARY}
    rec = create_recommender_pipeline(
        config={
            "recommender_type": "ranking",
            "scorer_type": "mixed_type_multi_target",
            "scorer_config": {"target_specs": ts},
            "estimator_config": {
                "ml_task": "multi_target",
                "multi_target": {
                    "mode": "independent",
                    "independent": {
                        "defaults": {
                            "binary": {"estimator_type": "lightgbm", "params": {"n_estimators": 5, "verbose": -1}},
                        },
                    },
                },
            },
        }
    )
    # Both scorer and estimator must hold the SAME dict object (no copies).
    assert rec.scorer.target_specs is ts
    assert rec.scorer.estimator.target_specs is ts


# ====================================================================== #
# Factory v2-list #11: contract_from_dataframe({}, target_specs={})
# ====================================================================== #


def test_factory_11_contract_from_dataframe_empty_inputs():
    """contract_from_dataframe with an empty DataFrame must raise (no
    contract detectable); with target_specs={} the empty-dict check
    must not trip the truthiness branch."""
    with pytest.raises(ValueError, match="Cannot detect"):
        contract_from_dataframe(pd.DataFrame())
    with pytest.raises(ValueError, match="Cannot detect"):
        contract_from_dataframe(pd.DataFrame(), target_specs={})


# ====================================================================== #
# Fix 2: factory recognizes the two v3 conditional modes
# ====================================================================== #


def test_fix2_capability_matrix_lists_conditional_modes():
    cm = capability_matrix()
    assert "conditional_joint_mlp" in cm["multi_target_model_types"]
    assert "conditional_joint_transformer" in cm["multi_target_model_types"]
    assert len(cm["multi_target_model_types"]) == 5
    assert MULTI_TARGET_MODEL_TYPES == cm["multi_target_model_types"]


def test_fix2_create_estimator_conditional_joint_mlp():
    ts = {"ITEM_a": TargetType.BINARY}
    est = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "conditional_joint_mlp",
                "params": {"epochs": 1, "hidden_dim": 8, "num_layers": 1, "mask_prob": 0.5},
            },
        },
        target_specs=ts,
    )
    assert isinstance(est, ConditionalJointMultiTargetMLPEstimator)


def test_fix2_create_estimator_conditional_joint_transformer():
    ts = {"ITEM_a": TargetType.BINARY}
    est = create_estimator(
        {
            "estimator_type": "tabular",
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "conditional_joint_transformer",
                "params": {
                    "epochs": 1,
                    "d_model": 8,
                    "n_heads": 2,
                    "n_layers": 1,
                    "ffn_dim": 16,
                    "mask_prob": 0.5,
                },
            },
        },
        target_specs=ts,
    )
    assert isinstance(est, ConditionalJointMultiTargetTransformerEstimator)


# ====================================================================== #
# Fix 3: logged_rewards per-column type validation
# ====================================================================== #


def _build_recommender_for_validation():
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = {
        "ITEM_clicked": (X["f0"] > 0).astype(int).to_numpy(),
        "ITEM_revenue": X["f1"].to_numpy(),
        "ITEM_action": np.array(["A", "B", "C"])[np.column_stack([X["f0"], X["f1"], X["f2"]]).argmax(axis=1)],
    }
    ts = {
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
        "ITEM_action": TargetType.MULTICLASS,
    }
    est = JointMultiTargetMLPEstimator(
        target_specs=ts,
        params={"epochs": 1, "hidden_dim": 8, "num_layers": 1, "batch_size": 16},
    )
    est.fit(X, y)
    return RankingRecommender(scorer=MixedTypeMultiTargetScorer(estimator=est, target_specs=ts)), X, y, ts


# --- P0-7: independent.defaults upfront coverage validation ---


def test_fix_p0_7_independent_missing_default_raises_upfront():
    """Pre-fix: missing default surfaced as a per-target ValueError deep
    in the per-target loop. Post-fix: a single upfront error lists EVERY
    missing default key so the user can fix the config in one round-trip."""
    from skrec.orchestrator import create_estimator

    target_specs = {
        "ITEM_a": TargetType.BINARY,
        "ITEM_b": TargetType.REGRESSION,
        "ITEM_action": TargetType.MULTICLASS,
    }
    with pytest.raises(ValueError) as exc:
        create_estimator(
            {
                "estimator_type": "tabular",
                "ml_task": "multi_target",
                "multi_target": {
                    "mode": "independent",
                    "independent": {
                        # Only declare 'binary'; regression + multiclass missing.
                        "defaults": {
                            "binary": {"estimator_type": "xgboost", "params": {}},
                        },
                    },
                },
            },
            target_specs=target_specs,
        )
    msg = str(exc.value)
    assert "missing coverage" in msg.lower()
    # Single error names BOTH missing default keys, not just the first one.
    assert "regression" in msg
    assert "multiclass" in msg


def test_p1_18_contract_from_dataframe_honors_target_specs_intent():
    """contract_from_dataframe must return wide_mixed_type_multi_target
    whenever target_specs is passed, even for all-BINARY specs."""
    from skrec.orchestrator import contract_from_dataframe

    df = pd.DataFrame(
        {
            "USER_ID": [1, 2, 3],
            "ITEM_a": [0, 1, 0],
            "ITEM_b": [1, 1, 0],
        }
    )
    # Without target_specs → default to wide_multioutput.
    assert contract_from_dataframe(df) == "wide_multioutput"
    # With all-BINARY target_specs → honor the explicit intent.
    ts = {"ITEM_a": TargetType.BINARY, "ITEM_b": TargetType.BINARY}
    assert contract_from_dataframe(df, target_specs=ts) == "wide_mixed_type_multi_target"


def test_p1_5_capability_matrix_derives_from_scorer_class_attr():
    """scorer_supports_observed_conditioning must be derived from each
    scorer class's supports_observed_conditioning attribute, not a
    hand-edited tuple."""
    from skrec.scorer.base_scorer import BaseScorer
    from skrec.scorer.mixed_type_multi_target import MixedTypeMultiTargetScorer

    assert BaseScorer.supports_observed_conditioning is False
    assert MixedTypeMultiTargetScorer.supports_observed_conditioning is True

    caps = capability_matrix()
    assert "mixed_type_multi_target" in caps["scorer_supports_observed_conditioning"]


def test_p1_14_independent_random_state_propagates():
    """multi_target.random_state must be plumbed into independent sub-
    estimator params when the sub-estimator's params don't set it."""
    target_specs = {"ITEM_a": TargetType.BINARY}

    est = create_estimator(
        estimator_config={
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "independent",
                "random_state": 42,
                "independent": {
                    "defaults": {
                        "binary": {
                            "estimator_type": "lightgbm",
                            "params": {"n_estimators": 5, "verbose": -1},
                        },
                    },
                },
            },
        },
        scorer_type="mixed_type_multi_target",
        target_specs=target_specs,
    )
    # The sub-estimator's underlying LGBM model should carry
    # random_state=42 via the factory's propagation.
    sub = est.estimators["ITEM_a"]
    assert sub._model.get_params().get("random_state") == 42

    # Caller-supplied random_state must NOT be overwritten.
    est2 = create_estimator(
        estimator_config={
            "ml_task": "multi_target",
            "multi_target": {
                "mode": "independent",
                "random_state": 42,
                "independent": {
                    "defaults": {
                        "binary": {
                            "estimator_type": "lightgbm",
                            "params": {
                                "n_estimators": 5,
                                "verbose": -1,
                                "random_state": 7,
                            },
                        },
                    },
                },
            },
        },
        scorer_type="mixed_type_multi_target",
        target_specs=target_specs,
    )
    assert est2.estimators["ITEM_a"]._model.get_params()["random_state"] == 7


# ====================================================================== #
# Post-P1 follow-up: evaluate path routes through preprocess_inputs
# (no longer bypasses interactions_schema coercion)
# ====================================================================== #


def test_independent_target_compat_includes_multilabel_row():
    """Structural pin: dropping the MULTILABEL row would silently break
    the agent-side pre-flight surface that reads this table. The
    compose loop fans multilabel members into BINARY entries before
    consulting the compat dict, so the MULTILABEL row is otherwise
    unreachable from the factory itself. This test guards the dict
    from a 'looks unused → delete' refactor."""
    assert TargetType.MULTILABEL in _INDEPENDENT_TARGET_COMPAT, (
        "MULTILABEL row dropped from _INDEPENDENT_TARGET_COMPAT — "
        "agent-side pre-flight validation (capability_matrix()["
        "'independent_target_compat']) reads this. Restore it."
    )
    assert _INDEPENDENT_TARGET_COMPAT[TargetType.MULTILABEL], (
        "MULTILABEL compat tuple is empty — must list at least one "
        "estimator_type the agent can suggest for multilabel groups."
    )
