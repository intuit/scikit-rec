# Mixed-type multi-target scorer + estimator families — v2 implementation plan

Supersedes `mixed_type_multi_target_plan.md` for v2 scope. Two changes from v1:

1. **Add an `IndependentMultiTargetEstimator` family** — per-target standalone estimators (XGB, LightGBM, sklearn, …) trained independently and stitched at the scorer level. Sits alongside the joint MLP/Transformer families and shares the scorer.
2. **Defer all `OBSERVED_*` (real-time-label conditioning) work to v3.** v2 ships vanilla-only. Conditional MLP, conditional Transformer, OBSERVED_* validators, schema-apply preservation hook, and the `_validate_inference_interactions` complexity are all out.

Single PR for v2 (confirmed). Conditional support lands as a purely additive v3 PR — see "Deferred to v3" at the bottom.

---

## Diff vs v1 (read this first)

| Area | v1 | v2 |
|---|---|---|
| Estimator families | 4: joint MLP/Transformer × vanilla/conditional | 3: joint MLP, joint Transformer, **independent** |
| `OBSERVED_*` columns | Scorer accepts (conditional) / rejects (vanilla) | Scorer **always rejects** with "deferred to v3" message |
| `BaseScorer.preserved_inference_columns()` hook | New hook added | Not added (deferred to v3) |
| `BaseRecommender.recommend_online` column preservation | Added around schema-apply | Not added (deferred to v3) |
| `_validate_inference_interactions` | Full OBSERVED-aware sibling validator | Trimmed: only rejects orphan ITEM_* columns and OBSERVED_* (with deferred-message) |
| Encoder Protocol | Shared `JointMultiTargetEncoder` Protocol with `label_input_dim` param | Same Protocol, `label_input_dim` param **retained at 0** to keep the v3 door open without implementing label inputs |
| Scorer-facing estimator contract | Implicit (estimator family) | **Explicit `MultiTargetEstimator` Protocol** unifies joint + independent |
| Factory `mode` enum | `joint_mlp` / `conditional_joint_mlp` / `joint_transformer` / `conditional_joint_transformer` | `joint_mlp` / `joint_transformer` / **`independent`** |
| Capability matrix `multi_target_model_types` | 4 entries | 3 entries (above) |
| Test files | 5 (scorer + 4 estimator families) | 6 (scorer + joint MLP + joint Transformer + independent + factory + **evaluation**) |
| Evaluation | Not specified in v1 plan | **New section, shipped in v2 PR**: per-`TargetType` metric dispatch in `RankingRecommender.evaluate()`, new `MULTICLASS_ACCURACY` metric, `score_per_target` escape hatch on the scorer |

Everything not listed above is unchanged from v1. Section numbering below is renumbered for v2 to read standalone; cross-references back to v1 are explicit where helpful.

---

## Goal (v2)

Add a scorer + estimator families to scikit-rec that support:

- Wide-format inputs with **heterogeneous target types per column** (binary, regression, multiclass, multilabel) trained either jointly (one shared model) or independently (one sub-estimator per target).
- A clean Protocol-based extension point so v3 can add conditional (real-time-label) variants without breaking v2 behavior.

Architectural superset of the existing `MultioutputScorer + MultiOutputClassifier` path; not a replacement.

## Locked decisions (v2)

| # | Decision | Choice |
|---|---|---|
| 1 | v1 → v2 scope shift | Drop both conditional variants; add `independent` family |
| 2 | Estimator families | Joint MLP, Joint Transformer (FT-Transformer-style), Independent (per-target sub-estimators) |
| 3 | Scorer ↔ estimator contract | New explicit `MultiTargetEstimator` Protocol (runtime-checkable). Scorer rejects non-implementers at init |
| 4 | Encoder abstraction (joint only) | Shared `JointMultiTargetEncoder` Protocol kept; `label_input_dim` param retained at 0 (v3 door) |
| 5 | Independent estimator construction | **Both paths supported:** (a) config-driven via factory (defaults + per-target overrides), (b) direct-construction with `estimators=dict[target_name, BaseEstimator]` for advanced users |
| 6 | Independent + multilabel | Fan-out: each member column becomes its own binary sub-estimator. Group inductive bias is unavailable in independent mode (documented trade-off) |
| 7 | Independent + multiclass | Sub-estimator must be a classifier supporting K>2 (XGB, LightGBM, LogReg). Validated at factory time |
| 8 | Independent + conditional | Not in v2. Not on the v3 roadmap either — semantics are awkward (per-target estimators receiving cross-target observed values as features). If revisited, it's v4+ |
| 9 | `OBSERVED_*` semantics | Scorer rejects all OBSERVED_* columns at inference with `NotImplementedError("OBSERVED_* conditioning is deferred to v3; use vanilla inference for now.")`. No schema-apply preservation, no inference validator complexity |
| 10 | Factory `multi_target.mode` enum | `"joint_mlp" \| "joint_transformer" \| "independent"`. No conditional values |
| 11 | Pairing strictness | Strict — scorer rejects estimators outside the `MultiTargetEstimator` Protocol (TypeError at init) |
| 12 | Dict-shaped y plumbing | Same as v1: override `_validate_for_fit` on joint estimators (dict y has no `.shape`). For independent estimator, override is also dict-aware but iterates per-target |
| 13 | Output column conventions | Same as v1 (see "Data shapes" below) |
| 14 | `predict_targets()` method | Same as v1 — wide one-column-per-target DataFrame |
| 15 | `score_fast` / `recommend_online` | Same as v1 wide-format pattern; minus all OBSERVED_* handling |
| 16 | `recommend()` | Same as v1 — `RankingRecommender` short-circuits via isinstance to `predict_targets`, ignores `top_k` with warning |
| 17 | PR strategy | Single PR (confirmed) |
| 18 | Evaluation contract | Per-`TargetType` metric dispatch; always returns `Dict[str, float]` (no scalar default — heterogeneous types can't be macro-averaged). Restricted to `RecommenderEvaluatorType.SIMPLE`. Ranking metrics rejected with explanatory error |
| 19 | New metric in v2 | `MULTICLASS_ACCURACY` (top-1). Mirrors `BaseClassificationMetric` shape via new `BaseMulticlassMetric` base class |
| 20 | Escape hatch | `MixedTypeMultiTargetScorer.score_per_target(y_true, metric_callables, ...)`: user-supplied per-`TargetType` or per-target-name callables; covers log-loss, macro-F1, business metrics, etc. |

---

## Public API

### `MultiTargetEstimator` Protocol (new — unifies joint + independent)

```python
@runtime_checkable
class MultiTargetEstimator(Protocol):
    """Scorer-facing contract. Joint and independent estimator families both
    implement this. The scorer only sees this surface; the actual training
    machinery (single neural net vs dict of sub-estimators) is internal."""

    target_specs: dict[str, TargetType | TargetGroupSpec]

    def fit(
        self,
        X: pd.DataFrame,
        y: dict[str, np.ndarray],            # one entry per target spec key
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[dict[str, np.ndarray]] = None,
    ) -> "MultiTargetEstimator": ...

    def predict_proba_dict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-target probabilities / values, keyed by target name.
        binary       → (n, 2) array (class probs)
        multiclass   → (n, K) array (class probs)
        multilabel   → fanned out: one (n, 2) entry per *member column*,
                       keyed by member column name (NOT the group key)
        regression   → (n,) array (predicted values, de-normalized if applicable)
        """

    def predict_targets_dict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-target point estimates, keyed by target name.
        binary       → (n,) array of 0/1
        multiclass   → (n,) array of class labels (strings or ints, preserving input dtype)
        multilabel   → fanned out: one (n,) binary array per member column
        regression   → (n,) array of values
        """
```

Lives in `skrec/estimator/classification/_multi_target_protocol.py` (shared module — both joint and independent estimator subpackages import it).

### `MixedTypeMultiTargetScorer` (unchanged from v1 except: trimmed validators, no preserved-columns hook)

```python
class MixedTypeMultiTargetScorer(BaseScorer):
    target_col = None  # full _validate_interactions override; no single target column

    def __init__(
        self,
        estimator: MultiTargetEstimator,   # joint or independent — Protocol-checked at init
        target_specs: dict[str, TargetType | TargetGroupSpec],
    ) -> None:
        if not isinstance(estimator, MultiTargetEstimator):
            raise TypeError(
                f"MixedTypeMultiTargetScorer requires an estimator implementing "
                f"MultiTargetEstimator. Got {type(estimator).__name__}."
            )
        # ... target_specs consistency check (estimator.target_specs must equal target_specs)
        ...

    def process_datasets(self, users_df=None, items_df=None, interactions_df=None,
                        is_training=True) -> tuple[pd.DataFrame, dict[str, np.ndarray]]: ...

    # No train_model override — BaseScorer.train_model is a passthrough.
    # Dict-shaped y propagates through to estimator.fit unchanged.

    def score_items(self, interactions=None, users=None) -> pd.DataFrame:
        """Wide proba/value DataFrame; one block of columns per target.
        Internally: estimator.predict_proba_dict(X) → stitch to wide DataFrame
        following the output column conventions table below."""

    def predict_targets(self, interactions=None, users=None) -> pd.DataFrame:
        """Wide point-estimate DataFrame; one column per target.
        Internally: estimator.predict_targets_dict(X) → wide DataFrame."""

    def _validate_interactions(self, interactions_df: pd.DataFrame) -> None:
        """Training-time validator. 1-arg signature matches BaseScorer.
        Note: OUTCOME_* columns are stripped upstream by BaseRecommender
        before reaching the scorer."""

    def _validate_inference_interactions(self, interactions_df: pd.DataFrame) -> None:
        """Inference-time validator. v2 scope:
        - Reject any OBSERVED_* column with NotImplementedError naming v3.
        - Reject orphan ITEM_* feature columns that look like targets but
          aren't declared in target_specs.
        - (No partial-multilabel logic — that's a conditioning concern, deferred to v3.)
        """

    def score_fast(self, features: pd.DataFrame) -> pd.DataFrame:
        """Single-row predict_targets. features.shape[0] == 1.

        Fresh implementation calling predict_targets directly; does NOT route
        through _calculate_scores. v2: rejects any OBSERVED_* column with
        the deferred-to-v3 message."""

    def score_per_target(
        self,
        interactions: pd.DataFrame,
        y_true: pd.DataFrame,
        metric_callables: dict[Union[str, TargetType],
                               Callable[[np.ndarray, np.ndarray], float]],
    ) -> dict[str, float]:
        """Per-target evaluation escape hatch with user-supplied callables.
        Lookup precedence: target-name key > TargetType key. See Evaluation
        section for callable signature per TargetType."""

    def _calculate_scores(self, joined) -> NDArray:
        raise NotImplementedError(
            "Use score_items() instead. score_fast() is also a fresh "
            "implementation that does NOT route through _calculate_scores."
        )

    # NOTE: preserved_inference_columns() hook is NOT added in v2. Deferred to v3.
```

### Target specification (unchanged from v1)

```python
class TargetType(Enum):
    BINARY = "binary"
    REGRESSION = "regression"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"

class TargetGroupSpec(TypedDict):
    type: TargetType        # currently MULTILABEL only
    columns: list[str]      # ITEM_*-prefixed member columns
```

### Factory config shape (v2)

```
RecommenderConfig (extended)
├── recommender_type, scorer_type, recommender_params         # unchanged
├── scorer_config:                                            # NEW block (same as v1)
│     target_specs: { col: TargetType | TargetGroupSpec, ... }
└── estimator_config:
      estimator_type: "tabular"                               # unchanged
      ml_task: "multi_target"                                 # NEW value
      multi_target:                                           # NEW sub-config
        mode: "joint_mlp" | "joint_transformer" | "independent"

        # --- joint_mlp / joint_transformer only ---
        params: { hidden_dim, num_layers, dropout, batch_size, epochs, lr,
                  weight_decay, regression_normalize, device,
                  # transformer only:
                  d_model, n_heads, n_layers, ffn_dim, attn_dropout, ffn_dropout,
                  warmup_steps }

        # --- independent only ---
        independent:
          defaults:                              # used when per_target omits a type
            binary:      { estimator_type: "xgboost",  params: {...} }
            regression:  { estimator_type: "lightgbm", params: {...} }
            multiclass:  { estimator_type: "xgboost",  params: {...} }
            multilabel:  { estimator_type: "xgboost",  params: {...} }   # applied per-member
          per_target:                            # optional per-target overrides
            ITEM_revenue:
              estimator_type: "lightgbm"
              params: { n_estimators: 500 }
            # For multilabel groups: per_target key can be either the group key
            # OR an individual member column; member-level overrides take precedence.
```

`target_specs` (scorer concern) and `multi_target` (estimator concern) are passed independently. `create_recommender_pipeline` reads `target_specs` from `scorer_config["target_specs"]` once and threads it to both `create_estimator(..., target_specs=...)` and `create_scorer(..., target_specs=...)`. Factory composes the right estimator instance based on `mode`.

---

## Estimator families (v2)

### Family 1: Joint MLP

`JointMultiTargetMLPEstimator` — single network with `MLPEncoder` + per-target heads. All targets trained jointly with weighted multi-task loss. Same architecture as v1; conditional cousin removed.

### Family 2: Joint Transformer

`JointMultiTargetTransformerEstimator` — single network with `TransformerEncoder` (FT-Transformer-style: feature tokenization → MHSA blocks → CLS pooling) + per-target heads. Same architecture as v1; conditional cousin removed.

### Family 3: Independent (new)

`IndependentMultiTargetEstimator` — holds a `dict[target_name, BaseEstimator]` where each sub-estimator is an existing scikit-rec estimator instance:

- **binary target** → any `BaseClassifier` (XGB, LightGBM, LogReg, Sklearn universal)
- **regression target** → any `BaseRegressor` (XGB, LightGBM, Sklearn universal)
- **multiclass target** → any `BaseClassifier` supporting K>2 classes
- **multilabel group** → fan out: one binary `BaseClassifier` per member column. The dict stores per-member entries; the group key is metadata only

Two construction paths (both supported):

**A. Config-driven (orchestrator)**

```python
config: RecommenderConfig = {
    "recommender_type": "ranking",
    "scorer_type": "mixed_type_multi_target",
    "scorer_config": {
        "target_specs": {
            "ITEM_clicked": TargetType.BINARY,
            "ITEM_revenue": TargetType.REGRESSION,
            "ITEM_action": TargetType.MULTICLASS,
            "engagement_group": TargetGroupSpec(type=TargetType.MULTILABEL,
                                                columns=["ITEM_email_open",
                                                         "ITEM_app_open"]),
        },
    },
    "estimator_config": {
        "estimator_type": "tabular",
        "ml_task": "multi_target",
        "multi_target": {
            "mode": "independent",
            "independent": {
                "defaults": {
                    "binary":     {"estimator_type": "xgboost",  "params": {"n_estimators": 100}},
                    "regression": {"estimator_type": "lightgbm", "params": {"n_estimators": 200}},
                    "multiclass": {"estimator_type": "xgboost",  "params": {}},
                    "multilabel": {"estimator_type": "xgboost",  "params": {}},
                },
                "per_target": {
                    "ITEM_revenue": {"estimator_type": "lightgbm", "params": {"n_estimators": 500}},
                    "ITEM_email_open": {"estimator_type": "logreg", "params": {"max_iter": 200}},
                },
            },
        },
    },
}
recommender = create_recommender_pipeline(config)
```

**B. Direct-construction (advanced)**

```python
from skrec.estimator.classification.xgb_classifier import XGBClassifierEstimator
from skrec.estimator.classification.logreg_classifier import LogRegClassifierEstimator
from skrec.estimator.regression.lightgbm_regressor import LightGBMRegressorEstimator
from skrec.estimator.classification.independent_multi_target import (
    IndependentMultiTargetEstimator,
)

estimator = IndependentMultiTargetEstimator(
    target_specs={
        "ITEM_clicked": TargetType.BINARY,
        "ITEM_revenue": TargetType.REGRESSION,
        "engagement_group": TargetGroupSpec(
            type=TargetType.MULTILABEL,
            columns=["ITEM_email_open", "ITEM_app_open"],
        ),
    },
    estimators={
        "ITEM_clicked":    XGBClassifierEstimator(params={"n_estimators": 100}),
        "ITEM_revenue":    LightGBMRegressorEstimator(params={"n_estimators": 500}),
        "ITEM_email_open": LogRegClassifierEstimator(params={"max_iter": 200}),
        "ITEM_app_open":   XGBClassifierEstimator(params={}),
    },
)
scorer = MixedTypeMultiTargetScorer(estimator=estimator, target_specs=estimator.target_specs)
```

Direct-construction validates that `estimators` keys cover every target — for multilabel groups, every member column must be present (the group key itself must NOT be a key in `estimators`).

#### `IndependentMultiTargetEstimator.fit`

```python
def fit(self, X, y, X_valid=None, y_valid=None):
    self._validate_for_fit(X, y, X_valid, y_valid)   # dict-y aware
    self.feature_names = X.columns.tolist()
    for target_name, sub_est in self.estimators.items():
        y_target = self._extract_y_for_target(y, target_name)
        y_valid_target = self._extract_y_for_target(y_valid, target_name) if y_valid else None
        sub_est.fit(X, y_target,
                    X_valid=X_valid,
                    y_valid=y_valid_target)
    return self
```

`_extract_y_for_target` handles the multilabel fan-out: for a member column, it pulls the column out of the y dict entry keyed by the group name and serves it as a 1-D binary array.

#### `IndependentMultiTargetEstimator.predict_proba_dict` / `predict_targets_dict`

Just walks `self.estimators` and calls each sub-estimator's `predict_proba` / `predict`, returning per-target arrays. Multilabel groups fan out into one entry per member column. Output shape matches the Protocol contract exactly so the scorer's wide-format stitching is identical regardless of which family produced the dict.

#### Per-target type compatibility check (factory time)

```python
_COMPATIBLE_SUB_ESTIMATORS = {
    TargetType.BINARY:     ("xgboost", "lightgbm", "logreg", "sklearn"),
    TargetType.REGRESSION: ("xgboost", "lightgbm", "sklearn"),
    TargetType.MULTICLASS: ("xgboost", "lightgbm", "logreg"),     # support K>2
    TargetType.MULTILABEL: ("xgboost", "lightgbm", "logreg", "sklearn"),  # per member, binary
}
```

Factory validates the spec table before constructing any sub-estimator. Clean error names the offending target and the incompatible estimator type. Direct-construction path does the same validation in `IndependentMultiTargetEstimator.__init__`.

### Why one scorer, three estimator families (not separate scorers)

The scorer's job is the wide-format I/O contract: ITEM_* column unpacking at training, dict-y stitching, output column conventions, the `predict_targets` short-circuit in `RankingRecommender`. None of this depends on whether the underlying model is one network or N. The Protocol cleanly separates I/O from training machinery — keeping one scorer means one capability matrix row, one set of recommender wiring, one set of docs, and one decision-rule tree branch.

---

## Data shapes (v2 — simplified, no OBSERVED)

### Training (`interactions_df`)

- `USER_ID`, `ITEM_<name>` target columns (one per simple target; K per multilabel group), feature columns.
- One row per user.

### Inference (`interactions_df`)

- `USER_ID`, feature columns.
- **No `OBSERVED_*` columns.** v2 scorer rejects them with a `NotImplementedError` naming v3.

### Output column conventions (unchanged from v1)

`score_items` (wide DataFrame, no `USER_ID` column, one row per input user):

| Target type | Output columns |
|---|---|
| binary | `ITEM_<col>_0`, `ITEM_<col>_1` |
| regression | `ITEM_<col>` (de-normalized) |
| multiclass | `ITEM_<col>_<class_label>` per class |
| multilabel | `ITEM_<member>_0`, `ITEM_<member>_1` per member |

`predict_targets` (wide DataFrame, no `USER_ID` column, one row per input user):

| Target type | Output columns |
|---|---|
| binary | `ITEM_<col>` (predicted class label 0/1) |
| regression | `ITEM_<col>` (predicted value) |
| multiclass | `ITEM_<col>` (predicted class label) |
| multilabel | `ITEM_<member>` (predicted class label 0/1) per member |

---

## Evaluation

### Why no ranking metrics

`MixedTypeMultiTargetScorer` predicts per-target values, not item rankings — same situation as `MultioutputScorer`. NDCG, Recall@K, MAP, MRR all assume a long-format `(user, item, reward)` shape with a single comparable score per (user, item) pair. We don't have that — we have per-target predictions of different types. Ranking metrics on this scorer are rejected with a clean error pointing at the per-target metric path.

### Return shape: always `Dict[str, float]`

Unlike `MultioutputScorer` (which can macro-average to a scalar in classifier mode because all targets are binary), this scorer has **heterogeneous target types in the same model**. A binary AUC of 0.85 and a regression RMSE of 12.7 aren't on a common scale — averaging them would be meaningless. So `evaluate()` always returns `Dict[str, float]`, one entry per target (or per multilabel-member). No `per_label` flag — there's only one return shape.

If the caller wants a scalar (e.g., for HPO objective), they pick a primary target name or compose a weighted aggregate via `score_per_target` (escape hatch below). The plan deliberately does NOT bake a "primary metric" default into the scorer — that's a per-use-case decision.

### Per-`TargetType` metric dispatch

`RankingRecommender.evaluate()` gets a new `_evaluate_mixed_type_multi_target` branch (mirrors the existing `_evaluate_multioutput` pattern). The branch iterates declared targets and applies the appropriate metric per type. Compatibility table:

| Target type | Supported metric types (v2) | Score input shape | Ground-truth input shape |
|---|---|---|---|
| binary | `ROC_AUC`, `PR_AUC` | `(n,)` pos-class probabilities (sliced from `ITEM_<col>_1`) | `(n,)` of 0/1 |
| regression | `RMSE`, `MAE` | `(n,)` predicted values (from `ITEM_<col>`) | `(n,)` continuous |
| multiclass | `MULTICLASS_ACCURACY` (new) | `(n, K)` class probabilities (from `ITEM_<col>_<class>`) | `(n,)` of class labels |
| multilabel | `ROC_AUC`, `PR_AUC` (per member) | `(n,)` pos-class probabilities per member | `(n,)` of 0/1 per member |

Cross-type errors are explicit: `RMSE` on a binary target → `ValueError("RMSE requires a regression target; target 'ITEM_clicked' is declared BINARY. Use ROC_AUC or PR_AUC, or escape via score_per_target with a user-supplied callable.")`.

If a caller passes a single `metric_type`, it's broadcast across all targets — targets whose declared type doesn't support that metric raise an error naming each offender. If a caller passes a `Dict[str, RecommenderMetricType]` keyed by target name, per-target overrides are honored (e.g., `{"ITEM_clicked": ROC_AUC, "ITEM_revenue": RMSE, "ITEM_action": MULTICLASS_ACCURACY}`).

### New metric: `MULTICLASS_ACCURACY`

scikit-rec's existing metrics cover ranking, binary classification, and regression — there's no multiclass metric today. v2 adds one: top-1 accuracy, the canonical multiclass scalar.

**New files:**
- `skrec/metrics/multiclass_accuracy.py` — implements `MulticlassAccuracy(BaseMulticlassMetric)`. Takes `recommendation_scores (n, K)` and `modified_rewards (n,)` (class labels). Returns `np.mean(argmax(scores, axis=1) == labels)`.
- `skrec/metrics/base_metric.py` — add `BaseMulticlassMetric` base class. Parallels `BaseClassificationMetric`; documents that `modified_rewards` is `(n,)` of class labels (not one-hot, not `(n, K)`), `recommendation_scores` is `(n, K)`, ignores `recommendation_ranks` and `top_k`.

**Modified:**
- `skrec/metrics/datatypes.py` — add `MULTICLASS_ACCURACY` to `RecommenderMetricType` enum.
- `skrec/metrics/factory.py` — register `MulticlassAccuracy` in the metric factory.
- `skrec/metrics/__init__.py` — export `MulticlassAccuracy`, `BaseMulticlassMetric`.

Log-loss, macro-F1, and other multiclass metrics are not added in v2; users reach them via `score_per_target`. If they prove popular, follow-up PR adds them as named metric types.

### Escape hatch: `score_per_target`

```python
class MixedTypeMultiTargetScorer(BaseScorer):
    def score_per_target(
        self,
        interactions: pd.DataFrame,
        y_true: pd.DataFrame,                       # wide format; columns match predict_targets output
        metric_callables: dict[Union[str, TargetType],
                               Callable[[np.ndarray, np.ndarray], float]],
    ) -> dict[str, float]:
        """Per-target evaluation with user-supplied metric callables.

        Lookup precedence: target-name key beats TargetType key. A target with
        neither a name override nor a type-keyed default raises KeyError.

        Callable signature, by target type:
          binary           → (y_true_1d_0_1,  y_proba_2d_n_x_2)        -> float
          regression       → (y_true_1d_cont, y_pred_1d)               -> float
          multiclass       → (y_true_1d_labels, y_proba_2d_n_x_K)      -> float
          multilabel-member → (y_true_1d_0_1,  y_proba_2d_n_x_2)        -> float

        Use this for log-loss, macro-F1, business-specific metrics, or weighted
        aggregations. For named scikit-rec metrics, prefer `evaluate()`.
        """
```

Example:

```python
from sklearn.metrics import log_loss, f1_score, mean_absolute_percentage_error

metrics = scorer.score_per_target(
    interactions=valid_df,
    y_true=valid_targets_wide,
    metric_callables={
        TargetType.BINARY:     lambda y, p: log_loss(y, p[:, 1]),
        TargetType.REGRESSION: lambda y, p: mean_absolute_percentage_error(y, p),
        TargetType.MULTICLASS: lambda y, p: f1_score(y, p.argmax(axis=1), average="macro"),
        "ITEM_email_open":     lambda y, p: log_loss(y, p[:, 1]),  # member-level override
    },
)
# → {"ITEM_clicked": 0.42, "ITEM_revenue": 0.18, "ITEM_action": 0.71, "ITEM_email_open": 0.39, ...}
```

### `logged_rewards` shape and per-column type validation

Following `MultioutputScorer`'s pattern but type-aware:

- `logged_items (n_users, n_targets)`: target column names, same on every row.
- `logged_rewards (n_users, n_targets)`: ground-truth values **typed per declared `TargetType`**:
  - binary / multilabel-member columns: must be in `{0, 1}` (NaN allowed for ignore-mask)
  - regression columns: continuous floats (NaN allowed)
  - multiclass columns: class labels matching the labels seen at training (NaN allowed)

Per-column validation runs against `target_specs` at the eval-side gate. Mismatches produce errors naming the offending column and the expected type — symmetric with the training-side `_validate_interactions` and matches `MultioutputScorer`'s "fail fast at the eval boundary" pattern.

For multilabel groups: `logged_rewards` carries one column per member (the fanned-out form), matching `predict_targets` output. The group key is metadata only — it doesn't appear in `logged_rewards` columns.

### Restricted to `SimpleEvaluator`

Counterfactual evaluators (IPS, DR, SNIPS) assume a long-format ranking shape. They don't apply to per-target prediction. `evaluate()` rejects any `eval_type != RecommenderEvaluatorType.SIMPLE` with the same error MultioutputScorer uses today, adjusted for the new scorer name.

### Integration: new branch on `RankingRecommender.evaluate()`

```python
def evaluate(self, eval_type, metric_type, eval_top_k, ...):
    if isinstance(self.scorer, MixedTypeMultiTargetScorer):
        return self._evaluate_mixed_type_multi_target(
            eval_type=eval_type,
            metric_type=metric_type,
            score_items_kwargs=score_items_kwargs,
            eval_kwargs=eval_kwargs,
        )
    if isinstance(self.scorer, MultioutputScorer):
        return self._evaluate_multioutput(...)        # unchanged
    return super().evaluate(...)
```

`_evaluate_mixed_type_multi_target` parallels the structure of `_evaluate_multioutput`:

1. Reject `eval_type != SIMPLE` with a clear error.
2. Reject ranking metrics with a clear error pointing at per-target metric path.
3. Validate `score_items_kwargs` and `eval_kwargs` (logged_items / logged_rewards shape + per-column type compatibility).
4. Call `scorer.score_items(**score_items_kwargs)` once; cache wide proba/value DataFrame.
5. For each declared target: slice the relevant columns from score_items output + the matching logged_rewards column, call the dispatched metric, store under the target name.
6. Return `Dict[str, float]`.

If `metric_type` is a single `RecommenderMetricType`, broadcast across targets with per-type compatibility check. If `metric_type` is a `Dict[str, RecommenderMetricType]`, honor per-target overrides. (Signature update to `evaluate()` for this branch is type-only — the underlying field accepts both; the typed overloads in `RankingRecommender.evaluate` get a new variant.)

---

## Architecture

### Joint family (v2 — vanilla only)

```
JointMultiTargetEncoder Protocol (internal)
  ├── MLPEncoder
  └── TransformerEncoder  (FT-Transformer-style)

Per-target heads (shared across encoders):
  binary_head:     Linear(hidden_dim, 1)         + BCEWithLogitsLoss
  regression_head: Linear(hidden_dim, 1)         + MSELoss
  multiclass_head: Linear(hidden_dim, K_k)       + CrossEntropyLoss
  multilabel_head: Linear(hidden_dim, K_k)       + per-dim BCEWithLogitsLoss

JointMultiTargetMLPEstimator         → MLPEncoder + heads
JointMultiTargetTransformerEstimator → TransformerEncoder + heads
```

The `JointMultiTargetEncoder` Protocol keeps `label_input_dim: int = 0` as a default-zero parameter. In v2 it is always 0 (no label channel). In v3 the conditional estimators will pass `label_input_dim > 0`. This is the only v3 hook retained in v2 — it costs nothing and makes the conditional add purely additive.

### Independent family (v2 — new)

```
IndependentMultiTargetEstimator
  ├── target_specs:  dict[str, TargetType | TargetGroupSpec]
  └── estimators:    dict[target_or_member_name, BaseEstimator]
```

No encoder, no shared representation. Each sub-estimator sees the full feature matrix `X`. Multilabel groups are fanned out — the group key is stored on the estimator as metadata for `predict_proba_dict` / `predict_targets_dict` to reconstitute per-member outputs in a consistent order.

### Default hyperparameters

```python
COMMON_DEFAULTS = {
    "batch_size": 1024, "epochs": 10, "lr": 1e-3, "weight_decay": 0.0,
    "regression_normalize": True, "device": None, "seed": 42,
}
MLP_DEFAULTS = {**COMMON_DEFAULTS,
    "hidden_dim": 128, "num_layers": 3, "dropout": 0.1, "use_batch_norm": False,
}
TRANSFORMER_DEFAULTS = {**COMMON_DEFAULTS,
    "d_model": 64, "n_heads": 4, "n_layers": 3, "ffn_dim": 128,
    "attn_dropout": 0.1, "ffn_dropout": 0.1, "warmup_steps": 100,
}
INDEPENDENT_DEFAULTS = {
    "binary":     {"estimator_type": "xgboost",  "params": {"n_estimators": 100}},
    "regression": {"estimator_type": "lightgbm", "params": {"n_estimators": 200}},
    "multiclass": {"estimator_type": "xgboost",  "params": {}},
    "multilabel": {"estimator_type": "xgboost",  "params": {}},
}
# CONDITIONAL_EXTRAS removed; revives in v3.
```

---

## Files (v2)

### New

| Path | Contents |
|---|---|
| `skrec/scorer/mixed_type_multi_target.py` | Scorer, `TargetType` enum, `TargetGroupSpec`, validators, deferred-to-v3 OBSERVED_* rejection |
| `skrec/estimator/classification/_multi_target_protocol.py` | `MultiTargetEstimator` Protocol (shared between joint + independent) |
| `skrec/estimator/classification/_joint_multi_target_base.py` | `JointMultiTargetEncoder` Protocol (internal), per-target heads, shared training utilities, dict-y `_validate_for_fit` |
| `skrec/estimator/classification/_joint_multi_target_encoders.py` | `MLPEncoder`, `TransformerEncoder` |
| `skrec/estimator/classification/joint_multi_target_mlp.py` | `JointMultiTargetMLPEstimator` |
| `skrec/estimator/classification/joint_multi_target_transformer.py` | `JointMultiTargetTransformerEstimator` |
| `skrec/estimator/classification/independent_multi_target.py` | `IndependentMultiTargetEstimator` (new family) |
| `skrec/metrics/multiclass_accuracy.py` | `MulticlassAccuracy` top-1 metric (new `BaseMulticlassMetric` subclass) |
| `skrec/dataset/required_schemas/mixed_type_multi_target_schema_training.yaml` | Schema: `USER_ID: str` |
| `tests/test_mixed_type_multi_target_scorer.py` | Scorer-level tests |
| `tests/test_mixed_type_multi_target_evaluation.py` | Evaluation tests (per-type dispatch, cross-type rejection, escape hatch, logged_rewards validation, `MulticlassAccuracy` correctness) |
| `tests/test_joint_multi_target_mlp.py` | Joint MLP tests |
| `tests/test_joint_multi_target_transformer.py` | Joint Transformer tests |
| `tests/test_independent_multi_target.py` | Independent family tests |
| `examples/mixed_type_multi_target/notebook.ipynb` | End-to-end demo (auto-fetches synthetic data; all three families compared; committed with executed outputs) |
| `docs/user-guide/decision-rule.md` | "When to use which scorer" decision tree; calls out joint vs independent tradeoff |

### Modified

| Path | Change |
|---|---|
| `skrec/scorer/__init__.py` | Export `MixedTypeMultiTargetScorer`, `TargetType`, `TargetGroupSpec` |
| `skrec/estimator/classification/__init__.py` | Export `MultiTargetEstimator`, 3 new estimator classes |
| `skrec/dataset/interactions_dataset.py` | Add `InteractionMixedTypeMultiTargetDataset` |
| `skrec/orchestrator/factory.py` | (1) `SCORER_TYPES += ("mixed_type_multi_target",)` (2) `_TABULAR_SCORER_TYPES`/`_EMBEDDING_INCOMPATIBLE_SCORERS` updated (3) New `ScorerConfig` TypedDict with `target_specs` field; new `MultiTargetConfig` TypedDict with `mode`/`params`/`independent` (4) `RecommenderConfig` adds `scorer_config: ScorerConfig` field (5) `create_estimator(..., target_specs=None)`: `ml_task="multi_target"` branch routing on `multi_target.mode`; for `independent`, composes sub-estimators from defaults+per_target spec and validates type compatibility (6) `create_scorer(..., target_specs=None)`: passes target_specs into `MixedTypeMultiTargetScorer` (7) `create_recommender_pipeline`: reads `target_specs` from `config["scorer_config"]["target_specs"]` once, passes to both factories (8) Cross-cutting validation: `scorer_type="mixed_type_multi_target"` requires non-empty `scorer_config.target_specs`; `mode="independent"` requires either `independent.defaults` covering all target types in use OR `independent.per_target` covering every target (9) `capability_matrix()` adds `"multi_target_model_types": ("joint_mlp", "joint_transformer", "independent")` |
| `skrec/orchestrator/__init__.py` | Export `ScorerConfig`, `MultiTargetConfig` |
| `skrec/recommender/ranking/ranking_recommender.py` | (a) **Line 174–176 guard**: add `MixedTypeMultiTargetScorer` to the `users != None` rejection. (b) **After line 183**: parallel `elif isinstance(self.scorer, MixedTypeMultiTargetScorer):` block that warns top_k is ignored and returns `predict_targets(...)`. Do NOT lump into the existing line 177 conditional. (c) **Around line 583**: add `if isinstance(self.scorer, MixedTypeMultiTargetScorer):` branch before the existing `MultioutputScorer` branch in `evaluate()`. New helper `_evaluate_mixed_type_multi_target` (per-`TargetType` dispatch, returns `Dict[str, float]`, rejects ranking metrics and non-SIMPLE evaluators). Updated `evaluate()` typed overloads cover the new return shape and the optional `metric_type: Dict[str, RecommenderMetricType]` per-target form |
| `skrec/recommender/base_recommender.py` | Single change: add `elif isinstance(self.scorer, MixedTypeMultiTargetScorer): return self.scorer.score_fast(features)[list(active_item_names)]` between line 384 (MultioutputScorer arm) and line 386 (`_score_fast_np` fallback). Dispatch order matters. **No OBSERVED_* preservation logic added in v2** — deferred to v3 |
| `skrec/metrics/datatypes.py` | Add `MULTICLASS_ACCURACY` to `RecommenderMetricType` enum |
| `skrec/metrics/factory.py` | Register `MulticlassAccuracy` in the metric factory |
| `skrec/metrics/base_metric.py` | Add `BaseMulticlassMetric` base class (parallels `BaseClassificationMetric`; documents `(n, K)` scores + `(n,)` class-label rewards; ignores ranks/top_k) |
| `skrec/metrics/__init__.py` | Export `MulticlassAccuracy`, `BaseMulticlassMetric` |
| `tests/test_orchestrator_factory.py` | Update `test_capability_matrix_*`; add tests for `ml_task="multi_target"` happy paths and validation rejections for all 3 modes; add `MULTICLASS_ACCURACY` to the metric enum coverage tests |
| `docs/user-guide/scorers.md` | New section: `MixedTypeMultiTargetScorer`. Required sub-sections enumerated in "Documentation completeness" below |
| `docs/user-guide/capability-matrix.md` | New row in scorer × estimator table; new row in `recommend()` and `recommend_online()` tables; new `evaluate()` row; new `MULTICLASS_ACCURACY` × scorer compatibility table |
| `docs/recommender-types/ranking.md` | New sub-section "Mixed-type multi-target evaluation" describing the new pairing and the `Dict[str, float]` evaluate return |
| `docs/advanced/orchestration.md` | (a) Update `capability_matrix()` example output. (b) New sub-section "Multi-target capabilities" walking through the new published keys (`target_types`, `target_type_metric_compat`, `independent_target_compat`, `scorer_supports_observed_conditioning`, `scorer_config_keys["mixed_type_multi_target"]`) with example output and the agent-consumption pattern |
| `mkdocs.yml` | Insert `- Decision Rule: user-guide/decision-rule.md` in User Guide nav (after Scorers entry at [mkdocs.yml:55](mkdocs.yml#L55)) |
| `README.md` | Add bullet to `### Scorers` section listing `MixedTypeMultiTargetScorer` (one line; link to the user-guide section) |
| `CONTRIBUTING.md` | New sub-section "Adding a new multi-target estimator family" pointing at the `MultiTargetEstimator` Protocol pattern and showing the minimal class shape required |

### Untouched (delta from v1)

The following v1 modifications are **not in v2** (each becomes part of v3):

- `skrec/constants.py` — `OBSERVED_PREFIX = "OBSERVED_"` not added in v2
- `skrec/scorer/base_scorer.py` — `preserved_inference_columns()` hook not added in v2
- `skrec/recommender/base_recommender.py` — schema-apply column preservation not added in v2 (single dispatch change only, see table above)

### Untouched (same as v1)

- `skrec/recommender/training_coordinator.py` — dict-y propagates unchanged
- `pyproject.toml` — torch already in `[torch]` extra

---

## Build order (v2)

1. Schema YAML + `InteractionMixedTypeMultiTargetDataset`.
2. `TargetType` enum, `TargetGroupSpec`, validators (in scorer file).
3. `_multi_target_protocol.py` — `MultiTargetEstimator` Protocol.
4. `MixedTypeMultiTargetScorer` skeleton — `__init__` (Protocol-check estimator), `_validate_interactions` (training), `_validate_inference_interactions` (rejects OBSERVED_*, names v3), `process_datasets`, target/feature splitters. Stub `score_items` / `predict_targets` / `score_fast` until estimators exist.
5. `_joint_multi_target_base.py` — Encoder Protocol (internal, `label_input_dim=0`), per-target heads, shared training utilities, dict-y `_validate_for_fit`.
6. `_joint_multi_target_encoders.py` — `MLPEncoder`, `TransformerEncoder`.
7. `joint_multi_target_mlp.py` — `JointMultiTargetMLPEstimator` (implements `MultiTargetEstimator` Protocol).
8. `joint_multi_target_transformer.py` — `JointMultiTargetTransformerEstimator`.
9. `independent_multi_target.py` — `IndependentMultiTargetEstimator` (implements `MultiTargetEstimator` Protocol). Handles multilabel fan-out, direct-construction validation, and dict-y fit loop.
10. Wire scorer's `score_items` / `predict_targets` / `score_fast` to estimator dict outputs (post-process per target type into wide DataFrames).
11. Factory wiring: `SCORER_TYPES`, sets, new `ScorerConfig` and `MultiTargetConfig` TypedDicts, `RecommenderConfig.scorer_config` field, `create_estimator(..., target_specs=...)` branch with `mode` dispatch (including independent's defaults+per_target composition), `create_scorer(..., target_specs=...)` branch, `create_recommender_pipeline` reads `scorer_config["target_specs"]` and threads through, cross-cutting validation, `capability_matrix()` extension, exports.
12. `RankingRecommender` extension: line 174-176 guard adds new scorer; new `elif` branch after line 183 returns `predict_targets`.
13. `BaseRecommender.recommend_online` extension: new `elif` branch between existing `MultioutputScorer` arm (line 384) and `_score_fast_np` call (line 386). Dispatch order matters.
14. **Evaluation foundation:** `BaseMulticlassMetric` in `skrec/metrics/base_metric.py`; `MulticlassAccuracy` in `skrec/metrics/multiclass_accuracy.py`; `MULTICLASS_ACCURACY` enum value in `datatypes.py`; factory registration; exports.
15. **Scorer `score_per_target` escape hatch** in `MixedTypeMultiTargetScorer` (independent of `evaluate()` integration; can be exercised standalone in tests).
16. **`RankingRecommender.evaluate()` integration:** new `_evaluate_mixed_type_multi_target` branch + helper, mirroring the structure of `_evaluate_multioutput`. Per-`TargetType` metric dispatch, ranking-metric rejection, SIMPLE-only restriction, `logged_rewards` per-column type validation, support for both single-`metric_type` broadcast and per-target-keyed `Dict[str, RecommenderMetricType]` form.
17. Tests in order: scorer → joint MLP → joint Transformer → independent → factory → **evaluation**.
18. Verification gates (below).
19. Docs and example notebook (notebook must include an evaluation cell that shows `Dict[str, float]` output across all four target types).

---

## Tests

### `tests/test_mixed_type_multi_target_scorer.py`

1. Type validation: target declared `BINARY` but column has `{0, 1, 2}` → clean error.
2. `target_specs` validation: multiclass column with 1 unique value → error; multilabel group with 0 columns → error; orphan group member columns → error.
3. Capability rejection: scorer init with embedding estimator → `TypeError` (Protocol check).
4. Capability rejection: scorer init with non-family estimator (e.g., `XGBClassifierEstimator`) → `TypeError`.
5. Output column convention for `score_items`: every target type produces documented column names. Run once per estimator family (parametrized).
6. Output column convention for `predict_targets`: every target type produces one-column-per-target. Run once per estimator family.
7. `score_fast` rejects `features.shape[0] != 1`.
8. **OBSERVED_\* deferred-to-v3 rejection**: any `OBSERVED_*` column at inference (whether orphan or matching a target) → `NotImplementedError` with message naming v3. Covers both joint and independent estimators.
9. **OBSERVED_\* namespace-collision feature column** → same deferred-to-v3 error.
10. **Target-specs consistency check**: scorer with `target_specs={"A": BINARY}` and estimator with `target_specs={"B": BINARY}` → clean error at init.
11. **Dataset schema enforcement** (`test_dataset_schema_enforced`): `InteractionMixedTypeMultiTargetDataset` constructs successfully on a happy frame; rejects a frame missing `USER_ID`.
12. **`score_fast` ndarray rejection** (`test_score_fast_rejects_ndarray_input`): passing `np.ndarray` instead of `DataFrame` → clean error naming the expected type.
13. **`score_fast` column-order invariance** (`test_score_fast_column_order_invariant`): swapping feature column order → identical wide output.
14. **`score_fast` extra-column rejection** (`test_score_fast_rejects_extra_columns`): a feature column not in `feature_names` → clean error.
15. **`recommend()` top_k warning emission** (`test_recommend_top_k_warning_emitted`): passing `top_k=10` emits a `UserWarning` whose message names `MixedTypeMultiTargetScorer` and explains that top_k is ignored.
16. **`recommend()` users-kwarg rejection** (`test_recommend_rejects_users_kwarg`): passing a non-`None` `users` argument → clean error before any scoring runs (asserts the line 174–176 guard fires).
17. **`recommend_online` dispatch order** (`test_recommend_online_dispatch_uses_new_arm_not_score_fast_np`): mock `_score_fast_np` to raise; assert `recommend_online` on a `MixedTypeMultiTargetScorer`-wrapping recommender does NOT hit the mock — i.e., the new `elif` arm captures dispatch before fallback.
18. **Empty `target_specs` rejected** (`test_empty_target_specs_rejected`): scorer init with `target_specs={}` → clean error.
19. **Single-target `target_specs` round-trip** (`test_single_target_spec_round_trip`): `target_specs={"foo": BINARY}` runs end-to-end through `score_items` / `predict_targets`; output is a 2-column / 1-column wide frame respectively.
20. **Iteration-order independence** (`test_output_column_order_independent_of_target_specs_dict_order`): two scorers with same `target_specs` content but different insertion order produce the same wide-output column **set** (and document whether order is fixed via canonical sort or is itself insertion-order — pin one).
21. **Multilabel group single-member** (`test_multilabel_group_single_member`): `TargetGroupSpec(type=MULTILABEL, columns=["ITEM_x"])` (one member) → fan-out still produces a `ITEM_x_0` / `ITEM_x_1` pair; no degenerate behavior.
22. **Multilabel group zero-member rejection** (`test_multilabel_group_zero_members_rejected`): empty `columns=[]` → clean error at scorer init (parallel to test #2 but explicit).
23. **Multilabel group / simple-target key collision** (`test_target_specs_group_key_collides_with_simple_target_rejected`): `target_specs={"foo": BINARY, "foo": TargetGroupSpec(...)}` → since Python dict-shadowing silently drops the first entry, the scorer's init scans `target_specs` values for the collision pattern (e.g., a group key matching any declared member column or simple-target name) and rejects explicitly with a message naming the collision.
24. **`MultiTargetEstimator` Protocol isinstance strictness** (`test_multi_target_protocol_isinstance_strictness`): positive — `isinstance(est, MultiTargetEstimator)` is True for each of the 3 family instances. Negative — a class that has `target_specs`/`fit` but NOT `predict_proba_dict` / `predict_targets_dict` fails the isinstance check (asserts the Protocol attribute set is complete and not accidentally widened).

(v1 tests 10, 11, 12 dropped — all OBSERVED_*-related, deferred to v3.)

### `tests/test_joint_multi_target_mlp.py` (vanilla joint MLP)

1. Happy path: train on synthetic with all 4 target types, `epochs=2`; no errors; output shapes correct.
2. Per-type metric sanity (separable synthetic): binary AUC > 0.6, regression RMSE drops over epochs, multiclass top-1 > random + 0.1, multilabel per-dim AUC > 0.6.
3. Pickle round-trip: save → load → predict; outputs match.
4. Determinism: train twice with same seed → identical predictions. The estimator's `seed` argument plumbs through to `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `np.random.seed`, AND a seeded `torch.Generator` driving the `torch.randperm` batcher. v2 does NOT use a `DataLoader` — the batcher is a flat `torch.randperm`-based shuffler over the in-memory training tensors, which sidesteps DataLoader worker-fork RNG quirks. GPU determinism is NOT guaranteed (warn_only lets non-deterministic CUDA ops degrade with a warning). Test pins CPU; calls out the GPU caveat in the docstring.
5. Loss decreases monotonically over 5 epochs on synthetic.
6. `predict_proba_dict` returns the documented per-target shapes (multilabel members are fanned out).
7. `predict_targets_dict` returns the documented per-target shapes (multilabel members are fanned out).

(v1 tests 3, 4 — conditional/masked-related — dropped.)

### `tests/test_joint_multi_target_transformer.py`

Mirror tests 1–7 from joint MLP, parametrized for transformer architecture.

### `tests/test_independent_multi_target.py` (new family)

1. **Happy path config-driven**: build via factory with defaults covering all 4 target types; train on synthetic; output shapes correct.
2. **Happy path direct-construction**: build with `estimators=dict[...]`; train; output shapes correct.
3. **Per-target metric sanity** (matches joint MLP test 2 thresholds, separable synthetic): binary AUC > 0.6, regression RMSE drops, multiclass top-1 > random + 0.1, multilabel per-member AUC > 0.6.
4. **Per-target estimator-type compatibility validation**:
   - regression target + classifier-only sub-estimator → clean error at factory time AND at direct-construction `__init__`.
   - binary target + regressor sub-estimator → clean error.
   - multiclass target + LogReg with `solver="liblinear"` (binary-only) → clean error.
5. **Per-target overrides**: factory config has `defaults.regression=lightgbm` but `per_target.ITEM_revenue.estimator_type=xgboost`; assert the resulting sub-estimator for `ITEM_revenue` is XGB, others are LightGBM.
6. **Multilabel fan-out**: a multilabel group with 3 members → 3 distinct binary sub-estimator instances; each can be overridden individually via member-keyed `per_target` entries.
7. **Direct-construction validation**: `estimators` missing a target member → clean error naming the missing key. Includes the group-key-vs-member-key distinction (group key in `estimators` instead of members → error).
8. **Mixed sub-estimator types in one model**: one target uses LightGBM, another XGBoost, another LogReg; assert all fit and predict correctly. Sanity check that per-target metrics differ meaningfully from a uniform-XGB baseline (proves the routing is real, not silent fallback).
9. **Pickle round-trip across heterogeneous sub-estimators**: parametrized across at least 3 (target_type, estimator_type) combinations — `(BINARY, xgboost)`, `(REGRESSION, lightgbm)`, `(BINARY, logreg)`, `(MULTICLASS, sklearn)`. Save → load → predict; assert each sub-estimator's output (not just the aggregated wide frame) matches pre-pickle. Catches booster-handle / tensor-state pickle quirks that disappear in homogeneous tests.
10. **Determinism**: with each sub-estimator's random_state set, train twice → identical predictions.
11. **`predict_proba_dict` / `predict_targets_dict` shapes**: same documented shapes as joint family; multilabel members fanned out under member-column keys.
12. **Loss-balance N/A**: independent estimators have no shared loss → skip joint-loss-balance test. Document the absence explicitly in the test docstring so it's not mistaken for a coverage gap.
13. **Partial-fit failure cleanup** (`test_partial_fit_failure_cleans_state`): mock sub-estimator at index 2-of-5 to raise during `.fit`; assert `.predict_proba_dict` / `.predict_targets_dict` subsequently raise a clear "not fitted" error rather than producing partial output. Pins the no-half-fit-state invariant.
14. **Unused defaults silently ignored** (`test_independent_unused_defaults_silently_ignored`): `defaults` covers all 4 target types but `target_specs` declares only BINARY + REGRESSION; assert factory constructs only the 2 sub-estimators used and does NOT instantiate or warn about the unused MULTICLASS/MULTILABEL defaults. (Pin the chosen semantics — "silently ignored" is the proposed default; if the team prefers a warning, flip this assertion.)

### `tests/test_orchestrator_factory.py` updates

1. `test_capability_matrix_has_expected_keys` updated for new `multi_target_model_types` key (3 entries).
2. `test_capability_matrix_reflects_private_maps` updated.
3. New: `test_create_estimator_multi_target_joint_mlp` end-to-end.
4. New: `test_create_estimator_multi_target_joint_transformer` end-to-end.
5. New: `test_create_estimator_multi_target_independent_with_defaults` (no per_target overrides).
6. New: `test_create_estimator_multi_target_independent_with_per_target_overrides`.
7. New: `test_create_recommender_pipeline_multi_target_*` end-to-end happy path for all 3 modes with `RecommenderConfig`.
8. New: validation rejections:
   - `multi_target.mode` invalid → error.
   - `mode="independent"` + sub-estimator-type incompatible with declared target type → error.
   - `target_specs` missing → error.
   - `scorer_config.target_specs` and `estimator.target_specs` inconsistent → error at scorer init.
9. `RecommenderMetricType.MULTICLASS_ACCURACY` round-trips through metric factory (`create(MULTICLASS_ACCURACY)` returns a `BaseMulticlassMetric` instance).
10. **`capability_matrix()` publishes `scorer_config_keys["mixed_type_multi_target"]`** (`test_capability_matrix_publishes_mixed_type_scorer_config_keys`): asserts the key exists, is a non-empty tuple, and contains `"target_specs"`. This is the agent-consumed contract; missing it forces the agent into the legacy fallback path.
11. **`capability_matrix()` publishes `target_types`, `target_type_metric_compat`, `independent_target_compat`, `scorer_supports_observed_conditioning`** (`test_capability_matrix_publishes_multi_target_keys`): asserts each new key is present with the documented shape. Compares `target_type_metric_compat` and `independent_target_compat` against the in-code source-of-truth constants (`_TARGET_TYPE_TO_METRICS`, `_COMPATIBLE_SUB_ESTIMATORS`) — drift detector.

### `tests/test_mixed_type_multi_target_evaluation.py` (new — evaluation coverage)

End-to-end via `RankingRecommender.evaluate()` unless noted otherwise.

1. **`MulticlassAccuracy` numerical correctness** — Against sklearn's `accuracy_score(y_true, argmax(y_score, axis=1))` on a synthetic K=4 multiclass target. Direct metric-class test (no recommender).
2. **Per-`TargetType` metric dispatch (single `metric_type` broadcast)** — Build a scorer with one of each target type; evaluate with `metric_type=ROC_AUC`. Assert returns `Dict[str, float]` with entries for binary + multilabel members only; regression and multiclass targets are absent (or surface as a rejection — see test #3 for the choice; v2 rejects with a clear error rather than silently dropping).
3. **Cross-type rejection (broadcast form)** — `metric_type=ROC_AUC` with a regression target in `target_specs` → `ValueError` naming the offending target and pointing at compatible metrics. Same for `RMSE` with a binary target, `MULTICLASS_ACCURACY` with anything other than multiclass.
4. **Per-target-keyed `Dict[str, RecommenderMetricType]` form** — Pass `metric_type={"ITEM_clicked": ROC_AUC, "ITEM_revenue": RMSE, "ITEM_action": MULTICLASS_ACCURACY, "ITEM_email_open": ROC_AUC, "ITEM_app_open": ROC_AUC}`. Assert each target gets the correct metric and the return is `Dict[str, float]` covering all keys.
5. **Per-target dict with type mismatch** — `metric_type={"ITEM_revenue": ROC_AUC, ...}` (incompatible) → `ValueError` naming the target. (Validates the per-target form has the same compatibility checks as the broadcast form.)
6. **Per-target dict with missing target** — Caller passes a dict that omits some declared targets → `ValueError` naming the missing targets. Symmetric and explicit; do not silently skip.
7. **Per-target dict with unknown target name** — Caller passes a dict with a key not in `target_specs` → `ValueError` naming the unknown key.
8. **Ranking-metric rejection** — `metric_type=NDCG_AT_K` → `ValueError` with the "per-target prediction scorer, ranking metrics inapplicable" message pointing at `predict_targets` / `score_per_target`.
9. **Non-SIMPLE evaluator rejection** — `eval_type=RecommenderEvaluatorType.IPS` → `ValueError` mirroring `_evaluate_multioutput`'s SIMPLE-only error.
10. **`logged_rewards` per-column type validation** — Binary column with `[0, 0.5, 1]` (continuous-ish) → `ValueError` naming the column and declared `BINARY` type. Regression column with NaNs → accepted. Multiclass column with a class label not seen at training → `ValueError`.
11. **`logged_rewards` shape mismatch** — `logged_rewards.shape[1]` ≠ number of (fanned-out) targets → `ValueError`. `logged_rewards.shape[0]` ≠ `interactions` rows → `ValueError`.
12. **Multilabel-member columns appear separately in `logged_rewards`** — Build a scorer with a multilabel group of 3 members; assert `logged_rewards` expects 3 columns (one per member), not 1 (the group key). Symmetric with `predict_targets` output.
13. **`score_per_target` happy path** — User-supplied callables (sklearn `log_loss`, `mean_absolute_percentage_error`, `f1_score`); assert returned `Dict[str, float]` matches direct sklearn computation on the same inputs.
14. **`score_per_target` name-override beats type-default** — Dict with both `TargetType.BINARY: <log_loss>` and `"ITEM_clicked": <custom_callable>`; assert `ITEM_clicked` uses the custom callable and other binary targets use `log_loss`.
15. **`score_per_target` missing callable for a target** — Target with neither name override nor type-keyed default → `KeyError` naming the target.
16. **Joint vs independent return-shape AND dispatch-key equivalence** — Train both a `JointMultiTargetMLPEstimator`-backed scorer and an `IndependentMultiTargetEstimator`-backed scorer on the same synthetic data with the same `target_specs`. Evaluate both with the same per-target metric dict. Assert returned `Dict[str, float]` has (a) the **same key set** and (b) the **same `RecommenderMetricType` per key** as resolved by the dispatch table (introspect `_TARGET_TYPE_TO_METRICS` and verify both runs resolved each key to the same metric type — protects against silent dispatch-table divergence between code paths).
17. **`logged_rewards` NaN per-target as ignore-mask** (`test_logged_rewards_nan_per_target_treated_as_ignore_mask`): regression column with 20% NaN ground truth → metric computed on the non-NaN subset only; assert against direct sklearn computation on the same masked subset.
18. **`logged_rewards` all-NaN target** (`test_logged_rewards_all_nan_for_target_returns_nan`): a target column with 100% NaN ground truth → returned `Dict[str, float]` entry for that target is `float("nan")`, NOT a sklearn explosion or a 0.0. Document the contract.
19. **`score_per_target` y_true column-set mismatch** (`test_score_per_target_y_true_column_set_mismatch`): `y_true` missing a declared target column → `ValueError` naming the missing column. `y_true` with an extra column not in `target_specs` → `ValueError` naming the unknown column.
20. **`score_per_target` y_true column reorder aligns by name** (`test_score_per_target_y_true_column_reordered_aligns_by_name`): pass `y_true` with columns in reverse insertion order → results are identical to original order (proves alignment is by column name, not position).

---

## Verification gates (run before declaring done)

1. **Protocol gate** — All 3 estimator classes satisfy `MultiTargetEstimator` via `runtime_checkable` `isinstance`. Scorer accepts each at init.
2. **Encoder Protocol gate** — Both joint encoders satisfy the (internal) Protocol; both produce same `hidden_dim` shape for same input.
3. **Default-sanity gate** — Each of 3 estimators with default params on a synthetic QBO-shaped dataset (1 binary + 1 regression + 1 multiclass + 1 multilabel-group, ~5K rows, 3 epochs for joint, default n_estimators for independent). Each per-target metric must beat random by a clear margin. If any fails, defaults are wrong; fix before merge.
4. **Loss-curve sanity (joint only)** — Per-epoch loss plot (training + validation) per joint estimator in the example notebook. Visual inspection passes.
5. **Family-equivalence smoke** — On a tightly-correlated-targets synthetic where joint is expected to win, assert joint MLP beats independent on at least one metric (proves joint adds value). On a decorrelated synthetic where independent should be at least competitive, assert independent within 5% of joint MLP. This isn't a benchmarking guarantee — it's a sanity check that both families are wired up correctly and the joint vs independent decision tree has empirical grounding for the docs.
6. **Evaluation correctness gate** — On the same synthetic from gate 3, run `recommender.evaluate(metric_type={"ITEM_<binary>": ROC_AUC, "ITEM_<regression>": RMSE, "ITEM_<multiclass>": MULTICLASS_ACCURACY, "ITEM_<multilabel_member>": ROC_AUC, ...})`. Independently compute the same metrics via sklearn on the scorer's `predict_targets` / `score_items` output. Assert agreement to floating-point precision (or documented tolerance). Run for all 3 estimator families. Catches per-target column slicing bugs, metric routing mismatches, and `logged_rewards` reordering errors.
7. **Dispatch-table consistency gate** — Parse the per-`TargetType` metric dispatch table out of `docs/user-guide/decision-rule.md` (the human-canonical source); compare to the in-code constant `_TARGET_TYPE_TO_METRICS` and to `capability_matrix()["target_type_metric_compat"]`. All three must agree exactly. Run as part of the factory test suite. Catches doc drift, copy-paste mistakes, and stale capability-matrix mirrors.
8. **Agent-surface gate** — Build the full `capability_matrix()` dict; assert every key documented in the "New keys published" section is present with the documented shape and is JSON-serializable (so the agent can stream it through tool envelopes without custom encoders). Run as part of the factory test suite.

(v1 gate 2 "Mandatory leakage test" is N/A in v2 — no conditional, no label channel to leak. Revives in v3.)

---

## Capability matrix updates

`docs/user-guide/capability-matrix.md`:

**Scorer × estimator plane** — new row after `MultioutputScorer`:

| **MixedTypeMultiTargetScorer** | **Yes** (only `MultiTargetEstimator` Protocol implementers — joint MLP, joint Transformer, independent) | **No** (raises at init) | **No** |

**`recommend()` table** — note that `MixedTypeMultiTargetScorer` short-circuits to `predict_targets`, top_k ignored. Same pattern as MultioutputScorer.

**`recommend_online()` table** — new row:

| **MixedTypeMultiTargetScorer** | **Partial** | Returns DataFrame of point estimates per target column; `top_k` is ignored. `OBSERVED_*` columns rejected (deferred to v3) |

**`evaluate()` table** — new row:

| **MixedTypeMultiTargetScorer** | **Per-target** | Restricted to `RecommenderEvaluatorType.SIMPLE`. Returns `Dict[str, float]` (always — heterogeneous types can't macro-average). Metric dispatched per declared `TargetType`: binary/multilabel-member → `ROC_AUC` / `PR_AUC`; regression → `RMSE` / `MAE`; multiclass → `MULTICLASS_ACCURACY` (new v2 metric). Ranking metrics rejected. Caller can pass a single `metric_type` (broadcast) or a `Dict[str, RecommenderMetricType]` (per-target overrides). For metrics outside the named set, use `scorer.score_per_target(metric_callables=...)` |

**Metric × scorer table** — add the new `MULTICLASS_ACCURACY` column:

| Scorer | MULTICLASS_ACCURACY |
|---|---|
| UniversalScorer / IndependentScorer / MultioutputScorer / MulticlassScorer / SequentialScorer / HierarchicalScorer | ❌ (binary/ranking shapes only) |
| **MixedTypeMultiTargetScorer** | ✅ (per multiclass target) |

### New keys published by `capability_matrix()`

The agent (scikit-rec-agent) consumes `capability_matrix()` for pre-flight validation. v2 publishes four new keys so the agent can validate `target_specs`, metric selection, and sub-estimator type choices without hardcoding mirrors of in-scikit-rec dispatch tables (drift hazard otherwise).

```python
{
    # Existing keys unchanged.
    ...
    # NEW in v2:
    "target_types": ("binary", "regression", "multiclass", "multilabel"),
    "multi_target_model_types": ("joint_mlp", "joint_transformer", "independent"),
    "target_type_metric_compat": {
        "binary":     ("roc_auc", "pr_auc"),
        "regression": ("rmse", "mae"),
        "multiclass": ("multiclass_accuracy",),
        "multilabel": ("roc_auc", "pr_auc"),
    },
    "independent_target_compat": {
        "binary":     ("xgboost", "lightgbm", "logreg", "sklearn"),
        "regression": ("xgboost", "lightgbm", "sklearn"),
        "multiclass": ("xgboost", "lightgbm", "logreg"),
        "multilabel": ("xgboost", "lightgbm", "logreg", "sklearn"),
    },
    "scorer_supports_observed_conditioning": (),  # empty in v2; populated with scorer names in v3
    "scorer_config_keys": {
        # Existing entries unchanged.
        ...,
        "mixed_type_multi_target": ("target_specs",),
    },
}
```

All four new keys are mirrored from a single in-code source-of-truth constant per key (`_TARGET_TYPE_TO_METRICS`, `_COMPATIBLE_SUB_ESTIMATORS`, etc.). Factory tests #11 (above) assert mirror consistency.

---

## Documentation completeness

This section enumerates the doc surfaces the v2 plan touches and the minimum contents each must carry. Without explicit outlines, doc-writers (or future-us) skip the parts that bridge to existing scorers and to scikit-rec-agent.

### File-level design docstrings (required on every new code file)

The in-repo template is `skrec/scorer/multioutput.py` lines 1–31 — a 20–50-line header comment with vocabulary notes, design choices, and contract decisions. v2 requires equivalent headers on at least:

- **`skrec/scorer/mixed_type_multi_target.py`** — Cover: why one scorer for three families (Protocol-based polymorphism); the no-`preserved_inference_columns`-hook-in-v2 decision and where the v3 add lands; the two-validators-not-one-flag rationale (training vs inference); dict-y rationale; the `target_specs` consistency check between scorer and estimator.
- **`skrec/estimator/classification/_multi_target_protocol.py`** — Cover: why a runtime-checkable Protocol (so the scorer can validate at init without inheritance); the strict attribute set required (`target_specs`, `fit`, `predict_proba_dict`, `predict_targets_dict`) and why no `predict_with_observed` here (deferred to v3 — would widen the contract prematurely).
- **`skrec/estimator/classification/independent_multi_target.py`** — Cover: why a single concrete class (not a family) for the independent path; multilabel fan-out semantics and the group-key-as-metadata-only invariant; the two construction paths (factory vs direct) and the validation symmetry between them; the partial-fit-no-half-state invariant.
- **`skrec/estimator/classification/_joint_multi_target_base.py`** — Cover: encoder Protocol's `label_input_dim=0` v3 hook (explicit `# Note: v3 hook` comment naming the v3 plan section that activates it); per-target head architecture and the dict-y `_validate_for_fit` override.
- **`skrec/metrics/multiclass_accuracy.py`** — Cover: top-1 contract; what `recommendation_scores` and `modified_rewards` shapes mean for this metric (different from `BaseClassificationMetric` defaults); why this is the only multiclass metric in v2 (escape hatch via `score_per_target`).

### In-file `# Note: v3 hook` comments (required)

Every v3 hook point in v2 carries an inline `# Note: v3 hook — <reason>` comment naming the v3 plan section that activates it. Required at minimum:

- Encoder Protocol's `label_input_dim: int = 0` default.
- Scorer's OBSERVED_* rejection sites in `_validate_inference_interactions` and `score_fast`.
- `scorer_supports_observed_conditioning: ()` in `capability_matrix()`.
- The trimmed `_validate_inference_interactions` body (note that v3 flips reject-to-delegate).

Without these inline markers, the v3 refactor will silently re-implement the wrong things or miss the hooks entirely.

### Docstring discipline (required)

Every public class and method on the new files MUST have a numpy-style docstring with Args / Returns / Raises blocks matching the existing repo style (e.g., `skrec/scorer/multioutput.py` `score_items`, `predict_classes`). The agent introspects via `inspect.signature` and `__doc__`; missing or thin docstrings degrade the agent's error-diagnosis and config-suggestion quality.

### `docs/user-guide/scorers.md` new section — required sub-sections

1. **Contract overview** — input shape table, output shape table (cross-reference the "Data shapes" tables in this plan), one-paragraph "what this scorer is for."
2. **`target_specs` syntax** — concrete examples for all 4 `TargetType` values plus a `TargetGroupSpec` (multilabel).
3. **Estimator family pairing matrix** — which of joint MLP / joint Transformer / independent to pick, with one sentence per row on the trade-off.
4. **Output column conventions** — the wide-frame tables from "Data shapes" lifted into the user guide.
5. **Evaluation contract** — `Dict[str, float]` return; the per-`TargetType` dispatch table; `score_per_target` escape hatch with one example (log-loss).
6. **v3 deferral callout** — explicit "if you pass `OBSERVED_*` columns today, you get this error; here's the v3 roadmap link."
7. **Comparison to `MultioutputScorer`** — when to stay on the old scorer (all-binary homogeneous; macro-averaged scalar wanted) vs switch (heterogeneous types; per-target dict needed). A two-column table is enough.

### `docs/user-guide/decision-rule.md` (new file) — required sub-sections

1. **Decision tree visual** — text-rendered tree from data-shape question through to recommended scorer + estimator family.
2. **Per-`TargetType` metric dispatch table** — this is the **single source of truth** for human readers (the in-code constant is the source for the runtime). All other docs link here.
3. **When to add a new target type vs. encode-as-existing** — guidance on the binary-vs-multiclass and regression-vs-multiclass edges.
4. **v3 OBSERVED_* preview** — "if you need real-time-label conditioning, here's what's coming and the rough timeline."

### Migration / comparison guide

Either as a final sub-section of `scorers.md` or a short `docs/user-guide/multioutput-vs-mixed-type.md`. Covers: contract differences (homogeneous binary vs heterogeneous typed); evaluation return shape (`float` macro vs `Dict[str, float]`); when each is the right call.

### DRY for the per-`TargetType` metric dispatch table

The table appears in: (1) `_TARGET_TYPE_TO_METRICS` module-level constant (runtime dispatch in `_evaluate_mixed_type_multi_target`), (2) `capability_matrix()["target_type_metric_compat"]` (agent-readable), (3) `docs/user-guide/decision-rule.md` (human-readable canonical), (4) `docs/user-guide/scorers.md` (referenced), (5) `docs/user-guide/capability-matrix.md` (referenced). Required invariants:

- The constant is the single source of truth in code.
- `capability_matrix()` reads it directly (not a copy).
- A unit test asserts the decision-rule doc table (parsed) matches the constant exactly.
- `scorers.md` and `capability-matrix.md` reference the decision-rule doc table by link rather than restating it (or use mkdocs-include if available).

---

## Orchestrator surface for scikit-rec-agent

scikit-rec-agent (at `/Users/ssankararam/Shankar/Personal/RecSys/scikit-rec-agent`) consumes scikit-rec via the orchestrator factory and the `capability_matrix()` introspection surface. To make `MixedTypeMultiTargetScorer` + the new estimator families + evaluation contract first-class in the agent without forcing the agent to import deep from `skrec.scorer.*`, v2 must publish the following surfaces:

### Re-exports from `skrec.orchestrator`

```python
# skrec/orchestrator/__init__.py — new exports in v2:
from skrec.orchestrator.factory import (
    ScorerConfig,          # NEW (TypedDict)
    MultiTargetConfig,     # NEW (TypedDict)
)
from skrec.scorer.mixed_type_multi_target import (
    TargetType,            # re-export — NEW
    TargetGroupSpec,       # re-export — NEW
)
from skrec.estimator.classification import (
    MultiTargetEstimator,  # re-export — NEW (Protocol)
)
```

The agent imports from `skrec.orchestrator` exclusively (matches the existing pattern; `skrec.scorer.*` is not part of the agent's import surface). Re-exporting from the orchestrator package keeps the dependency line clean.

### `capability_matrix()` keys consumed by the agent

See the "New keys published by `capability_matrix()`" subsection above for the full list. The agent uses them as follows:

| Key | Agent consumption point |
|---|---|
| `target_types` | `tools/datasets.py` validation of `target_specs` values; `prompts/_capability.py` enum-tuple injection into system prompt |
| `multi_target_model_types` | `tools/training.py` pre-flight validation of `multi_target.mode`; `tools/sweep.py` `_AUTO_SWEEPS` table population |
| `target_type_metric_compat` | `tools/evaluation.py` pre-flight validation of metric/target compatibility (replaces hardcoded binary/regression partitions for this scorer) |
| `independent_target_compat` | `tools/training.py` pre-flight validation of `multi_target.independent.{defaults,per_target}.*.estimator_type` per declared target type |
| `scorer_supports_observed_conditioning` | `tools/training.py` pre-flight rejection of `OBSERVED_*` columns in v2; forward-compat flag for v3 |
| `scorer_config_keys["mixed_type_multi_target"]` | `tools/training.py` validation of the `scorer_config` block keys (existing pattern, extended for this scorer) |

### HPO search-space shape for `mode="independent"`

`skrec/orchestrator/hpo.py` and the agent's `run_hpo` / `sweep_methods` tools need a defined search-space shape for the independent mode. Plan-level decision: **flat keys with dotted-path navigation**:

```python
param_space = {
    "multi_target.independent.per_target.ITEM_revenue.params.n_estimators": [200, 500, 1000],
    "multi_target.independent.per_target.ITEM_revenue.params.learning_rate": [0.01, 0.1],
    "multi_target.independent.per_target.ITEM_clicked.params.max_depth": [3, 5, 7],
    # ... per-target hyperparameters as separate flat keys
}
```

Rationale: flat keys match the existing HPO infrastructure pattern (see `skrec/orchestrator/hpo.py` param-space examples); nested shapes would require HPO refactoring out of scope for v2. Per-target overrides via dotted paths give the agent and end users a uniform way to express "tune this sub-estimator's hyperparameters."

Example HPO config + verification round-trip in `examples/mixed_type_multi_target/notebook.ipynb` (mandatory).

### Dataset contract registration

`skrec/dataset/interactions_dataset.py` adds `InteractionMixedTypeMultiTargetDataset`. The agent's `_CONTRACT_TO_DATASET_TYPE` map and `contract_from_dataframe` helper today recognize `wide_multioutput` for any "1-row-per-user + ≥2 ITEM_*" shape — which would silently misroute mixed-type data to `MultioutputScorer` (which rejects non-binary at fit time).

Plan-level decision: **publish a `contract_from_dataframe` helper in `skrec.orchestrator`** so both scikit-rec and the agent share one detection routine. Signature:

```python
def contract_from_dataframe(
    df: pd.DataFrame,
    target_specs: Optional[dict[str, TargetType | TargetGroupSpec]] = None,
) -> str:
    """Returns one of: 'long_interactions', 'long_with_timestamp',
    'wide_multioutput', 'wide_mixed_type_multi_target', 'multiclass',
    'prebuilt_sequences', 'sessions'.

    Auto-detects shape from dtypes for long/sequence/multiclass paths.
    For wide formats: if `target_specs` is provided AND any value has
    a TargetType other than BINARY, returns 'wide_mixed_type_multi_target';
    otherwise returns 'wide_multioutput'. Without `target_specs`, defaults
    to 'wide_multioutput' (the legacy heuristic).
    """
```

Without this helper, the agent's contract detection silently misroutes mixed-type wide data. Co-locating the helper in `skrec.orchestrator` (not in the agent) means both repos stay aligned as new contracts are added.

---

## Agent-side follow-ups (out of v2 scope; required for first-class agent support)

These changes live in `scikit-rec-agent`, NOT in scikit-rec. v2 plan documents them so the follow-up PR has a clear punch list and so reviewers understand the full picture:

1. **`_scorer_is_mixed_type_multi_target(handle)` detector** in `scikit_rec_agent/tools/evaluation.py` — parallels existing `_scorer_is_classifier_multioutput`. Routes `Dict[str, float]` return shapes to the agent's existing dict-handling branch (already present for `MultioutputScorer per_label=True`).
2. **`MissingDecision` gate adjustment** — the existing MUST-ASK `per_label` gate for `MultioutputScorer` is N/A for the new scorer (the return is always per-target). The detector must short-circuit the gate.
3. **`_build_eval_kwargs_from_validation`** — new branch for `bundle.dataset_type == "interaction_mixed_type_multi_target"` constructing `logged_items` / `logged_rewards` with the multilabel-member fan-out convention.
4. **`_AUTO_SWEEPS["wide_mixed_type_multi_target"]`** — new 3-entry default sweep: joint MLP / joint Transformer / independent-with-default-sub-estimators. Without this entry, `sweep_methods(methods="auto")` errs on mixed-type data.
5. **`profile_data` recognition of "wide with heterogeneous ITEM_* dtypes"** — when detected, the agent recommends `MixedTypeMultiTargetScorer` (joint MLP as default) rather than `MultioutputScorer`. Surfaces via `system.py` `_HEURISTICS` table.
6. **`prompts/_capability.py`** — render the new `capability_matrix()` keys into the system prompt so the LLM knows the new scorer + estimator modes + evaluation contract are available.
7. **`tools/datasets.py` `column_mapping`** — extend to accept a `target_specs` dict so the agent can pass `target_specs` through `create_datasets` for the new contract detection.

All seven are tracked in the v2 plan but ship in a separate scikit-rec-agent PR (cross-repo coordination — the agent PR can land any time after this v2 PR merges).

---

## Open implementation details (not blocking)

- Regression z-score scaler (joint only): per-target `(mean, std)` stored on the estimator, serialized with model state. Independent regressors handle their own scaling internally per sub-estimator.
- Transformer warmup schedule: linear warmup for first `warmup_steps`, then constant. Cosine schedule deferred.
- Per-target loss weighting (joint only): rely on `regression_normalize=True` default. Expose explicit per-target weights only if loss-balance smoke fails.
- TunedEstimator (sklearn-CV-based) wrapping is not supported in v2 for joint family (dict y won't work through sklearn CV). Independent family can in principle wrap each sub-estimator in its existing `Tuned*` cousin, but tuning is a future enhancement — out of scope for v2. HPO for the new family uses `skrec/orchestrator/hpo.py` end-to-end pipeline sweeps with flat dotted-path param keys (see "Orchestrator surface for scikit-rec-agent" → "HPO search-space shape").
- Determinism scope: `seed` plumbing in v2 is CPU + single-process only. The training loop uses `torch.randperm` with a seeded `torch.Generator` over in-memory tensors — no `DataLoader`, no `num_workers` concept (so there's nothing to set). GPU determinism is best-effort via `torch.use_deterministic_algorithms(True, warn_only=True)`; CUDA ops without deterministic implementations degrade to a warning. A future GPU/multi-worker mode would need DataLoader-based batching with `worker_init_fn` + CUDA-specific guards. Test docstring + user-facing docstring on the estimator's `seed` param call out the scope.

---

## Risks watched

1. **Loss balance** under realistic dollar-scale regression (joint only). Mitigation: default-sanity gate.
2. **Transformer training instability** (NaN gradients on first batch). Mitigation: gradient clipping, LR warmup, careful attention init.
3. **`target_specs` validation surface** — being exhaustive at init is far cheaper than failing at training time.
4. **Independent + multilabel** — fan-out loses the group inductive bias. Risk: users pick independent when joint would serve them better. Mitigation: decision-rule doc explicitly calls this out; default-sanity gate uses both families to make the trade-off concrete in the example notebook.
5. **Per-target sub-estimator compatibility** — runtime errors deep inside sub-estimator `.fit` are unfriendly. Mitigation: factory and direct-construction validate the `target_type × estimator_type` compatibility table up-front.
6. **Protocol drift** — joint and independent estimators must agree on `predict_proba_dict` / `predict_targets_dict` output shapes (especially multilabel fan-out keys). Mitigation: a shared parametrized scorer test (test #5/#6) runs the wide-format stitching against each family and asserts identical column layout for the same `target_specs`.
7. **v3 additivity** — risk that v3 conditional refactoring breaks v2 API. Mitigation: encoder Protocol's `label_input_dim` param retained at 0; scorer's `_validate_inference_interactions` already structured to flip from "reject" to "delegate to estimator" when v3 lands; OBSERVED_* error message names v3 so users won't be surprised when behavior changes.
8. **Heterogeneous metric aggregation** — callers used to MultioutputScorer's macro-averaged scalar may want a scalar from this scorer too (e.g., for HPO objective). v2 deliberately does not provide one — there's no honest aggregation across binary AUC + regression RMSE + multiclass accuracy. Mitigation: decision-rule doc + scorer-section doc explicitly walk through the "pick a primary target" and "compose a weighted aggregate via `score_per_target`" patterns. Also: HPO sweeps using this scorer must point `primary_metric` at a single target name, not a scorer-wide aggregate — verify via factory-level HPO config test.
9. **Metric-coverage drift** — the per-`TargetType` → metric dispatch table appears in 5 mirrors (runtime constant, `capability_matrix()`, decision-rule doc, scorers doc, capability-matrix doc). Mitigation: extract the dispatch table to a single module-level constant `_TARGET_TYPE_TO_METRICS` in `mixed_type_multi_target.py`; have `capability_matrix()` read it directly (not copy); have factory tests assert decision-rule doc parses to the same table; have `scorers.md` and `capability-matrix.md` link to the decision-rule doc rather than restate (mkdocs-include where supported).
10. **scikit-rec-agent drift** — agent-side mirrors of skrec internals (contract detection, metric/scorer compatibility, sub-estimator-type compatibility) drift as new scorers and metrics land. Mitigation: every drift-prone surface in v2 is published through `capability_matrix()` (see "New keys published" section); the agent consumes those keys at runtime rather than maintaining a code mirror; a co-located `contract_from_dataframe` helper lives in `skrec.orchestrator` so both repos share one detection routine.
11. **Determinism on non-CPU setups** — joint estimator determinism tests pin CPU + single-process. The training loop uses `torch.randperm` over in-memory tensors (no `DataLoader`, no `num_workers`), so same-seed reproducibility on CPU is fully pinned by `torch.manual_seed` + `torch.use_deterministic_algorithms(True, warn_only=True)` + seeded `torch.Generator` + `np.random.seed`. GPU is best-effort (`warn_only` lets non-deterministic CUDA ops degrade with a warning). Mitigation: test docstring calls out the GPU limitation; user-facing docstring on the estimator's `seed` param links to the PyTorch reproducibility doc.
12. **Partial-fit state in independent estimator** — if sub-estimator k of K raises during `.fit`, the estimator must not be left in a half-fit state that produces partial output on `.predict_*`. Mitigation: `IndependentMultiTargetEstimator.fit` either completes all sub-fits or rolls back to unfit state (no half-state); test #13 in independent tests pins this invariant.

---

## Polish items shipped in v2

1. Worked end-to-end example notebook (per repo memory: auto-fetches data, committed with executed outputs) comparing all 3 families on the same dataset; includes an `evaluate()` cell showing per-target `Dict[str, float]` output and a `score_per_target` cell showing the user-supplied-callable path (log-loss + macro-F1 example).
2. Default hyperparameter sanity test (gate 3).
3. Family-equivalence smoke (gate 5).
4. Pickle round-trip tests for all 3 families.
5. Decision-rule doc at `docs/user-guide/decision-rule.md`, including joint-vs-independent guidance, the "no scalar default" evaluation note, and the "pick a primary target for HPO objective" pattern.

---

## Deferred to v3

A separate plan + PR will add real-time-label conditioning. v3 is purely additive — no v2 API changes. Scope sketch (lifted from v1, adjusted):

- `OBSERVED_PREFIX = "OBSERVED_"` in `skrec/constants.py`.
- Two new joint estimator classes:
  - `ConditionalJointMultiTargetMLPEstimator`
  - `ConditionalJointMultiTargetTransformerEstimator`
- Per-target label encoding (`is_observed` flag + value/one-hot/binary-vector encoding).
- Encoder Protocol's `label_input_dim > 0` path actually used by conditional estimators.
- `_predict_with_observed(X, observed)` on conditional estimators.
- `MixedTypeMultiTargetScorer._validate_inference_interactions` upgraded to OBSERVED-aware: permits NaN, rejects orphans, rejects partial-multilabel-group observation per row, and delegates the "is OBSERVED_* allowed here?" decision to the estimator (vanilla rejects; conditional accepts).
- `BaseScorer.preserved_inference_columns(self) -> list[str]` hook returning `[]` by default; `MixedTypeMultiTargetScorer` overrides to declare OBSERVED_* columns matching declared targets (unconditional — see v1 plan note for rationale).
- `BaseRecommender.recommend_online` sets aside `scorer.preserved_inference_columns()` before `interactions_schema.apply()` and re-attaches after — protects OBSERVED_* from being silently stripped.
- `RankingRecommender` line 174–176 guard unchanged; line 183 elif unchanged. No new branches.
- Factory `multi_target.mode` enum extends with `"conditional_joint_mlp"` and `"conditional_joint_transformer"`. `capability_matrix()` `multi_target_model_types` extends to 5 entries.
- Independent + conditional: not on v3 roadmap. Cross-target observed-as-features is a structurally different feature; if revisited, it's v4+.
- Verification gate added: **mandatory leakage test** (per-row, per-target masked positions have `is_observed=0` and `value=0` in the label-input tensor).
- Tests added: `tests/test_conditional_joint_multi_target_mlp.py`, `tests/test_conditional_joint_multi_target_transformer.py` (mirror v1's coverage).
- Scorer test #8 / #9 / #10 from v2 (OBSERVED_*-deferred rejection) flip semantics: instead of always raising, they become OBSERVED-aware happy/sad paths.

The v2 plan deliberately keeps the surface area small enough that v3 is a clean diff — no behavior changes to vanilla code paths, only additions.

**Evaluation in v3**: no contract changes. Conditional estimators reuse the v2 evaluation surface unchanged — `evaluate()` still dispatches per `TargetType`, `score_per_target` still works, `MULTICLASS_ACCURACY` still applies. The only v3 evaluation question is whether to add new metrics that quantify the lift from conditioning vs vanilla (e.g., conditional-AUC, an "uplift over vanilla baseline" metric). That's a v3 design question, not a v2 blocker.
