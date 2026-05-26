# Decision rule: which scorer + estimator for which data?

This page is the single source of truth for two recurring questions:

1. **Which scorer fits my data shape and target types?**
2. **Within `MixedTypeMultiTargetScorer`, do I want joint or independent?**

It also publishes the canonical per-`TargetType` metric dispatch table that the runtime, [`capability_matrix()["target_type_metric_compat"]`](capability-matrix.md), [scorers.md](scorers.md), and scikit-rec-agent all reference.

---

## Decision tree

```
Q1: Long format (USER_ID, ITEM_ID, OUTCOME triples) or wide (one row per user with ITEM_* columns)?
├── long
│   ├── Need timestamps? → InteractionsDataset + TIMESTAMP + SequentialScorer / HierarchicalScorer
│   ├── Single multi-class target (ITEM_ID IS the class)? → MulticlassScorer
│   └── Otherwise → UniversalScorer or IndependentScorer
└── wide
    ├── All-binary targets, single mode → MultioutputScorer (classifier mode)
    ├── All-continuous targets, single mode → MultioutputScorer (regressor mode)
    └── Heterogeneous targets (mix of binary, regression, multiclass, multilabel)
                                                                              │
                                                                              ▼
                                              MixedTypeMultiTargetScorer + one of:
                                                                              │
                                              ┌───────────────────────────────┴───────────────────────────────┐
                                              │                                                                │
                                          Joint family                                                  Independent
                                              │                                                                │
                              Targets are correlated;                                       Targets are independent; OR
                              want a shared representation;                                 you want per-target estimator
                              moderate target count.                                        type flexibility (e.g. XGB on
                              Multilabel groups carry an                                    one target, LightGBM on another).
                              inductive bias.                                               Multilabel group inductive bias
                                              │                                             is lost.
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                       JointMultiTargetMLP             JointMultiTargetTransformer
                       Default; small to                FT-Transformer-style; better
                       moderate features.               when pairwise feature
                                                        interactions matter.
```

---

## Per-`TargetType` metric dispatch (canonical table)

The runtime constant `TARGET_TYPE_TO_METRICS` in `skrec.scorer.mixed_type_multi_target` is the source of truth; `capability_matrix()["target_type_metric_compat"]` mirrors it; gate 7 (`tests/test_mixed_type_multi_target_gates.py`) asserts both agree with this table.

| Target type | Compatible metrics | Notes |
|---|---|---|
| `BINARY` | `ROC_AUC`, `PR_AUC` | Standard binary classification metrics |
| `REGRESSION` | `RMSE`, `MAE` | Per-target prediction error |
| `MULTICLASS` | `MULTICLASS_ACCURACY` | Top-1 accuracy (new v2 metric). For log-loss / macro-F1, use `score_per_target` |
| `MULTILABEL` member | `ROC_AUC`, `PR_AUC` | Each fanned-out member is binary |

Metrics outside this table reach via `scorer.score_per_target(metric_callables=...)` — see [scorers.md](scorers.md#5-mixedtypemultitargetscorer) for the callable contract per `TargetType`.

---

## Joint vs independent: when to pick which

| Question | Lean joint | Lean independent |
|---|---|---|
| Are your targets correlated? | Yes — joint shares a learned representation that improves prediction on each | No — independent estimators avoid the joint loss-balancing complexity |
| Do you have a multilabel group? | Yes — joint preserves group inductive bias | No — independent fans out into per-member binaries |
| Do you want per-target estimator-type flexibility (XGB here, LightGBM there)? | No | Yes — independent's `per_target` dict supports this directly |
| Is your data feature-rich (>20 features)? | Joint Transformer if pairwise feature interactions matter; joint MLP otherwise | Either; sub-estimator handles its own features |
| Do you have very different target scales (e.g. dollar regression + binary)? | Joint MLP / Transformer with `regression_normalize=True` (default) | Independent: each sub-estimator has its own scaling internals |
| Is HPO important? | Both work via `skrec/orchestrator/hpo.py`; joint has one model to tune, independent has K sub-estimators | Both work; per-target tuning is more granular in independent mode |

When in doubt, start with **joint MLP** (smaller default network, low setup cost) and compare against **independent with XGB defaults** as a baseline. The v2 quickstart notebook does this comparison side-by-side.

---

## When to add a new target type vs. encode as existing

| Situation | Choose |
|---|---|
| Multi-class target with 2 classes only | `BINARY` (multiclass machinery is overkill) |
| Continuous target bounded to [0, 1] (e.g. probability) | `REGRESSION` — model handles the bound; switching to BINARY loses information |
| Multi-class target with class imbalance | `MULTICLASS` — top-1 accuracy is the v2 metric; for class-weighted F1, use `score_per_target` |
| Several related binary outcomes (engagement signals) | `TargetGroupSpec(type=MULTILABEL, columns=[...])` — joint families preserve the group bias |

---

## "No scalar default" for evaluation

`MixedTypeMultiTargetScorer.evaluate()` always returns `Dict[str, float]`. There is no honest macro aggregation across heterogeneous target types (binary AUC of 0.85 and regression RMSE of 12.7 aren't on a common scale).

If you need a single number for HPO or model comparison:

- **Pick a primary target** (the one your downstream task cares about most) and use that target's metric as the objective.
- **Compose a weighted aggregate** via `score_per_target` with user-supplied callables:

```python
weights = {"ITEM_clicked": 0.5, "ITEM_revenue": 0.5}
result = scorer.score_per_target(...)  # per-target metrics
score = sum(weights[k] * v for k, v in result.items() if k in weights)
```

This is intentional — baking a "primary metric" default into the scorer would mask a per-use-case choice that belongs in the caller's hands.

---

## Real-time-label conditioning (v3, available)

For scenarios where the caller has observed the ground truth for **some** targets at inference and wants those values to condition predictions for the rest, use a conditional estimator:

- `ConditionalJointMultiTargetMLPEstimator`
- `ConditionalJointMultiTargetTransformerEstimator`

Both implement the runtime-checkable `ConditionalMultiTargetEstimator` Protocol subclass. The scorer's inference validator delegates to the estimator: vanilla estimators reject `OBSERVED_*` with a clean error; conditional estimators permit it and validate the multilabel-group "members must mask together per row" rule. Predictions flow through `predict_with_observed(X, observed)` when `OBSERVED_*` columns are present; vanilla `predict_proba_dict(X)` is equivalent to passing an empty observed dict.

`capability_matrix()["scorer_supports_observed_conditioning"]` is now `("mixed_type_multi_target",)`.

`OBSERVED_*` columns are auto-preserved through `interactions_schema.apply()` via the `BaseScorer.preserved_inference_columns()` hook even when the client schema doesn't declare them, so `recommend_online` works out of the box.

**Independent + conditional is NOT supported** (v3 locked decision #1). Cross-target observed-as-features is structurally different from joint masking; if revisited it lands in v4+. Use the joint families when you need conditioning.

See [`mixed_type_multi_target_plan_v3.md`](https://github.com/intuit/scikit-rec/blob/main/mixed_type_multi_target_plan_v3.md) for the four locked design decisions and the v3 implementation details.
