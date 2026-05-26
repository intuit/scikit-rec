# Mixed-type multi-target — v3 conditional inference (delta from v2)

Adds real-time-label conditioning to the v2 scorer. Purely additive: every v2 vanilla code path keeps working unchanged. v3 lands after v2 merges, in this same repo. Supersedes the conditional portions of `mixed_type_multi_target_plan.md` (v1).

## Locked decisions (from v2→v3 review)

| # | Decision | Choice |
|---|---|---|
| 1 | Conditional independent estimator | **Not in v3.** Joint MLP + joint Transformer only. Cross-target observed-as-features is structurally different; if revisited, v4+ |
| 2 | Conditional dispatch surface | **New Protocol subclass** `ConditionalMultiTargetEstimator(MultiTargetEstimator)` with `predict_with_observed(X, observed)`. Scorer's `_validate_inference_interactions` checks `isinstance(estimator, ConditionalMultiTargetEstimator)` to decide whether `OBSERVED_*` is allowed |
| 3 | `mask_prob=1.0 ≈ vanilla` equivalence test | **Keep.** Safety net for label-channel-leakage and masking bugs |
| 4 | Multilabel group masking schedule | **Together per row.** All members observed together OR all NaN together per row. Partial-group observation → clean error |

## Diff vs v2

| Area | v2 | v3 |
|---|---|---|
| Estimator families | 3 (joint MLP, joint Transformer, independent) | **5** (adds conditional joint MLP, conditional joint Transformer) |
| `MultiTargetEstimator` Protocol | runtime-checkable contract on all 3 families | unchanged; v3 adds subclass below |
| `ConditionalMultiTargetEstimator` Protocol | — | **NEW** runtime-checkable subclass adding `predict_with_observed(X, observed) -> dict[str, np.ndarray]` |
| `OBSERVED_*` columns | scorer rejects with "deferred to v3" | scorer **delegates** to estimator: conditional → accepts (NaN = unobserved per row); vanilla → still rejects (unchanged error wording, just trimmed of "v3" mention) |
| `BaseScorer.preserved_inference_columns()` hook | not added | **added**; returns `[]` by default; `MixedTypeMultiTargetScorer` overrides to declare `OBSERVED_*` columns matching declared targets — **unconditional** (vanilla AND conditional estimators) |
| `BaseRecommender.recommend_online` schema-apply | unchanged | **sets aside** `scorer.preserved_inference_columns()` before `interactions_schema.apply()` and re-attaches after |
| `_validate_inference_interactions` | rejects all `OBSERVED_*` | OBSERVED-aware: permits NaN, rejects orphans, rejects partial-multilabel-group observation per row, rejects any OBSERVED_* for vanilla estimators |
| Factory `multi_target.mode` enum | 3 values | **5** values (adds `conditional_joint_mlp`, `conditional_joint_transformer`) |
| `capability_matrix()["multi_target_model_types"]` | 3 entries | 5 entries |
| `capability_matrix()["scorer_supports_observed_conditioning"]` | empty tuple `()` | **`("mixed_type_multi_target",)`** |
| Encoder Protocol `label_input_dim` | `=0` always (v3 hook) | activated: conditional estimators pass `label_input_dim > 0` |

Everything not listed above is unchanged from v2.

## Public API additions

### `ConditionalMultiTargetEstimator` Protocol

```python
@runtime_checkable
class ConditionalMultiTargetEstimator(MultiTargetEstimator, Protocol):
    """Subclass of MultiTargetEstimator that supports real-time-label conditioning.

    Conditional estimators accept an `observed` dict at inference time:
    - For declared targets where the caller has observed the ground truth in
      real time, the value is used to condition predictions for other targets.
    - For targets where the value is NaN (per row), the estimator predicts
      from features alone, as in vanilla mode.

    At training, the estimator implements a masked-target curriculum
    (Bernoulli mask per (row, target)) to learn both conditional and
    unconditional prediction modes.
    """

    def predict_with_observed(
        self,
        X: pd.DataFrame,
        observed: dict[str, np.ndarray],   # keys match target_specs; NaN = unobserved per row
    ) -> dict[str, np.ndarray]:
        """Per-target predictions conditioned on `observed`. Same output shape
        contract as `predict_proba_dict`. Vanilla `predict_proba_dict(X)` is
        equivalent to `predict_with_observed(X, all-NaN observed dict)`."""
```

Lives in `skrec/estimator/classification/_multi_target_protocol.py` alongside the v2 base Protocol.

### Per-target label encoding (conditional only)

| Target type | Encoded shape (per target) |
|---|---|
| binary | 2 dims: `[is_observed, value]` |
| regression | 2 dims: `[is_observed, z-score-normalized value]` |
| multiclass | `1 + K_k` dims: `[is_observed, one-hot]` |
| multilabel (group) | `1 + K_k` dims: `[is_observed_group, binary vector per member]` — single `is_observed` flag covers the whole group (per locked decision #4) |

Per-target linear projection to `label_embedding_dim`; concatenated across targets and fed alongside `X` into the encoder. **Masked positions zero both `is_observed` AND the value/one-hot/binary-vector** (defense in depth; tested by the mandatory leakage gate).

### Conditional estimator classes

| Class | Encoder | Conditional? |
|---|---|---|
| `ConditionalJointMultiTargetMLPEstimator` | MLP | Yes |
| `ConditionalJointMultiTargetTransformerEstimator` | Transformer | Yes |

Both implement `ConditionalMultiTargetEstimator`. They reuse the v2 encoder Protocol (passing `label_input_dim > 0` for the first time) and the v2 per-target heads. The training loop adds:

- Bernoulli masking per (row, target) at probability `mask_prob` (default 0.5).
- Loss only on **masked** positions (the estimator must predict masked targets given unmasked ones).
- `mask_prob=0.0` rejected at init (no learning signal); `mask_prob=1.0` accepted (equivalent to vanilla — covered by equivalence test).

### Scorer changes

```python
class MixedTypeMultiTargetScorer(BaseScorer):
    # v3-modified methods:

    def _validate_inference_interactions(self, interactions_df: pd.DataFrame) -> None:
        """v3 scope:
        - If estimator is ConditionalMultiTargetEstimator: permit OBSERVED_* columns,
          reject orphans (no matching target), reject partial-multilabel-group
          observation per row (members must mask together).
        - If estimator is vanilla MultiTargetEstimator: reject any OBSERVED_*
          column with a clean error pointing at the conditional estimator classes.
        """

    def preserved_inference_columns(self) -> list[str]:
        """v3 override of new BaseScorer hook. Returns OBSERVED_<suffix> column
        names matching the declared targets — UNCONDITIONALLY, regardless of
        whether self.estimator is vanilla or conditional.

        Rationale (from v1 plan): the hook's job is "what columns must survive
        schema apply so the scorer can validate them?" — a scorer property.
        Returning [] for vanilla would let schema.apply() silently strip
        OBSERVED_* with a generic warning, hiding the user's intent and leaving
        the scorer unable to raise a clean error. With unconditional
        preservation, vanilla + OBSERVED_* at inference is rejected explicitly
        by _validate_inference_interactions with an actionable error.
        """

    def score_items(self, interactions=None, users=None) -> pd.DataFrame:
        """v3: if estimator is conditional and interactions has OBSERVED_*
        columns, build the `observed` dict and call predict_with_observed.
        Otherwise (vanilla, or conditional without OBSERVED_* columns) calls
        predict_proba_dict unchanged."""

    def predict_targets(self, interactions=None, users=None) -> pd.DataFrame:
        """Same OBSERVED-aware dispatch as score_items, but uses
        predict_targets_dict (which conditional estimators also implement —
        equivalent semantics)."""

    def score_fast(self, features: pd.DataFrame) -> pd.DataFrame:
        """v3: single-row OBSERVED-aware path. Conditional estimator honors
        observed values in the single row; vanilla rejects any OBSERVED_*."""
```

### `OBSERVED_*` column convention

Mapping: `ITEM_workflow_engaged` (target) ↔ `OBSERVED_workflow_engaged` (observed input). Suffix is target name with `ITEM_` stripped, prefixed with `OBSERVED_`.

For multilabel groups: per locked decision #4, members must observe together. The OBSERVED_* columns are per-member (matching the fanned-out target columns), but the validator rejects rows where some members of a group are observed and others are NaN.

## Files

### New

| Path | Contents |
|---|---|
| `skrec/estimator/classification/conditional_joint_multi_target_mlp.py` | `ConditionalJointMultiTargetMLPEstimator` |
| `skrec/estimator/classification/conditional_joint_multi_target_transformer.py` | `ConditionalJointMultiTargetTransformerEstimator` |
| `skrec/estimator/classification/_conditional_label_encoding.py` | Per-target label encoding utilities (is_observed flag + value encoding + projection); masking helpers; group-mask validators |
| `tests/test_conditional_joint_multi_target_mlp.py` | Conditional MLP tests |
| `tests/test_conditional_joint_multi_target_transformer.py` | Conditional Transformer tests |
| `examples/mixed_type_multi_target/conditional.ipynb` | v3 conditional inference demo (extends the v2 quickstart with `OBSERVED_*` columns at inference) |

### Modified

| Path | Change |
|---|---|
| `skrec/constants.py` | Add `OBSERVED_PREFIX = "OBSERVED_"` |
| `skrec/estimator/classification/_multi_target_protocol.py` | Add `ConditionalMultiTargetEstimator(MultiTargetEstimator, Protocol)` Protocol subclass with `predict_with_observed` |
| `skrec/estimator/classification/__init__.py` | Export `ConditionalMultiTargetEstimator`, both new conditional estimator classes |
| `skrec/scorer/base_scorer.py` | Add `preserved_inference_columns(self) -> list[str]` hook returning `[]` by default |
| `skrec/scorer/mixed_type_multi_target.py` | Flip `_validate_inference_interactions` from reject-all-OBSERVED to OBSERVED-aware dispatch (vanilla rejects; conditional permits, validates group-mask-together, rejects orphans). Override `preserved_inference_columns()`. Add OBSERVED → `observed` dict construction in `score_items` / `predict_targets` / `score_fast` with isinstance dispatch on the estimator |
| `skrec/recommender/base_recommender.py` | In `recommend_online`, around `interactions_schema.apply()`: set aside columns from `scorer.preserved_inference_columns()` before apply; re-attach after |
| `skrec/orchestrator/factory.py` | (1) `multi_target.mode` enum adds `conditional_joint_mlp` + `conditional_joint_transformer`. (2) `capability_matrix()["multi_target_model_types"]` adds the two new entries. (3) `capability_matrix()["scorer_supports_observed_conditioning"]` flips from `()` to `("mixed_type_multi_target",)`. (4) `create_estimator` branch for the two conditional modes (composes the conditional estimator class with `mask_prob` + `label_embedding_dim` defaults) |
| `docs/user-guide/scorers.md` | Update v3-deferral callout — flip to "conditional support is now available" + add `OBSERVED_*` usage example |
| `docs/user-guide/decision-rule.md` | Update v3 OBSERVED_* preview section — flip from "coming" to "available" with usage guidance |
| `docs/user-guide/capability-matrix.md` | Update `multi_target_model_types` row to list 5 modes; update `scorer_supports_observed_conditioning` |
| `docs/advanced/orchestration.md` | Update `capability_matrix()` example output for `scorer_supports_observed_conditioning` and the 5-entry `multi_target_model_types` |

### Untouched (delta from v2)

`MixedTypeMultiTargetScorer.__init__`, `process_datasets`, `_validate_interactions` (training-time), `_calculate_scores`, `score_per_target`, all 3 v2 estimator classes (joint MLP, joint Transformer, independent), `RankingRecommender.evaluate` and its `_evaluate_mixed_type_multi_target` branch (the evaluation contract doesn't change — conditional models still produce per-target predictions; eval still dispatches per `TargetType`), `MULTICLASS_ACCURACY` metric, all v2 tests.

## Build order

1. **V3-M1: Protocol + OBSERVED_PREFIX + preserved-columns hook + schema-apply preservation.** Add `ConditionalMultiTargetEstimator` Protocol subclass; `OBSERVED_PREFIX` constant; `BaseScorer.preserved_inference_columns()` default; `MixedTypeMultiTargetScorer.preserved_inference_columns()` override; `BaseRecommender.recommend_online` schema-apply preservation. Minimal tests: hook default returns `[]`, scorer override returns expected OBSERVED_* set, schema-apply round-trip preserves OBSERVED_* columns even when client schema doesn't declare them.
2. **V3-M2: Conditional joint MLP + conditional joint Transformer + label encoding.** Implement label encoding helpers (`_conditional_label_encoding.py`) — per-target encoding shapes, mask-aware zeroing, group-mask validators. Implement both conditional estimator classes. Both implement `ConditionalMultiTargetEstimator`. `mask_prob=0.0` rejected at init.
3. **V3-M3: Scorer validator flip + dispatch updates.** Flip `_validate_inference_interactions` from reject-all-OBSERVED to OBSERVED-aware. Update `score_items` / `predict_targets` / `score_fast` to construct `observed` dict and dispatch via isinstance check on the estimator. Factory wiring for the two new modes.
4. **V3-M4: v3 tests.** All conditional MLP tests (8), all conditional Transformer tests (mirror), mandatory leakage gate, `mask_prob=1.0 ≈ vanilla` equivalence test, scorer's `OBSERVED_*`-permit / orphan-reject / partial-group-reject tests, `recommend_online` end-to-end with non-declaring client schema.
5. **V3-M5: Docs update + conditional notebook.** Flip v2's "deferred to v3" callouts to "available in v3." Add `examples/mixed_type_multi_target/conditional.ipynb` extending the v2 quickstart with `OBSERVED_*` inference.

## Tests

### `tests/test_conditional_joint_multi_target_mlp.py`

1. Happy path: train with `mask_prob=0.5`, `epochs=2`; predict with mixed observed/NaN; no errors; output shapes correct.
2. **Conditioning has measurable effect**: synthetic with strongly correlated targets A, B; assert prediction quality on B differs significantly when A observed vs NaN.
3. **`mask_prob=1.0 ≈ vanilla`** (locked decision #3): on the same synthetic, conditional with `mask_prob=1.0` produces predictions within tolerance of a v2 vanilla joint MLP. Equivalence is "within 10% on per-target AUC/RMSE" — not bit-exact (label embedding contributions exist) but functionally close.
4. **`mask_prob=0.0` rejected at init**: clean error.
5. **Multilabel group group-mask-together** (locked decision #4): observe one member, NaN the rest → clean error at inference; observe all members → conditioning works; NaN all members → falls back to feature-only prediction.
6. **Label-channel zeroing (mandatory leakage gate, gate 2 from v1)**: construct a batch where masked positions have non-zero raw label values. Forward and assert label-input tensor at masked positions has `is_observed=0` AND `value=0` (defense in depth).
7. **Loss-balance smoke**: train with one regression target on dollar scale (~1e6) plus one binary target. Assert no NaN gradients in first 3 epochs; both losses decrease.
8. **Pickle round-trip** preserves regression z-score scaler params and masking config; predictions match post-load.
9. **Single-row conditional inference** (`score_fast`): single-row DataFrame with both feature columns and `OBSERVED_*` columns; assert observed values condition the prediction vs same row with `OBSERVED_*` set to NaN.
10. **`recommend_online` with constraining schema**: build a recommender with a client inference schema that does NOT declare `OBSERVED_*` columns. Call `recommend_online(interactions=row_with_OBSERVED)`. Assert the conditioning still works (preserved-columns hook shields `OBSERVED_*` from schema-apply's silent strip).

### `tests/test_conditional_joint_multi_target_transformer.py`

Mirror all 10 tests above, parametrized for transformer architecture.

### Scorer-level tests (updates to `tests/test_mixed_type_multi_target_scorer.py`)

The v2 tests #8 / #9 (OBSERVED_* deferred-to-v3 rejection) flip semantics:

- **#8 → #8a (vanilla)**: OBSERVED_* with a vanilla estimator → clean error pointing at conditional estimator classes (no longer "deferred to v3"; the v3 path is now available).
- **#8 → #8b (conditional, NEW)**: OBSERVED_* with a conditional estimator → accepted; observed values flow into `predict_with_observed`.
- **#9 (namespace collision)**: feature column starting with OBSERVED_ that doesn't match any target → clean error regardless of estimator type (vanilla or conditional).

Plus new tests:

- **`preserved_inference_columns()` returns OBSERVED set unconditionally**: vanilla AND conditional estimator both produce the same OBSERVED_* set. (v1 scorer test 11.)
- **Partial-multilabel-group observation per row → clean error** (v1 scorer test 10): observe one member, NaN the rest within a row → error at scorer's `_validate_inference_interactions`.

### Verification gates (v3 additions)

- **Gate 9: Mandatory leakage test** (returns from v1): label-channel zeroing at masked positions — both estimators. Test #6 above is the implementation; gate 9 is the named verification checkpoint.
- **Gate 10: `mask_prob=1.0 ≈ vanilla` equivalence** (locked decision #3): both conditional estimators with `mask_prob=1.0` produce per-target metrics within 10% of v2 vanilla baselines on the same synthetic.

## Risks watched

(Inherits all v2 risks. v3-specific additions:)

13. **Label-channel leakage** — masked positions could pass label information through. Mitigation: gate 9 + zero-both-flag-and-value defense in depth.
14. **Conditional vs vanilla divergence at `mask_prob=1.0`** — conditional architecture's label channel may not fully no-op. Mitigation: gate 10 with documented tolerance.
15. **Schema-apply silently strips OBSERVED_*** — client schema doesn't declare OBSERVED_*; `interactions_schema.apply()` warns and drops; user thinks conditioning worked but it silently didn't. Mitigation: `preserved_inference_columns()` hook + `recommend_online` set-aside-and-reattach + test #10 in conditional tests.
16. **Protocol drift** — `ConditionalMultiTargetEstimator` is a subclass Protocol; runtime_checkable Protocol subclassing is well-defined but rarely exercised in practice. Mitigation: isinstance strictness tests assert positive (both conditional estimators) and negative (vanilla estimators NOT a `ConditionalMultiTargetEstimator`).

## Out of v3 scope

- Conditional independent estimator (locked decision #1; v4+ if ever).
- Mask schedule curriculum beyond constant Bernoulli (e.g., annealing, per-target rates).
- Per-target loss weighting beyond `regression_normalize=True` default.
- Counterfactual evaluation of conditional models (e.g., does conditioning improve uplift estimates?). Evaluation contract is unchanged from v2.

## Post-v3 follow-up: agent-changes doc

After v3 merges, write `agent_changes_after_v3.md` capturing the 7-item agent follow-up list (carried forward from v2 plan's "Agent-side follow-ups" section). That doc bridges v2+v3 changes into the scikit-rec-agent repo. Out of scope for this repo's work.
