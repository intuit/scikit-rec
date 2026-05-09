# Wide-format multi-output scorer. Routes ITEM_<name> target columns through
# either sklearn.MultiOutputClassifier (per-target categorical) or
# sklearn.MultiOutputRegressor (per-target continuous) depending on the
# estimator type passed at construction.
#
# Vocabulary note: the wide-multioutput contract uses **target** to mean
# "one ITEM_<name> column = one supervised
# learning response variable". This deliberately avoids overloading **item**,
# which in long-format scoring means "a real catalogue entity (product, video)
# referenced by ITEM_ID." Newer public API names (DegenerateTargetPolicy,
# on_degenerate_target, the new error messages added in the binary-only
# rework) prefer "target". A few legacy validation messages still say
# "item column" / "ITEM column" — those predate the rework and reach
# external callers via stable contracts; we left them rather than churn
# the public error strings. Internal state preserves the older scikit-rec
# convention (item_names / item_subset / item_count) for back-compat with
# the rest of BaseScorer.
#
# No-LabelEncoder design note: classifier mode requires every ITEM_<name>
# target to be numeric and binary-valued in {0, 1} (or {0.0, 1.0}). Strings
# and signed-integer encodings (e.g. {-1, 1}) are NOT accepted — pre-encode
# at the caller (e.g. ``df[col] = (df[col] == "yes").astype(float)``). Bool
# columns ARE accepted because Python `bool` ⊂ `int`: `True == 1` and
# `False == 0` in set membership, so a `{True, False}` column collapses
# to `{1, 0}` and passes the binary numeric check. Dropping the per-target
# LabelEncoder means: column names in score_items output are uniformly
# ``ITEM_<name>_0`` / ``ITEM_<name>_1`` regardless of input dtype;
# positive_proba_column_name returns ``ITEM_<name>_1``; predict_classes
# returns 0/1 directly (callers do their own inverse mapping if they want
# original labels). Regressor mode passes continuous values through
# unchanged, same as before.
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import skrec.constants as C
from skrec.estimator.classification.base_classifier import BaseClassifier
from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator
from skrec.estimator.regression.base_regressor import BaseRegressor
from skrec.scorer.base_scorer import BaseScorer
from skrec.util.logger import get_logger

logger = get_logger(__name__)


class DegenerateTargetPolicy(str, Enum):
    """Policy for ``ITEM_<name>`` target columns that have a single unique
    value in the training slice (no learning signal).

    A "degenerate" target is one where ``df[col].nunique() < 2`` — every row
    carries the same value. ``sklearn.LogisticRegression`` and friends refuse
    to fit on such ``y``; ``XGBClassifier`` accepts the fit but emits
    ``predict_proba`` with a column count that disagrees with the binary
    contract, causing a downstream pandas shape error. This enum picks how
    :class:`MultioutputScorer` reacts when it detects this case.

    Members:
        RAISE: Default. ``train()`` raises ``ValueError`` listing the
            offending column(s) and the only observed value. Caller is
            expected to drop or backfill the column(s) before retrying.
        CONSTANT: Permissive opt-in. Excludes the degenerate column from the
            underlying classifier fit and substitutes a constant prediction
            (probability 1.0 for the only seen class) at score time.
            ``MultioutputScorer.degenerate_targets`` exposes the manifest of
            columns that fell back to this behaviour. Use this when you have
            many targets, accept that some will be degenerate in the train
            slice, and want training to succeed on the rest.
    """

    RAISE = "raise"
    CONSTANT = "constant"


# The two valid values for any binary ITEM_* target column. Used in the
# fit-time validation gate and in degenerate-target constant-prediction
# reconstruction.
_BINARY_VALUES: frozenset = frozenset({0, 1, 0.0, 1.0})


class MultioutputScorer(BaseScorer):
    """Wide-format multi-output scorer.

    Each ``ITEM_<name>`` column in the training frame is one supervised
    target. The scorer routes those targets through either
    ``sklearn.MultiOutputClassifier`` (categorical targets) or
    ``sklearn.MultiOutputRegressor`` (continuous targets) based on the
    estimator type:

    - **Classifier mode** (``BaseClassifier`` estimator, e.g.
      :class:`~skrec.estimator.classification.multioutput_classifier.MultiOutputClassifierEstimator`):
      every ``ITEM_<name>`` target must be **binary numeric** — values
      strictly in ``{0, 1}`` (or ``{0.0, 1.0}``). Pre-encode non-numeric
      labels at the caller (e.g. ``df[col] = (df[col] == "yes").astype(float)``).
      ``score_items`` returns per-class probabilities — DataFrame columns
      are uniformly ``ITEM_<name>_0`` / ``ITEM_<name>_1``, shape
      ``(n_users, 2 * n_targets)``. ``predict_classes`` returns the argmax
      class (0 or 1) per target. Multi-class targets (3+ classes) and
      non-numeric labels are rejected at fit time — see ``_validate_targets``.
    - **Regressor mode** (``BaseRegressor`` estimator, e.g.
      :class:`~skrec.estimator.regression.multioutput_regressor.MultiOutputRegressorEstimator`):
      values pass through unencoded and ``score_items`` returns predicted
      values — DataFrame columns are the original ``ITEM_<name>``, shape
      ``(n_users, n_targets)``. ``predict_targets`` is the regressor-mode
      analogue of ``predict_classes``.

    Degenerate targets (single-class in the training slice) are handled
    according to ``on_degenerate_target`` — see :class:`DegenerateTargetPolicy`.

    .. warning::
        **Not thread-safe.** Like :class:`~skrec.scorer.independent.IndependentScorer`,
        ``MultioutputScorer`` carries shared mutable state across calls
        (``item_subset`` via :meth:`set_item_subset` / :meth:`clear_item_subset`,
        ``item_names``, ``item_count``, ``degenerate_targets``,
        ``_fitted_target_order``, ``_validating_post_drop``). Concurrent
        ``train()``, ``score_items()``, ``set_item_subset()``, or
        ``predict_classes()`` calls on the same instance can corrupt each
        other's state. Use one instance per thread or serialize calls.

    Args:
        estimator: A :class:`BaseClassifier` or :class:`BaseRegressor`.
            Embedding estimators are explicitly rejected — use
            :class:`~skrec.scorer.universal.UniversalScorer` for those.
        on_degenerate_target: How to react when a target column is
            single-class in the training data. ``RAISE`` (default) fails
            loudly; ``CONSTANT`` falls back to a constant predictor for
            the offending columns. See :class:`DegenerateTargetPolicy`.
            Accepts either the enum member or its string value.
    """

    def __init__(
        self,
        estimator: Union[BaseClassifier, BaseRegressor],
        on_degenerate_target: Union[DegenerateTargetPolicy, str] = DegenerateTargetPolicy.RAISE,
    ) -> None:
        if isinstance(estimator, BaseEmbeddingEstimator):
            raise TypeError(
                "MultioutputScorer does not support BaseEmbeddingEstimator. "
                "Use UniversalScorer for embedding estimators (e.g. MatrixFactorizationEstimator, NCFEstimator)."
            )
        if not isinstance(estimator, (BaseClassifier, BaseRegressor)):
            raise TypeError(
                f"MultioutputScorer requires a BaseClassifier or BaseRegressor estimator, "
                f"got {type(estimator).__name__}."
            )
        super().__init__(estimator)
        self.is_classifier = isinstance(estimator, BaseClassifier)
        self.on_degenerate_target = DegenerateTargetPolicy(on_degenerate_target)
        # Manifest of targets that fell back to a constant predictor under
        # DegenerateTargetPolicy.CONSTANT. Always empty under RAISE (training
        # would have aborted). Maps column name → the only observed value
        # (a float in {0.0, 1.0}).
        self.degenerate_targets: Dict[str, float] = {}
        # Snapshot of the column order the underlying classifier was fit on,
        # captured in `_validate_targets` after the degenerate filter.
        # `_calculate_scores` uses this to do a name-keyed lookup into the
        # predict_proba list rather than positional iteration — pins the
        # implicit "predict_proba returns one entry per fitted column in
        # fit-order" contract in named state. If a future sklearn release
        # ever changes that order, the explicit lookup still uses the
        # snapshot (so output would be wrong) but downstream tests on a
        # fresh fit would catch the drift via inverted predictions.
        self._fitted_target_order: List[str] = []
        # Selects the ITEM_* count floor in `_validate_interactions`. False
        # (default) enforces the structural ≥2 rule on the user's input;
        # True relaxes to ≥1 for the post-degenerate-drop call inside
        # super().process_datasets(). Toggled with try/finally in
        # `process_datasets` so a partial-degeneracy training succeeds but
        # the structural rule is preserved against the original input
        # frame the caller hands in.
        self._validating_post_drop: bool = False

    def process_datasets(
        self,
        users_df: Optional[pd.DataFrame] = None,
        items_df: Optional[pd.DataFrame] = None,
        interactions_df: Optional[pd.DataFrame] = None,
        is_training: Optional[bool] = True,
    ) -> Tuple[pd.DataFrame, NDArray]:
        """Validate and prepare wide-format interaction data.

        Args:
            users_df: Must be ``None``.
            items_df: Must be ``None``.
            interactions_df: Wide-format DataFrame with ``USER_ID_NAME`` and at
                least two ``ITEM_*`` columns. One row per user.
            is_training: When ``True``, validates targets and initialises item state.

        Returns:
            ``(X, y)`` ready for the underlying estimator's ``fit``.
        """
        if users_df is not None or items_df is not None:
            raise ValueError("Item Dataset and User Dataset will not be used in MultioutputScorer.")

        # Copy interactions_df only when _validate_targets might mutate it —
        # specifically classifier-mode training. Inference and regressor
        # mode never touch the frame, so paying for a full DataFrame copy
        # of a multi-million-row wide frame on every score-time call is
        # wasted work. Without the copy, _validate_targets's drop under
        # CONSTANT policy would silently shrink the caller's DataFrame.
        #
        # Invariant: regressor-mode `_init_item_state` is currently
        # frame-pure (only reads columns; never drops). If a future change
        # ever adds drop logic to the regressor path (e.g. an "drop
        # all-NaN target" feature), widen this guard to include the
        # regressor case — otherwise the caller's frame would be silently
        # mutated.
        if is_training and self.is_classifier and interactions_df is not None:
            interactions_df = interactions_df.copy()

        self._validate_interactions(interactions_df)
        if is_training:
            if self.is_classifier:
                self._validate_targets(interactions_df)
            else:
                self._init_item_state(interactions_df)

        # Tell the inherited _validate_interactions (called inside
        # super().process_datasets) to use the post-drop floor (≥1 ITEM_*)
        # instead of the structural floor (≥2). The structural rule has
        # already been enforced against the user's input above. try/finally
        # so the flag resets even if super raises.
        self._validating_post_drop = True
        try:
            return super().process_datasets(
                users_df=None,
                items_df=None,
                interactions_df=interactions_df,
                is_training=is_training,
            )
        finally:
            self._validating_post_drop = False

    def positive_proba_column_name(self, label: str) -> str:
        """Return the ``score_items`` column name carrying ``P(label = 1)``.

        Defined for classifier mode only. Always returns ``f"{label}_1"`` —
        no per-target lookup needed because the binary-only contract pins
        the positive class to ``1``.

        Raises:
            RuntimeError: when called in regressor mode (no per-class
                columns exist there — use ``label`` directly).
            KeyError: when ``label`` is not a known target.
        """
        if not self.is_classifier:
            raise RuntimeError(
                "positive_proba_column_name is only defined for classifier mode; "
                "in regressor mode, score_items columns are the bare ITEM_<name>."
            )
        if label not in self.item_names:
            raise KeyError(f"Unknown target {label!r}. Known targets: {list(self.item_names)}")
        return f"{label}_1"

    def _reset_target_state(self) -> None:
        """Clear per-target state. Called from `_validate_targets` and
        `_init_item_state` so the two entry points start from the same
        baseline regardless of which mode is active."""
        self.item_count = 0
        self.item_names = []
        self.degenerate_targets = {}
        self._fitted_target_order = []

    def _generate_X_y(self, joined_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Degenerate-classifier targets are dropped from joined_data in
        # _validate_targets so they do not enter y; everything still on the
        # frame with the ITEM_ prefix is a fitted target.
        list_labels = [x for x in joined_data.columns if x.startswith(C.ITEM_PREFIX)]
        y = joined_data[list_labels]
        X = joined_data.drop(list_labels + [C.USER_ID_NAME], axis=1, errors="ignore")
        return X, y

    def _process_items(self, items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> Tuple[NDArray, pd.DataFrame]:
        # Use the cached item_names captured before any degenerate-target drop
        # so the public catalogue always reflects every ITEM_* the caller
        # declared, regardless of training-slice degeneracy.
        if self.item_names is not None and len(self.item_names) > 0:
            return np.array(self.item_names), None
        items_list = np.array([x for x in interactions_df.columns if x.startswith(C.ITEM_PREFIX)])
        return items_list, None

    def _join_data_train(
        self, users_df: pd.DataFrame, items_df: pd.DataFrame, interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        return interactions_df.copy()

    def _validate_interactions(self, interactions_df: pd.DataFrame) -> None:
        """Wide-format validation: USER_ID + ITEM_* columns, no duplicate users.

        ``process_datasets`` runs this twice — once on the user's input frame
        (structural floor: ≥2 ITEM_* columns required) and a second time on
        the post-degenerate-drop frame inside ``super().process_datasets()``
        (fittability floor: ≥1 ITEM_* column required for sklearn's
        MultiOutputClassifier to accept the y matrix). The
        ``_validating_post_drop`` flag selects between the two floors.
        Without this distinction, a 1-alive + N-dead frame under CONSTANT
        policy would raise a misleading "<2 ITEM columns" error on the
        second call after CONSTANT legitimately dropped the dead columns.
        """
        self._validate_interactions_base(interactions_df)

        if C.USER_ID_NAME not in interactions_df.columns:
            raise ValueError("Interaction Dataset must contain USER_ID column for Multioutput Scorer.")

        item_cols = [col for col in interactions_df.columns if col.startswith(C.ITEM_PREFIX)]

        for col in item_cols:
            null_count = interactions_df[col].isnull().sum()
            if null_count > 0:
                raise ValueError(
                    f"item column '{col}' contains {null_count} null value(s). Remove or impute before training."
                )

        if self._validating_post_drop:
            # Second call (inside super().process_datasets) — floor is ≥1
            # so single-target post-drop frames still fit. The all-degenerate
            # case is caught earlier in _validate_targets, before the drop,
            # so reaching here with 0 columns would indicate a bug.
            if len(item_cols) < 1:
                raise ValueError(
                    "MultioutputScorer post-drop frame has 0 ITEM_* columns — internal "
                    "invariant violated; the all-degenerate guard in _validate_targets "
                    "should have caught this earlier. Report as a bug."
                )
        else:
            # First call (against the user's input frame) — structural floor.
            if len(item_cols) < 2:
                raise ValueError("Interaction Dataset must contain at least 2 ITEM columns for Multioutput Scorer.")

        if interactions_df[C.USER_ID_NAME].duplicated().any():
            raise ValueError("Multioutput Scorer only accepts one row per user.")

    def _init_item_state(self, interactions_df: pd.DataFrame) -> None:
        """Regressor-mode equivalent of `_validate_targets`.

        No encoding is applied — continuous targets pass through. Only sets up
        the catalogue (``item_names``, ``item_count``, etc.) used by the
        scoring path.
        """
        self._reset_target_state()
        item_cols = [c for c in interactions_df.columns if c.startswith(C.ITEM_PREFIX)]
        self.item_count = len(item_cols)
        self.item_names = list(item_cols)
        # Regressor mode has no degenerate-target machinery — every column
        # is fit. Snapshot is just item_names.
        self._fitted_target_order = list(item_cols)

    def _validate_targets(self, interactions_df: pd.DataFrame) -> None:
        """Classifier-mode validation: every ITEM_<name> column is binary numeric.

        Without a LabelEncoder layer, the underlying ``MultiOutputClassifier``
        sees the ITEM_<name> columns directly as ``y``. Binary numeric
        ``{0, 1}`` is the contract — string, bool, or signed-int encodings
        must be pre-converted by the caller.

        Two-pass collection so multi-class and degenerate targets each get a
        single error message naming every offending column rather than one
        error per retry.

        Under ``DegenerateTargetPolicy.CONSTANT``, single-class columns are
        recorded in ``self.degenerate_targets`` and **dropped from
        ``interactions_df``** in place. The underlying classifier is fit
        only on targets it can learn; ``item_names`` still contains every
        target so the public catalogue and ``score_items`` shape stay
        invariant. Degenerate columns get a constant prediction
        reconstructed at score time from the recorded seen value.
        """
        self._reset_target_state()

        bad_non_numeric: Dict[str, set] = {}
        degenerate: Dict[str, float] = {}

        for col in [c for c in interactions_df.columns if c.startswith(C.ITEM_PREFIX)]:
            self.item_count += 1
            self.item_names.append(col)
            unique = set(interactions_df[col].unique())
            extra = unique - _BINARY_VALUES
            if extra:
                bad_non_numeric[col] = extra
                continue
            if len(unique) > 2:
                # Unreachable: the previous `extra = unique - _BINARY_VALUES`
                # filter rejects any column with values outside {0, 1}, so by
                # the time we get here len(unique) ≤ 2 always. If a future
                # change widens _BINARY_VALUES to include, say, {-1, +1},
                # this guard would activate and catch the resulting 3+-class
                # case before the empty bad_multi handling at the bottom
                # silently lets it through. Raising AssertionError makes the
                # invariant explicit rather than relying on a comment-as-spec.
                raise AssertionError(
                    f"unreachable branch in _validate_targets: column {col!r} has "
                    f"{len(unique)} unique values within _BINARY_VALUES — implies "
                    f"_BINARY_VALUES has been widened beyond binary. Update this "
                    f"validator to handle the new contract."
                )
            if len(unique) < 2:
                # Single-class — degenerate. Record either way; resolve below per policy.
                only = next(iter(unique))
                degenerate[col] = float(only)

        # Order of error precedence: non-numeric/multi-class first (they're
        # contract violations the caller must fix), then degenerate (only
        # raised under RAISE policy).
        if bad_non_numeric:
            details = "; ".join(
                f"{col!r} (saw values: {sorted(values, key=str)})" for col, values in bad_non_numeric.items()
            )
            raise ValueError(
                f"MultioutputScorer (classifier mode) requires every ITEM_<name> target "
                f"to be binary numeric — values strictly in {{0, 1}} (or {{0.0, 1.0}}). "
                f"The following column(s) contain non-binary values: {details}. "
                f"Pre-encode at the caller, e.g.: "
                f"df['ITEM_x'] = (df['ITEM_x'] == 'yes').astype(float). "
                f"For multi-class targets specifically, see migration paths: (1) "
                f"MulticlassScorer in long format for a single multi-class target; "
                f"(2) one-hot encode multi-class targets into binary columns; (3) wait "
                f"for the planned mixed-type multi-target scorer."
            )
        if degenerate and self.on_degenerate_target == DegenerateTargetPolicy.RAISE:
            details = ", ".join(f"{col!r} (only value: {value!r})" for col, value in degenerate.items())
            raise ValueError(
                f"MultioutputScorer: target column(s) with a single class in the training "
                f"slice cannot be fit: {details}. Drop the column(s) before training, or pass "
                f"on_degenerate_target=DegenerateTargetPolicy.CONSTANT (or the string 'constant') "
                f"to fall back to a constant predictor for them."
            )

        # Pre-mutation guard: under CONSTANT, if EVERY target is degenerate
        # there's nothing for the underlying classifier to fit on. The
        # downstream `_validate_interactions` second-call would catch this
        # with "must contain at least 2 ITEM columns" — technically loud
        # but misleadingly named (the targets WERE present, they just had
        # no signal). Check BEFORE mutating self.degenerate_targets or
        # interactions_df, so a caught ValueError leaves the scorer state
        # and the (already-copied) DataFrame both pristine.
        if degenerate and len(degenerate) == self.item_count:
            details = ", ".join(f"{col!r} (only value: {value!r})" for col, value in degenerate.items())
            raise ValueError(
                f"MultioutputScorer: every target column is degenerate (single-class) in "
                f"the training slice — there's nothing for the underlying classifier to "
                f"fit on. on_degenerate_target='constant' would drop all of them and "
                f"leave an empty y matrix. Affected columns: {details}. Drop the columns "
                f"and use a different evaluation strategy (e.g. predict_classes returns "
                f"the constant per-target labels), or fix the training-slice imbalance "
                f"(e.g. use a stratified split that retains both classes per target)."
            )

        # Either no degenerate targets, or policy is CONSTANT and at least
        # one non-degenerate target remains — drop the degenerate columns
        # from interactions_df so they bypass the underlying
        # MultiOutputClassifier.fit, and remember which value to emit at
        # score time. One df.drop with the full column list (O(n_cols))
        # instead of N drops in a loop (O(n_cols²)).
        if degenerate:
            self.degenerate_targets.update(degenerate)
            interactions_df.drop(columns=list(degenerate.keys()), inplace=True)
            for col, seen_value in degenerate.items():
                logger.warning(
                    "MultioutputScorer: target column %r has only one class (%r) in training "
                    "data; on_degenerate_target='constant' is active so a constant predictor will "
                    "be used. Per-target classification metrics will be undefined for this target.",
                    col,
                    seen_value,
                )

        # Snapshot the fitted-target order in named state. This is the order
        # the underlying MultiOutputClassifier sees as columns of y at fit
        # time (item_names minus degenerate). predict_proba on the same
        # estimator returns its list in this order per sklearn's documented
        # contract; _calculate_scores reads back via name → index lookup
        # against this snapshot, so the implicit positional dependency is
        # made explicit and visible to anyone reading the code.
        self._fitted_target_order = [c for c in self.item_names if c not in self.degenerate_targets]

    def _calculate_scores(
        self, joined: Union[pd.DataFrame, NDArray]
    ) -> Union[List[NDArray[np.float64]], NDArray[np.float64]]:
        if not self.is_classifier:
            return np.asarray(self.estimator.predict(joined))

        raw = self.estimator.predict_proba(joined)

        # Defensive count guard: predict_proba should return exactly one
        # block per fitted target. If sklearn ever drops or duplicates
        # blocks between fit and predict, this catches the count drift
        # before it produces silently misrouted predictions. Order drift
        # (same length, shuffled) is not catchable here — the name-keyed
        # lookup below pins our dependence on the documented contract in
        # named state (`_fitted_target_order`) so a maintainer can see the
        # assumption and downstream tests on a fresh fit would surface
        # any inversion via wrong predictions.
        n_fitted_expected = len(self._fitted_target_order)
        if len(raw) != n_fitted_expected:
            raise RuntimeError(
                f"MultiOutputClassifier.predict_proba returned {len(raw)} target blocks "
                f"but {n_fitted_expected} non-degenerate target(s) were fit "
                f"({len(self.item_names)} total - {len(self.degenerate_targets)} degenerate). "
                f"Two known causes: (1) sklearn's predict_proba contract changed and is "
                f"now returning a different number of blocks than at fit time — please "
                f"file a bug; (2) `scorer.degenerate_targets` or `scorer._fitted_target_order` "
                f"was mutated externally after `train()` — don't do that, fit the scorer "
                f"freshly instead. The per-target name-keyed lookup in _calculate_scores "
                f"depends on the snapshot pinned at fit time."
            )

        # Build a name → raw-index map from the fit-time snapshot so the
        # mapping is by-name rather than by-position. With sklearn's current
        # contract these are equivalent, but the named version is the spec
        # the rest of the code reads against.
        fitted_index = {name: i for i, name in enumerate(self._fitted_target_order)}

        if not self.degenerate_targets:
            # Fast path: no interleave needed, but still iterate in
            # self.item_names order via the named lookup so the no-shuffle
            # contract is visible at every call site.
            return [raw[fitted_index[col_name]] for col_name in self.item_names]

        # Interleave constant predictions for degenerate targets so the
        # returned list aligns with self.item_names. For each degenerate
        # target, emit (n_rows, 2): probability 1.0 for the seen class,
        # 0.0 for the never-observed class. This preserves shape uniformity
        # downstream (every target has 2 proba columns).
        n_rows = len(joined) if hasattr(joined, "__len__") else joined.shape[0]
        full: List[NDArray[np.float64]] = []
        for col_name in self.item_names:
            if col_name in self.degenerate_targets:
                # `seen` is in {0.0, 1.0} per `_validate_targets` — the
                # int() cast is exact (not lossy rounding) and resolves
                # to the column index in proba where the constant 1.0
                # belongs.
                seen = self.degenerate_targets[col_name]
                proba = np.zeros((n_rows, 2), dtype=np.float64)
                proba[:, int(seen)] = 1.0
                full.append(proba)
            else:
                full.append(raw[fitted_index[col_name]])
        return full

    def score_fast(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return predictions for all items for a single pre-merged user row.

        Returns predicted class labels (classifier mode) or predicted values
        (regressor mode), with one column per item.

        Raises:
            ValueError: If ``features`` has more than one row.
        """
        if features.shape[0] != 1:
            batch_method = "predict_classes()" if self.is_classifier else "predict_targets()"
            raise ValueError(
                f"score_fast() expects exactly 1 row, got {features.shape[0]}. Use {batch_method} for batch scoring."
            )
        drop_cols = [col for col in [C.USER_ID_NAME, C.ITEM_ID_NAME, C.LABEL_NAME] if col in features.columns]
        if drop_cols:
            features = features.drop(columns=drop_cols)
        scores = self._calculate_scores(features)
        return self._create_df_from_scores(scores)

    def predict_classes(
        self,
        interactions: Optional[pd.DataFrame] = None,
        users: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predicted class label per item (classifier mode only).

        Returns a DataFrame of shape ``(n_users, n_items)`` with one column
        per ``ITEM_<name>`` carrying the argmax class — ``0`` or ``1`` —
        per user. Callers who want their original (pre-encoded) labels back
        keep their own mapping; the scorer itself doesn't store one because
        the binary-only contract pins targets to numeric ``{0, 1}``.

        For per-class probabilities, call ``score_items`` instead.
        """
        if not self.is_classifier:
            raise NotImplementedError(
                "predict_classes is only defined for classifier estimators. "
                "For regressor estimators use predict_targets / score_items."
            )
        if users is not None:
            raise ValueError("Multioutput Scorer cannot accept Users Dataframe, set it to None!")
        return super().score_items(interactions, users)

    def predict_targets(
        self,
        interactions: Optional[pd.DataFrame] = None,
        users: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predicted continuous values per target (regressor mode only).

        Returns a DataFrame of shape ``(n_users, n_items)`` with one column per
        ``ITEM_<name>`` carrying the regressor's predicted value.
        """
        if self.is_classifier:
            raise NotImplementedError(
                "predict_targets is only defined for regressor estimators. "
                "For classifier estimators use predict_classes / score_items."
            )
        if users is not None:
            raise ValueError("Multioutput Scorer cannot accept Users Dataframe, set it to None!")
        return super().score_items(interactions, users)

    def score_items(
        self,
        interactions: Optional[pd.DataFrame] = None,
        users: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Per-target scores.

        Classifier mode: per-class probability columns, uniformly named
        ``ITEM_<name>_0`` and ``ITEM_<name>_1`` — shape
        ``(n_users, 2 * n_targets)``.

        Regressor mode: predicted values, one column per target named
        ``ITEM_<name>`` — shape ``(n_users, n_items)``.
        """
        if users is not None:
            raise ValueError("Multioutput Scorer cannot accept Users Dataframe, set it to None!")
        user_interactions_df = self._get_user_interactions_df(interactions=interactions, users=users)
        scores = self._calculate_scores(user_interactions_df)
        if self.is_classifier:
            return self._create_proba_df(scores)
        return self._create_value_df(scores)

    def _create_df_from_scores(self, scores: Union[List[NDArray[np.float64]], NDArray[np.float64]]) -> pd.DataFrame:
        """Predicted class label / value per item — one column per item.

        Output column order honours ``item_subset`` when one is set
        (alphabetical, per ``BaseScorer._process_item_subset``); otherwise
        uses ``item_names`` order.
        """
        if not self.is_classifier:
            return self._create_value_df(scores)
        order = list(self.item_subset) if self.item_subset else list(self.item_names)
        name_to_idx = {name: i for i, name in enumerate(self.item_names)}
        result: Dict[str, NDArray] = {}
        for col_name in order:
            col_num = name_to_idx[col_name]
            if col_name in self.degenerate_targets:
                # Constant prediction: always the seen class. `seen` is in
                # {0.0, 1.0} per `_validate_targets` so int() is exact.
                n_rows = scores[col_num].shape[0]
                result[col_name] = np.full(n_rows, int(self.degenerate_targets[col_name]))
            else:
                result[col_name] = np.argmax(scores[col_num], axis=1)
        return pd.DataFrame(result)

    def _create_proba_df(self, scores: List[NDArray[np.float64]]) -> pd.DataFrame:
        """Per-class probability columns (classifier mode).

        Column names are uniformly ``ITEM_<name>_0`` / ``ITEM_<name>_1``
        regardless of input dtype — the binary-only contract guarantees
        every target has exactly these two classes.
        """
        order = list(self.item_subset) if self.item_subset else list(self.item_names)
        name_to_idx = {name: i for i, name in enumerate(self.item_names)}
        dfs = []
        for col_name in order:
            col_num = name_to_idx[col_name]
            col_names = [f"{col_name}_0", f"{col_name}_1"]
            dfs.append(pd.DataFrame(scores[col_num], columns=col_names, dtype=np.float64))
        return pd.concat(dfs, axis=1)

    def _create_value_df(self, scores: NDArray[np.float64]) -> pd.DataFrame:
        """One column per target with predicted values (regressor mode)."""
        scores_arr = np.atleast_2d(np.asarray(scores))
        if scores_arr.shape[1] != len(self.item_names):
            raise ValueError(
                f"Fitted target count vs. predicted target count mismatch: regressor "
                f"predict returned shape {scores_arr.shape} but the scorer's catalogue "
                f"has {len(self.item_names)} target(s) ({self.item_names}). The "
                f"≥2 ITEM_* fit-time check should make this unreachable for the "
                f"scorer's normal flow — if you're hitting this, please report as a "
                f"bug with the call sequence that triggered it."
            )
        order = list(self.item_subset) if self.item_subset else list(self.item_names)
        name_to_idx = {name: i for i, name in enumerate(self.item_names)}
        result: Dict[str, NDArray] = {}
        for col_name in order:
            col_num = name_to_idx[col_name]
            result[col_name] = scores_arr[:, col_num].astype(np.float64)
        return pd.DataFrame(result)

    def score_items_per_target(
        self,
        interactions: Optional[pd.DataFrame] = None,
        users: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Per-target relevance scores — one column per target.

        Classifier mode: returns ``P(target = 1)`` per target. DataFrame shape
        ``(n_users, n_targets)``, columns are the original ``ITEM_<name>``
        identifiers. This is the canonical "relevance score per label" used
        for cross-target ranking (recommend, NDCG, Precision@K, ...).

        Regressor mode: returns the predicted continuous values, identical
        to ``score_items``.

        Args:
            interactions: DataFrame with interaction features.
            users: Must be ``None``.

        Returns:
            DataFrame of shape ``(n_users, n_targets)`` with one
            relevance / value per target.

        Notes:
            Classifier-mode positive-class extraction relies on sklearn's
            documented ``MultiOutputClassifier`` contract that each fitted
            estimator's ``classes_`` is sorted ascending — so ``predict_proba``
            column 1 is ``P(class=1)``. The binary-only fit-time contract
            pins target values to ``{0, 1}``, so the positive class is
            always at index 1 regardless of input dtype. If you upgrade
            sklearn and predictions look inverted (e.g. metrics drop
            sharply with no model change), check whether the
            ``classes_``-ascending convention still holds in your sklearn
            version.
        """
        if users is not None:
            raise ValueError("Multioutput Scorer cannot accept Users Dataframe, set it to None!")
        if not self.is_classifier:
            return self.score_items(interactions=interactions, users=users)

        # Hot path — `recommend()` calls this per-request. Avoid the round
        # trip through score_items's per-class DataFrame (which builds a
        # (N, 2T) frame just to extract T columns). Pull the positive-class
        # probability directly from the per-target arrays in
        # _calculate_scores, which is already (N, 2) per target.
        user_interactions_df = self._get_user_interactions_df(interactions=interactions, users=users)
        scores = self._calculate_scores(user_interactions_df)
        order = list(self.item_subset) if self.item_subset else list(self.item_names)
        name_to_idx = {name: i for i, name in enumerate(self.item_names)}
        result: Dict[str, NDArray[np.float64]] = {}
        for col_name in order:
            col_num = name_to_idx[col_name]
            # Positive-class column is index 1: sklearn sorts classes_
            # ascending and our binary-only fit-time contract pins targets
            # to {0, 1}, so column 1 is always P(class=1). For degenerate
            # targets _calculate_scores emits (N, 2) with the seen class at
            # probability 1.0, so [:, 1] correctly returns 0.0 for an
            # all-zero degenerate and 1.0 for an all-one one.
            result[col_name] = scores[col_num][:, 1].astype(np.float64)
        return pd.DataFrame(result)

    # ------------------------------------------------------------------
    # _score_items_np override for ranking-aware downstream use
    # ------------------------------------------------------------------
    # BaseScorer._score_items_np returns score_items().to_numpy(), which for
    # the classifier mode is shape (N, 2 * n_targets) — not usable for
    # ranking. We override here so that downstream callers that expect
    # (N, n_items) (e.g. RankingRecommender.recommend's base path, the
    # BaseRecommender eval-score-bundle builder) get the per-target
    # positive-class score matrix instead. score_items's public per-class
    # contract is unchanged for direct callers.
    def _score_items_np(
        self,
        interactions: Optional[pd.DataFrame] = None,
        users: Optional[pd.DataFrame] = None,
    ) -> NDArray[np.float64]:
        return self.score_items_per_target(interactions=interactions, users=users).to_numpy()
