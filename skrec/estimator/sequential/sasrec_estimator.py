from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from skrec.util.logger import get_logger

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from skrec.constants import ITEM_ID_NAME, USER_ID_NAME
from skrec.estimator.sequential.base_sequential_estimator import SequentialEstimator

logger = get_logger(__name__)

# Column names for sequence data (produced by SequentialRecommender, consumed by estimator)
ITEM_SEQUENCE_COL = "ITEM_SEQUENCE"
OUTCOME_SEQUENCE_COL = "OUTCOME_SEQUENCE"


_nn_module = nn.Module if nn is not None else object

class _SASRecTransformerBlock(_nn_module):
    """Single transformer block for SASRec: pre-norm, causal self-attention + FFN."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.attention_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.ffn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate),
        )
        self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        real_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm self-attention with residual.
        # attn_mask is a float (B*H, L, L) additive mask combining causal + padding-key blocking
        # with large finite negatives (-1e9) instead of -inf. This prevents
        # softmax([-inf,...,-inf]) = NaN for padding queries (all-masked rows).
        normed = self.attention_layernorm(x)
        attn_out, _ = self.attention(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            need_weights=False,
        )
        # Zero out padding positions so they don't contaminate real positions via residual.
        attn_out = attn_out * real_mask
        x = x + self.attn_dropout(attn_out)

        # Pre-norm FFN with residual
        normed = self.ffn_layernorm(x)
        ffn_out = self.ffn(normed)
        ffn_out = ffn_out * real_mask
        x = x + ffn_out
        return x


class _SASRecModule(_nn_module):
    """
    SASRec (Self-Attentive Sequential Recommendation) PyTorch module.

    Uses a stack of transformer blocks with causal masking to encode item sequences.
    Item 0 is reserved as the padding token.

    Reference:
        Kang, W. C., & McAuley, J. (2018). Self-attentive sequential recommendation.
        In ICDM 2018.
    """

    def __init__(
        self,
        num_items: int,
        hidden_units: int,
        max_len: int,
        num_blocks: int,
        num_heads: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.max_len = max_len

        # Item 0 is padding; real items are indexed 1..num_items
        self.item_embedding = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_units)

        self.transformer_blocks = nn.ModuleList(
            [_SASRecTransformerBlock(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)]
        )
        self.output_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    _INIT_STD = 0.02
    _MASK_VALUE = -1e9

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self._INIT_STD)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self._INIT_STD)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode item sequence through transformer blocks.

        Args:
            item_seq: (batch_size, seq_len) item indices, 0 for padding.

        Returns:
            (batch_size, seq_len, hidden_units) sequence representations.
        """
        batch_size, seq_len = item_seq.shape

        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)  # (1, seq_len)
        # Input embedding: item + position, no scaling.
        # The original SASRec paper (Kang & McAuley, 2018) does not scale input embeddings.
        # Applying sqrt(hidden_units) here amplifies the input-path gradient 7× relative to
        # the scoring-path gradient for weight-tied embeddings, causing item embeddings to
        # optimize only for sequence context and collapse to near-uniform scoring vectors.
        x = self.item_embedding(item_seq) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Build a combined float attention mask (additive, not boolean) that encodes
        # both causal masking and padding-key masking using -1e9 instead of -inf.
        #
        # Why: PyTorch's key_padding_mask adds -inf for masked keys. When a padding QUERY
        # has ALL its allowed keys masked (right-aligned padding, early positions), softmax
        # receives all -inf inputs → NaN. Even with nan_to_num in the forward pass, the
        # backward through softmax computes 0 * NaN = NaN, corrupting embeddings after
        # the first optimizer step. Using a large finite value (-1e9) instead means
        # softmax([c,...,c]) = [1/n,...,1/n] (uniform, finite) for all-masked rows.

        # Causal part: upper triangle → -1e9 for j > i (shape: L×L)
        causal_bias = torch.triu(
            torch.full((seq_len, seq_len), self._MASK_VALUE, device=item_seq.device),
            diagonal=1,
        )

        # Padding-key part: -1e9 for positions j where item_seq[:,j] == 0 (shape: B×1×L)
        key_pad_bias = item_seq.eq(0).float().unsqueeze(1) * self._MASK_VALUE  # (B, 1, L)

        # Combined: (B, L, L) — broadcast-sum of (1, L, L) causal and (B, 1, L) padding-key
        combined_mask = causal_bias.unsqueeze(0) + key_pad_bias  # (B, L, L)

        # Expand to (B*num_heads, L, L) as required by MultiheadAttention for per-sample masks
        num_heads = self.transformer_blocks[0].attention.num_heads
        combined_mask = (
            combined_mask.unsqueeze(1)  # (B, 1, L, L)
            .expand(-1, num_heads, -1, -1)  # (B, H, L, L)
            .reshape(batch_size * num_heads, seq_len, seq_len)  # (B*H, L, L)
        )

        # Real-position mask for zeroing padding outputs (1 for real, 0 for padding)
        real_mask = item_seq.ne(0).float().unsqueeze(-1)  # (B, L, 1)

        for block in self.transformer_blocks:
            x = block(x, combined_mask, real_mask)

        # Zero out padding positions in final output (their layernorm output = bias, not meaningful)
        return self.output_layernorm(x) * real_mask  # (batch_size, seq_len, hidden_units)

    def score_all_items(self, seq_repr: torch.Tensor) -> torch.Tensor:
        """
        Score all real items (excluding padding token at index 0).

        Args:
            seq_repr: (batch_size, hidden_units) user representations.

        Returns:
            (batch_size, num_items) scores.
        """
        # item_embedding.weight shape: (num_items+1, hidden_units)
        # Skip index 0 (padding), score real items 1..num_items
        item_weights = self.item_embedding.weight[1:]  # (num_items, hidden_units)
        return seq_repr @ item_weights.T  # (batch_size, num_items)


class _SASRecBaseEstimator(SequentialEstimator):
    """
    Base estimator for SASRec. Handles sequence data preparation, training loop,
    and inference. Subclasses differ only in loss function.

    Unlike NCF/other embedding estimators, SASRec:
    - Has no user embedding table (user is represented by their item sequence)
    - Receives sequences_df (USER_ID, ITEM_SEQUENCE, OUTCOME_SEQUENCE) not raw interactions
    - Produces (num_users, num_items) scores at inference (no cross-join needed)
    """

    def __init__(
        self,
        hidden_units: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
        num_negatives: int = 1,
        max_len: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        optimizer_name: Union[str, Type[torch.optim.Optimizer]] = "adam",
        loss_fn_name: Union[str, nn.Module] = "bce",
        weight_decay: float = 0.0,
        early_stopping_patience: Optional[int] = None,
        restore_best_weights: bool = True,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            loss_fn_name=loss_fn_name,
            weight_decay=weight_decay,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_negatives = num_negatives
        self.max_len = max_len
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_weights = restore_best_weights

        # Set during fit_embedding_model
        self._item_to_idx: Dict[str, int] = {}  # item_id → 1-indexed embedding index

    def _build_pytorch_model(self) -> _SASRecModule:
        return _SASRecModule(
            num_items=self.num_items,
            hidden_units=self.hidden_units,
            max_len=self.max_len,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )

    def fit_embedding_model(
        self,
        users: Optional[pd.DataFrame],
        items: Optional[pd.DataFrame],
        interactions: pd.DataFrame,
        valid_users: Optional[pd.DataFrame] = None,
        valid_interactions: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Train SASRec on sequence data.

        Args:
            users: Ignored (SASRec has no user embeddings).
            items: DataFrame with ITEM_ID column. Used to build the item vocabulary.
            interactions: sequences_df with USER_ID, ITEM_SEQUENCE, OUTCOME_SEQUENCE columns.
                          Produced by SequentialRecommender._build_sequences().
            valid_users: Ignored.
            valid_interactions: Optional sequences_df for validation (same format as interactions).
                                Required when early_stopping_patience is set.
                                Produced by SequentialRecommender from the leave-last-out
                                validation split (all interactions minus the held-out test item).
        """
        self._user_data_truncated = False

        if self.early_stopping_patience is not None and valid_interactions is None:
            raise ValueError("early_stopping_patience requires valid_interactions to be provided.")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        # Build item vocabulary (1-indexed; index 0 is padding).
        # Normalize all item IDs to strings so the vocabulary key type is consistent
        # regardless of whether the caller passes int64 (from InteractionsDataset) or
        # str (from raw pandas DataFrames). This prevents silent all-zero seq_tensors
        # at inference when the caller's ITEM_ID dtype differs from the vocabulary type.
        if items is not None and not items.empty:
            item_ids = sorted({str(item) for item in items[ITEM_ID_NAME]})
        else:
            item_ids = sorted({str(item) for seq in interactions[ITEM_SEQUENCE_COL] for item in seq})

        self.item_id_index = pd.Index(item_ids)
        self.num_items = len(self.item_id_index)
        self._item_to_idx = {item: idx + 1 for idx, item in enumerate(self.item_id_index)}

        # User index (used for get_user_embeddings compatibility; SASRec doesn't learn user embeddings)
        self.user_id_index = pd.Index(interactions[USER_ID_NAME].values)
        self.num_users = len(self.user_id_index)
        self.unknown_user_idx = self.num_users
        self.unknown_item_idx = 0  # padding index doubles as unknown

        # Build padded input / target / outcome tensors
        seq_tensor, target_tensor, outcome_tensor = self._build_padded_tensors(interactions)

        # Build and move model to device
        self.model = self._build_pytorch_model()
        self.model.to(self.device)

        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.to(self.device)

        optimizer = self.optimizer_cls(  # type: ignore[call-arg]
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        val_seq_tensor: Optional[torch.Tensor] = None
        val_target_tensor: Optional[torch.Tensor] = None
        val_outcome_tensor: Optional[torch.Tensor] = None
        if valid_interactions is not None:
            val_seq_tensor, val_target_tensor, val_outcome_tensor = self._build_padded_tensors(valid_interactions)

        self._sasrec_training_loop(
            seq_tensor,
            target_tensor,
            outcome_tensor,
            optimizer,
            val_seq_tensor,
            val_target_tensor,
            val_outcome_tensor,
        )

    def _build_padded_tensors(self, sequences_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert sequences_df rows into right-aligned input/target/outcome tensors.

        Follows the original SASRec paper's training protocol exactly:
        - Input  (seq_tensor):    [item_1, ..., item_{n-1}]  — excludes the last item
        - Target (target_tensor): [item_2, ..., item_n]      — excludes the first item
        Both are right-aligned to max_len so that item_{n-1} always sits at position
        max_len-1. This is identical to the evaluation input, so positional embeddings
        are consistent between training and inference.

        The last training target (at position max_len-1) is item_n — the held-out test
        item. The model is explicitly trained to predict it from the full preceding
        history, which is exactly the task evaluated at test time.

        SequentialRecommender._build_sequences() truncates each history to max_len+1
        before passing it here, so input_seq (full_seq[:-1]) has exactly max_len items
        for users with long histories — every positional slot is fully utilized.

        Args:
            sequences_df: DataFrame with ITEM_SEQUENCE_COL and optionally
                          OUTCOME_SEQUENCE_COL.  Each row's ITEM_SEQUENCE must
                          contain at least 2 items; single-item rows are skipped.

        Returns:
            seq_tensor:    (num_users, max_len) input item indices (0 = padding).
            target_tensor: (num_users, max_len) target item indices (0 = padding).
            outcome_tensor:(num_users, max_len) reward for each target (default 1.0).
        """
        n_users = len(sequences_df)
        max_len: int = self.max_len
        has_outcome = OUTCOME_SEQUENCE_COL in sequences_df.columns

        seq_array = np.zeros((n_users, max_len), dtype=np.int64)
        target_array = np.zeros((n_users, max_len), dtype=np.int64)
        outcome_array = np.ones((n_users, max_len), dtype=np.float32)

        item_seqs = sequences_df[ITEM_SEQUENCE_COL].tolist()
        outcome_seqs = sequences_df[OUTCOME_SEQUENCE_COL].tolist() if has_outcome else None

        # Collect flat arrays for a single vectorized item-to-index lookup.
        # Building position metadata in Python is unavoidable for variable-length sequences,
        # but the expensive part — string comparison for vocabulary lookup — is deferred to
        # a single pd.Index.get_indexer() call which runs at C speed.
        flat_user_idx: List[np.ndarray] = []
        flat_positions: List[np.ndarray] = []
        flat_input_items: List[str] = []
        flat_target_items: List[str] = []
        flat_outcome_vals: List[float] = []

        for i, full_seq in enumerate(item_seqs):
            # Sequences are pre-truncated to max_len+1 by _build_sequences;
            # the guard below handles edge cases (e.g. direct API calls).
            full_seq = full_seq[-(max_len + 1) :]
            if len(full_seq) < 2:
                continue

            input_seq = full_seq[:-1]
            target_seq = full_seq[1:]
            seq_len = len(input_seq)
            start = max_len - seq_len

            flat_user_idx.append(np.full(seq_len, i, dtype=np.int64))
            flat_positions.append(np.arange(start, max_len, dtype=np.int64))
            flat_input_items.extend(str(x) for x in input_seq)
            flat_target_items.extend(str(x) for x in target_seq)

            if outcome_seqs is not None:
                full_out = outcome_seqs[i][-(max_len + 1) :]
                flat_outcome_vals.extend(float(v) for v in full_out[1:])

        if flat_user_idx:
            user_idx = np.concatenate(flat_user_idx)
            positions = np.concatenate(flat_positions)

            # Vectorized C-level vocabulary lookup; unknown items map to -1 → +1 = 0 (padding)
            input_idx = self.item_id_index.get_indexer(flat_input_items) + 1
            target_idx = self.item_id_index.get_indexer(flat_target_items) + 1

            seq_array[user_idx, positions] = input_idx
            target_array[user_idx, positions] = target_idx

            if flat_outcome_vals:
                outcome_array[user_idx, positions] = flat_outcome_vals

        return (
            torch.from_numpy(seq_array),
            torch.from_numpy(target_array),
            torch.from_numpy(outcome_array),
        )

    def _build_inference_tensor(self, sequences_df: pd.DataFrame) -> torch.Tensor:
        """Build right-aligned padded sequence tensor for inference (no outcome needed)."""
        n_users = len(sequences_df)
        max_len: int = self.max_len

        seq_array = np.zeros((n_users, max_len), dtype=np.int64)

        item_seqs = sequences_df[ITEM_SEQUENCE_COL].tolist()

        flat_user_idx: List[np.ndarray] = []
        flat_positions: List[np.ndarray] = []
        flat_items: List[str] = []

        for i, seq in enumerate(item_seqs):
            seq = seq[-max_len:]
            seq_len = len(seq)
            if seq_len == 0:
                continue
            start = max_len - seq_len
            flat_user_idx.append(np.full(seq_len, i, dtype=np.int64))
            flat_positions.append(np.arange(start, max_len, dtype=np.int64))
            flat_items.extend(str(x) for x in seq)

        if flat_user_idx:
            user_idx = np.concatenate(flat_user_idx)
            positions = np.concatenate(flat_positions)
            # Vectorized C-level lookup; unknown items map to -1 → +1 = 0 (padding)
            item_idx = self.item_id_index.get_indexer(flat_items) + 1
            seq_array[user_idx, positions] = item_idx

        return torch.from_numpy(seq_array)

    def _compute_sequence_loss(
        self,
        hidden: torch.Tensor,
        seq: torch.Tensor,
        target_items: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute pointwise loss over all valid (source position, target item) pairs.

        ``seq`` and ``target_items`` share the same right-aligned positional layout:
        at position t, seq[t] is the input item and target_items[t] is the item the
        model should predict next.  Both were built by ``_build_padded_tensors`` so
        they have the same non-zero span.

        Args:
            hidden:       (batch_size, max_len, hidden_units)
            seq:          (batch_size, max_len) input item indices (0 = padding)
            target_items: (batch_size, max_len) target item indices (0 = padding)
            outcomes:     (batch_size, max_len) reward for each target item

        Returns:
            Scalar loss tensor, or None if no valid positions in the batch.
        """
        # A position is valid when both the source item (seq) and its target are real.
        valid_mask = (seq != 0) & (target_items != 0)  # (B, L)
        if not valid_mask.any():
            return None

        repr_flat = hidden[valid_mask]  # (N_valid, H)
        target_flat = target_items[valid_mask]  # (N_valid,)
        reward_flat = outcomes[valid_mask]  # (N_valid,)

        item_emb = self.model.item_embedding.weight  # type: ignore[union-attr]  # (num_items+1, H)

        # Positive: score for the actual target item, supervised by its reward
        pos_emb = item_emb[target_flat]  # (N_valid, H)
        pos_scores = (repr_flat * pos_emb).sum(dim=1, keepdim=True)  # (N_valid, 1)
        pos_loss = self.loss_fn(pos_scores, reward_flat.unsqueeze(1))

        if self.num_negatives == 0:
            return pos_loss

        # Random negatives: sample items not seen by the user (approximate)
        neg_items = torch.randint(
            1,
            self.num_items + 1,
            (repr_flat.shape[0], self.num_negatives),
            device=self.device,
        )  # (N_valid, num_negatives)
        neg_emb = item_emb[neg_items]  # (N_valid, num_negatives, H)
        neg_scores = (repr_flat.unsqueeze(1) * neg_emb).sum(dim=2)  # (N_valid, num_negatives)
        neg_targets = torch.zeros_like(neg_scores)
        neg_loss = self.loss_fn(neg_scores, neg_targets)

        # Normalise by num_negatives so the total negative gradient contribution
        # stays 1:1 with positive regardless of how many negatives are sampled.
        # At num_negatives=1 this is a no-op; without it, higher num_negatives
        # up-weights the push-down signal and can cause embedding collapse for
        # frequently-sampled popular items.
        return pos_loss + neg_loss / self.num_negatives

    def _sasrec_training_loop(
        self,
        seq_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        outcome_tensor: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        val_seq_tensor: Optional[torch.Tensor] = None,
        val_target_tensor: Optional[torch.Tensor] = None,
        val_outcome_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Run the SASRec mini-batch training loop.

        Each epoch shuffles users, iterates over mini-batches, computes the
        next-item prediction loss via ``_compute_sequence_loss``, and
        back-propagates.  Optionally evaluates a held-out validation split and
        applies early stopping with best-weight restoration.

        Args:
            seq_tensor: Padded input-sequence tensor of shape
                ``(n_users, max_len)`` with item indices (0 = padding).
            target_tensor: Padded target-sequence tensor of shape
                ``(n_users, max_len)`` with item indices (0 = padding).
            outcome_tensor: Padded outcome tensor of shape
                ``(n_users, max_len)`` with float labels (0 for padding
                positions).
            optimizer: Configured optimizer wrapping the SASRec model
                parameters.
            val_seq_tensor: Optional validation input-sequence tensor.
            val_target_tensor: Optional validation target-sequence tensor.
            val_outcome_tensor: Optional validation outcome tensor.
        """
        n_users = seq_tensor.shape[0]
        indices = np.arange(n_users)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state: Optional[dict] = None
        _warned_no_val_batches = False

        self.model.train()  # type: ignore[union-attr]
        for epoch in range(self.epochs):
            if self.random_state is not None:
                np.random.seed(self.random_state + epoch)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n_users, self.batch_size):
                end = min(start + self.batch_size, n_users)
                batch_idx = indices[start:end]

                batch_seq = seq_tensor[batch_idx].to(self.device)
                batch_targets = target_tensor[batch_idx].to(self.device)
                batch_outcomes = outcome_tensor[batch_idx].to(self.device)

                optimizer.zero_grad()

                hidden = self.model(batch_seq)  # type: ignore[operator]  # (B, max_len, H)
                loss = self._compute_sequence_loss(hidden, batch_seq, batch_targets, batch_outcomes)

                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

            # Compute validation loss if validation tensors are provided
            val_loss_str = ""
            if val_seq_tensor is not None and val_target_tensor is not None and val_outcome_tensor is not None:
                self.model.eval()  # type: ignore[union-attr]
                val_loss_total = 0.0
                val_batches = 0
                n_val = val_seq_tensor.shape[0]
                with torch.no_grad():
                    for start in range(0, n_val, self.batch_size):
                        end = min(start + self.batch_size, n_val)
                        v_seq = val_seq_tensor[start:end].to(self.device)
                        v_targets = val_target_tensor[start:end].to(self.device)
                        v_outcomes = val_outcome_tensor[start:end].to(self.device)
                        v_hidden = self.model(v_seq)  # type: ignore[operator]
                        v_loss = self._compute_sequence_loss(v_hidden, v_seq, v_targets, v_outcomes)
                        if v_loss is not None:
                            val_loss_total += v_loss.item()
                            val_batches += 1
                self.model.train()  # type: ignore[union-attr]
                if val_batches == 0:
                    if not _warned_no_val_batches:
                        logger.warning(
                            "No valid validation batches found — skipping early stopping check. "
                            "This warning will not repeat for subsequent epochs."
                        )
                        _warned_no_val_batches = True
                    continue
                val_loss = val_loss_total / val_batches
                val_loss_str = f", Val Loss: {val_loss:.4f}"

                if self.early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        if self.restore_best_weights:
                            best_state = copy.deepcopy(self.model.state_dict())  # type: ignore[union-attr]
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.early_stopping_patience:
                            if self.verbose > 0:
                                logger.info(
                                    f"Early stopping at epoch {epoch + 1}/{self.epochs} — "
                                    f"val loss did not improve for {self.early_stopping_patience} epoch(s). "
                                    f"Best val loss: {best_val_loss:.4f}"
                                )
                            if self.restore_best_weights:
                                if best_state is not None:
                                    self.model.load_state_dict(best_state)  # type: ignore[union-attr]
                                else:
                                    logger.warning(
                                        "restore_best_weights=True but no improvement was ever recorded "
                                        "(val loss never decreased from its initial value). "
                                        "Model weights are from the final epoch."
                                    )
                            break

            if self.verbose > 0 and num_batches > 0:
                logger.info(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss / num_batches:.4f}{val_loss_str}")

    def predict_proba_with_embeddings(
        self,
        interactions: pd.DataFrame,
        users: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Score all items for each user sequence.

        Args:
            interactions: sequences_df with USER_ID and ITEM_SEQUENCE columns.
            users: Ignored (SASRec has no user embedding lookup).

        Returns:
            (num_users, num_items) numpy array of scores, aligned with self.item_id_index order.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit_embedding_model() first.")

        seq_tensor = self._build_inference_tensor(interactions)

        self.model.eval()  # type: ignore[union-attr]
        all_scores: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(seq_tensor), self.batch_size):
                end = min(start + self.batch_size, len(seq_tensor))
                batch_seq = seq_tensor[start:end].to(self.device)

                hidden = self.model(batch_seq)  # type: ignore[operator]  # (B, max_len, H)

                # With right-aligned padding, the most recent item is always at the last position.
                seq_repr = hidden[:, -1, :]  # (B, H)

                scores = self.model.score_all_items(seq_repr)  # type: ignore[union-attr]  # (B, num_items)
                all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores, axis=0)  # (num_users, num_items)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "hidden_units": self.hidden_units,
                "num_blocks": self.num_blocks,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "num_negatives": self.num_negatives,
                "max_len": self.max_len,
                "early_stopping_patience": self.early_stopping_patience,
                "restore_best_weights": self.restore_best_weights,
            }
        )
        return config


class SASRecClassifierEstimator(_SASRecBaseEstimator):
    """
    SASRec for classification-style rewards.

    Use when OUTCOME is binary (0/1) or discrete ratings (1-5).
    Default loss: Binary Cross-Entropy (reward treated as label probability).

    Customer data modes:
    - Positives only (traditional SASRec): all OUTCOME=1, use num_negatives >= 1
    - Binary 0/1: 0s serve as natural hard negatives via reward signal
    - Ratings 1-5: normalise to [0, 1] in data prep for soft-label BCE
    """

    def __init__(
        self,
        hidden_units: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
        num_negatives: int = 1,
        max_len: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        optimizer_name: Union[str, Type[torch.optim.Optimizer]] = "adam",
        loss_fn_name: Union[str, nn.Module] = "bce",
        weight_decay: float = 0.0,
        early_stopping_patience: Optional[int] = None,
        restore_best_weights: bool = True,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            hidden_units=hidden_units,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            num_negatives=num_negatives,
            max_len=max_len,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            loss_fn_name=loss_fn_name,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            restore_best_weights=restore_best_weights,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )


class SASRecRegressorEstimator(_SASRecBaseEstimator):
    """
    SASRec for continuous rewards.

    Use when OUTCOME is a continuous value (e.g., revenue, time-spent, rating as float).
    Default loss: Mean Squared Error (predicted score regresses toward reward value).

    Customer data modes:
    - Continuous rewards: MSE loss, reward magnitude directly supervises the model
    - num_negatives > 0: random unseen items receive target=0.0
    """

    def __init__(
        self,
        hidden_units: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
        num_negatives: int = 1,
        max_len: int = 50,
        learning_rate: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        optimizer_name: Union[str, Type[torch.optim.Optimizer]] = "adam",
        loss_fn_name: Union[str, nn.Module] = "mse",
        weight_decay: float = 0.0,
        early_stopping_patience: Optional[int] = None,
        restore_best_weights: bool = True,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            hidden_units=hidden_units,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            num_negatives=num_negatives,
            max_len=max_len,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            loss_fn_name=loss_fn_name,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            restore_best_weights=restore_best_weights,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
