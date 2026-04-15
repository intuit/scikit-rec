from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from skrec.constants import ITEM_ID_NAME, USER_ID_NAME
from skrec.estimator.sequential.base_sequential_estimator import SequentialEstimator
from skrec.util.logger import get_logger

logger = get_logger(__name__)

# Column names for session-structured data (produced by HierarchicalSequentialRecommender)
SESSION_SEQUENCES_COL = "SESSION_SEQUENCES"
SESSION_OUTCOMES_COL = "SESSION_OUTCOMES"


_nn_module = nn.Module if nn is not None else object


class _HRNNModule(_nn_module):
    """
    HRNN PyTorch module with a two-level GRU hierarchy.

    - Session GRU: processes item embeddings within one session.
    - User GRU: processes terminal session states across sessions.

    The user GRU's output at session s seeds the session GRU's h_0 at session s+1,
    allowing long-term preference context to influence within-session dynamics.

    Sessions are right-aligned: oldest session at index 0, most recent at index max_sessions-1.
    Items within each session are also right-aligned: last real item at max_session_len-1.
    Item index 0 is reserved as the padding token.

    Reference:
        Quadrana, M., Cremonesi, P., & Jannach, D. (2017).
        Personalizing session-based recommendations with hierarchical recurrent neural networks.
        In Proceedings of the Eleventh ACM Conference on Recommender Systems (RecSys 2017).
    """

    def __init__(
        self,
        num_items: int,
        hidden_units: int,
        num_layers: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        # Item 0 is padding; real items are indexed 1..num_items
        self.item_embedding = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)

        # Session GRU: encodes item sequence within one session.
        # h_0 is seeded by the user GRU output from the previous session.
        self.session_gru = nn.GRU(
            hidden_units,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # User GRU: updates user state from each session's terminal representation.
        # Single-layer; updates layer 0 of the session GRU's h_0.
        self.user_gru = nn.GRU(hidden_units, hidden_units, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    _INIT_STD = 0.02

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self._INIT_STD)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, sessions_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode all sessions for a batch of users.

        Args:
            sessions_tensor: (B, max_sessions, max_session_len) item indices.
                             0 for padding. Right-aligned in both dimensions.

        Returns:
            (B, max_sessions, max_session_len, hidden_units) hidden states.
            Padding positions are zeroed out.
        """
        B, max_sessions, max_session_len = sessions_tensor.shape
        H = self.hidden_units

        # Embed all items at once: (B, max_sessions, max_session_len, H)
        item_emb = self.dropout(self.item_embedding(sessions_tensor))

        # Collect per-session hidden states in a list and stack at the end.
        # Avoids in-place writes to a pre-allocated tensor, which break autograd
        # when the computation graph spans multiple iterations of this loop.
        session_hidden_states: List[torch.Tensor] = []

        # user_state: (num_layers, B, H) — h_0 for the session GRU.
        # Layer 0 is updated by the user GRU after each real session;
        # deeper layers start at zero and evolve per-session without cross-session memory.
        user_state = torch.zeros(self.num_layers, B, H, device=sessions_tensor.device)

        for s in range(max_sessions):
            session_items = sessions_tensor[:, s, :]  # (B, max_session_len)
            session_emb = item_emb[:, s, :]  # (B, max_session_len, H)

            # Run session GRU with user_state as initial hidden state
            session_h, _ = self.session_gru(session_emb, user_state)  # (B, max_session_len, H)

            # Zero out hidden states at padding positions so they don't corrupt the loss
            # or the terminal state extracted below.
            real_mask = session_items.ne(0).float().unsqueeze(-1)  # (B, max_session_len, 1)
            session_h = session_h * real_mask  # out-of-place: new tensor each iteration

            session_hidden_states.append(session_h)

            # Terminal state: last position is the last real item (right-aligned).
            terminal = session_h[:, -1:, :]  # (B, 1, H)

            # Update layer 0 of user_state via the user GRU.
            _, new_h0 = self.user_gru(terminal, user_state[0:1])  # (1, B, H)

            # Only update user_state for users where this session contains real items.
            # Padding sessions (all zeros) must not corrupt the user state.
            # Build user_state as a NEW tensor (out-of-place) — in-place slice assignment
            # (user_state[0:1] = ...) would corrupt autograd when this loop runs under
            # backward(), because user_state[0:1] is a view of a tensor already in the graph.
            session_is_real = session_items.ne(0).any(dim=1).float()  # (B,)
            mask = session_is_real.view(1, -1, 1)  # (1, B, 1)
            updated_h0 = user_state[0:1] * (1.0 - mask) + new_h0 * mask  # (1, B, H)
            if self.num_layers == 1:
                user_state = updated_h0
            else:
                user_state = torch.cat([updated_h0, user_state[1:]], dim=0)

        # Stack along the session dimension: (B, max_sessions, max_session_len, H)
        return torch.stack(session_hidden_states, dim=1)

    def score_all_items(self, seq_repr: torch.Tensor) -> torch.Tensor:
        """
        Score all real items against a batch of sequence representations.

        Args:
            seq_repr: (batch_size, hidden_units)

        Returns:
            (batch_size, num_items) scores.
        """
        item_weights = self.item_embedding.weight[1:]  # (num_items, H); skip padding at 0
        return seq_repr @ item_weights.T


class _HRNNBaseEstimator(SequentialEstimator):
    """
    Base estimator for HRNN. Handles session-structured data, training, and inference.

    Unlike NCF/SASRec estimators, HRNN:
    - Has no user embedding table (user is represented by their session history)
    - Receives sessions_df (USER_ID, SESSION_SEQUENCES, SESSION_OUTCOMES) not flat sequences
    - Produces (num_users, num_items) scores in a single forward pass
    """

    def __init__(
        self,
        hidden_units: int = 50,
        num_layers: int = 1,
        dropout_rate: float = 0.2,
        num_negatives: int = 1,
        max_sessions: int = 10,
        max_session_len: int = 20,
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
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_negatives = num_negatives
        self.max_sessions = max_sessions
        self.max_session_len = max_session_len
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_weights = restore_best_weights

        self._item_to_idx: Dict[str, int] = {}

    def _build_pytorch_model(self) -> _HRNNModule:
        return _HRNNModule(
            num_items=self.num_items,
            hidden_units=self.hidden_units,
            num_layers=self.num_layers,
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
        Train HRNN on session-structured interaction data.

        Args:
            users: Ignored (HRNN has no user embedding table).
            items: DataFrame with ITEM_ID column. Used to build the item vocabulary.
            interactions: sessions_df with USER_ID, SESSION_SEQUENCES, SESSION_OUTCOMES columns.
                          Produced by HierarchicalSequentialRecommender._build_session_sequences().
            valid_users: Ignored.
            valid_interactions: Optional sessions_df for validation (same format as interactions).
                                Required when early_stopping_patience is set.
                                Produced by HierarchicalSequentialRecommender from the
                                leave-last-out validation split.
        """
        self._user_data_truncated = False

        if self.early_stopping_patience is not None and valid_interactions is None:
            raise ValueError("early_stopping_patience requires valid_interactions to be provided.")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        # Build item vocabulary (1-indexed; 0 is the padding token).
        # Normalize all IDs to str for type-consistency (ItemsDataset returns int64).
        if items is not None and not items.empty:
            item_ids = sorted({str(item) for item in items[ITEM_ID_NAME]})
        else:
            item_ids = sorted(
                {
                    str(item)
                    for session_list in interactions[SESSION_SEQUENCES_COL]
                    for session in session_list
                    for item in session
                }
            )

        self.item_id_index = pd.Index(item_ids)
        self.num_items = len(self.item_id_index)
        self._item_to_idx = {item: idx + 1 for idx, item in enumerate(self.item_id_index)}

        # user_id_index is used for pickle compatibility (SequentialEstimator.__setstate__)
        self.user_id_index = pd.Index(interactions[USER_ID_NAME].values)
        self.num_users = len(self.user_id_index)
        self.unknown_user_idx = self.num_users
        self.unknown_item_idx = 0  # padding index doubles as unknown

        sessions_tensor, target_tensor, outcome_tensor = self._build_padded_session_tensors(interactions)

        self.model = self._build_pytorch_model()
        self.model.to(self.device)

        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.to(self.device)

        optimizer = self.optimizer_cls(  # type: ignore[call-arg]
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        val_sess_tensor: Optional[torch.Tensor] = None
        val_target_tensor: Optional[torch.Tensor] = None
        val_outcome_tensor: Optional[torch.Tensor] = None
        if valid_interactions is not None:
            val_sess_tensor, val_target_tensor, val_outcome_tensor = self._build_padded_session_tensors(
                valid_interactions
            )

        self._hrnn_training_loop(
            sessions_tensor,
            target_tensor,
            outcome_tensor,
            optimizer,
            val_sess_tensor,
            val_target_tensor,
            val_outcome_tensor,
        )

    def _build_padded_session_tensors(
        self, sessions_df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert sessions_df into right-aligned 3D input/target/outcome tensors.

        Layout (right-aligned in both dimensions):
            sessions_tensor[u, s, t] = item index at position t of session s for user u (0=padding)
            target_tensor[u, s, t]   = next item within session s at position t+1 (0 at last real
                                       position or padding — no cross-session targets)
            outcome_tensor[u, s, t]  = reward for the target item (1.0 default)

        The valid training signal at position (s, t) requires:
            sessions_tensor[u,s,t] != 0  AND  target_tensor[u,s,t] != 0

        Args:
            sessions_df: DataFrame with SESSION_SEQUENCES_COL and optionally SESSION_OUTCOMES_COL.

        Returns:
            sessions_tensor: (n_users, max_sessions, max_session_len)
            target_tensor:   (n_users, max_sessions, max_session_len)
            outcome_tensor:  (n_users, max_sessions, max_session_len)
        """
        n_users = len(sessions_df)
        has_outcome = SESSION_OUTCOMES_COL in sessions_df.columns

        sessions_array = np.zeros((n_users, self.max_sessions, self.max_session_len), dtype=np.int64)
        target_array = np.zeros((n_users, self.max_sessions, self.max_session_len), dtype=np.int64)
        outcome_array = np.ones((n_users, self.max_sessions, self.max_session_len), dtype=np.float32)

        all_session_lists = sessions_df[SESSION_SEQUENCES_COL].tolist()
        all_outcome_lists = sessions_df[SESSION_OUTCOMES_COL].tolist() if has_outcome else None

        short_outcome_count = 0  # sessions where outcome list was shorter than the item sequence

        for u, session_list in enumerate(all_session_lists):
            # Keep the most recent max_sessions sessions
            session_list = session_list[-self.max_sessions :]
            n_sess = len(session_list)
            outcome_list = all_outcome_lists[u][-self.max_sessions :] if all_outcome_lists else None

            for s_offset, session in enumerate(session_list):
                # Right-align sessions: most recent at max_sessions-1
                sess_idx = self.max_sessions - n_sess + s_offset

                # Truncate each session to max_session_len
                session = list(session[-self.max_session_len :])
                sess_len = len(session)
                if sess_len == 0:
                    continue

                # Vectorized vocabulary lookup (C-level via pd.Index.get_indexer)
                item_strs = [str(x) for x in session]
                item_idx = self.item_id_index.get_indexer(item_strs) + 1  # unknown → 0 (padding)

                # Right-align items within session
                start = self.max_session_len - sess_len
                sessions_array[u, sess_idx, start : start + sess_len] = item_idx

                # Targets: item[1], ..., item[sess_len-1] at positions start..start+sess_len-2
                # The last real position has no within-session target (target stays 0).
                if sess_len > 1:
                    target_array[u, sess_idx, start : start + sess_len - 1] = item_idx[1:]

                if has_outcome and outcome_list is not None:
                    outcomes = list(outcome_list[s_offset][-self.max_session_len :])[:sess_len]
                    if len(outcomes) < sess_len:
                        short_outcome_count += 1
                        # Pad with 1.0 (positive/neutral signal) — consistent with implicit
                        # feedback conventions where observed interactions are treated as positive.
                        outcomes = outcomes + [1.0] * (sess_len - len(outcomes))
                    if sess_len > 1:
                        # Reward for target at position t = reward of item[t+1]
                        outcome_array[u, sess_idx, start : start + sess_len - 1] = [
                            float(v) for v in outcomes[1:sess_len]
                        ]

        if short_outcome_count > 0:
            logger.warning(
                f"{short_outcome_count} session(s) had outcome lists shorter than their item sequences "
                "and were padded with 1.0. Check that your OUTCOME column is fully populated — "
                "missing values will be treated as positive signals during training."
            )

        return (
            torch.from_numpy(sessions_array),
            torch.from_numpy(target_array),
            torch.from_numpy(outcome_array),
        )

    def _build_inference_session_tensor(self, sessions_df: pd.DataFrame) -> torch.Tensor:
        """Build (n_users, max_sessions, max_session_len) tensor for inference (no targets needed)."""
        n_users = len(sessions_df)
        sessions_array = np.zeros((n_users, self.max_sessions, self.max_session_len), dtype=np.int64)

        all_session_lists = sessions_df[SESSION_SEQUENCES_COL].tolist()

        for u, session_list in enumerate(all_session_lists):
            session_list = session_list[-self.max_sessions :]
            n_sess = len(session_list)

            for s_offset, session in enumerate(session_list):
                sess_idx = self.max_sessions - n_sess + s_offset
                session = list(session[-self.max_session_len :])
                sess_len = len(session)
                if sess_len == 0:
                    continue

                item_strs = [str(x) for x in session]
                item_idx = self.item_id_index.get_indexer(item_strs) + 1

                start = self.max_session_len - sess_len
                sessions_array[u, sess_idx, start : start + sess_len] = item_idx

        return torch.from_numpy(sessions_array)

    def _compute_hrnn_loss(
        self,
        all_session_h: torch.Tensor,
        sessions: torch.Tensor,
        targets: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Pointwise loss over all valid (session, position) pairs.

        A position is valid when both the source item and its within-session target are real
        (non-zero). The last real item in each session has target=0 and is excluded.

        Args:
            all_session_h: (B, max_sessions, max_session_len, H)
            sessions:      (B, max_sessions, max_session_len) input item indices
            targets:       (B, max_sessions, max_session_len) target item indices
            outcomes:      (B, max_sessions, max_session_len) rewards for each target

        Returns:
            Scalar loss tensor, or None if no valid positions exist in the batch.
        """
        valid_mask = (sessions != 0) & (targets != 0)  # (B, max_sessions, max_session_len)
        if not valid_mask.any():
            return None

        repr_flat = all_session_h[valid_mask]  # (N_valid, H)
        target_flat = targets[valid_mask]  # (N_valid,)
        reward_flat = outcomes[valid_mask]  # (N_valid,)

        item_emb = self.model.item_embedding.weight  # type: ignore[union-attr]  # (num_items+1, H)

        pos_emb = item_emb[target_flat]
        pos_scores = (repr_flat * pos_emb).sum(dim=1, keepdim=True)  # (N_valid, 1)
        pos_loss = self.loss_fn(pos_scores, reward_flat.unsqueeze(1))

        if self.num_negatives == 0:
            return pos_loss

        neg_items = torch.randint(
            1,
            self.num_items + 1,
            (repr_flat.shape[0], self.num_negatives),
            device=self.device,
        )
        neg_emb = item_emb[neg_items]  # (N_valid, num_negatives, H)
        neg_scores = (repr_flat.unsqueeze(1) * neg_emb).sum(dim=2)  # (N_valid, num_negatives)
        neg_loss = self.loss_fn(neg_scores, torch.zeros_like(neg_scores))

        return pos_loss + neg_loss / self.num_negatives

    def _hrnn_training_loop(
        self,
        sessions_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        outcome_tensor: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        val_sess_tensor: Optional[torch.Tensor] = None,
        val_target_tensor: Optional[torch.Tensor] = None,
        val_outcome_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Run the HRNN mini-batch training loop.

        Each epoch shuffles users, iterates over mini-batches, computes the
        next-item prediction loss via ``_compute_hrnn_loss``, and
        back-propagates.  Optionally evaluates a held-out validation split and
        applies early stopping with best-weight restoration.

        Args:
            sessions_tensor: Padded sessions tensor of shape
                ``(n_users, n_sessions, session_len)`` with item indices
                (0 = padding).
            target_tensor: Padded target-item tensor of shape
                ``(n_users, n_sessions, session_len)`` with item indices
                (0 = padding).
            outcome_tensor: Padded outcome tensor of shape
                ``(n_users, n_sessions, session_len)`` with float labels
                (0 for padding positions).
            optimizer: Configured optimizer wrapping the HRNN model parameters.
            val_sess_tensor: Optional validation sessions tensor.
            val_target_tensor: Optional validation target tensor.
            val_outcome_tensor: Optional validation outcome tensor.
        """
        n_users = sessions_tensor.shape[0]
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

                batch_sess = sessions_tensor[batch_idx].to(self.device)
                batch_targets = target_tensor[batch_idx].to(self.device)
                batch_outcomes = outcome_tensor[batch_idx].to(self.device)

                optimizer.zero_grad()

                all_session_h = self.model(batch_sess)  # type: ignore[operator]
                loss = self._compute_hrnn_loss(all_session_h, batch_sess, batch_targets, batch_outcomes)

                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

            # Compute validation loss if validation tensors are provided
            val_loss_str = ""
            if val_sess_tensor is not None and val_target_tensor is not None and val_outcome_tensor is not None:
                self.model.eval()  # type: ignore[union-attr]
                val_loss_total = 0.0
                val_batches = 0
                n_val = val_sess_tensor.shape[0]
                with torch.no_grad():
                    for start in range(0, n_val, self.batch_size):
                        end = min(start + self.batch_size, n_val)
                        v_sess = val_sess_tensor[start:end].to(self.device)
                        v_targets = val_target_tensor[start:end].to(self.device)
                        v_outcomes = val_outcome_tensor[start:end].to(self.device)
                        v_h = self.model(v_sess)  # type: ignore[operator]
                        v_loss = self._compute_hrnn_loss(v_h, v_sess, v_targets, v_outcomes)
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
        Score all items for each user's session history.

        Args:
            interactions: sessions_df with USER_ID and SESSION_SEQUENCES columns.
            users: Ignored (HRNN has no user embedding lookup).

        Returns:
            (num_users, num_items) numpy array of scores, aligned with self.item_id_index.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit_embedding_model() first.")

        sess_tensor = self._build_inference_session_tensor(interactions)

        self.model.eval()  # type: ignore[union-attr]
        all_scores: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(sess_tensor), self.batch_size):
                end = min(start + self.batch_size, len(sess_tensor))
                batch_sess = sess_tensor[start:end].to(self.device)

                all_session_h = self.model(batch_sess)  # type: ignore[operator]

                # Most recent session is at index -1 (right-aligned); last item at position -1.
                seq_repr = all_session_h[:, -1, -1, :]  # (B, H)

                scores = self.model.score_all_items(seq_repr)  # type: ignore[union-attr]
                all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores, axis=0)  # (num_users, num_items)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "hidden_units": self.hidden_units,
                "num_layers": self.num_layers,
                "dropout_rate": self.dropout_rate,
                "num_negatives": self.num_negatives,
                "max_sessions": self.max_sessions,
                "max_session_len": self.max_session_len,
                "early_stopping_patience": self.early_stopping_patience,
                "restore_best_weights": self.restore_best_weights,
            }
        )
        return config


class HRNNClassifierEstimator(_HRNNBaseEstimator):
    """
    HRNN for classification-style rewards.

    Use when OUTCOME is binary (0/1) or discrete ratings (1-5).
    Default loss: Binary Cross-Entropy (reward treated as label probability).

    Data modes:
    - Positives only: all OUTCOME=1, use num_negatives >= 1 for random negatives
    - Binary 0/1: 0s serve as natural hard negatives via reward signal
    - Ratings 1-5: normalise to [0, 1] in data prep for soft-label BCE

    Reference:
        Quadrana, M., Cremonesi, P., & Jannach, D. (2017).
        Personalizing session-based recommendations with hierarchical recurrent neural networks.
        In Proceedings of the Eleventh ACM Conference on Recommender Systems (RecSys 2017).
    """

    def __init__(
        self,
        hidden_units: int = 50,
        num_layers: int = 1,
        dropout_rate: float = 0.2,
        num_negatives: int = 1,
        max_sessions: int = 10,
        max_session_len: int = 20,
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
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            num_negatives=num_negatives,
            max_sessions=max_sessions,
            max_session_len=max_session_len,
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


class HRNNRegressorEstimator(_HRNNBaseEstimator):
    """
    HRNN for continuous rewards.

    Use when OUTCOME is a continuous value (e.g., revenue, time-spent, rating as float).
    Default loss: Mean Squared Error (predicted score regresses toward reward value).

    Data modes:
    - Continuous rewards: MSE loss, reward magnitude directly supervises the model
    - num_negatives > 0: random unseen items receive target=0.0

    Reference:
        Quadrana, M., Cremonesi, P., & Jannach, D. (2017).
        Personalizing session-based recommendations with hierarchical recurrent neural networks.
        In Proceedings of the Eleventh ACM Conference on Recommender Systems (RecSys 2017).
    """

    def __init__(
        self,
        hidden_units: int = 50,
        num_layers: int = 1,
        dropout_rate: float = 0.2,
        num_negatives: int = 1,
        max_sessions: int = 10,
        max_session_len: int = 20,
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
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            num_negatives=num_negatives,
            max_sessions=max_sessions,
            max_session_len=max_session_len,
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
