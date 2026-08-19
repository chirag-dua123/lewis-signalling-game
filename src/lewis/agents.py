"""
agents.py — Sender and Receiver neural networks.

Sender
------
  Input : target object one-hot vector  (B, input_dim)
  Output: message — L symbols, each a soft categorical over vocab V
          Shape (B, msg_len, vocab_size)
          During training : soft Gumbel-Softmax vectors (differentiable)
          During eval     : hard one-hot (argmax), effectively discrete

Receiver
--------
  Input : message (B, msg_len, vocab_size), candidates (B, N, input_dim)
  Output: log-probabilities over N candidates  (B, N)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

class Sender(nn.Module):
    """
    Maps an observed object to a fixed-length discrete message via
    Gumbel-Softmax sampling.

    Architecture
    ------------
    object → [Linear → ReLU → Linear → ReLU] → hidden
    hidden → [Linear_i]  for i in 0..msg_len-1  → logits_i → GS sample
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        msg_len: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.msg_len = msg_len

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # One independent linear head per symbol position
        self.symbol_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, vocab_size) for _ in range(msg_len)]
        )

    def forward(
        self, x: Tensor, temperature: float = 1.0, hard: bool = False
    ) -> Tensor:
        """
        Parameters
        ----------
        x           : (B, input_dim) one-hot object
        temperature : Gumbel-Softmax temperature τ (lower → more discrete)
        hard        : if True, use straight-through hard one-hot (for eval/analysis)

        Returns
        -------
        message : (B, msg_len, vocab_size)  soft or hard categorical
        """
        h = self.encoder(x)
        symbols = [
            F.gumbel_softmax(head(h), tau=temperature, hard=hard)
            for head in self.symbol_heads
        ]
        return torch.stack(symbols, dim=1)   # (B, msg_len, vocab_size)

    @torch.no_grad()
    def hard_message(self, x: Tensor) -> Tensor:
        """
        Produce hard discrete messages (argmax). Returns (B, msg_len) int64 indices.
        Used for analysis and visualisation.
        """
        soft = self.forward(x, temperature=0.01, hard=True)   # (B, L, V)
        return soft.argmax(dim=-1)                             # (B, L)


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class Receiver(nn.Module):
    """
    Given a message and a set of candidate objects, selects the most likely target.

    Architecture
    ------------
    message   → flatten → [Linear → ReLU → Linear] → msg_emb   (B, emb_dim)
    candidate → [Linear → ReLU → Linear]             → obj_emb  (B, N, emb_dim)
    score_i   = dot(msg_emb, obj_emb_i)
    output    = log_softmax(scores)                              (B, N)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        msg_len: int,
        n_candidates: int,
    ) -> None:
        super().__init__()
        self.n_candidates = n_candidates
        msg_input_dim = msg_len * vocab_size

        self.msg_encoder = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obj_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, message: Tensor, candidates: Tensor) -> Tensor:
        """
        Parameters
        ----------
        message    : (B, msg_len, vocab_size) soft or hard message
        candidates : (B, n_candidates, input_dim)

        Returns
        -------
        log_probs : (B, n_candidates)
        """
        # Encode message
        msg_flat = message.flatten(1)                         # (B, msg_len * vocab_size)
        msg_emb = self.msg_encoder(msg_flat)                  # (B, hidden_dim)

        # Encode every candidate object
        # candidates: (B, N, input_dim)
        B, N, _ = candidates.shape
        cand_flat = candidates.view(B * N, -1)                # (B*N, input_dim)
        obj_emb = self.obj_encoder(cand_flat).view(B, N, -1)  # (B, N, hidden_dim)

        # Dot-product similarity between message embedding and each object embedding
        msg_emb_exp = msg_emb.unsqueeze(2)                    # (B, hidden_dim, 1)
        logits = torch.bmm(obj_emb, msg_emb_exp).squeeze(2)  # (B, N)

        return F.log_softmax(logits, dim=-1)                  # (B, N)
