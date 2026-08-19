"""
game.py — Lewis Signalling Game loop.

Ties together the World, Sender, and Receiver into a single forward pass
that computes loss and accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from lewis.world import World
from lewis.agents import Sender, Receiver


@dataclass
class GameConfig:
    """Hyperparameters for the Lewis game (architecture + training)."""
    # World
    n_attributes: int = 3
    n_values: int = 4
    n_distractors: int = 3
    # Agent architecture
    hidden_dim: int = 128
    vocab_size: int = 10
    msg_len: int = 3
    # Training
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 2000
    # Gumbel-Softmax temperature annealing
    tau_start: float = 5.0
    tau_min: float = 0.1
    tau_decay: float = 0.998


class LewisGame:
    """
    Coordinates a single forward pass of the Lewis Signalling Game.

    Usage
    -----
    game = LewisGame(world, sender, receiver)
    result = game.step(batch_size=512, temperature=1.0)
    result["loss"].backward()
    """

    def __init__(self, world: World, sender: Sender, receiver: Receiver) -> None:
        self.world = world
        self.sender = sender
        self.receiver = receiver

    def step(self, batch_size: int, temperature: float = 1.0) -> dict:
        """
        Run one forward pass.

        Parameters
        ----------
        batch_size  : number of episodes per batch
        temperature : Gumbel-Softmax temperature (annealed over training)

        Returns
        -------
        dict with keys:
            loss        : scalar Tensor  (cross-entropy, ready for .backward())
            accuracy    : float  (fraction of correct picks)
            message     : (B, msg_len, vocab_size)  soft message Tensor
            target      : (B, input_dim)
            candidates  : (B, n_candidates, input_dim)
            target_idx  : (B,)
        """
        target, candidates, target_idx = self.world.sample(batch_size)

        # Sender observes target and produces a message
        message = self.sender(target, temperature=temperature)  # (B, L, V)

        # Receiver sees message + candidates, outputs log-probs
        log_probs = self.receiver(message, candidates)          # (B, N)

        # Cross-entropy loss: maximise log P(correct candidate)
        loss = F.nll_loss(log_probs, target_idx)

        with torch.no_grad():
            acc = (log_probs.argmax(dim=-1) == target_idx).float().mean().item()

        return {
            "loss": loss,
            "accuracy": acc,
            "message": message.detach(),
            "target": target.detach(),
            "candidates": candidates.detach(),
            "target_idx": target_idx.detach(),
        }


def compute_temperature(epoch: int, cfg: GameConfig) -> float:
    """Exponential temperature annealing schedule."""
    tau = cfg.tau_start * (cfg.tau_decay ** epoch)
    return max(tau, cfg.tau_min)
