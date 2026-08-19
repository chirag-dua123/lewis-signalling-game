"""
metrics.py — Evaluation metrics for emergent communication.

Metrics
-------
- accuracy          : fraction of correct picks (computed in the game loop)
- topographic_similarity : Spearman ρ between semantic and message distances
- symbol_entropy    : Shannon entropy of symbol usage per position
- symbol_frequencies: raw counts / probabilities per (position, symbol)
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from scipy.stats import spearmanr

from lewis.world import World
from lewis.agents import Sender


# ---------------------------------------------------------------------------
# Topographic Similarity
# ---------------------------------------------------------------------------

def topographic_similarity(world: World, sender: Sender) -> float:
    """
    Compute Topographic Similarity (TopSim) between the semantic space
    and the emergent language's message space.

    TopSim = Spearman ρ(semantic_pairwise_distances, message_pairwise_distances)

    A higher value indicates a more compositional / structured language.

    Returns
    -------
    float in [-1, 1]; random language ≈ 0, compositional language > 0.4
    """
    sender.eval()
    device = next(sender.parameters()).device

    all_objects = world.all_objects().to(device)       # (N, input_dim)
    all_cat = world.all_objects_categorical().to(device)  # (N, n_attr)

    with torch.no_grad():
        messages = sender.hard_message(all_objects)    # (N, msg_len)  int64

    messages_np = messages.cpu().numpy()               # (N, msg_len)
    cat_np = all_cat.cpu().numpy()                     # (N, n_attr)

    N = len(cat_np)

    # Pairwise semantic distance (Hamming over categorical attributes)
    semantic_dists = []
    message_dists = []
    for i in range(N):
        for j in range(i + 1, N):
            s_dist = int((cat_np[i] != cat_np[j]).sum())
            m_dist = int((messages_np[i] != messages_np[j]).sum())
            semantic_dists.append(s_dist)
            message_dists.append(m_dist)

    if len(set(message_dists)) == 1:
        # All messages identical → degenerate case
        return 0.0

    rho, _ = spearmanr(semantic_dists, message_dists)
    sender.train()
    return float(rho) if not math.isnan(rho) else 0.0


# ---------------------------------------------------------------------------
# Symbol Statistics
# ---------------------------------------------------------------------------

def symbol_frequencies(world: World, sender: Sender) -> np.ndarray:
    """
    Compute the frequency distribution of symbols for each message position.

    Returns
    -------
    freq : (msg_len, vocab_size) ndarray of probabilities summing to 1 per row
    """
    sender.eval()
    device = next(sender.parameters()).device

    all_objects = world.all_objects().to(device)
    with torch.no_grad():
        messages = sender.hard_message(all_objects)   # (N, msg_len)

    messages_np = messages.cpu().numpy()
    msg_len = messages_np.shape[1]
    vocab_size = sender.vocab_size

    freq = np.zeros((msg_len, vocab_size), dtype=np.float64)
    for pos in range(msg_len):
        for sym in range(vocab_size):
            freq[pos, sym] = (messages_np[:, pos] == sym).sum()
    freq /= freq.sum(axis=1, keepdims=True) + 1e-9

    sender.train()
    return freq


def symbol_entropy(freq: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy for each message position.

    Parameters
    ----------
    freq : (msg_len, vocab_size) frequency array from symbol_frequencies()

    Returns
    -------
    entropies : (msg_len,) entropy in bits per position
    """
    eps = 1e-9
    log2_freq = np.log2(freq + eps)
    return -(freq * log2_freq).sum(axis=1)


def mean_entropy(freq: np.ndarray) -> float:
    """Mean entropy across all message positions (scalar)."""
    return float(symbol_entropy(freq).mean())


# ---------------------------------------------------------------------------
# Per-object message table
# ---------------------------------------------------------------------------

def build_message_table(world: World, sender: Sender) -> list[dict]:
    """
    Return a list of dicts, one per object in the catalogue.

    Each dict has:
        object_str : human-readable label
        cat_idx    : catalogue index
        message    : list of int (discrete symbol indices)
        message_str: e.g.  "[3, 0, 7]"
    """
    sender.eval()
    device = next(sender.parameters()).device

    all_objects = world.all_objects().to(device)
    with torch.no_grad():
        messages = sender.hard_message(all_objects)   # (N, msg_len)

    rows = []
    for idx in range(world.config.n_objects):
        msg = messages[idx].tolist()
        rows.append({
            "cat_idx": idx,
            "object_str": world.object_to_str(idx),
            "message": msg,
            "message_str": str(msg),
            "first_symbol": msg[0] if msg else 0,
        })
    sender.train()
    return rows
