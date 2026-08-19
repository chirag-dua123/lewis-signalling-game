"""Tests for the metrics module."""

import math

import numpy as np
import pytest
import torch

from lewis.agents import Sender
from lewis.metrics import (
    topographic_similarity,
    symbol_frequencies,
    symbol_entropy,
    mean_entropy,
    build_message_table,
)
from lewis.world import World, WorldConfig


@pytest.fixture
def world():
    cfg = WorldConfig(n_attributes=2, n_values=3, n_distractors=2)
    return World(cfg)


@pytest.fixture
def sender(world):
    return Sender(
        input_dim=world.config.input_dim,
        hidden_dim=32,
        vocab_size=6,
        msg_len=2,
    )


# ---------------------------------------------------------------------------
# symbol_frequencies
# ---------------------------------------------------------------------------

def test_symbol_frequencies_shape(world, sender):
    freq = symbol_frequencies(world, sender)
    assert freq.shape == (sender.msg_len, sender.vocab_size)


def test_symbol_frequencies_sum_to_one(world, sender):
    freq = symbol_frequencies(world, sender)
    row_sums = freq.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_symbol_frequencies_nonneg(world, sender):
    freq = symbol_frequencies(world, sender)
    assert (freq >= 0).all()


# ---------------------------------------------------------------------------
# symbol_entropy
# ---------------------------------------------------------------------------

def test_entropy_uniform():
    """Uniform distribution should yield maximum entropy."""
    vocab_size = 8
    freq = np.ones((3, vocab_size)) / vocab_size
    entropies = symbol_entropy(freq)
    expected = math.log2(vocab_size)
    assert np.allclose(entropies, expected, atol=0.01)


def test_entropy_degenerate():
    """Deterministic (one-hot) distribution should yield zero entropy."""
    vocab_size = 6
    freq = np.zeros((2, vocab_size))
    freq[:, 0] = 1.0
    entropies = symbol_entropy(freq)
    assert np.allclose(entropies, 0.0, atol=0.01)


def test_mean_entropy_scalar(world, sender):
    freq = symbol_frequencies(world, sender)
    ent = mean_entropy(freq)
    assert isinstance(ent, float)
    assert ent >= 0.0


# ---------------------------------------------------------------------------
# topographic_similarity
# ---------------------------------------------------------------------------

def test_topsim_returns_float(world, sender):
    ts = topographic_similarity(world, sender)
    assert isinstance(ts, float)


def test_topsim_in_range(world, sender):
    """TopSim (Spearman ρ) must be in [-1, 1]."""
    ts = topographic_similarity(world, sender)
    assert -1.0 <= ts <= 1.0


def test_topsim_compositional_language():
    """
    Hand-crafted perfectly compositional language:
    object (a, b) → message [a, b].
    This should yield near-maximum TopSim.
    """
    world_cfg = WorldConfig(n_attributes=2, n_values=4, n_distractors=2)
    world = World(world_cfg)

    # Build a Sender and hard-wire it to emit a deterministic compositional message
    sender = Sender(
        input_dim=world_cfg.input_dim,
        hidden_dim=32,
        vocab_size=world_cfg.n_values,  # vocab same size as n_values
        msg_len=world_cfg.n_attributes,
    )

    # Patch hard_message to return compositional encoding
    def _perfect_hard(x):
        # For each object, return its attribute values as the message
        all_cat = world.all_objects_categorical()  # (N, n_attr)
        return all_cat.to(torch.int64)

    import types
    sender.hard_message = types.MethodType(
        lambda self, x: _perfect_hard(x), sender
    )

    ts = topographic_similarity(world, sender)
    assert ts > 0.8, f"Expected TopSim > 0.8 for perfect language, got {ts:.3f}"


def test_topsim_random_language():
    """Random sender messages should yield near-zero TopSim."""
    world_cfg = WorldConfig(n_attributes=2, n_values=4, n_distractors=2)
    world = World(world_cfg)

    sender = Sender(
        input_dim=world_cfg.input_dim,
        hidden_dim=32,
        vocab_size=8,
        msg_len=2,
    )
    # Don't train — random weights → random messages → TopSim ≈ 0
    # (may be slightly nonzero due to randomness, but should not be high)
    ts = topographic_similarity(world, sender)
    assert ts < 0.6, f"Expected low TopSim for random language, got {ts:.3f}"


# ---------------------------------------------------------------------------
# build_message_table
# ---------------------------------------------------------------------------

def test_message_table_length(world, sender):
    rows = build_message_table(world, sender)
    assert len(rows) == world.config.n_objects


def test_message_table_fields(world, sender):
    rows = build_message_table(world, sender)
    for row in rows:
        assert "cat_idx" in row
        assert "object_str" in row
        assert "message" in row
        assert "message_str" in row
        assert len(row["message"]) == sender.msg_len
