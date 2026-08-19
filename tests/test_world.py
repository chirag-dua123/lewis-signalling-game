"""Tests for the World module."""

import pytest
import torch

from lewis.world import World, WorldConfig


@pytest.fixture
def default_world():
    cfg = WorldConfig(n_attributes=3, n_values=4, n_distractors=3)
    return World(cfg)


def test_catalogue_size(default_world):
    """Catalogue should contain n_values^n_attributes objects."""
    w = default_world
    assert w._catalogue.shape == (w.config.n_objects, w.config.input_dim)


def test_input_dim(default_world):
    """input_dim should equal n_attributes * n_values."""
    cfg = default_world.config
    assert cfg.input_dim == cfg.n_attributes * cfg.n_values


def test_one_hot_valid(default_world):
    """Each object vector should be a valid concatenated one-hot."""
    cat = default_world._catalogue
    cfg = default_world.config
    # Each attribute block sums to 1
    for attr_i in range(cfg.n_attributes):
        block = cat[:, attr_i * cfg.n_values:(attr_i + 1) * cfg.n_values]
        assert torch.allclose(block.sum(dim=1), torch.ones(cfg.n_objects))


def test_sample_shapes(default_world):
    """sample() should return tensors of the correct shapes."""
    B = 32
    target, candidates, target_idx = default_world.sample(B)
    cfg = default_world.config

    assert target.shape == (B, cfg.input_dim)
    assert candidates.shape == (B, cfg.n_candidates, cfg.input_dim)
    assert target_idx.shape == (B,)


def test_target_in_candidates(default_world):
    """The target object should always be among the candidates."""
    B = 64
    target, candidates, target_idx = default_world.sample(B)

    for b in range(B):
        slot = target_idx[b].item()
        assert torch.allclose(target[b], candidates[b, slot])


def test_no_duplicate_candidates(default_world):
    """All candidates in a single episode should be distinct objects."""
    B = 32
    _, candidates, _ = default_world.sample(B)
    N = default_world.config.n_candidates

    for b in range(B):
        cands = candidates[b]  # (N, input_dim)
        for i in range(N):
            for j in range(i + 1, N):
                assert not torch.allclose(cands[i], cands[j]), (
                    f"Duplicate candidates at batch {b}, slots {i} and {j}"
                )


def test_target_idx_in_range(default_world):
    """target_idx values must be within [0, n_candidates)."""
    B = 64
    _, _, target_idx = default_world.sample(B)
    assert (target_idx >= 0).all()
    assert (target_idx < default_world.config.n_candidates).all()


def test_all_objects_shape(default_world):
    """all_objects() and all_objects_categorical() return correct shapes."""
    w = default_world
    all_oh = w.all_objects()
    all_cat = w.all_objects_categorical()
    assert all_oh.shape == (w.config.n_objects, w.config.input_dim)
    assert all_cat.shape == (w.config.n_objects, w.config.n_attributes)


def test_object_to_str_runs(default_world):
    """object_to_str should return a non-empty string for all catalogue indices."""
    for idx in range(default_world.config.n_objects):
        s = default_world.object_to_str(idx)
        assert isinstance(s, str) and len(s) > 0
