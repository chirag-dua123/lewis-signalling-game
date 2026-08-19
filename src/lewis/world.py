"""
world.py — Object/attribute world for the Lewis Signalling Game.

Objects are described by n_attributes categorical dimensions, each with
n_values possible values.  They are encoded as concatenated one-hot vectors.

Example (n_attributes=3, n_values=4):
    object (1, 0, 2)  →  [0,1,0,0, 1,0,0,0, 0,0,1,0]   (dim = 12)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class WorldConfig:
    """Configuration for the attribute world."""
    n_attributes: int = 3   # number of object dimensions (e.g., color, shape, size)
    n_values: int = 4       # number of values per attribute
    n_distractors: int = 3  # number of distractor objects shown to Receiver

    @property
    def n_candidates(self) -> int:
        """Total candidates shown to Receiver = target + distractors."""
        return self.n_distractors + 1

    @property
    def input_dim(self) -> int:
        """Dimension of the one-hot encoded object vector."""
        return self.n_attributes * self.n_values

    @property
    def n_objects(self) -> int:
        """Total number of distinct objects in the world."""
        return self.n_values ** self.n_attributes


class World:
    """
    Samples batches of (target, candidates, target_idx) tuples.

    Attributes
    ----------
    config : WorldConfig
    device : torch.device
    """

    def __init__(self, config: WorldConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        # Pre-compute the full object catalogue as one-hot tensors
        self._catalogue = self._build_catalogue()   # (n_objects, input_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """
        Draw a batch of referential game episodes.

        Returns
        -------
        target     : (B, input_dim)          — one-hot target object
        candidates : (B, n_candidates, input_dim) — target at random slot
        target_idx : (B,)                    — ground-truth index into candidates
        """
        cfg = self.config
        B = batch_size
        N = cfg.n_candidates

        # Sample target indices from the catalogue
        target_cat_idx = torch.randint(0, cfg.n_objects, (B,), device=self.device)
        target = self._catalogue[target_cat_idx]  # (B, input_dim)

        # Sample distractors (different from target)
        candidates = torch.zeros(B, N, cfg.input_dim, device=self.device)
        target_slot = torch.randint(0, N, (B,), device=self.device)  # where target goes

        for b in range(B):
            # Pool of objects excluding the target
            pool = [i for i in range(cfg.n_objects) if i != target_cat_idx[b].item()]
            distractor_indices = torch.tensor(
                pool[:cfg.n_distractors],  # just take first n_distractors from shuffled pool
                device=self.device,
            )
            # Shuffle pool
            perm = torch.randperm(len(pool), device=self.device)[:cfg.n_distractors]
            distractor_indices = torch.tensor(
                [pool[p] for p in perm.tolist()], device=self.device
            )
            dist_slot = 0
            for slot in range(N):
                if slot == target_slot[b].item():
                    candidates[b, slot] = target[b]
                else:
                    candidates[b, slot] = self._catalogue[distractor_indices[dist_slot]]
                    dist_slot += 1

        return target, candidates, target_slot

    def all_objects(self) -> Tensor:
        """Return all objects in the catalogue. Shape: (n_objects, input_dim)."""
        return self._catalogue.clone()

    def all_objects_categorical(self) -> Tensor:
        """
        Return all objects as integer attribute vectors.
        Shape: (n_objects, n_attributes), values in [0, n_values).
        """
        cfg = self.config
        combos = list(itertools.product(range(cfg.n_values), repeat=cfg.n_attributes))
        return torch.tensor(combos, device=self.device)  # (n_objects, n_attributes)

    def object_to_str(self, cat_idx: int) -> str:
        """Human-readable label for an object by catalogue index."""
        cfg = self.config
        attrs = []
        remaining = cat_idx
        for _ in range(cfg.n_attributes):
            attrs.append(remaining % cfg.n_values)
            remaining //= cfg.n_values
        attr_names = ["color", "shape", "size", "texture", "pattern"]
        value_names = [
            ["red", "green", "blue", "yellow"],
            ["circle", "square", "triangle", "star"],
            ["small", "medium", "large", "xlarge"],
            ["plain", "striped", "dotted", "checkered"],
            ["solid", "hollow", "outline", "gradient"],
        ]
        parts = []
        for i, v in enumerate(attrs):
            name = attr_names[i] if i < len(attr_names) else f"attr{i}"
            vals = value_names[i] if i < len(value_names) else [str(j) for j in range(cfg.n_values)]
            val_label = vals[v] if v < len(vals) else str(v)
            parts.append(f"{name}={val_label}")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_catalogue(self) -> Tensor:
        """Build the full object catalogue as one-hot tensors."""
        cfg = self.config
        combos = list(itertools.product(range(cfg.n_values), repeat=cfg.n_attributes))
        one_hots = []
        for combo in combos:
            vec = torch.zeros(cfg.input_dim)
            for attr_i, val in enumerate(combo):
                vec[attr_i * cfg.n_values + val] = 1.0
            one_hots.append(vec)
        return torch.stack(one_hots).to(self.device)  # (n_objects, input_dim)
