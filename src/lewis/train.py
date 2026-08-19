"""
train.py — CLI training script for the Lewis Signalling Game.

Usage
-----
    python -m lewis.train [options]
    lewis-train [options]

Output
------
    <output_dir>/
        metrics.jsonl       — one JSON line per epoch (loss, accuracy, topsim, …)
        checkpoint_best.pt  — best model (highest accuracy)
        checkpoint_last.pt  — last epoch model
        config.json         — hyperparameters used for this run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from lewis.agents import Sender, Receiver
from lewis.game import GameConfig, LewisGame, compute_temperature
from lewis.metrics import topographic_similarity, symbol_frequencies, mean_entropy
from lewis.world import World, WorldConfig


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: GameConfig, output_dir: Path, topsim_every: int = 50) -> None:
    """
    Run the full training loop.

    Parameters
    ----------
    cfg         : GameConfig with all hyperparameters
    output_dir  : directory for checkpoints and metrics log
    topsim_every: compute TopSim every N epochs (expensive O(N²))
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build world
    world_cfg = WorldConfig(
        n_attributes=cfg.n_attributes,
        n_values=cfg.n_values,
        n_distractors=cfg.n_distractors,
    )
    world = World(world_cfg, device=device)
    print(f"World: {world_cfg.n_objects} objects "
          f"({cfg.n_attributes} attrs × {cfg.n_values} values), "
          f"{cfg.n_distractors} distractors")

    # Build agents
    sender = Sender(
        input_dim=world_cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        vocab_size=cfg.vocab_size,
        msg_len=cfg.msg_len,
    ).to(device)

    receiver = Receiver(
        input_dim=world_cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        vocab_size=cfg.vocab_size,
        msg_len=cfg.msg_len,
        n_candidates=world_cfg.n_candidates,
    ).to(device)

    params = list(sender.parameters()) + list(receiver.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    game = LewisGame(world, sender, receiver)

    # Training
    metrics_path = output_dir / "metrics.jsonl"
    best_acc = 0.0
    t0 = time.time()

    with open(metrics_path, "w") as mf:
        for epoch in range(cfg.epochs):
            sender.train()
            receiver.train()

            tau = compute_temperature(epoch, cfg)
            result = game.step(cfg.batch_size, temperature=tau)

            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()

            acc = result["accuracy"]
            loss_val = result["loss"].item()

            # Compute TopSim periodically
            topsim = None
            entropy = None
            if epoch % topsim_every == 0:
                try:
                    freq = symbol_frequencies(world, sender)
                    entropy = mean_entropy(freq)
                    topsim = topographic_similarity(world, sender)
                except Exception as e:
                    print(f"  [warn] metrics failed at epoch {epoch}: {e}")

            # Log
            record = {
                "epoch": epoch,
                "loss": round(loss_val, 6),
                "accuracy": round(acc, 4),
                "temperature": round(tau, 4),
                "topsim": round(topsim, 4) if topsim is not None else None,
                "entropy": round(entropy, 4) if entropy is not None else None,
            }
            mf.write(json.dumps(record) + "\n")
            mf.flush()

            # Console output
            if epoch % 100 == 0 or epoch == cfg.epochs - 1:
                elapsed = time.time() - t0
                ts_str = f"TopSim={topsim:.3f}" if topsim is not None else ""
                ent_str = f"Entropy={entropy:.2f}" if entropy is not None else ""
                print(
                    f"Epoch {epoch:5d}/{cfg.epochs}  "
                    f"loss={loss_val:.4f}  acc={acc:.3f}  "
                    f"tau={tau:.3f}  {ts_str}  {ent_str}  "
                    f"[{elapsed:.0f}s]"
                )

            # Save best checkpoint
            if acc > best_acc:
                best_acc = acc
                _save_checkpoint(output_dir / "checkpoint_best.pt",
                                 sender, receiver, optimizer, epoch, record)

    # Save final checkpoint
    _save_checkpoint(output_dir / "checkpoint_last.pt",
                     sender, receiver, optimizer, cfg.epochs - 1, record)

    print(f"\nTraining complete. Best accuracy: {best_acc:.3f}")
    print(f"Outputs saved to: {output_dir}")


def _save_checkpoint(path: Path, sender, receiver, optimizer, epoch, metrics) -> None:
    torch.save({
        "epoch": epoch,
        "sender_state": sender.state_dict(),
        "receiver_state": receiver.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train two neural agents to invent a language via Lewis Signalling Game",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # World
    p.add_argument("--n-attributes", type=int, default=3,
                   help="Number of object attribute dimensions")
    p.add_argument("--n-values", type=int, default=4,
                   help="Number of values per attribute")
    p.add_argument("--n-distractors", type=int, default=3,
                   help="Number of distractor objects shown to Receiver")
    # Architecture
    p.add_argument("--hidden-dim", type=int, default=128,
                   help="MLP hidden layer width")
    p.add_argument("--vocab-size", type=int, default=10,
                   help="Number of symbols in the vocabulary")
    p.add_argument("--msg-len", type=int, default=3,
                   help="Number of symbols per message")
    # Training
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--batch-size", type=int, default=512, help="Episodes per batch")
    p.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    # Temperature annealing
    p.add_argument("--tau-start", type=float, default=5.0,
                   help="Initial Gumbel-Softmax temperature")
    p.add_argument("--tau-min", type=float, default=0.1,
                   help="Minimum temperature (floor)")
    p.add_argument("--tau-decay", type=float, default=0.998,
                   help="Per-epoch temperature decay factor")
    # Output
    p.add_argument("--output-dir", type=str, default="./runs/exp1",
                   help="Directory for checkpoints and metrics")
    p.add_argument("--topsim-every", type=int, default=50,
                   help="Compute TopSim every N epochs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = GameConfig(
        n_attributes=args.n_attributes,
        n_values=args.n_values,
        n_distractors=args.n_distractors,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        msg_len=args.msg_len,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        tau_start=args.tau_start,
        tau_min=args.tau_min,
        tau_decay=args.tau_decay,
    )
    train(cfg, output_dir=Path(args.output_dir), topsim_every=args.topsim_every)


if __name__ == "__main__":
    main()
