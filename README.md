# Lewis Signalling Game 🗣️

Two neural agents that **invent a shared language from scratch** to solve a cooperative object-referencing task.

## What is a Lewis Signalling Game?

A Lewis Signalling Game (Lewis, 1969) is a minimal cooperative communication game:

1. **Sender** observes an object described by attributes (e.g., `color=red, shape=circle, size=large`).
2. **Sender** produces a **message** — a sequence of discrete symbols from a vocabulary it shares with the Receiver.
3. **Receiver** sees the message plus a set of candidate objects (one target + several distractors).
4. **Receiver** picks the object it thinks the Sender was describing.
5. Both agents receive **+1 reward** if correct, **0** otherwise.

No language is pre-specified. The agents must **co-invent one from scratch** via end-to-end gradient training (Gumbel-Softmax relaxation).

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Train via CLI (saves metrics + checkpoints to ./runs/exp1)
python -m lewis.train --epochs 2000 --output-dir ./runs/exp1

# 3. Launch the interactive Streamlit dashboard
streamlit run app.py
```

## Project Structure

```
lewis-signalling-game/
├── pyproject.toml
├── README.md
├── app.py                    # Streamlit dashboard (4 tabs)
├── src/lewis/
│   ├── world.py              # Object/attribute world + sampling
│   ├── agents.py             # Sender + Receiver neural networks
│   ├── game.py               # Game loop + loss computation
│   ├── train.py              # CLI training script
│   ├── metrics.py            # Accuracy, TopSim, entropy
│   └── visualize.py          # Plotly chart helpers
└── tests/
    ├── test_world.py
    ├── test_agents.py
    └── test_metrics.py
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--n-attributes` | 3 | Number of object dimensions (e.g., color, shape, size) |
| `--n-values` | 4 | Number of values per attribute |
| `--n-distractors` | 3 | Distractors shown to Receiver (harder = more) |
| `--vocab-size` | 10 | Size of the symbol vocabulary |
| `--msg-len` | 3 | Number of symbols per message |
| `--hidden-dim` | 128 | MLP hidden layer width |
| `--epochs` | 2000 | Training epochs |
| `--lr` | 1e-3 | Learning rate (Adam) |
| `--tau-start` | 5.0 | Initial Gumbel-Softmax temperature |
| `--tau-min` | 0.1 | Minimum temperature after annealing |
| `--tau-decay` | 0.998 | Per-epoch temperature decay factor |

## Metrics

- **Accuracy**: Fraction of episodes where Receiver selects the correct object.
- **Topographic Similarity (TopSim)**: Spearman ρ between semantic and message-space distances — measures compositionality.
- **Symbol Entropy**: Shannon entropy of symbol usage — high entropy means an efficient, balanced vocabulary.

## Expected Training Dynamics

| Epoch | Accuracy | TopSim |
|---|---|---|
| 0 | ~25% (random) | ~0.0 |
| 500 | ~70–85% | ~0.1–0.3 |
| 1000 | ~90–98% | ~0.3–0.6 |
| 2000 | >95% | ~0.4–0.7 |
