"""
app.py — Interactive Streamlit dashboard for the Lewis Signalling Game.

Tabs
----
1. Configure & Train  — set hyperparameters, launch training, watch live progress
2. Training Curves    — accuracy, loss, temperature, TopSim plots
3. Language Analysis  — symbol heatmap, entropy, TopSim score, message explorer
4. Emergent Language  — full table of all objects → their emergent messages
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import torch

# ── Page configuration must come first ──────────────────────────────────────
st.set_page_config(
    page_title="Lewis Signalling Game",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Local imports (after set_page_config) ───────────────────────────────────
from lewis.agents import Sender, Receiver
from lewis.game import GameConfig, LewisGame, compute_temperature
from lewis.metrics import (
    topographic_similarity,
    symbol_frequencies,
    symbol_entropy,
    mean_entropy,
    build_message_table,
)
from lewis.visualize import (
    plot_training_curves,
    plot_topsim_curve,
    plot_symbol_heatmap,
    plot_entropy_bars,
    plot_message_scatter,
)
from lewis.world import World, WorldConfig

# ── Session state defaults ───────────────────────────────────────────────────
_DEFAULTS = {
    "training_running": False,
    "metrics_log": [],
    "run_dir": "./runs/dashboard_run",
    "world": None,
    "sender": None,
    "receiver": None,
    "cfg": None,
    "world_cfg": None,
    "train_thread": None,
    "stop_flag": False,
    "epoch_done": 0,
    "best_acc": 0.0,
    "checkpoint_loaded": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# Training thread (runs in background)
# ═══════════════════════════════════════════════════════════════════════════

def _training_loop(
    cfg: GameConfig,
    world: World,
    sender: Sender,
    receiver: Receiver,
    run_dir: Path,
    topsim_every: int,
) -> None:
    """Background training loop that updates session state in place."""
    run_dir.mkdir(parents=True, exist_ok=True)

    device = next(sender.parameters()).device
    params = list(sender.parameters()) + list(receiver.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    game = LewisGame(world, sender, receiver)

    metrics_path = run_dir / "metrics.jsonl"
    best_acc = 0.0
    st.session_state.metrics_log = []

    with open(metrics_path, "w") as mf:
        for epoch in range(cfg.epochs):
            if st.session_state.stop_flag:
                break

            sender.train()
            receiver.train()

            tau = compute_temperature(epoch, cfg)
            result = game.step(cfg.batch_size, temperature=tau)

            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()

            acc = result["accuracy"]
            loss_val = result["loss"].item()

            topsim = None
            entropy_val = None
            if epoch % topsim_every == 0:
                try:
                    freq = symbol_frequencies(world, sender)
                    entropy_val = mean_entropy(freq)
                    topsim = topographic_similarity(world, sender)
                except Exception:
                    pass

            record = {
                "epoch": epoch,
                "loss": round(loss_val, 6),
                "accuracy": round(acc, 4),
                "temperature": round(tau, 4),
                "topsim": round(topsim, 4) if topsim is not None else None,
                "entropy": round(entropy_val, 4) if entropy_val is not None else None,
            }
            mf.write(json.dumps(record) + "\n")
            mf.flush()

            st.session_state.metrics_log.append(record)
            st.session_state.epoch_done = epoch

            if acc > best_acc:
                best_acc = acc
                st.session_state.best_acc = best_acc
                torch.save({
                    "epoch": epoch,
                    "sender_state": sender.state_dict(),
                    "receiver_state": receiver.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": cfg.__dict__,
                    "world_cfg": world.config.__dict__,
                }, run_dir / "checkpoint_best.pt")

    torch.save({
        "epoch": epoch,
        "sender_state": sender.state_dict(),
        "receiver_state": receiver.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.__dict__,
        "world_cfg": world.config.__dict__,
    }, run_dir / "checkpoint_last.pt")

    st.session_state.training_running = False
    st.session_state.stop_flag = False


# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════

st.title("🗣️ Lewis Signalling Game")
st.markdown(
    "Two neural agents **invent a shared language from scratch** "
    "to solve a cooperative object-referencing task. "
    "No language is pre-given — they must discover one through play."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Configure & Train",
    "📈 Training Curves",
    "🔬 Language Analysis",
    "📖 Emergent Language Viewer",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Configure & Train
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Game & Agent Configuration")

    col_world, col_agent, col_train, col_gs = st.columns(4)

    with col_world:
        st.subheader("🌍 World")
        n_attributes = st.slider("Attributes", 2, 5, 3,
            help="Number of object dimensions (e.g., color, shape, size)")
        n_values = st.slider("Values per attribute", 2, 6, 4,
            help="Number of distinct values each attribute can take")
        n_distractors = st.slider("Distractors", 1, 7, 3,
            help="Distractor objects shown to Receiver — more = harder")
        n_objects = n_values ** n_attributes
        st.info(f"**{n_objects}** unique objects total")

    with col_agent:
        st.subheader("🧠 Agents")
        hidden_dim = st.select_slider("Hidden dim", [64, 128, 256, 512], 128)
        vocab_size = st.slider("Vocab size (V)", 2, 50, 10,
            help="Number of symbols in the shared vocabulary")
        msg_len = st.slider("Message length (L)", 1, 6, 3,
            help="Number of symbols per message")
        st.caption(f"Message space: V^L = **{vocab_size**msg_len}** combinations")

    with col_train:
        st.subheader("🏋️ Training")
        epochs = st.number_input("Epochs", 100, 10000, 2000, step=100)
        batch_size = st.select_slider("Batch size", [128, 256, 512, 1024], 512)
        lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 3e-3, 1e-2], 1e-3,
                              format_func=lambda x: f"{x:.0e}")
        topsim_every = st.slider("TopSim every N epochs", 10, 200, 50,
            help="TopSim is O(N²) — compute less often for large worlds")

    with col_gs:
        st.subheader("🌡️ Temperature")
        tau_start = st.slider("τ start", 0.5, 10.0, 5.0, step=0.5,
            help="Initial Gumbel-Softmax temperature")
        tau_min = st.slider("τ min", 0.01, 1.0, 0.1, step=0.01,
            help="Temperature floor after annealing")
        tau_decay = st.slider("τ decay", 0.990, 0.9999, 0.998, step=0.001,
            help="Per-epoch decay factor")
        half_life = int(math.log(0.5) / math.log(tau_decay)) if tau_decay < 1 else 9999
        st.caption(f"Temperature halves every **{half_life}** epochs")

    run_dir_str = st.text_input("Output directory", "./runs/dashboard_run")

    st.divider()

    # ── Buttons ─────────────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

    with btn_col1:
        start_btn = st.button(
            "▶️ Start Training",
            disabled=st.session_state.training_running,
            use_container_width=True,
            type="primary",
        )
    with btn_col2:
        stop_btn = st.button(
            "⏹ Stop",
            disabled=not st.session_state.training_running,
            use_container_width=True,
        )
    with btn_col3:
        load_col1, load_col2 = st.columns([3, 1])
        with load_col1:
            load_path = st.text_input("Load checkpoint", placeholder="path/to/checkpoint.pt",
                                      label_visibility="collapsed")
        with load_col2:
            load_btn = st.button("📂 Load", use_container_width=True)

    # ── Start training ───────────────────────────────────────────────────────
    if start_btn and not st.session_state.training_running:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        world_cfg = WorldConfig(
            n_attributes=n_attributes,
            n_values=n_values,
            n_distractors=n_distractors,
        )
        cfg = GameConfig(
            n_attributes=n_attributes,
            n_values=n_values,
            n_distractors=n_distractors,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            msg_len=msg_len,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            tau_start=tau_start,
            tau_min=tau_min,
            tau_decay=tau_decay,
        )

        world = World(world_cfg, device=device)
        sender = Sender(
            input_dim=world_cfg.input_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            msg_len=msg_len,
        ).to(device)
        receiver = Receiver(
            input_dim=world_cfg.input_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            msg_len=msg_len,
            n_candidates=world_cfg.n_candidates,
        ).to(device)

        st.session_state.world = world
        st.session_state.sender = sender
        st.session_state.receiver = receiver
        st.session_state.cfg = cfg
        st.session_state.world_cfg = world_cfg
        st.session_state.run_dir = run_dir_str
        st.session_state.training_running = True
        st.session_state.stop_flag = False
        st.session_state.epoch_done = 0
        st.session_state.best_acc = 0.0
        st.session_state.checkpoint_loaded = False

        thread = threading.Thread(
            target=_training_loop,
            args=(cfg, world, sender, receiver, Path(run_dir_str), topsim_every),
            daemon=True,
        )
        add_script_run_ctx(thread)
        thread.start()
        st.session_state.train_thread = thread
        st.rerun()

    # ── Stop training ────────────────────────────────────────────────────────
    if stop_btn and st.session_state.training_running:
        st.session_state.stop_flag = True
        st.warning("⏹ Stop signal sent — finishing current epoch…")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    if load_btn and load_path and not st.session_state.training_running:
        ckpt_path = Path(load_path)
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                cfg_d = ckpt.get("cfg", {})
                wc_d = ckpt.get("world_cfg", {})

                world_cfg = WorldConfig(**wc_d)
                cfg = GameConfig(**cfg_d)

                world = World(world_cfg)
                sender = Sender(
                    input_dim=world_cfg.input_dim,
                    hidden_dim=cfg.hidden_dim,
                    vocab_size=cfg.vocab_size,
                    msg_len=cfg.msg_len,
                )
                receiver = Receiver(
                    input_dim=world_cfg.input_dim,
                    hidden_dim=cfg.hidden_dim,
                    vocab_size=cfg.vocab_size,
                    msg_len=cfg.msg_len,
                    n_candidates=world_cfg.n_candidates,
                )
                sender.load_state_dict(ckpt["sender_state"])
                receiver.load_state_dict(ckpt["receiver_state"])
                sender.eval()
                receiver.eval()

                # Try to load associated metrics log
                log_path = ckpt_path.parent / "metrics.jsonl"
                metrics_log = []
                if log_path.exists():
                    with open(log_path) as f:
                        for line in f:
                            try:
                                metrics_log.append(json.loads(line))
                            except Exception:
                                pass

                st.session_state.world = world
                st.session_state.sender = sender
                st.session_state.receiver = receiver
                st.session_state.cfg = cfg
                st.session_state.world_cfg = world_cfg
                st.session_state.metrics_log = metrics_log
                st.session_state.checkpoint_loaded = True
                st.success(f"✅ Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
            except Exception as e:
                st.error(f"Failed to load checkpoint: {e}")
        else:
            st.error(f"File not found: {ckpt_path}")

    # ── Live status ──────────────────────────────────────────────────────────
    st.divider()
    if st.session_state.training_running or st.session_state.metrics_log:
        st.subheader("Live Status")
        total = st.session_state.cfg.epochs if st.session_state.cfg else 1
        done = st.session_state.epoch_done
        st.progress(min(done / max(total - 1, 1), 1.0),
                    text=f"Epoch {done} / {total}")

        log = st.session_state.metrics_log
        if log:
            latest = log[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{latest['accuracy']:.1%}")
            m2.metric("Loss", f"{latest['loss']:.4f}")
            m3.metric("Temperature τ", f"{latest['temperature']:.3f}")
            ts_pts = [m for m in log if m.get("topsim") is not None]
            m4.metric("TopSim", f"{ts_pts[-1]['topsim']:.3f}" if ts_pts else "—")
            m1.metric("Best Accuracy", f"{st.session_state.best_acc:.1%}")

        if st.session_state.training_running:
            st.info("🔄 Training in progress — switch to **Training Curves** tab to watch live.")
            time.sleep(1)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Training Curves
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Training Curves")
    log = st.session_state.metrics_log

    if not log:
        st.info("Start training in the **Configure & Train** tab to see curves here.")
    else:
        st.plotly_chart(plot_training_curves(log), use_container_width=True)
        st.plotly_chart(plot_topsim_curve(log), use_container_width=True)

        if st.session_state.training_running:
            time.sleep(1)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Language Analysis
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Language Analysis")

    sender = st.session_state.sender
    world = st.session_state.world
    cfg = st.session_state.cfg

    if sender is None or world is None:
        st.info("Start training or load a checkpoint to analyse the emergent language.")
    else:
        # Compute metrics
        try:
            freq = symbol_frequencies(world, sender)
            entropies = symbol_entropy(freq)
            max_ent = math.log2(cfg.vocab_size) if cfg else math.log2(10)

            # TopSim
            log = st.session_state.metrics_log
            ts_pts = [m for m in log if m.get("topsim") is not None]
            topsim_val = ts_pts[-1]["topsim"] if ts_pts else None

            # ── Summary metrics ────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Symbol Entropy",
                      f"{entropies.mean():.2f} / {max_ent:.2f} bits")
            c2.metric("Topographic Similarity",
                      f"{topsim_val:.3f}" if topsim_val is not None else "—",
                      help="Spearman ρ: 0=random, >0.5=compositional")
            acc_pts = [m["accuracy"] for m in log] if log else []
            c3.metric("Current Accuracy",
                      f"{acc_pts[-1]:.1%}" if acc_pts else "—")

            # ── TopSim interpretation ──────────────────────────────────────
            if topsim_val is not None:
                if topsim_val >= 0.5:
                    st.success(f"🎉 **Compositional language!** TopSim = {topsim_val:.3f} "
                               "— similar objects are described by similar messages.")
                elif topsim_val >= 0.2:
                    st.warning(f"⚠️ **Weakly structured.** TopSim = {topsim_val:.3f} "
                               "— some compositionality emerging, keep training.")
                else:
                    st.error(f"❌ **Random-like.** TopSim = {topsim_val:.3f} "
                             "— language not yet compositional.")

            st.divider()

            # ── Heatmap + Entropy ──────────────────────────────────────────
            h1, h2 = st.columns(2)
            with h1:
                st.plotly_chart(plot_symbol_heatmap(freq), use_container_width=True)
            with h2:
                st.plotly_chart(plot_entropy_bars(entropies, max_ent),
                                use_container_width=True)

            st.divider()

            # ── Message explorer ───────────────────────────────────────────
            st.subheader("🔍 Message Explorer")
            st.markdown(
                "Select an object by its attributes to see what message the Sender assigns it."
            )
            world_cfg = world.config
            attr_names = ["color", "shape", "size", "texture", "pattern"]
            value_names = [
                ["red", "green", "blue", "yellow"],
                ["circle", "square", "triangle", "star"],
                ["small", "medium", "large", "xlarge"],
                ["plain", "striped", "dotted", "checkered"],
                ["solid", "hollow", "outline", "gradient"],
            ]
            attr_vals = []
            exp_cols = st.columns(world_cfg.n_attributes)
            for i in range(world_cfg.n_attributes):
                name = attr_names[i] if i < len(attr_names) else f"attr{i}"
                vals = (value_names[i][:world_cfg.n_values]
                        if i < len(value_names) else [str(j) for j in range(world_cfg.n_values)])
                chosen = exp_cols[i].selectbox(name.capitalize(), vals)
                attr_vals.append(vals.index(chosen))

            # Build one-hot for the selected object
            import itertools
            one_hot = torch.zeros(world_cfg.input_dim)
            for ai, v in enumerate(attr_vals):
                one_hot[ai * world_cfg.n_values + v] = 1.0
            device = next(sender.parameters()).device
            x = one_hot.unsqueeze(0).to(device)
            with torch.no_grad():
                msg = sender.hard_message(x)[0].tolist()

            st.markdown(f"**Message**: `{msg}`")
            st.markdown(
                "  ".join([
                    f"<span style='background:#1f4e79;padding:6px 12px;"
                    f"border-radius:6px;font-size:1.3em;font-weight:bold'>{s}</span>"
                    for s in msg
                ]),
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Analysis failed: {e}")

        if st.session_state.training_running:
            time.sleep(2)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Emergent Language Viewer
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Emergent Language Viewer")
    st.markdown(
        "Full table of every object in the world and the message the Sender assigns it. "
        "Objects with the same **first symbol** are highlighted in the same colour — "
        "look for clusters that correspond to shared attributes."
    )

    sender = st.session_state.sender
    world = st.session_state.world
    cfg = st.session_state.cfg

    if sender is None or world is None:
        st.info("Start training or load a checkpoint to see the emergent language table.")
    else:
        try:
            rows = build_message_table(world, sender)
            st.plotly_chart(plot_message_scatter(rows, world.config.n_values),
                            use_container_width=True)

            st.divider()
            st.subheader("Full Message Table")

            # Colour-code rows by first symbol
            n_symbols = cfg.vocab_size if cfg else 10
            palette = [
                "#1f4e79", "#7f1d1d", "#14532d", "#78350f", "#312e81",
                "#701a75", "#134e4a", "#1e3a5f", "#4a1942", "#2d3748",
            ]

            for i, row in enumerate(rows):
                bg = palette[row["first_symbol"] % len(palette)]
                sym_badges = " ".join([
                    f"<span style='background:{bg};padding:3px 8px;"
                    f"border-radius:4px;font-weight:bold'>{s}</span>"
                    for s in row["message"]
                ])
                st.markdown(
                    f"<div style='padding:4px 8px;margin:2px 0;"
                    f"border-left:4px solid {bg};border-radius:4px;"
                    f"background:rgba(255,255,255,0.03)'>"
                    f"<code>{row['object_str']}</code> → {sym_badges}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Could not build language table: {e}")

        if st.session_state.training_running:
            time.sleep(2)
            st.rerun()
