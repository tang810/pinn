"""
Notebook-friendly eval file for DEV-GNN4Science.

Online Jupyter usage:
1. Upload train_model_surrogate.npz as a public asset.
2. Upload simul_gnn_graph.npz as a public asset (optional, for comparison).
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the GNN surrogate model, visualizes the graph node features,
and shows figures with plt.show(). It does not save images.

Self-contained — only depends on the .npz files.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = ""
MODEL_HASH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/train_model_surrogate.npz")


# =========================
# 2. Data generation / loading
# =========================

def make_simul_graph(seed=42, n_nodes=64):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 1.0 + 0.25 * np.sin(3 * theta) + 0.08 * rng.normal(size=n_nodes)
    x = radius * np.cos(theta); y = radius * np.sin(theta)
    pos = np.stack([x, y], axis=1).astype(np.float32)
    edges = []
    for i in range(n_nodes):
        edges.append((i, (i + 1) % n_nodes))
        if i % 3 == 0: edges.append((i, (i + 7) % n_nodes))
    edges = np.array(edges, dtype=np.int32)
    node_feat = np.stack([
        x, y,
        np.sin(2 * theta) + 0.1 * rng.normal(size=n_nodes),
        np.cos(3 * theta) + 0.1 * rng.normal(size=n_nodes),
    ], axis=1).astype(np.float32)
    return pos, edges, node_feat


def load_or_create_data(data_path, seed=42):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path and os.path.isfile(data_path):
        d = np.load(data_path)
        return d["pos"], d["edges"], d["node_feat"]
    return make_simul_graph(seed=seed)


def load_model(model_path):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return np.load(model_path)["node_feat"]


# =========================
# 3. Visualization
# =========================

def show_graph(pos, edges, node_feat, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    for s, t in edges:
        ax.plot([pos[s, 0], pos[t, 0]], [pos[s, 1], pos[t, 1]],
                color="lightgray", linewidth=0.7, zorder=1)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=node_feat[:, 2], cmap="viridis", s=40, zorder=2)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="node feature (channel 2)")
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box"); plt.tight_layout(); plt.show()


def show_eval_result(pos, edges, ref_feat, pred_feat):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, feat, title in [(ax1, ref_feat, "Reference"), (ax2, pred_feat, "Model (Surrogate)")]:
        for s, t in edges:
            ax.plot([pos[s, 0], pos[t, 0]], [pos[s, 1], pos[t, 1]],
                    color="lightgray", linewidth=0.7, zorder=1)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=feat[:, 2], cmap="viridis", s=40, zorder=2)
        plt.colorbar(sc, ax=ax, shrink=0.8, label="node feature (ch 2)")
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    plt.suptitle("DEV-GNN4Science Eval — Reference vs Model", fontsize=13)
    plt.tight_layout(); plt.show()

    mae = float(np.mean(np.abs(ref_feat[:, 2:] - pred_feat[:, 2:])))
    print(f"MAE (channels 2:): {mae:.4e}")
    return mae


# =========================
# 4. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH, seed=42):
    pos, edges, ref_feat = load_or_create_data(data_path, seed=seed)
    pred_feat = load_model(model_path)

    print("Eval result"); print(f"model_path: {model_path}")
    print(f"nodes: {len(pos)}, edges: {len(edges)}")

    mae = show_eval_result(pos, edges, ref_feat, pred_feat)
    return {"mae": mae}


eval_result = evaluate()
