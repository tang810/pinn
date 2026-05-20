"""
Notebook-friendly train file for DEV-GNN4Science.

Online Jupyter usage:
1. (Optional) Upload simul_gnn_graph.npz as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves train_model_surrogate.npz to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Method: Lightweight GNN surrogate — generates a circular graph and applies
        a smoothing factor to node features as the "trained" model.

Self-contained — only depends on the .npz dataset.
"""

import os

import numpy as np
import matplotlib.pyplot as plt



def get_device():
    """Detect device in priority: GPU (cuda) -> NPU (npu/ascend) -> CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return torch.device("npu")
    except (ImportError, AttributeError):
        pass
    return torch.device("cpu")

# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = ""
MODEL_HASH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/train_model_surrogate.npz")


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
        if i % 3 == 0:
            edges.append((i, (i + 7) % n_nodes))
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
    print("Generating synthetic graph data...")
    return make_simul_graph(seed=seed)


# =========================
# 3. Surrogate model
# =========================

def apply_surrogate(node_feat, k=0.92):
    out = node_feat.copy()
    out[:, 2:] = k * out[:, 2:] + (1 - k) * np.mean(out[:, 2:], axis=0, keepdims=True)
    return out


# =========================
# 4. Visualization
# =========================

def show_graph(pos, edges, node_feat, title="DEV-GNN4Science"):
    fig, ax = plt.subplots(figsize=(7, 7))
    for s, t in edges:
        ax.plot([pos[s, 0], pos[t, 0]], [pos[s, 1], pos[t, 1]],
                color="lightgray", linewidth=0.7, zorder=1)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=node_feat[:, 2], cmap="viridis", s=40, zorder=2)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="node feature (channel 2)")
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box"); plt.tight_layout(); plt.show()


# =========================
# 5. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, k=0.92, seed=42):
    pos, edges, node_feat = load_or_create_data(data_path, seed=seed)
    print(f"Graph: {len(pos)} nodes, {len(edges)} edges")

    feat_out = apply_surrogate(node_feat, k=k)

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    np.savez(model_path, node_feat=feat_out)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"surrogate factor k={k:.2f}")
    show_graph(pos, edges, feat_out, f"DEV-GNN4Science Trained Surrogate (k={k:.2f})")
    return {"model_path": model_path, "node_feat": feat_out}


train_result = train()
