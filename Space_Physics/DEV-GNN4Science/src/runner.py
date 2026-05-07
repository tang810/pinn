import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dirs(args):
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)


def _make_simul_graph(path, seed=42, n_nodes=64):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 1.0 + 0.25 * np.sin(3 * theta) + 0.08 * rng.normal(size=n_nodes)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    edges = []
    for i in range(n_nodes):
        edges.append((i, (i + 1) % n_nodes))
        if i % 3 == 0:
            edges.append((i, (i + 7) % n_nodes))
    edges = np.array(edges, dtype=np.int32)

    node_feat = np.stack(
        [
            x,
            y,
            np.sin(2 * theta) + 0.1 * rng.normal(size=n_nodes),
            np.cos(3 * theta) + 0.1 * rng.normal(size=n_nodes),
        ],
        axis=1,
    ).astype(np.float32)
    np.savez(path, pos=np.stack([x, y], axis=1).astype(np.float32), edges=edges, node_feat=node_feat)
    return np.stack([x, y], axis=1), edges, node_feat


def _load_or_create_data(args):
    path = Path(args.data_path)
    if args.data == "simul":
        return path, *_make_simul_graph(path, seed=args.seed)
    if not path.exists():
        raise FileNotFoundError(f"Real data not found: {path}")
    d = np.load(path)
    return path, d["pos"], d["edges"], d["node_feat"]


def _save_graph_plot(result_dir, pos, edges, node_feat, title):
    result_dir = Path(result_dir)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    for s, t in edges:
        ax.plot([pos[s, 0], pos[t, 0]], [pos[s, 1], pos[t, 1]], color="lightgray", linewidth=0.7, zorder=1)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=node_feat[:, 2], cmap="viridis", s=28, zorder=2)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="node_feature_2")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(result_dir / "quick_result.png")
    plt.close(fig)


def run_quick(args):
    _ensure_dirs(args)
    data_path, pos, edges, node_feat = _load_or_create_data(args)
    _save_graph_plot(args.result_dir, pos, edges, node_feat, f"GNN Quick Graph ({args.task})")
    np.savez(Path(args.result_dir) / "quick_graph_fields.npz", pos=pos, edges=edges, node_feat=node_feat)

    summary = {
        "mode": "quick",
        "task": args.task,
        "data": args.data,
        "data_path": str(data_path),
        "result_image": str(Path(args.result_dir) / "quick_result.png"),
    }
    (Path(args.result_dir) / "quick_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[quick] Result image saved to {Path(args.result_dir) / 'quick_result.png'}")


def _build_delegate_command(args):
    script_map = {
        "airfoil": ("airfoil_train.py", "airfoil_infer.py"),
        "cardiovascular": ("cardiovascular_train.py", "cardiovascular_infer.py"),
        "cylinder_flow": ("cylinder_flow_train.py", "cylinder_flow_infer.py"),
        "flag_simple": ("flag_simple_train.py", "flag_simple_infer.py"),
        "mol_doc_pdb": ("mol_doc_pdb_train.py", "mol_doc_pdb_infer.py"),
        "mol_gen_qm9": ("mol_gen_qm9_train.py", "mol_gen_qm9_infer.py"),
    }
    train_script, _ = script_map[args.task]
    script_path = Path("src/demo") / train_script
    return [sys.executable, str(script_path)]


def run_train(args):
    _ensure_dirs(args)
    if args.delegate_original:
        cmd = _build_delegate_command(args)
        print("[train] Delegating to original script:", " ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Original script failed with code {completed.returncode}")
    else:
        data_path, pos, edges, node_feat = _load_or_create_data(args)
        feat_out = node_feat.copy()
        feat_out[:, 2:] = 0.92 * feat_out[:, 2:] + 0.08 * np.mean(feat_out[:, 2:], axis=0, keepdims=True)
        np.savez(Path(args.model_dir) / "train_model_surrogate.npz", node_feat=feat_out)
        _save_graph_plot(args.result_dir, pos, edges, feat_out, f"GNN Train Surrogate ({args.task})")
        (Path(args.result_dir) / "train_summary.json").write_text(
            json.dumps(
                {
                    "mode": "train",
                    "task": args.task,
                    "data_path": str(data_path),
                    "model": str(Path(args.model_dir) / "train_model_surrogate.npz"),
                    "result_image": str(Path(args.result_dir) / "quick_result.png"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print("[train] Done.")

