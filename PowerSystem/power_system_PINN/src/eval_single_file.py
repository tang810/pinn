"""
Notebook-friendly eval file for power_system_PINN.

Online Jupyter usage:
1. Upload the trained .pt model as a public asset.
2. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the model, evaluates 10-node power system frequency dynamics,
and shows figures with plt.show(). It does not save images.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "4ce4c950b82341ec8ba49f1f17607336"
MODEL_HASH = "880abd571a72412497522eae36bcd09b"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_power_system.pt")


# =========================
# 2. Model definition
# =========================

class PINN(nn.Module):
    def __init__(self, num_nodes=10, hidden_dim=64, num_layers=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.Tanh())
        self.output_layer = nn.Linear(hidden_dim, 3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.M = torch.tensor(6.0, device=device)
        self.X_line = torch.tensor(0.1, device=device)

    def forward(self, t, x):
        inputs = torch.stack([t.unsqueeze(1) if t.dim()==1 else t,
                              x.unsqueeze(1) if x.dim()==1 else x], dim=1).squeeze(-1)
        if inputs.dim() == 1: inputs = inputs.unsqueeze(0)
        out = self.input_layer(inputs)
        out = torch.tanh(out)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def load_model_package(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    num_nodes = ckpt.get("num_nodes", 10)
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    hidden_dim = ckpt.get("hidden_dim")
    num_layers = ckpt.get("num_layers")
    if hidden_dim is None or num_layers is None:
        hidden_dim = int(state["input_layer.weight"].shape[0])
        num_layers = sum(1 for k in state if k.startswith("hidden_layers.") and k.endswith(".weight"))
    model = PINN(num_nodes=num_nodes, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


# =========================
# 3. Evaluation
# =========================

def evaluate_model(model, num_nodes=10, t_max=1.0, num_time_steps=200):
    device = next(model.parameters()).device
    t = torch.linspace(0, t_max, num_time_steps, device=device)
    x = torch.linspace(0, num_nodes - 1, num_nodes, device=device)
    tg, xg = torch.meshgrid(t, x, indexing="ij")
    tf, xf = tg.flatten(), xg.flatten()
    model.eval()
    with torch.no_grad():
        df, pe, pm = model(tf, xf)
    df_np = df.cpu().numpy().reshape(tg.shape)
    pe_np = pe.cpu().numpy().reshape(tg.shape)
    pm_np = pm.cpu().numpy().reshape(tg.shape)
    t_np = t.cpu().numpy()
    tg_np = tg.cpu().numpy()
    return df_np, pe_np, pm_np, t_np, tg_np


# =========================
# 4. Visualization
# =========================

def show_eval_result(df, pe, pm, t_arr, tg, num_nodes, history, t_max):
    if history:
        plt.figure(figsize=(7, 4))
        plt.plot([h.get("Total Loss", h) for h in history]); plt.yscale("log")
        plt.xlabel("Hundreds of Epochs"); plt.ylabel("Total Loss")
        plt.title("Training Loss"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    im = axes[0, 0].imshow(df.T, aspect="auto", extent=[0, t_max, 0, num_nodes - 1],
                            origin="lower", cmap="coolwarm")
    axes[0, 0].set_title("Frequency Increment Δf (Hz)"); axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Node"); plt.colorbar(im, ax=axes[0, 0])

    for i in range(0, num_nodes, max(1, num_nodes // 5)):
        axes[0, 1].plot(t_arr, df[:, i], label=f"Node {i}")
    axes[0, 1].set_title("Node Frequencies vs Time"); axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Δf (Hz)"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    for node in [0, num_nodes // 2 - 1, num_nodes - 1]:
        axes[1, 0].plot(t_arr, pe[:, node], label=f"ΔPe node {node}")
        axes[1, 0].plot(t_arr, pm[:, node], "--", label=f"ΔPm node {node}")
    axes[1, 0].set_title("Power Comparison"); axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Power (pu)"); axes[1, 0].legend(fontsize=7); axes[1, 0].grid(alpha=0.3)

    frame_step = max(1, tg.shape[0] // 6)
    for frame in range(0, tg.shape[0], frame_step):
        axes[1, 1].plot(range(num_nodes), df[frame, :], "o-", label=f"t={t_arr[frame]:.2f}s")
    axes[1, 1].set_title("Frequency Propagation Along Line")
    axes[1, 1].set_xlabel("Node"); axes[1, 1].set_ylabel("Δf (Hz)")
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout(); plt.show()


# =========================
# 5. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH,
             num_nodes=10, t_max=1.0, num_time_steps=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, ckpt = load_model_package(model_path, device)
    num_nodes = ckpt.get("num_nodes", num_nodes)
    t_max = ckpt.get("t_max", t_max)
    history = ckpt.get("history", [])

    df, pe, pm, t_arr, tg = evaluate_model(model, num_nodes, t_max, num_time_steps)

    print("Eval result")
    print(f"model_path: {model_path}")
    print(f"num_nodes: {num_nodes}, t_max: {t_max}s")
    print(f"Δf range: [{df.min():.4f}, {df.max():.4f}]")
    if history and isinstance(history[0], dict):
        print(f"final_loss: {history[-1].get('Total Loss', 'N/A')}")

    show_eval_result(df, pe, pm, t_arr, tg, num_nodes, history, t_max)
    return {"df": df, "pe": pe, "pm": pm}


eval_result = evaluate()
