"""
Notebook-friendly eval file for Multiphysics_1 (PINNsformer).

Online Jupyter usage:
1. Upload trained_model_multiphysics1.pt as a public asset.
2. Upload data_7246.txt as a public asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the PINNsformer model, evaluates temperature prediction,
and shows figures with plt.show(). It does not save images.

Self-contained PyTorch-only.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = ""
MODEL_HASH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_multiphysics1.pt")


# =========================
# 2. PINNsformer Model (minimal)
# =========================

class WaveAct(nn.Module):
    def __init__(self): super().__init__(); self.w1 = nn.Parameter(torch.ones(1)); self.w2 = nn.Parameter(torch.ones(1))
    def forward(self, x): return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(d_model, d_ff), WaveAct(), nn.Linear(d_ff, d_ff), WaveAct(), nn.Linear(d_ff, d_model))
    def forward(self, x): return self.linear(x)

class EncoderLayer(nn.Module):
    def __init__(self, dm, h): super().__init__(); self.attn = nn.MultiheadAttention(dm, h, batch_first=True); self.ff = FeedForward(dm); self.act1 = WaveAct(); self.act2 = WaveAct()
    def forward(self, x): x2 = self.act1(x); x = x + self.attn(x2, x2, x2)[0]; x2 = self.act2(x); x = x + self.ff(x2); return x

class DecoderLayer(nn.Module):
    def __init__(self, dm, h): super().__init__(); self.attn = nn.MultiheadAttention(dm, h, batch_first=True); self.ff = FeedForward(dm); self.act1 = WaveAct(); self.act2 = WaveAct()
    def forward(self, x, e): x2 = self.act1(x); x = x + self.attn(x2, e, e)[0]; x2 = self.act2(x); x = x + self.ff(x2); return x

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(N)])
        self.act = WaveAct()
    def forward(self, x):
        for l in self.layers: x = l(x); return self.act(x)

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(N)])
        self.act = WaveAct()
    def forward(self, x, e):
        for l in self.layers: x = l(x, e); return self.act(x)

class PINNsformer(nn.Module):
    def __init__(self, d_out=1, d_model=64, d_hidden=512, N=1, heads=2):
        super().__init__()
        self.linear_emb = nn.Linear(4, d_model)
        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(nn.Linear(d_model, d_hidden), WaveAct(), nn.Linear(d_hidden, d_hidden), WaveAct(), nn.Linear(d_hidden, d_out))
    def forward(self, x, y, z, t):
        src = self.linear_emb(torch.cat([x, y, z, t], dim=-1)); e = self.encoder(src); d = self.decoder(src, e)
        return self.linear_out(d)


# =========================
# 3. Model loading
# =========================

def load_model_package(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]; dm = ckpt.get("d_model", 64); dh = ckpt.get("d_hidden", 512)
        N = ckpt.get("N", 1); h = ckpt.get("heads", 2); hist = ckpt.get("history", [])
    else:
        state = ckpt; dm = state["linear_emb.weight"].shape[0]; dh = state["linear_out.0.weight"].shape[0]
        N = 1; h = 2; hist = []
    model = PINNsformer(d_out=1, d_model=dm, d_hidden=dh, N=N, heads=h).to(device)
    model.load_state_dict(state); model.eval()
    return model, hist


# =========================
# 4. Visualization
# =========================

def show_eval_result(model, device, data_path, history):
    if history:
        plt.figure(figsize=(7,4)); plt.plot(history); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    data_path = os.path.expanduser(data_path) if data_path else "data/data_7246.txt"
    if not os.path.isfile(data_path):
        print("Data file not found, skipping eval plots.")
        return
    df = pd.read_csv(data_path, delimiter=' ', header=None).to_numpy()
    idx = np.linspace(0, len(df)-1, min(8000, len(df))).astype(int)
    x = torch.tensor(df[idx, 0:1], dtype=torch.float32, device=device)
    y = torch.tensor(df[idx, 1:2], dtype=torch.float32, device=device)
    z = torch.tensor(df[idx, 2:3], dtype=torch.float32, device=device)
    t = torch.tensor(df[idx, 3:4], dtype=torch.float32, device=device)
    T_true = df[idx, 4]

    with torch.no_grad():
        T_pred = model(x, y, z, t).cpu().numpy().reshape(-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sc = axes[0].scatter(T_true, T_pred, c=t.cpu().numpy().reshape(-1), s=3, alpha=0.4, cmap="plasma")
    axes[0].plot([T_true.min(), T_true.max()], [T_true.min(), T_true.max()], "r--", lw=1)
    axes[0].set_xlabel("True T (K)"); axes[0].set_ylabel("Pred T (K)"); axes[0].set_title("True vs Pred")
    axes[0].grid(alpha=0.3); plt.colorbar(sc, ax=axes[0], label="t")
    err = T_pred - T_true
    axes[1].hist(err, bins=50); axes[1].set_xlabel("Error (K)"); axes[1].set_title("Error Distribution")
    axes[2].scatter(T_true, err, s=2, alpha=0.4); axes[2].axhline(0, color='r', ls='--')
    axes[2].set_xlabel("True T (K)"); axes[2].set_ylabel("Error (K)"); axes[2].set_title("Error vs True T")
    axes[2].grid(alpha=0.3)
    plt.suptitle("Multiphysics_1 PINNsformer Eval", fontsize=14); plt.tight_layout(); plt.show()

    rl2 = np.linalg.norm(err) / (np.linalg.norm(T_true) + 1e-12)
    mae = float(np.mean(np.abs(err))); rmse = float(np.sqrt(np.mean(err**2)))
    print(f"Relative L2: {rl2:.4e}, MAE: {mae:.4f} K, RMSE: {rmse:.4f} K")
    return rl2, mae, rmse


# =========================
# 5. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, history = load_model_package(model_path, device)
    print("Eval result"); print(f"model_path: {model_path}")

    rl2, mae, rmse = show_eval_result(model, device, data_path, history)
    return {"rl2": rl2, "mae": mae, "rmse": rmse}


eval_result = evaluate()
