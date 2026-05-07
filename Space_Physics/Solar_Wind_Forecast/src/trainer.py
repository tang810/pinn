import numpy as np
import torch

from .constants import FEATURES
from .data_utils import inverse_z_torch


def rmse(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def physics_penalty_original_units(yhat_norm, alpha, mu_t, sig_t):
    """
    yhat_norm: (batch, 7) in normalized space
    alpha: scalar in original space
    mu_t, sig_t: torch tensors (7,) in original space
    """
    yhat_raw = inverse_z_torch(yhat_norm, mu_t, sig_t)

    E = yhat_raw[:, 0].abs()
    V = yhat_raw[:, 1:4]
    B = yhat_raw[:, 4:7]
    cross = torch.cross(V, B, dim=1)
    cross_norm = torch.linalg.vector_norm(cross, ord=2, dim=1)

    g = E - alpha * cross_norm
    return torch.relu(g).mean()


@torch.no_grad()
def predict_all(model, loader, device="cpu"):
    model.eval()
    ys, yhats = [], []
    for x, y in loader:
        x = x.to(device)
        yhat = model(x)
        ys.append(y.cpu().numpy())
        yhats.append(yhat.cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(yhats, axis=0)


@torch.no_grad()
def eval_metrics(model, loader, alpha, mu_t, sig_t, device="cpu"):
    model.eval()
    ys, yhats = [], []
    rmses, phys = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        ys.append(y.cpu().numpy())
        yhats.append(yhat.cpu().numpy())
        rmses.append(float(rmse(yhat, y).cpu()))
        phys.append(float(physics_penalty_original_units(yhat, alpha, mu_t, sig_t).cpu()))

    y = np.concatenate(ys, axis=0)
    yhat = np.concatenate(yhats, axis=0)

    ss_res = np.sum((y - yhat) ** 2, axis=0)
    ss_tot = np.sum((y - y.mean(axis=0)) ** 2, axis=0) + 1e-9
    r2 = 1 - ss_res / ss_tot

    return {
        "rmse": float(np.mean(rmses)),
        "physics_penalty": float(np.mean(phys)),
        "r2_per_feature": dict(zip(FEATURES, r2.tolist())),
        "r2_macro": float(np.mean(r2)),
    }


def train_one_epoch(model, loader, opt, alpha, mu_t, sig_t, lam=0.2, device="cpu", clip=1.0):
    model.train()
    total, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        yhat = model(x)

        mse = torch.mean((yhat - y) ** 2)
        phy = physics_penalty_original_units(yhat, alpha, mu_t, sig_t)
        loss = (1 - lam) * mse + lam * phy

        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        total += loss.item() * x.size(0)
        n += x.size(0)

    return total / max(n, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    alpha,
    mu_t,
    sig_t,
    epochs=20,
    lam=0.2,
    device="cpu",
    clip=1.0,
):
    history = {"train_loss": [], "val_rmse": [], "val_r2": [], "val_phys": []}
    best_val = -1e9
    best_state = None

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            opt=optimizer,
            alpha=alpha,
            mu_t=mu_t,
            sig_t=sig_t,
            lam=lam,
            device=device,
            clip=clip,
        )
        val_m = eval_metrics(model, val_loader, alpha=alpha, mu_t=mu_t, sig_t=sig_t, device=device)

        history["train_loss"].append(tr_loss)
        history["val_rmse"].append(val_m["rmse"])
        history["val_r2"].append(val_m["r2_macro"])
        history["val_phys"].append(val_m["physics_penalty"])

        print(
            f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_r2_macro={val_m['r2_macro']:.4f} "
            f"| val_rmse={val_m['rmse']:.4f} | val_phys={val_m['physics_penalty']:.4f}"
        )

        if val_m["r2_macro"] > best_val:
            best_val = val_m["r2_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
