import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import FEATURES


def save_metrics_json(metrics: dict, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_predictions_npz(y, yhat, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, y=y, yhat=yhat)


def plot_results(
    history,
    y,
    yhat,
    alpha,
    mu,
    sig,
    results_dir,
    feature_names=FEATURES,
    max_points_scatter=3000,
    max_points_ts=300,
    show=False,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    def _savefig(name):
        plt.tight_layout()
        plt.savefig(results_dir / name, dpi=150)
        if show:
            plt.show()
        plt.close()

    # 1) train curves
    plt.figure()
    plt.plot(epochs, history["train_loss"])
    plt.title("Train loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    _savefig("train_loss.png")

    # plt.figure()
    # plt.plot(epochs, history["val_rmse"])
    # plt.title("Validation RMSE over epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("RMSE")
    # _savefig("val_rmse.png")

    # plt.figure()
    # plt.plot(epochs, history["val_r2"])
    # plt.title("Validation R2 (macro) over epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("R2")
    # plt.ylim(-1, 1)
    # _savefig("val_r2_macro.png")

    plt.figure()
    plt.plot(epochs, history["val_phys"])
    plt.title("Validation physics penalty over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Penalty")
    _savefig("val_physics_penalty.png")

    # 2) scatter for selected features
    def scatter_true_pred(feat):
        i = feature_names.index(feat)
        N = y.shape[0]
        m = min(max_points_scatter, N)
        idx = np.linspace(0, N - 1, m).astype(int)

        plt.figure()
        plt.scatter(y[idx, i], yhat[idx, i], s=8)
        plt.title(f"True vs Pred scatter: {feat} (normalized)")
        plt.xlabel("True")
        plt.ylabel("Pred")
        _savefig(f"scatter_{feat}.png")

    for feat in [ "Vx"]:
        scatter_true_pred(feat)

    # # 3) time series overlay for E
    # iE = feature_names.index("E")
    # n = min(max_points_ts, y.shape[0])
    # plt.figure()
    # plt.plot(y[:n, iE], label="True")
    # plt.plot(yhat[:n, iE], label="Pred")
    # plt.title("Time series overlay (E) on test segment (normalized)")
    # plt.xlabel("Sample index (test)")
    # plt.ylabel("E")
    # plt.legend()
    # _savefig("timeseries_E.png")

    # 4) physics check in original units
    yhat_raw = yhat * sig.values[None, :] + mu.values[None, :]
    E_abs = np.abs(yhat_raw[:, 0])
    V = yhat_raw[:, 1:4]
    B = yhat_raw[:, 4:7]
    rhs = alpha * np.linalg.norm(np.cross(V, B), axis=1)
    g = E_abs - rhs

    N = yhat.shape[0]
    m = min(max_points_scatter, N)
    idx = np.linspace(0, N - 1, m).astype(int)

    plt.figure()
    plt.scatter(rhs[idx], E_abs[idx], s=8)
    mm = max(rhs[idx].max(), E_abs[idx].max()) if len(idx) > 0 else 1.0
    plt.plot([0, mm], [0, mm])
    plt.title("Physics check (ORIGINAL UNITS): |E| vs alpha*||V x B||")
    plt.xlabel("alpha * ||V x B||")
    plt.ylabel("|E|")
    _savefig("physics_check_scatter_raw.png")

    plt.figure()
    plt.hist(g, bins=60)
    plt.title("Physics violation g distribution (ORIGINAL UNITS)")
    plt.xlabel("g = |E| - alpha||VxB||")
    plt.ylabel("Count")
    _savefig("physics_violation_hist_raw.png")

    # 5) R2 per feature
    ss_res = np.sum((y - yhat) ** 2, axis=0)
    ss_tot = np.sum((y - y.mean(axis=0)) ** 2, axis=0) + 1e-9
    r2 = 1 - ss_res / ss_tot

    plt.figure()
    plt.bar(np.arange(len(feature_names)), r2)
    plt.title("R2 per feature (test, normalized)")
    plt.xlabel("Feature")
    plt.ylabel("R2")
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45)
    plt.ylim(-1, 1)
    _savefig("r2_per_feature.png")

    summary = {
        "violation_rate_g_gt_0_raw": float(np.mean(g > 0)),
        "mean_relu_g_raw": float(np.mean(np.maximum(g, 0))),
        "r2_macro_normalized": float(np.mean(r2)),
    }
    save_metrics_json(summary, results_dir / "plot_summary.json")

    print("\n[Plot summary]")
    print("Violation rate (g>0) [original units]:", summary["violation_rate_g_gt_0_raw"])
    print("Mean ReLU(g) [original units]:", summary["mean_relu_g_raw"])
    print("R2 macro (normalized):", summary["r2_macro_normalized"])
