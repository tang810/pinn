from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .constants import FEATURES
from .data_utils import make_synthetic_hourly, normalize_z, split_by_time, fit_alpha_raw
from .datasets import WindowDataset
from .model import GRUForecast
from .trainer import train_model, eval_metrics, predict_all
from .postprocess import plot_results, save_metrics_json, save_predictions_npz


def ensure_case_dirs(case_root):
    case_root = Path(case_root)
    data_dir = case_root / "data"
    model_dir = case_root / "model"
    results_dir = case_root / "results"
    src_dir = case_root / "src"

    for d in [data_dir, model_dir, results_dir, src_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "case_root": case_root,
        "data_dir": data_dir,
        "model_dir": model_dir,
        "results_dir": results_dir,
        "src_dir": src_dir,
    }


def build_dataloaders(train_arr, val_arr, test_arr, span, prior, batch_size):
    train_ds = WindowDataset(train_arr, span=span, prior=prior)
    val_ds = WindowDataset(val_arr, span=span, prior=prior)
    test_ds = WindowDataset(test_arr, span=span, prior=prior)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_ld, val_ld, test_ld


def prepare_data(
    n_hours=24 * 240,
    seed=0,
    span=24,
    prior=12,
    data_dir=None,
    save_csv=True,
):
    """
    生成合成数据 + 切分 + 标准化 + DataLoader 所需数组。
    """
    df_raw = make_synthetic_hourly(n_hours=n_hours, seed=seed, alpha_true=0.02)

    if data_dir is not None and save_csv:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(Path(data_dir) / "synthetic_hourly_raw.csv", index=True)

    train_raw, val_raw, test_raw = split_by_time(df_raw, train_ratio=0.60, val_ratio=0.30)
    alpha = fit_alpha_raw(train_raw)

    train_n, mu, sig = normalize_z(train_raw)
    val_n = (val_raw - mu) / sig
    test_n = (test_raw - mu) / sig

    train_arr = train_n[FEATURES].to_numpy(dtype=np.float32)
    val_arr = val_n[FEATURES].to_numpy(dtype=np.float32)
    test_arr = test_n[FEATURES].to_numpy(dtype=np.float32)

    return {
        "df_raw": df_raw,
        "train_raw": train_raw,
        "val_raw": val_raw,
        "test_raw": test_raw,
        "mu": mu,
        "sig": sig,
        "alpha": alpha,
        "train_arr": train_arr,
        "val_arr": val_arr,
        "test_arr": test_arr,
        "span": span,
        "prior": prior,
    }


def build_model_and_optimizer(input_dim, hidden, num_layers, dropout, lr, weight_decay, device):
    model = GRUForecast(
        input_dim=input_dim,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=input_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model


def run_train_and_test(config: dict, case_root="."):
    dirs = ensure_case_dirs(case_root)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    data_pack = prepare_data(
        n_hours=config["n_hours"],
        seed=config["seed"],
        span=config["span"],
        prior=config["prior"],
        data_dir=dirs["data_dir"],
        save_csv=True,
    )

    print("alpha (fit on raw) =", data_pack["alpha"])

    train_ld, val_ld, test_ld = build_dataloaders(
        train_arr=data_pack["train_arr"],
        val_arr=data_pack["val_arr"],
        test_arr=data_pack["test_arr"],
        span=config["span"],
        prior=config["prior"],
        batch_size=config["batch_size"],
    )

    model, optimizer = build_model_and_optimizer(
        input_dim=len(FEATURES),
        hidden=config["hidden"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        device=device,
    )

    mu_t = torch.tensor(data_pack["mu"][FEATURES].to_numpy(dtype=np.float32), device=device)
    sig_t = torch.tensor(data_pack["sig"][FEATURES].to_numpy(dtype=np.float32), device=device)

    model, history = train_model(
        model=model,
        train_loader=train_ld,
        val_loader=val_ld,
        optimizer=optimizer,
        alpha=data_pack["alpha"],
        mu_t=mu_t,
        sig_t=sig_t,
        epochs=config["epochs"],
        lam=config["lam"],
        device=device,
        clip=config["clip"],
    )

    ckpt_path = dirs["model_dir"] / "solar_wind_prediction_best.pt"
    save_checkpoint(model, ckpt_path)

    test_metrics = eval_metrics(
        model, test_ld,
        alpha=data_pack["alpha"], mu_t=mu_t, sig_t=sig_t,
        device=device
    )

    print("\n=== TEST ===")
    print("R2 macro:", test_metrics["r2_macro"])
    print("R2 per feature:", test_metrics["r2_per_feature"])
    print("RMSE:", test_metrics["rmse"])
    print("Physics penalty:", test_metrics["physics_penalty"])

    y_test, yhat_test = predict_all(model, test_ld, device=device)

    save_predictions_npz(y_test, yhat_test, dirs["results_dir"] / "test_predictions.npz")
    save_metrics_json(test_metrics, dirs["results_dir"] / "test_metrics.json")
    save_metrics_json(history, dirs["results_dir"] / "train_history.json")

    plot_results(
        history=history,
        y=y_test,
        yhat=yhat_test,
        alpha=data_pack["alpha"],
        mu=data_pack["mu"],
        sig=data_pack["sig"],
        results_dir=dirs["results_dir"],
        show=config["show_plots"],
    )

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "paths": dirs,
        "checkpoint": str(ckpt_path),
    }


def run_eval_only(config: dict, case_root="."):
    """
    评估模式：重新生成相同合成数据（依赖相同 seed/config），加载模型做测试与出图。
    """
    dirs = ensure_case_dirs(case_root)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    data_pack = prepare_data(
        n_hours=config["n_hours"],
        seed=config["seed"],
        span=config["span"],
        prior=config["prior"],
        data_dir=dirs["data_dir"],
        save_csv=False,
    )

    _, _, test_ld = build_dataloaders(
        train_arr=data_pack["train_arr"],
        val_arr=data_pack["val_arr"],
        test_arr=data_pack["test_arr"],
        span=config["span"],
        prior=config["prior"],
        batch_size=config["batch_size"],
    )

    model, _ = build_model_and_optimizer(
        input_dim=len(FEATURES),
        hidden=config["hidden"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        device=device,
    )

    ckpt_path = Path(dirs["model_dir"]) / "solar_wind_prediction_best.pt"

    print("实际查找路径:", ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型权重文件：{ckpt_path}")

    model = load_checkpoint(model, ckpt_path, map_location=device)
    model.to(device)

    mu_t = torch.tensor(data_pack["mu"][FEATURES].to_numpy(dtype=np.float32), device=device)
    sig_t = torch.tensor(data_pack["sig"][FEATURES].to_numpy(dtype=np.float32), device=device)

    test_metrics = eval_metrics(
        model, test_ld,
        alpha=data_pack["alpha"], mu_t=mu_t, sig_t=sig_t,
        device=device
    )

    print("\n=== TEST (EVAL-ONLY) ===")
    print("R2 macro:", test_metrics["r2_macro"])
    print("R2 per feature:", test_metrics["r2_per_feature"])
    print("RMSE:", test_metrics["rmse"])
    print("Physics penalty:", test_metrics["physics_penalty"])

    y_test, yhat_test = predict_all(model, test_ld, device=device)
    save_predictions_npz(y_test, yhat_test, dirs["results_dir"] / "test_predictions_eval_only.npz")
    save_metrics_json(test_metrics, dirs["results_dir"] / "test_metrics_eval_only.json")

    return {
        "model": model,
        "test_metrics": test_metrics,
        "paths": dirs,
        "checkpoint": str(ckpt_path),
    }
