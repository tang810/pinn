import json
import os
import pickle
from typing import Dict, Tuple

import numpy as np

from .data_utils import ensure_dirs, ensure_simul_dataset, load_dataset


def _split(x: np.ndarray, y: np.ndarray, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    n_train = int(n * ratio)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def _fit_linear(x: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> Dict[str, np.ndarray]:
    n, d = x.shape
    out_dim = y.shape[1]
    w = np.zeros((d, out_dim), dtype=np.float32)
    b = np.zeros((1, out_dim), dtype=np.float32)
    losses = []

    for _ in range(epochs):
        pred = x @ w + b
        err = pred - y
        loss = float(np.mean(err * err))
        losses.append(loss)
        grad_w = (2.0 / n) * (x.T @ err)
        grad_b = (2.0 / n) * np.sum(err, axis=0, keepdims=True)
        w -= lr * grad_w.astype(np.float32)
        b -= lr * grad_b.astype(np.float32)

    return {"w": w, "b": b, "losses": np.array(losses, dtype=np.float32)}


def _predict(x: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    return x @ model["w"] + model["b"]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2) + 1e-12)
    return 1.0 - ss_res / ss_tot


def _save_model(model_path: str, model: Dict[str, np.ndarray]) -> None:
    with open(model_path, "wb") as f:
        pickle.dump({"w": model["w"], "b": model["b"]}, f)


def _load_model(model_path: str) -> Dict[str, np.ndarray]:
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return {"w": obj["w"], "b": obj["b"]}


def run_train(args) -> None:
    ensure_dirs(args.result_dir, args.model_dir, "data")
    if args.data == "simul":
        ensure_simul_dataset(args.data_path, seed=args.seed)
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data not found: {args.data_path}")

    dataset = load_dataset(args.data_path)
    x_train, y_train, x_val, y_val = _split(dataset["x"], dataset["y"])
    model = _fit_linear(x_train, y_train, epochs=args.epochs, lr=args.lr)
    y_val_pred = _predict(x_val, model)
    val_mse = float(np.mean((y_val_pred - y_val) ** 2))
    val_r2 = _r2(y_val, y_val_pred)

    model_path = os.path.join(args.model_dir, "dev_dit4science_surrogate.pt")
    _save_model(model_path, model)

    metrics = {
        "mode": "train",
        "data_path": args.data_path,
        "epochs": args.epochs,
        "lr": args.lr,
        "final_train_loss": float(model["losses"][-1]),
        "val_mse": val_mse,
        "val_r2": val_r2,
        "model_path": model_path,
    }

    np.save(os.path.join(args.result_dir, "train_loss.npy"), model["losses"])
    with open(os.path.join(args.result_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[train] model saved: {model_path}")
    print(f"[train] val_mse={val_mse:.6e}, val_r2={val_r2:.6f}")
    print(f"[train] metrics saved: {os.path.join(args.result_dir, 'train_metrics.json')}")


def run_quick(args) -> None:
    ensure_dirs(args.result_dir, args.model_dir, "data")
    if args.data == "simul":
        ensure_simul_dataset(args.data_path, seed=args.seed)
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data not found: {args.data_path}")

    dataset = load_dataset(args.data_path)
    _, _, x_test, y_test = _split(dataset["x"], dataset["y"])
    model_path = os.path.join(args.model_dir, "dev_dit4science_surrogate.pt")

    if not os.path.exists(model_path):
        warmup = _fit_linear(dataset["x"], dataset["y"], epochs=80, lr=args.lr)
        _save_model(model_path, warmup)
        print(f"[quick] warmup model created: {model_path}")

    model = _load_model(model_path)
    y_pred = _predict(x_test, model)
    mse = float(np.mean((y_pred - y_test) ** 2))
    r2 = _r2(y_test, y_pred)

    np.savez(os.path.join(args.result_dir, "quick_predictions.npz"), y_true=y_test, y_pred=y_pred)
    with open(os.path.join(args.result_dir, "quick_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "quick",
                "data_path": args.data_path,
                "mse": mse,
                "r2": r2,
                "model_path": model_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[quick] mse={mse:.6e}, r2={r2:.6f}")
    print(f"[quick] results saved to {args.result_dir}")
