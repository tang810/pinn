import os
import sys
from pathlib import Path

os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import gen_testdata
from .net import create_network
from .pinn_solver import get_bc_ic, pde


class suppress_stdout_stderr:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def _resolve_model_path(model_dir, model_name="trained_model.pth"):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / model_name


def _resolve_result_dir(result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def run_pinn_training_mode(
    epochs=1000,
    lr=1e-3,
    user_data=None,
    model_dir="model",
    result_dir="results",
):
    if user_data is not None and "X" in user_data and "Y" in user_data:
        observe_x, y = user_data["X"], user_data["Y"]
    else:
        observe_x, y = gen_testdata()

    geom = dde.geometry.Interval(-20, 20)
    timedomain = dde.geometry.TimeDomain(0, 4)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        get_bc_ic(geomtime, observe_x, y),
        num_domain=254,
        num_boundary=80,
        num_initial=160,
        num_test=254,
    )

    model = dde.Model(data, create_network())
    b = dde.Variable(0.1)
    rho1 = dde.Variable(1.0)
    rho2 = dde.Variable(1.0)
    with suppress_stdout_stderr():
        model.compile("adam", lr=lr, external_trainable_variables=[b, rho1, rho2])
        losshistory, _ = model.train(iterations=epochs, display_every=0)

    model_path = _resolve_model_path(model_dir)
    torch.save(model.net.state_dict(), str(model_path))

    result_dir = _resolve_result_dir(result_dir)
    train_losses = np.array(losshistory.loss_train)
    total_train_loss = train_losses.sum(axis=1)
    plt.figure(figsize=(7, 4))
    plt.plot(total_train_loss)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("KdV Inverse Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(result_dir / "loss_curve.png")
    plt.close()

    x_line = np.linspace(-20, 20, 200)
    t_line = np.full_like(x_line, 1.0)
    X_line = np.vstack((x_line, t_line)).T
    y_line = model.predict(X_line)

    plt.figure(figsize=(8, 5))
    plt.plot(x_line, y_line, label="PINN Prediction (t=1.0)")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Train Fit Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "train_fit_curve.png")
    plt.close()

    return {
        "status": "ok",
        "model_path": str(model_path),
        "loss_plot": str(result_dir / "loss_curve.png"),
    }


def run_quick_inference(
    model_dir="model",
    result_dir="results",
    user_data=None,
):
    model_path = _resolve_model_path(model_dir)
    if not model_path.exists():
        return None

    geom = dde.geometry.Interval(-20, 20)
    timedomain = dde.geometry.TimeDomain(0, 4)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    dummy_data = dde.data.TimePDE(geomtime, lambda x, y: 0 * x, [], num_domain=10, num_boundary=2, num_initial=2)

    model = dde.Model(dummy_data, create_network())
    with suppress_stdout_stderr():
        model.compile("adam", lr=1e-3, loss="MSE")
    checkpoint = torch.load(str(model_path), map_location="cpu")
    model.net.load_state_dict(checkpoint)
    model.net.eval()

    x = np.linspace(-20, 20, 200)
    t = np.linspace(0, 4, 100)
    X, T = np.meshgrid(x, t)
    X_test = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    y_pred = model.predict(X_test)

    x_line = np.linspace(-20, 20, 200)
    t_fixed = 1.0
    X_line = np.vstack((x_line, np.full_like(x_line, t_fixed))).T
    y_line = model.predict(X_line)

    result_dir = _resolve_result_dir(result_dir)
    plt.figure(figsize=(8, 5))
    plt.plot(x_line, y_line, "b-", label=f"PINN Prediction (t={t_fixed})")
    try:
        if user_data is not None and "X" in user_data and "Y" in user_data:
            X_obs, y_true = user_data["X"], user_data["Y"]
        else:
            X_obs, y_true = gen_testdata()
        mask_t = np.isclose(X_obs[:, 1], t_fixed, atol=0.05)
        ref_x = X_obs[mask_t, 0]
        ref_y = y_true[mask_t]
        sort_idx = np.argsort(ref_x)
        if len(sort_idx) > 0:
            plt.plot(ref_x[sort_idx], ref_y[sort_idx], "g--", label=f"Reference (t={t_fixed})")
    except Exception:
        pass
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Quick Inference Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "quick_inference_curve.png")
    plt.close()

    np.savetxt(result_dir / "pred_results.txt", y_pred[:1000], fmt="%.6e")
    return y_pred

