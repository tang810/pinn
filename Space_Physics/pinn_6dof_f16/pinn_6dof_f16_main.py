import argparse
import os
from pathlib import Path

import numpy as np
import torch

from src.config import get_device
from src.data_utils import generate_flight, save_dataset_npz, load_dataset_npz, build_nn_inputs
from src.train import train_pinn
from src.eval import evaluate_coefficients
from src.plotting import plot_loss, plot_coeff_compare
from src.model import AeroNet
from src.io_utils import load_model, save_model

def parse_args():
    p = argparse.ArgumentParser(description="PINN 6DOF F-16-like aircraft: multi-file case")
    # NOTE: 为了便于在 PyCharm 里直接点运行（不填参数也能跑），
    # 这里不强制要求传入模式参数；若用户未指定，则默认 quick + simul。
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--train", action="store_true", help="训练模式")
    mode.add_argument("--quick", action="store_true", help="快速模式：少量 epoch 或仅评估")

    data_src = p.add_mutually_exclusive_group(required=False)
    data_src.add_argument("--simul", action="store_true", help="仿真生成数据")
    data_src.add_argument("--load", action="store_true", help="从文件加载数据（npz）")

    p.add_argument("--data_path", type=str, default="data/flight_dataset.npz", help="数据文件路径（load/simul都会用到保存/读取）")
    p.add_argument("--ckpt", type=str, default="model/best_model.pt", help="模型权重路径")
    p.add_argument("--epochs", type=int, default=5000, help="训练 epoch 数（train 模式）")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--noise", type=float, default=0.02, help="仿真噪声强度（simul 模式）")
    p.add_argument("--t_end", type=float, default=20.0, help="仿真时长（秒）")
    p.add_argument("--dt", type=float, default=0.02, help="仿真积分步长（秒）")
    p.add_argument("--seed", type=int, default=0, help="随机种子（影响噪声）")
    p.add_argument("--no_resume", action="store_true", help="不从 ckpt 恢复优化器状态")
    p.add_argument("--verbose_every", type=int, default=500, help="打印/保存间隔（epoch）")
    p.add_argument("--data", type=str, default="simul", help="仿真数据")
    p.add_argument("--mode", type=str, default="quick", help="快速开始")
    args = p.parse_args()

    # 默认行为：不带参数时，按“快速 + 仿真生成数据”跑通流程
    if (not args.train) and (not args.quick):
        args.quick = True
    if (not args.simul) and (not args.load):
        args.simul = True

    return args

def ensure_dirs():
    for d in ["data", "model", "results"]:
        Path(d).mkdir(parents=True, exist_ok=True)

def to_torch_tensors(X_in: np.ndarray, ds, extras, device: torch.device):
    X_t = torch.tensor(X_in, dtype=torch.float32, device=device)
    states = ds.states_meas
    acc = ds.acc_meas

    tensors = {
        "u": torch.tensor(states[:,0], dtype=torch.float32, device=device),
        "v": torch.tensor(states[:,1], dtype=torch.float32, device=device),
        "w": torch.tensor(states[:,2], dtype=torch.float32, device=device),
        "p": torch.tensor(states[:,3], dtype=torch.float32, device=device),
        "q": torch.tensor(states[:,4], dtype=torch.float32, device=device),
        "r": torch.tensor(states[:,5], dtype=torch.float32, device=device),
        "phi": torch.tensor(states[:,6], dtype=torch.float32, device=device),
        "theta": torch.tensor(states[:,7], dtype=torch.float32, device=device),
        "u_dot": torch.tensor(acc[:,0], dtype=torch.float32, device=device),
        "v_dot": torch.tensor(acc[:,1], dtype=torch.float32, device=device),
        "w_dot": torch.tensor(acc[:,2], dtype=torch.float32, device=device),
        "p_dot": torch.tensor(acc[:,3], dtype=torch.float32, device=device),
        "q_dot": torch.tensor(acc[:,4], dtype=torch.float32, device=device),
        "r_dot": torch.tensor(acc[:,5], dtype=torch.float32, device=device),
    }
    return X_t, tensors

def main():
    args = parse_args()
    ensure_dirs()
    device = get_device()
    print("Device:", device)

    # 1) 数据准备：simul 生成 或 load 读取
    if args.simul:
        ds = generate_flight(t_end=args.t_end, dt=args.dt, noise_level=args.noise, seed=args.seed)
        save_dataset_npz(ds, args.data_path)
        print(f"Saved dataset to: {args.data_path}")
    else:
        ds = load_dataset_npz(args.data_path)
        print(f"Loaded dataset from: {args.data_path}")

    X_in, extras = build_nn_inputs(ds)
    X_t, tensors = to_torch_tensors(X_in, ds, extras, device=device)

    # 2) 训练 or 快速
    ckpt_path = args.ckpt

    if args.train:
        out = train_pinn(
            X_t, tensors, device=device,
            epochs=args.epochs,
            lr=args.lr,
            ckpt_path=ckpt_path,
            resume=(not args.no_resume),
            verbose_every=args.verbose_every
        )
        # 训练输出保存
        np.save("results/loss_hist.npy", out.loss_hist)
        plot_loss(out.loss_hist, "results/loss_curve.png")
        save_model(out.model, ckpt_path)
        net = out.model
    else:
        # quick: 如果 ckpt 存在则加载，不存在则只跑一个很短训练
        net = AeroNet().to(device)
        if os.path.exists(ckpt_path):
            net = load_model(net, ckpt_path, device=device)
            print(f"Loaded model from: {ckpt_path}")
        else:
            # 为了保持“默认行为 = 原始单文件脚本参数”，这里使用 args.epochs / args.verbose_every
            # 原始代码默认 n_epochs=5000，lr=1e-3。
            print(f"No checkpoint found; training with default params (epochs={args.epochs}) ...")
            out = train_pinn(
                X_t, tensors,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                ckpt_path=ckpt_path,
                resume=False,
                verbose_every=args.verbose_every
            )
            net = out.model
            np.save("results/loss_hist.npy", out.loss_hist)
            plot_loss(out.loss_hist, "results/loss_curve.png")
            save_model(net, ckpt_path)

    # 3) 评估与结果保存
    eval_out = evaluate_coefficients(net, X_t, extras)
    np.save("results/coeff_pred.npy", eval_out.coeff_pred)
    np.save("results/coeff_true.npy", eval_out.coeff_true)
    plot_coeff_compare(ds.t, eval_out.coeff_true, eval_out.coeff_pred, "results/coeff_compare.png")

    print("\nAero coefficient RMSEs:")
    for k, v in eval_out.rmse.items():
        print(f"{k} RMSE = {v:.4e}")

    print("\nSaved outputs to results/ and model/")

if __name__ == "__main__":
    main()
