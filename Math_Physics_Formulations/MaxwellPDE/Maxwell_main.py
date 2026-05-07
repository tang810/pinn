# main.py
import os
import torch
import argparse
import numpy as np

from src import (
    get_device,
    generate_training_data,
    HighPrecisionPINN,
    Loss,
    plot_history,
)


def main(mode="normal", data_source="simul", show_plots=False):
    # =========================================================
    # 文件夹
    # =========================================================
    os.makedirs("model", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # =========================================================
    # 设备
    # =========================================================
    device = get_device()

    # =========================================================
    # 数据
    # =========================================================
    data_path = "/data/AI4PDE_CN/src/PINN4Science/Math_Physics_Formulations/MaxwellPDE/data/training_data.npy"
    model_path = "/data/AI4PDE_CN/src/PINN4Science/Math_Physics_Formulations/MaxwellPDE/model/high_precision_maxwell_model.pth"
    
    if data_source == "simul":
        # ⭐ 只有 normal + simul 才生成并保存数据
        X = generate_training_data(device=device)

        if mode == "normal":
            np.save(data_path, X.detach().cpu().numpy())
            print(f"[data] Training data saved to {data_path}")

    elif data_source == "load":
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"[load] 数据文件不存在: {data_path}，请先运行 normal + simul"
            )
        X = torch.tensor(
            np.load(data_path),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        print(f"[data] Loaded training data from {data_path}")

    else:
        raise ValueError("Invalid data_source (choose 'simul' or 'load')")

    N = X.shape[0]

    # =========================================================
    # 模型
    # =========================================================
    net = HighPrecisionPINN().to(device)


    # =========================================================
    # QUICK 模式：用【训练数据】画 loss（不训练）
    # =========================================================
    if mode == "quick":
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[quick] 模型文件不存在: {model_path}，请先运行 normal 模式"
            )

        net.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        net.eval()
        print(f"[quick] Loaded model from {model_path}")

        x_all = X.clone().detach().requires_grad_(True)

        loss, lpde, lbc = Loss.total(net, x_all, device)

        print(
            f"[quick] Loss | "
            f"Total={loss.item():.3e} | "
            f"PDE={lpde:.3e} | "
            f"BC={lbc:.3e}"
        )

        history = {
            "total": [loss.item()],
            "pde": [lpde],
            "bc": [lbc],
        }

        #plot_history(
           # history,
          #  save_path="results/quick_loss_on_train_data.png",
         #   show=False,
        #)

        print("[quick] Finished.")
        return

    # =========================================================
    # NORMAL 模式：训练
    # =========================================================
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    epochs = 3000
    batch_size = 64

    history = {"total": [], "pde": [], "bc": []}
    print(f"Training: epochs={epochs}, batch={batch_size}")

    for epoch in range(epochs):
        perm = torch.randperm(N)

        sum_loss = 0.0
        sum_pde = 0.0
        sum_bc = 0.0
        nb = 0

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            x_batch = X[idx].clone().detach().requires_grad_(True)

            optimizer.zero_grad()
            loss, lpde, lbc = Loss.total(net, x_batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            sum_loss += loss.item()
            sum_pde += lpde
            sum_bc += lbc
            nb += 1

        avg_loss = sum_loss / nb
        avg_pde = sum_pde / nb
        avg_bc = sum_bc / nb

        history["total"].append(avg_loss)
        history["pde"].append(avg_pde)
        history["bc"].append(avg_bc)

        scheduler.step(avg_loss)

        if epoch % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:5d} | "
                f"Loss {avg_loss:.3e} | "
                f"PDE {avg_pde:.3e} | "
                f"BC {avg_bc:.3e} | "
                f"LR {lr:.1e}"
            )
            if show_plots:
                plot_history(history, show=True)

    # =========================================================
    # 保存模型 & 训练曲线
    # =========================================================
    torch.save(net.state_dict(), model_path)
    print(f"[model] Saved model to {model_path}")

    plot_history(
        history,
        save_path="/data/AI4PDE_CN/src/PINN4Science/Math_Physics_Formulations/MaxwellPDE/results/training_history.png",
        show=False,
    )

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maxwell PINN / Quick Mode")
    parser.add_argument("--mode", choices=["normal", "quick"], default="normal")
    parser.add_argument("--data", choices=["simul", "load"], default="simul")
    parser.add_argument("--show_plots", action="store_true")
    args = parser.parse_args()

    main(
        mode=args.mode,
        data_source=args.data,
        show_plots=args.show_plots,
    )
