import os
import matplotlib.pyplot as plt
import numpy as np

# 固定保存路径为当前目录下的 viz 文件夹
VIZ_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(VIZ_DIR, exist_ok=True)


def plot_results(data, pred=None, mode="train"):
    """
    画参考解和预测解，并保存到 ./viz 文件夹
    """
    x = data["x"][0]  # 取第一个样本的 x
    ref = data["ref"][0]

    plt.figure(figsize=(8, 5))
    plt.plot(x, ref, label="Reference", color="black", linewidth=2)

    if pred is not None:
        plt.plot(x, pred[0], "--", label="Prediction", color="red", linewidth=1.5)

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"KdV Solution ({mode})")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(VIZ_DIR, f"{mode}_result.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📊 结果图已保存: {save_path}")


def plot_loss_curve(loss_history):
    """
    画训练 Loss 曲线，保存到 ./viz 文件夹
    """
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss Curve")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(VIZ_DIR, "loss_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📉 Loss 曲线已保存: {save_path}")
