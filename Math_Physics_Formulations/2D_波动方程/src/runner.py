import os
import torch
import numpy as np
from src.numerical_solver import WaveEquation
from src.pinn_model import MLP
from src.trainer import PINNTrainer
from src.postprocess import (
    plot_2d_snapshots,
    plot_3d_snapshots,
    create_3d_animation,
    save_numpy_result,
    save_interactive_plot,
)

def run_numerical_solver(N=30, T=1.0):
    solver = WaveEquation(N, T)
    u_sol = solver.solve()

    x = np.arange(-1, 1 + 1/16, 1/16)
    y = x.copy()
    dt = 6 / N**2
    t = np.arange(0, T + dt, dt)

    #print("数值求解完成。")
    return {"x": x, "y": y, "t": t, "u_sol": u_sol}

def run_pinn_training(N=30, T=1.0, epochs=1000, device="cpu"):
    num_result = run_numerical_solver(N, T)
    x, y, t, u_sol = num_result["x"], num_result["y"], num_result["t"], num_result["u_sol"]

    model = MLP(in_features=3, out_features=1, num_layers=5, num_neurons=100)
    domain_bounds = (np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, T]))
    trainer = PINNTrainer(model, domain_bounds, device=device)

    def u_initial(x_pts, y_pts):
        return np.exp(-40 * ((x_pts - 0.4) ** 2 + y_pts ** 2))

    trainer.prepare_data(x_range=(-1, 1), y_range=(-1, 1), t_range=(0, T), u_initial_fn=u_initial)
    loss_record = trainer.train(epochs=epochs)
    trainer.plot_loss(loss_record)

    # 保存模型
    os.makedirs("/data/AI4PDE_CN/src/PINN4Science/Math_Physics_Formulations/2D_波动方程/model", exist_ok=True)
    torch.save(model.state_dict(), "/data/AI4PDE_CN/src/PINN4Science/Math_Physics_Formulations/2D_波动方程/model/wave_pinn_model.pt")
    print("训练完成，模型已保存至 /data/AI4PDE_CN/src/PINN4Science/Math_Physics_Formulations/2D_波动方程/model/wave_pinn_model.pt")

    return {"model": model, "x": x, "y": y, "t": t, "u_sol": u_sol, "loss_record": loss_record}

def run_quick_inference(x, y, t, model_path="/data/AI4PDE_CN/src/PINN4Science/Math_Physics_Formulations/2D_波动方程/model/wave_pinn_model.pt", device="cpu"):
    model = MLP(in_features=3, out_features=1, num_layers=5, num_neurons=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    X, Y = np.meshgrid(x, y)
    XY = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    T_all = np.expand_dims(np.repeat(t, len(XY)), axis=1)
    XY_tiled = np.tile(XY, (len(t), 1))
    X_input = np.hstack((XY_tiled, T_all))

    with torch.no_grad():
        input_tensor = torch.tensor(X_input, dtype=torch.float32, device=device)
        output = model(input_tensor).cpu().numpy()

    u_pred = output.reshape(len(t), len(x), len(y))
    print("推理完成，结果已获取。")
    return u_pred

def export_all_results(u_sol, u_pred, x, y, t, output_dir="/data/AI4PDE_CN/src/PINN4Science/Math_Physics_Formulations/2D_波动方程/results"):
    os.makedirs(output_dir, exist_ok=True)

    save_numpy_result(u_sol, f"{output_dir}/u_sol.npy")
    save_numpy_result(u_pred, f"{output_dir}/u_pred.npy")

    plot_2d_snapshots(u_sol, t, title_prefix="Numerical Solution", save_path=f"{output_dir}/numerical_snapshots.png")
    plot_2d_snapshots(u_pred, t, title_prefix="PINN Prediction", save_path=f"{output_dir}/pinn_snapshots.png")

    plot_3d_snapshots(u_pred, x, y, t, save_path=f"{output_dir}/pinn_3d_snapshots.png")
    create_3d_animation(u_pred, x, y, t, save_path=f"{output_dir}/pinn_animation.gif")
    #save_interactive_plot(u_pred, x, y, t, save_html=f"{output_dir}/pinn_interactive.html")

    #print(f"结果已导出至 {output_dir}")
