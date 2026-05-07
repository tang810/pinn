import os
import torch
import matplotlib.pyplot as plt
import src.rans_data as rdata
from src.pinn_solver_2D import PysicsInformedNeuralNetwork

def plot_comparison_all(x_star, y_star, u_true, v_true, u_pred, v_pred, mode="Train"):
    os.makedirs("src/PINN4Science/Fluid_Mechanics/rans_backstep/results/", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    # Ground Truth u
    sc1 = axes[0, 0].scatter(x_star, y_star, c=u_true, cmap='jet', s=1)
    axes[0, 0].set_title("Ground Truth: u-velocity")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")
    plt.colorbar(sc1, ax=axes[0, 0])

    # Predicted u
    sc2 = axes[0, 1].scatter(x_star, y_star, c=u_pred, cmap='jet', s=1)
    axes[0, 1].set_title(f"{mode} Mode: u-velocity")
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")
    plt.colorbar(sc2, ax=axes[0, 1])

    # Ground Truth v
    sc3 = axes[1, 0].scatter(x_star, y_star, c=v_true, cmap='jet', s=1)
    axes[1, 0].set_title("Ground Truth: v-velocity")
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    plt.colorbar(sc3, ax=axes[1, 0])

    # Predicted v
    sc4 = axes[1, 1].scatter(x_star, y_star, c=v_pred, cmap='jet', s=1)
    axes[1, 1].set_title(f"{mode} Mode: v-velocity")
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    plt.colorbar(sc4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f"src/PINN4Science/Fluid_Mechanics/rans_backstep/results/uv_velocity_{mode.lower()}_comparison——test.png", dpi=300)
    plt.close()


def run_pinn_training(device="cpu", model_dir="src/PINN4Science/Fluid_Mechanics/rans_backstep/model/", data_source=None, datapath=None,
                      n_interior=40000, n_boundary=1000, epochs=10000,
                      nu_0=1e-5, U_0=1.0, rho_0=1.0, L_0=1.0,
                      alpha_evm=0.03, beta_evm=10.0,
                      layers=4, layers_1=4, hidden_size=120, hidden_size_1=40, lr=1e-3,
                      save_model=True, user_data=None):

    os.makedirs(model_dir, exist_ok=True)

    # 创建 PINN
    PINN = PysicsInformedNeuralNetwork(
        nu_0=nu_0,
        U_0=U_0,
        rho_0=rho_0,
        L_0=L_0,
        layers=layers,
        layers_1=layers_1,
        hidden_size=hidden_size,
        hidden_size_1=hidden_size_1,
        N_f=n_interior,
        alpha_evm=alpha_evm,
        beta_evm=beta_evm,
        checkpoint_path=model_dir,
        device=device
    )

    # 数据加载
    dataloader = rdata.DataLoader(path="src/PINN4Science/Fluid_Mechanics/rans_backstep/datasets/", N_f=n_interior, N_b=n_boundary)

    if user_data is None:
        # 默认数据路径
        uvk_wall_data, uvk_io_data, p_b_data = dataloader.loading_boundary_data(
            bc_UV="src/PINN4Science/Fluid_Mechanics/rans_backstep/data/Re1e5_backStep_BC_UV_L8.mat",
            bc_P="src/PINN4Science/Fluid_Mechanics/rans_backstep/data/Re1e5_backStep_BC_P_L8.mat",
            nu_0=nu_0
        )
        x_star, y_star, u_star, v_star, p_star, k_star, o_star, n_star = dataloader.loading_evaluate_data(
            "src/PINN4Science/Fluid_Mechanics/rans_backstep/data/Re1e5_backStep_L8.mat"
        )
    else:
        # 使用用户数据
        uvk_wall_data, uvk_io_data, p_b_data = dataloader.loading_boundary_data(user_data, nu_0=nu_0)
        x_star, y_star, u_star, v_star, p_star, k_star, o_star, n_star = dataloader.loading_evaluate_data(user_data)

    PINN.set_io_boundary_data(uvk_io_data)
    PINN.set_pres_boundary_data(p_b_data)
    PINN.set_wall_boundary_data(uvk_wall_data)

    # 训练数据
    training_data = dataloader.loading_training_data()
    PINN.set_eq_training_data(training_data)

    # 设置测试数据
    PINN.set_testing_data(x_star, y_star, u_star, v_star, p_star, k_star, o_star)

    # 训练
    PINN.train(start_epoch=0, num_epoch=epochs, lr=lr)

    # 保存模型
    if save_model:
        model_file = os.path.join(model_dir, "winged_block_backstep_model.pt")
        torch.save(PINN.net.state_dict(), model_file)
        print(f"✅ 模型已保存到 {model_file}")

    # 训练后预测
    with torch.no_grad():
        x_tensor = torch.tensor(x_star, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_star, dtype=torch.float32).to(device)
        uv_pred = PINN.net(torch.cat([x_tensor, y_tensor], dim=1))
        u_pred = uv_pred[:, 0].cpu().numpy()
        v_pred = uv_pred[:, 1].cpu().numpy()

    pinn_result = {"u": u_pred, "v": v_pred, "x": x_star, "y": y_star}

    plot_comparison_all(x_star, y_star, u_star, v_star,
                        pinn_result['u'], pinn_result['v'],
                        mode="Train")

    return pinn_result


def run_quick_inference(device="cpu", model_path=None, data_source=None, datapath=None):
    """
    快速推理模式，直接使用训练好的模型生成预测结果并绘图
    """
    print("🧠 初始化 PINN 模型...")

    PINN = PysicsInformedNeuralNetwork(
        nu_0=1e-5,
        U_0=1.0,
        rho_0=1.0,
        L_0=1.0,
        layers=4,
        layers_1=4,
        hidden_size=120,
        hidden_size_1=40,
        N_f=40000,
        alpha_evm=0.05,
        beta_evm=10.0,
        checkpoint_path='src/PINN4Science/Fluid_Mechanics/rans_backstep/model/',
        device=device
    )

    # 加载模型
    if model_path is None:
        model_path = 'src/PINN4Science/Fluid_Mechanics/rans_backstep/model/model_loop90000.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"训练好的模型文件不存在: {model_path}")
    PINN.net.load_state_dict(torch.load(model_path, map_location=device))
    # print(f"✅ 模型权重从 {model_path} 加载完成")

    # 测试数据
    dataloader = rdata.DataLoader(path='src/PINN4Science/Fluid_Mechanics/rans_backstep/datasets/', N_f=40000, N_b=1000)
    filename = datapath if datapath is not None else 'src/PINN4Science/Fluid_Mechanics/rans_backstep/data/Re1e5_backStep_L8.mat'
    x_star, y_star, u_star, v_star, p_star, k_star, o_star, n_star = dataloader.loading_evaluate_data(filename)

    PINN.set_testing_data(x_star, y_star, u_star, v_star, p_star, k_star, o_star)

    # 预测
    print("📊 生成预测结果...")
    with torch.no_grad():
        x_tensor = torch.tensor(x_star, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_star, dtype=torch.float32).to(device)
        uv_pred = PINN.net(torch.cat([x_tensor, y_tensor], dim=1))
        u_pred = uv_pred[:, 0].cpu().numpy()
        v_pred = uv_pred[:, 1].cpu().numpy()

    pinn_result = {"u": u_pred, "v": v_pred, "x": x_star, "y": y_star}

    plot_comparison_all(x_star, y_star, u_star, v_star,
                        pinn_result['u'], pinn_result['v'],
                        mode="Quick")
    print("✅ 快速推理完成")
    return pinn_result




def run_numerical_solver(n_interior=40000, n_boundary=1000, nu_0=1e-5, U_0=1.0):
    # TODO: 这里可以实现或调用已有的数值解求解器
    return {"u": None, "v": None, "x": None, "y": None}
