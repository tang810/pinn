import argparse
import os

from src.runner import (
    run_numerical_solver,
    run_pinn_training,
    run_quick_inference,
    export_all_results,
)
from src.numerical_solver import load_wave_equation_from_csv,save_wave_equation_to_csv
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

def main(
    mode="quick",
    data="simul",
    data_path=None,
    device="cpu",
    epochs=1000,
    N=30,
    T=1.0,
    output_dir="/data/AI4PDE_CN/src/PINN4Science/Math_Physics_Formulations/2D_波动方程/results"
):
    """
    主函数入口，支持多种运行模式：
    - mode: ["numerical", "train", "quick"]
    - data: ["simul", "load"]
    """
    #print(f"🚀 模式选择: mode = {mode} | data = {data}")

    # ========== 1. 加载数据 ==========
    if data == "load":
        if not data_path or not os.path.exists(data_path):
            print("❗ 未提供数据文件或文件不存在，将切换为模拟数据模式。")
            data = "simul"
        else:
            try:
                print(f"🔹 加载上传数据...")
                u_data, x, y, t = load_wave_equation_from_csv(data_path)
                result = {"u_sol": u_data, "x": x, "y": y, "t": t}

                # ✅ 打印数据加载信息
                print("✅ 上传数据加载成功：")
                print(f"  - u_sol 形状: {u_data.shape} （时间步 × x × y）")
                print(f"  - x 长度: {x.shape[0]}，范围: [{x[0]:.2f}, {x[-1]:.2f}]")
                print(f"  - y 长度: {y.shape[0]}，范围: [{y[0]:.2f}, {y[-1]:.2f}]")
                print(f"  - t 长度: {t.shape[0]}，范围: [{t[0]:.2f}, {t[-1]:.2f}]")
            except Exception as e:
                print(f"❗ 数据文件加载失败（{e}），将切换为模拟数据模式。")
                data = "simul"

    # fallback to simulate mode
    if data == "simul":
        print("🔹 使用模拟数据")
        result = run_numerical_solver(N=N, T=T)

        # ✅ 打印模拟数据结构信息
        print("✅ 模拟数据已成功构建：")
        print(f" * x 范围: [{result['x'][0]:.2f}, {result['x'][-1]:.2f}]，共 {len(result['x'])} 个点")
        print(f" * y 范围: [{result['y'][0]:.2f}, {result['y'][-1]:.2f}]，共 {len(result['y'])} 个点")
        print(f" * t 范围: [{result['t'][0]:.2f}, {result['t'][-1]:.2f}]，共 {len(result['t'])} 个时间步")
        print(f" * u_sol 形状: {result['u_sol'].shape} （时间步 × x × y）    \n")
        print("%Line Break%")
    
    # ========== 2. 模式执行 ==========
    if mode == "numerical":
        print("🔹 当前为数值求解模式，导出模拟解结果")
        export_all_results(result["u_sol"], result["u_sol"], result["x"], result["y"], result["t"], output_dir=output_dir)

        # 数值求解完成后额外保存为 CSV（方便用于 --data load 模式测试）
        save_wave_equation_to_csv(
            u_list=result["u_sol"],
            x=result["x"],
            y=result["y"],
            dt=result["t"],
            save_path=f"{output_dir}/wave_data.csv"
        )
        #print(f"📁 已将模拟结果导出为 CSV 文件: {output_dir}/wave_data.csv")


    elif mode == "train":
        print("🔹 开始模型训练")
        train_result = run_pinn_training(N=N, T=T, epochs=epochs, device=device)
        u_pred = run_quick_inference(train_result["x"], train_result["y"], train_result["t"], device=device)
        export_all_results(train_result["u_sol"], u_pred, train_result["x"], train_result["y"], train_result["t"], output_dir=output_dir)

    elif mode == "quick":
        print("\n🔹 快速推理中")
        u_pred = run_quick_inference(result["x"], result["y"], result["t"], device=device)
        export_all_results(result["u_sol"], u_pred, result["x"], result["y"], result["t"], output_dir=output_dir)

    else:
        raise ValueError(f"❌ 不支持的 mode 模式: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Wave Equation PINN Main Runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["numerical", "train", "quick"],
                        help="运行模式：numerical=数值模拟, train=训练, quick=推理")
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "load"],
                        help="数据模式：simul=模拟生成, load=加载CSV文件")
    parser.add_argument("--data_path", type=str, default=None,
                        help="数据文件路径（仅在 --data=load 时使用）")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备 (cpu / cuda)")
    parser.add_argument("--epochs", type=int, default=1000, help="训练轮数")
    parser.add_argument("--N", type=int, default=30, help="Chebyshev 网格数")
    parser.add_argument("--T", type=float, default=1.0, help="模拟时间长度")
    parser.add_argument("--output_dir", type=str, default="/data/AI4PDE_CN/src/PINN4Science/Math_Physics_Formulations/2D_波动方程/results", help="结果输出目录")

    args = parser.parse_args()

    main(
        mode=args.mode,
        data=args.data,
        data_path=args.data_path,
        device=args.device,
        epochs=args.epochs,
        N=args.N,
        T=args.T,
        output_dir=args.output_dir
    )
