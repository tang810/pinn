import argparse
import os
import torch
from src.runner import run_pinn_training_mode, run_quick_inference
from src.data_utils import generate_virtual_data, load_user_data, validate_user_data


# === 占位函数: 未来可以补充 KdV 的数值解 ===
def run_numerical_solver(n_interior=5000, n_boundary=600, E=1.0, mu=0.25):
    print("⚠️ [占位] 数值求解器尚未实现，返回空结果")
    return {"status": "not_implemented"}


def export_numerical_results(result, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "numerical_result.txt"), "w") as f:
        f.write(str(result))
    print(f"📁 数值解结果已保存到 {output_dir}")


def export_training_results(result, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_result.txt"), "w") as f:
        f.write(str(result))
    print(f"📁 训练结果已保存到 {output_dir}")


def export_all_results(num_result, pinn_result, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "all_results.txt"), "w") as f:
        f.write("数值解结果:\n")
        f.write(str(num_result))
        f.write("\n\nPINN 结果:\n")
        f.write(str(pinn_result))
    #print(f"📁 综合结果已保存到 {output_dir}")


def main(mode="quick", epochs=1000, lr=1e-3, 
         data_mode="simul", data_path=None,
         output_dir="results", device="cpu"):
    # 🔹 数据处理
    if data_mode == "load":
        if not data_path:
            print(" data=load 时必须提供 --data_path")
            data_mode = "simul"
            return
        user_data = load_user_data(data_path)
        if validate_user_data(user_data):
            print(f"🔹 使用用户数据, 样本数: {len(user_data['X'])}")
        else:
            print(" 用户数据无效，切换到虚拟数据")
            user_data = None
            data_mode = "simul"
    else:
        user_data = None
        print("🔄 使用虚拟仿真数据...")

    # === 模式选择 ===
    if mode == "train":
        print("🔹 训练模式")
        training_result = run_pinn_training_mode(epochs=epochs, lr=lr, user_data=user_data)
        export_training_results(training_result, output_dir=output_dir)
    elif mode == "quick":
        print("🔹 快速推理模式")
        pinn_result = run_quick_inference(model_dir="model", n_test_points=100)

        if pinn_result is not None:   # ✅ 改这里
            num_result = {"status": "skipped"}  # quick模式下占位
            export_all_results(num_result, pinn_result, output_dir=output_dir)
        else:
            print(" Quick 模式推理失败，结果为空")
    elif mode == "numerical":
        print("🔹 数值求解模式")
        numerical_result = run_numerical_solver()
        export_numerical_results(numerical_result, output_dir=output_dir)

    else:
        raise ValueError(f"不支持的模式: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="quick", choices=["train", "quick", "numerical"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)

    # 🔹 新逻辑
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "load"],
                        help="数据源选择: simul(仿真数据) 或 load(用户数据)")
    parser.add_argument("--datapath", type=str, default=None, help="用户提供的数据文件路径")

    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # CUDA 检查
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，切换到 CPU")
        args.device = "cpu"

    # load 时检查文件路径
    if args.data == "load" and (not args.datapath or not os.path.exists(args.datapath)):
        print(f"❌ data=load 时必须指定有效 --datapath 文件路径")
        raise SystemExit(1)

    main(mode=args.mode, epochs=args.epochs, lr=args.lr,
         data_mode=args.data, data_path=args.datapath,
         output_dir=args.output_dir, device=args.device)
