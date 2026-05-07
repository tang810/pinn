import argparse
import os
from src.runner import run_pinn_training, run_quick_inference, run_numerical_solver
from src.visualize_results import visualize_results

def get_model_path(model_dir="src/PINN4Science/Fluid_Mechanics/rans_backstep/model"):
    """返回模型路径，优先加载用户模型"""
    user_model_path = os.path.join(model_dir, "user_data_trained_model.pth")
    default_model_path = os.path.join(model_dir, "winged_block_backstep_model.pt")

    if os.path.exists(user_model_path):
        return os.path.abspath(user_model_path)
    elif os.path.exists(default_model_path):
        return os.path.abspath(default_model_path)
    else:
        return None


def main(mode="quick", data="simul", datapath=None, device="cpu", model_path=None):
    model_dir = "src/PINN4Science/Fluid_Mechanics/rans_backstep/model"

    # === 数据源逻辑 ===
    if data == "simul":
        print("🔄 使用虚拟仿真数据")
        user_data = None
        data_source = "simul"

    elif data == "load":
        if datapath is None or not os.path.exists(datapath):
            print("⚠️ 未找到用户数据文件，自动回退到仿真数据模式")
            user_data = None
            data_source = "simul"
        else:
            print(f"📂 加载用户数据: {datapath}")
            user_data = datapath
            data_source = "load"

    else:
        raise ValueError(f" 不支持的数据模式: {data}")

    # === 模式选择 ===
    if mode == "train":
        print("🔹 训练模式")
        run_pinn_training(device=device, model_dir=model_dir,
                          data_source=data_source, datapath=user_data)

    elif mode == "quick":
        print("🔹 快速推理模式")
        model_path = get_model_path(model_dir)
        if model_path is None:
            print(f" 在 {model_dir} 中找不到任何模型，请先运行 --mode train")
            return
        # run_quick_inference(device=device, model_path=model_path,
        #                     data_source=data_source, datapath=user_data)
        visualize_results(ref_file='src/PINN4Science/Fluid_Mechanics/rans_backstep/data/Re1e5_backStep_L8.mat',
                          result_file='src/PINN4Science/Fluid_Mechanics/rans_backstep/data/f1_result_epoch200000.mat',
                          folder_path='src/PINN4Science/Fluid_Mechanics/rans_backstep/results')

    
    elif mode == "numerical":
        print("🔹 数值求解模式")
        run_numerical_solver()

    else:
        raise ValueError(f"不支持的模式: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="quick", choices=["train", "quick", "numerical"])
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "load"],
                        help="数据模式：simul=仿真数据, load=用户数据")
    parser.add_argument("--datapath", type=str, default=None,
                        help="当 --data=load 时，必须指定用户数据文件路径")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_path", type=str, default="src/PINN4Science/Fluid_Mechanics/rans_backstep/model/model_loop90000.pth",
                        help="手动指定模型路径（默认自动查找）")
    args = parser.parse_args()

    main(mode=args.mode, data=args.data, datapath=args.datapath, device=args.device, model_path=args.model_path)
