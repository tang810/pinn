# ================================
# File: main.py
# ================================
import argparse
import os
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import torch
from src.models import PINNInverseMLP
from src.data_utils import PINNDataset
from src.data_utils import (
    load_data,
    build_eval_loader_from_prepared
)
from src.infer import load_model_bundle
from src.viz import output_viz_data
from src.trainer import run_training

# -----------------------------
# 解析参数（默认：mode=quick, data=simul）
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="微悬臂梁几何结构参数反演 - 主控脚本（标准案例模式）")
    parser.add_argument("--mode", type=str, default="quick", choices=["train", "quick"],
                        help="运行模式：train=训练, quick=推理可视化")
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "load"],
                        help="数据模式：simul=使用预训练模型内置的 test_loader；load=读取用户上传的 HDF5")
    parser.add_argument("--data_path", type=str, default="./wt_lt/data/wt_lt_big.h5",
                        help="数据文件路径（--data=load 时使用）")
    parser.add_argument("--scaler_path", type=str, default="./wt_lt/data/norm_scalers_big.pkl",
                        help="scaler 保存/读取路径（由 load_data 写入）")
    parser.add_argument("--model_path", type=str, default="./model/wtlt_big_pinn.pt",
                        help="预训练模型 bundle 文件路径（包含model、train_loader、test_loader、loss_history、scalers、constants）")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="结果输出目录（前端读取）")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备 (cpu / cuda)")
    parser.add_argument("--epochs", type=int, default=1000, help="训练轮数（仅在 --mode=train 时使用）")
    parser.add_argument("--batch", type=int, default=32, help="batch size（仅在 --mode=train / load 推理构建loader时使用）")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=4)
    return parser.parse_args()


# -----------------------------
# 主函数（遵循你的运行约定）
# -----------------------------
def main(
    mode="quick",
    data="simul",
    data_path="./data/wt_lt_big.h5",
    scaler_path="./data/norm_scalers_big.pkl",
    model_path="./model/wtlt_big_pinn.pt",
    output_dir="./results",
    device="cpu",
    epochs=1000,
    batch=32,
    hidden_dim=128,
    hidden_layers=4,
):
    """
    - 默认：mode='quick', data='simul'
    - data='simul'：直接使用预训练模型 bundle 内置的 test_loader 做快速推理与可视化；
    - data='load' ：读取用户上传的 HDF5，构造 eval_loader；根据 mode 执行 quick 推理或 train 训练；
    """
    os.makedirs(output_dir, exist_ok=True)

    # ========== A. simul 模式：仅使用 bundle 内置的 test_loader ==========
    if data == "simul":
        print("🔹 使用模拟数据验证")
        model, train_loader_b, test_loader_b, loss_history_b, scalers_b, constants_b = load_model_bundle(model_path)
        if test_loader_b is None:
            raise RuntimeError("❌ 预训练模型中未包含 test_loader，无法在 simul 模式下执行 quick 推理。请改用 --data load。")

        if mode == "quick":
            print("🔹 快速推理 (mode=quick) with bundle test_loader")
            # 使用 bundle 内置 scalers 做反归一化与可视化
            output_viz_data(
                model=model,
                test_loader=test_loader_b,
                loss_history=loss_history_b if isinstance(loss_history_b, dict) else {},
                scalers=scalers_b,
                folder_path=output_dir
            )
            print(f"📊 可视化结果已生成")
            return
        elif mode == "train":
            # simul + train 没有上传数据，不执行训练
            raise NotImplementedError("❌ simul 模式下不支持训练（缺少原始 HDF5）。请使用 --data load 并提供 HDF5 后再 --mode train。")
        else:
            raise ValueError(f"❌ 不支持的 mode: {mode}")

    # ========== B. load 模式：读取用户HDF5，构造 eval_loader ==========
    if data == "load":
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ --data=load 但未找到数据文件: {data_path}")

        print(f"🔹 加载用户数据: {data_path}")
        (omega_list, y_list, wt_list, lt_list, freq_list, mc_list,
         M_list, dkt_list, dk3t_list, Fac_list, C_list,
         phi_full, constants, phi,
         X_feat, Y_target, Y_phys, omega_norm_all, y_norm_all,
         scalers, per_sample_scalers) = load_data(file_path=data_path, scaler_path=scaler_path)

        # 构造 eval loader（与 Notebook 结构一致）
        eval_loader = build_eval_loader_from_prepared(
            X_feat=X_feat,
            Y_target=Y_target,
            Y_phys=Y_phys,
            phi_full=phi,
            omega_norm_all=list(omega_norm_all),
            y_norm_all=list(y_norm_all),
            batch_size=batch,
            shuffle=False
        )
        print("✅ 数据加载并构造 eval_loader 成功。")

        if mode == "quick":
            print("🔹 快速推理 (mode=quick) with user-loaded data")
            # 载入预训练模型（bundle）
            model, train_loader_b, test_loader_b, loss_history_b, scalers_b, constants_b = load_model_bundle(model_path)
            # 使用“当前数据对应的 scalers”做返归一化更合理（以当前数据域呈现）
            output_viz_data(
                model=model,
                test_loader=eval_loader,
                loss_history=loss_history_b if isinstance(loss_history_b, dict) else {},
                scalers=scalers,
                folder_path=output_dir
            )
            print(f"📊 已生成可视化结果")
            return

        if mode == "train":
            print("🔹 开始模型训练 (mode=train) with user-loaded data")
            # 需要提供 inverse_func；若你已有物理一致的实现，请替换此占位
            def example_inverse_func(pred_w_norm, pred_l_norm, scalers_dic, constants_dic):
                import numpy as np
                pw = np.asarray(pred_w_norm); pl = np.asarray(pred_l_norm)
                M_n = np.clip(0.6 * pw + 0.4 * pl, 0.0, 1.0)
                dkt_n = np.clip(0.5 * pw + 0.5 * pl, 0.0, 1.0)
                dk3t_n = np.clip(0.7 * pw + 0.3 * pl, 0.0, 1.0)
                c_n = np.clip(0.3 * pw + 0.7 * pl, 0.0, 1.0)
                return {"M_norm": M_n, "dkt_norm": dkt_n, "dk3t_norm": dk3t_n, "c_norm": c_n}

            model, train_loader, val_loader, loss_history = run_training(
                X_feat, Y_target, Y_phys, phi, omega_norm_all, y_norm_all,
                inverse_func=example_inverse_func,
                scalers=scalers,
                constants=constants,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                num_epochs=epochs,
                batch_size=batch,
                lr=1e-3,
                device=device
            )

            # 训练完成后，用验证/评估集导出前端可视化
            output_viz_data(
                model=model,
                test_loader=val_loader,
                loss_history=loss_history if isinstance(loss_history, dict) else {},
                scalers=scalers,
                folder_path=output_dir
            )
            print(f"📊 训练完成并导出可视化到: {output_dir}")

            # 按你的 bundle 方式保存（方便后续 simul/quick 直接用）
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            torch.save(
                {
                    "model": model.cpu(),
                    "train_loader": train_loader,
                    "test_loader": val_loader,
                    "loss_history": loss_history,
                    "scalers": scalers,
                    "constants": constants,
                },
                model_path
            )
            print(f"💾 已保存训练好的 bundle 到: {model_path}")
            return

        raise ValueError(f"❌ 不支持的 mode: {mode}")

    raise ValueError(f"❌ 不支持的数据模式: {data}")


if __name__ == "__main__":
    args = parse_args()
    main(
        mode=args.mode,
        data=args.data,
        data_path=args.data_path,
        scaler_path=args.scaler_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
    )
