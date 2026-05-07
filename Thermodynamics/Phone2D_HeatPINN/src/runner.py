# src/runner.py
# -*- coding: utf-8 -*-
"""
统一实验入口（手机 2D 散热）：
- 从 main 传进来的 args 整理成配置
- 调用 data_utils / pinn_model / trainer / postprocess 完成训练与可视化

模式说明：
    * train: 完整训练 + 保存模型 + 可视化
    * quick: 仅加载已有模型做推理与可视化（不再训练）
"""

from pathlib import Path
from typing import Dict
import argparse

import torch

from .data_utils import load_case_config, build_physical_params, set_random_seed
from .pinn_model import build_model
from .trainer import train
from .postprocess import (
    save_snapshots,
    save_center_temperature_curve,
)


def run_experiment(args: argparse.Namespace):
    """
    主调度函数：
    - train: 完整训练 + 保存模型 + 可视化
    - quick: 仅加载已有模型并做推理与可视化
    """
    # 项目根目录：Phone2D_HeatPINN/
    project_root = Path(__file__).resolve().parent.parent

    # ===== 1. 读取配置 & 物理参数 =====
    cfg_path = project_root / "data" / "phone_thermal_case.yaml"
    cfg: Dict = load_case_config(str(cfg_path))

    params: Dict = build_physical_params(cfg)

    # ===== 2. 设备与目录 =====
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_dir / "pinn_phone_thermal.pt"

    # ===== 3. 根据 mode 分支 =====
    if args.mode == "train":
        print("[Runner] Mode = train → 开始训练 PINN 热传导模型...")

        # 这里的 n_iters_override / print_every_override 可以按需从 yaml 里取
        n_iters_override = None
        print_every_override = None

        # trainer.train 内部应负责：
        # - 构建模型
        # - 训练
        # - 保存 ckpt 到 ckpt_path
        model, device_used = train(
            cfg=cfg,
            params=params,
            model_dir=str(model_dir),
            n_iters_override=n_iters_override,
            print_every_override=print_every_override,
        )
        device = device_used
        print(f"[Runner] 训练完成，模型已保存到: {ckpt_path}")

    elif args.mode == "quick":
        print("[Runner] Mode = quick → 仅加载已有模型并做推理/可视化（不训练）")

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"[Runner] 未找到预训练模型文件: {ckpt_path}\n"
                f"请先使用 --mode train 完成一次训练。"
            )

        # 构建同结构模型并加载权重
        builder_out = build_model(cfg)

        # 兼容两种写法：build_model(...) → model 或 (model, ...)
        if isinstance(builder_out, tuple):
            model = builder_out[0]
        else:
            model = builder_out

        model = model.to(device)

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"[INFO] Loaded model from {ckpt_path}")

    else:
        raise ValueError(f"Unknown mode: {args.mode!r}，请使用 'train' 或 'quick'")

    # ===== 4. 统一可视化：train 跑完 or quick 加载完都会走这里 =====
    save_snapshots(
        model=model,
        device=device,
        cfg=cfg,
        params=params,
        results_dir=str(output_dir),
    )

    save_center_temperature_curve(
        model=model,
        device=device,
        cfg=cfg,
        params=params,
        results_dir=str(output_dir),
    )

    print("[Runner] All done, figures saved to:", output_dir)
