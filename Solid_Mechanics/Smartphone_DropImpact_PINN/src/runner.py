# src/runner.py
# -*- coding: utf-8 -*-
"""
统一实验入口：
- 从 main 传入的 args 解析配置
- 调用 data_utils / pinn_model / trainer / postprocess
- 只保留两种模式：
    * train: 完整训练 + 保存模型 + 可视化
    * quick: 只加载已有模型做推理与可视化（不再训练）
"""

from pathlib import Path
import argparse

import yaml
import torch

from .data_utils import set_random_seed, get_device
from .pinn_model import build_model
from .trainer import train_model
from .postprocess import (
    find_best_time_for_bottom_contrast,
    compute_max_stress_time_curve,
    plot_max_stress_curve,
    plot_stress_phone,
    plot_multiple_frames,
)


# -----------------------------
# 配置加载与可视化辅助
# -----------------------------
def load_config(project_root: Path) -> dict:
    cfg_path = project_root / "data" / "phone_drop_case.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_time_lists(cfg: dict):
    vis_cfg = cfg.get("visualization", {})
    triptych_ms = vis_cfg.get("triptych_ms", [0.25, 1.0, 5.0])

    gif_cfg = vis_cfg.get("gif_ms", {})
    if isinstance(gif_cfg, dict):
        start = gif_cfg.get("start", 0.25)
        end = gif_cfg.get("end", 7.5)
        step = gif_cfg.get("step", 0.25)
        n = int((end - start) / step) + 1
        gif_ms = [start + i * step for i in range(n)]
    else:
        gif_ms = gif_cfg
    return triptych_ms, gif_ms


def run_eval_and_visualization(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
):
    """
    用当前模型做一次完整的推理与可视化：
    - 自动搜索“底部/顶部对比最佳时刻”
    - 最大等效应力-时间曲线
    - 若干关键帧 + 全时序帧
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 自动搜索“底部/顶部”应力对比最佳时间
    vis_cfg = cfg.get("visualization", {})
    t_min_ms = vis_cfg.get("t_min_search_ms", 0.5)
    t_best, score = find_best_time_for_bottom_contrast(
        model,
        cfg,
        device,
        num_t=80,
        bottom_ratio=0.2,
        top_ratio=0.2,
        t_min=t_min_ms * 1e-3,
    )
    print(
        f"[Auto] Best time for bottom contrast: "
        f"t={t_best*1e3:.2f} ms, bottom/top={score:.2f}"
    )

    # 2. 最大等效应力-时间曲线
    t_samples, max_stress_norm = compute_max_stress_time_curve(
        model, cfg, device, num_t=150
    )
    plot_max_stress_curve(
        t_samples,
        max_stress_norm,
        savepath=output_dir / "max_stress_vs_time.png",
    )

    # 3. 主视图（用 t_best）
    from .postprocess import evaluate_stress_field

    X, Y, Svm_star_t = evaluate_stress_field(
        model, cfg, device, t_best, nx=120, ny=240
    )
    Svm_norm = Svm_star_t / (Svm_star_t.max() + 1e-12)
    plot_stress_phone(
        cfg,
        X,
        Y,
        Svm_norm,
        t_best,
        savepath=output_dir / "phone_drop_t_best.png",
        show=False,
    )

    # 4. 多帧（triptych + 全时序）
    triptych_ms, gif_ms = build_time_lists(cfg)

    # Triptych 三张图
    for i, t_ms in enumerate(triptych_ms):
        t_val = t_ms * 1e-3
        X, Y, Svm_star_t = evaluate_stress_field(
            model, cfg, device, t_val, nx=120, ny=240
        )
        S_norm = Svm_star_t / (Svm_star_t.max() + 1e-12)
        plot_stress_phone(
            cfg,
            X,
            Y,
            S_norm,
            t_val,
            savepath=output_dir / f"stress_field_t{t_ms:.2f}ms_simple.png",
            show=False,
        )

    # 全时序帧（方便之后生成 GIF）
    prefix = output_dir / "phone_drop_"
    # plot_multiple_frames(model, cfg, device, gif_ms, prefix=prefix)


# -----------------------------
# 统一入口：只保留 quick / train 两种模式
# -----------------------------
def run_experiment(args: argparse.Namespace):
    """
    主调度函数：
    - train: 完整训练 + 保存模型 + 可视化
    - quick: 仅加载已有模型做推理与可视化（相当于原来的 eval）
    """
    # 推断 project_root = Smartphone_DropImpact_PINN/
    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(project_root)

    set_random_seed(args.seed)
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 构建模型骨架
    model = build_model(cfg, device)

    # 统一 checkpoint 路径
    ckpt_path = model_dir / "pinn_phone_drop.pt"

    if args.mode == "train":
        print("[Runner] Mode = train → 开始训练 PINN 模型...")
        model = train_model(
            model=model,
            cfg=cfg,
            device=device,
            mode="train",
            output_dir=output_dir,
            model_dir=model_dir,
        )
        print(f"[Runner] 训练完成，模型已保存到: {ckpt_path}")

    elif args.mode == "quick":
        print("[Runner] Mode = quick → 仅加载已有模型并做可视化（不再训练）")
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"[Runner] 未找到预训练模型文件: {ckpt_path}\n"
                f"请先使用 --mode train 完成一次训练。"
            )
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f"[INFO] Loaded model from {ckpt_path}")

    else:
        raise ValueError(f"Unknown mode: {args.mode!r}, 请使用 'train' 或 'quick'")

    # 统一：不管是 train 后还是 quick，最后都跑一遍可视化
    run_eval_and_visualization(model, cfg, device, output_dir)
    print("=== Done ===")
