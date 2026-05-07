#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smartphone_DropImpact_PINN_main.py

入口脚本：
- 负责解析命令行参数
- 调用 src.runner.run_experiment 完成训练 / 快速推理 / 可视化
"""

import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smartphone Drop-Impact Stress PINN Demo"
    )

    # 运行模式
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "train"],
        help="运行模式：quick=只做推理与可视化，train=完整训练",
    )

    # 数据模式：纯 PINN → 默认 simul
    parser.add_argument(
        "--data",
        type=str,
        default="simul",
        choices=["simul", "load"],
        help=(
            "数据模式：simul=基于 PDE 的物理仿真生成训练样本；"
            "load=从文件加载(本案例未实现 load，仅支持 simul)"
        ),
    )

    # 设备
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备：cuda 或 cpu（默认 cuda）",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（确保可复现）",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="结果输出目录（相对当前项目根目录）",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # 本案例严格为 PINN → 必须是 simul
    if args.data != "simul":
        raise ValueError(
            f"[Smartphone_DropImpact_PINN] 本案例为纯 PINN，不支持 --data={args.data!r}，"
            f"请使用 --data simul"
        )

    # 规范 output_dir
    project_root = Path(__file__).resolve().parent
    args.output_dir = str((project_root / args.output_dir).resolve())

    from src.runner import run_experiment  # 延迟导入，避免循环依赖

    run_experiment(args)


if __name__ == "__main__":
    main()
