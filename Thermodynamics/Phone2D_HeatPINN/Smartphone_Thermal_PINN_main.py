#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smartphone_Thermal_PINN_main.py

入口脚本：
- 负责解析命令行参数
- 调用 src.runner.run_experiment 完成训练 / 快速推理 / 可视化
"""

import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smartphone 2D Thermal PINN Demo"
    )

    # 运行模式：quick=只做推理，train=训练
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "train"],
        help="运行模式：quick=加载已有模型并推理可视化，train=完整训练+可视化",
    )

    # 数据模式：本案例是纯 PINN → 只能 simul
    parser.add_argument(
        "--data",
        type=str,
        default="simul",
        choices=["simul"],
        help="数据来源：本案例为纯 PINN，仅支持 simul（基于 PDE 点采样）",
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
        help="结果输出目录（相对项目根目录）",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.data != "simul":
        raise ValueError(
            f"[Phone2D_HeatPINN] 本案例为纯 PINN，不支持 --data={args.data!r}，"
            f"请使用 --data simul"
        )

    # 规范 output_dir 为绝对路径
    project_root = '/data/AI4PDE_CN/src/PINN4Science/Thermodynamics/Phone2D_HeatPINN'
    args.output_dir = '/data/AI4PDE_CN/src/PINN4Science/Thermodynamics/Phone2D_HeatPINN/results'
    # args.output_dir = str((project_root / args.output_dir).resolve())

    from src.runner import run_experiment  # 延迟导入，避免循环依赖
    run_experiment(args)


if __name__ == "__main__":
    main()
