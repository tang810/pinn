#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

目标：
- 保留“内层 main(...)”原样（mode ∈ {"synthetic","file"}，data_dir 仅在 file 模式使用）。
- 提供“外层 main_outer(...)”，其**参数结构与内层 main 相似**（仅将外层的 mode/data 语义映射到内层）。
- 将命令行解析逻辑整合进 main_outer()：当以脚本方式运行并传入命令行参数时，main_outer() 会解析 CLI；
  若直接在代码中调用 main_outer(... 以参数形式传入 ...)，则跳过 CLI，直接使用函数形参。

外层 CLI（脚本运行时）：
  --mode {train,quick}        # 外层：train→synthetic；quick→file
  --data {simul,load}         # 外层：simul 强制 synthetic；load 强制 file
  --data-dir PATH             # 当 --data=load 时使用；其它情况下忽略
其余参数与内层 main 完全一致（--outdir, --model-dir, --num-samples, ...）。

示例：
  合成数据：
    python main.py --mode train --data simul --outdir result/run_synth --model-dir model
  读取目录：
    python main.py --mode quick --data load --data-dir data/hrb_case1 --outdir result/run_file --model-dir model
"""

from __future__ import annotations

import sys
from pathlib import Path


# =========================================================
# 內層：保持原有签名与语义（不要改动）
# =========================================================
def main_in(
    mode: str = "synthetic",             # "synthetic" or "file"
    data_dir: str = "data",              # used when mode == "file"
    outdir: str = "result",
    model_dir: str = "model",
    num_samples: int = 200,
    burn: int = 100,
    L: int = 100,
    step_size: float = 1e-3,
    N_tr_u: int = 50,
    N_tr_f: int = 50,
    N_val_1d: int = 40,
    lb: float = 0.0,
    ub: float = 1.0,
    no_plots: bool = False,
    fast: bool = False,
):
    """
    內層主函数：
    - 相对路径统一以本文件所在目录为基准解析；
    - 调用 src/run_hmc.py 的 run_hmc_pipeline(...)。
    """
    here = Path(__file__).resolve().parent
    project_root = here  # 约定：main.py 所在目录为工程根
    sys.path.insert(0, str(project_root / "src"))

    # 延迟导入，避免无 src/ 环境时报错
    try:
        from run_hmc import run_hmc_pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(f"[Import] 无法导入 src/run_hmc.py 中的 run_hmc_pipeline: {e}")

    # 统一路径：把传入的相对路径解析为相对工程根的绝对路径
    def _resolve_relative(p: str) -> str:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str((project_root / pp).resolve())

    data_dir_resolved  = _resolve_relative(data_dir)
    outdir_resolved    = _resolve_relative(outdir)
    model_dir_resolved = _resolve_relative(model_dir)

    # 执行核心流水线
    return run_hmc_pipeline(
        mode=mode,
        data_dir=data_dir_resolved,
        outdir=outdir_resolved,
        model_dir=model_dir_resolved,
        num_samples=num_samples,
        burn=burn,
        L=L,
        step_size=step_size,
        N_tr_u=N_tr_u,
        N_tr_f=N_tr_f,
        N_val_1d=N_val_1d,
        lb=lb,
        ub=ub,
        no_plots=no_plots,
        fast=fast,
    )


# =========================================================
# 外层：main() —— 参数结构与“内层 main”相似；整合 CLI（可程序调用/可命令行解析）
# =========================================================
def main(
    # 与“内层 main”尽量一致的参数结构（仅 mode、data 的语义不同）：
    mode: str = "train",          # 外层：["train","quick"]，将映射为内层 ["synthetic","file"]
    data: str = "simul",          # 外层：["simul","load"]，优先决定内层模式与 data_dir 使用
    data_dir: str = "data",       # 当 data="load" 时传入内层；其余情况下忽略
    outdir: str = "result",
    model_dir: str = "model",
    num_samples: int = 200,
    burn: int = 100,
    L: int = 100,
    step_size: float = 1e-3,
    N_tr_u: int = 50,
    N_tr_f: int = 50,
    N_val_1d: int = 40,
    lb: float = 0.0,
    ub: float = 1.0,
    no_plots: bool = False,
    fast: bool = False,
):
    """
    使用方式：
    - 作为库函数：直接 main_outer(mode=..., data=..., data_dir=..., 其余同内层)，不会解析 CLI。
    - 作为脚本入口：若命令行存在 --xxx 形式的参数，则优先解析 CLI，覆盖上述形参。
    """
    # --- 判断是否需要解析命令行参数（当脚本运行并传入 --xxx 才解析；否则使用函数形参） ---
    if any(arg.startswith("--") for arg in sys.argv[1:]):
        import argparse

        p = argparse.ArgumentParser(
            description="Outer entry（集成 CLI）：参数结构贴近內層 main；将 {mode,data,data_dir} 语义映射为 內層 {mode,data_dir}。"
        )
        p.add_argument("--mode", choices=["train", "quick"], default=mode,
                       help="外层：train→synthetic；quick→file。")
        p.add_argument("--data", choices=["simul", "load"], default=data,
                       help="外层数据来源：simul(合成) 或 load(从 --data-dir 指定目录读取)。")
        p.add_argument("--data-dir", type=str, default=data_dir,
                       help="当 --data=load 时用于指定数据目录路径；其它情况下忽略。")

        p.add_argument("--outdir", type=str, default=outdir)
        p.add_argument("--model-dir", type=str, default=model_dir)

        p.add_argument("--num-samples", type=int, default=num_samples)
        p.add_argument("--burn", type=int, default=burn)
        p.add_argument("--L", type=int, default=L)
        p.add_argument("--step-size", type=float, default=step_size)

        p.add_argument("--N-tr-u", dest="N_tr_u", type=int, default=N_tr_u)
        p.add_argument("--N-tr-f", dest="N_tr_f", type=int, default=N_tr_f)
        p.add_argument("--N-val-1d", dest="N_val_1d", type=int, default=N_val_1d)

        p.add_argument("--lb", type=float, default=lb)
        p.add_argument("--ub", type=float, default=ub)

        p.add_argument("--no-plots", action="store_true", default=no_plots)
        p.add_argument("--fast", action="store_true", default=fast)

        a = p.parse_args()

        # 用 CLI 的值覆盖函数形参
        mode     = a.mode
        data     = a.data
        data_dir = a.data_dir
        outdir   = a.outdir
        model_dir= a.model_dir
        num_samples = a.num_samples
        burn        = a.burn
        L           = a.L
        step_size   = a.step_size
        N_tr_u      = a.N_tr_u
        N_tr_f      = a.N_tr_f
        N_val_1d    = a.N_val_1d
        lb          = a.lb
        ub          = a.ub
        no_plots    = a.no_plots
        fast        = a.fast

    # --- 外→内 映射逻辑：data 优先于 mode 决策 ---
    if data == "simul":
        inner_mode = "synthetic"
        inner_data_dir = "data"      # 占位；內層不会使用
    elif data == "load":
        inner_mode = "file"
        inner_data_dir = data_dir
    else:
        # 兜底（正常不走到这里）：按外层 mode 决策
        inner_mode = "synthetic" if mode == "train" else "file"
        inner_data_dir = data_dir if inner_mode == "file" else "data"


    # Quick 模式默认启用 fast 以走“预训练/缓存优先”的快速反演路径
    if mode == 'quick' and not fast:
        fast = True

    # 直接调用“内层 main(...)”
    return main_in(
        mode=inner_mode,
        data_dir=inner_data_dir,
        outdir=outdir,
        model_dir=model_dir,
        num_samples=num_samples,
        burn=burn,
        L=L,
        step_size=step_size,
        N_tr_u=N_tr_u,
        N_tr_f=N_tr_f,
        N_val_1d=N_val_1d,
        lb=lb,
        ub=ub,
        no_plots=no_plots,
        fast=fast,
    )


# =========================================================
# 入口：仅需调用 main_outer()
# =========================================================
if __name__ == "__main__":
    main()
