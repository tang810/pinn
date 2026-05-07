# -*- coding: utf-8 -*-
"""
run_hmc.py  —— 统一封装 HMC 训练/预测的“库模块”，可单独运行（包含 CLI）。
设计目标：
1) 暴露**一组可复用函数**（数据构建、functional 封装、采样、预测、保存等）；
2) 支持两种数据模式：synthetic（合成） / file（读取 data/ 下 npz 或 csv）；
3) 作为脚本运行时，提供命令行参数，直接跑完整流程；
4) 供新的 main.py（只有一个 main 函数）直接调用。
"""

from pathlib import Path
import importlib.util
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import hamiltorch  # 项目内本地包：src/hamiltorch/

# ===== 业务配置/依赖（使用项目的 src/* 模块） =====
from config import (
    device, DTYPE, DEVICE_STR_FOR_UTIL,
    tau_priors, tau_likes,
    step_size as CFG_STEP_SIZE, L as CFG_L, burn as CFG_BURN, num_samples as CFG_NS,
    pde, pinns, epochs,
    layer_sizes, activation,
    lb as CFG_LB, ub as CFG_UB,
    N_tr_u as CFG_NTRU, N_tr_f as CFG_NTRF, N_val_1d as CFG_NVAL,
    n_params_single, exact_single,
)
from data import build_train_data as build_train_data_synth, build_val_data
from model import build_nets
from loss import model_loss
from plotting import save_posterior_slices
# ==== 兼容保存：自动把 torch.Tensor 转成 numpy 再 np.save ====
def _save_npy(path, arr):
    try:
        import torch as _torch, numpy as _np
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        elif isinstance(arr, _torch.Tensor):
            arr = arr.cpu().numpy()
        return _np.save(str(path), arr)
    except Exception as e:
        print(f"[Warn] 保存 {path} 失败：{e}")

# ==== 预训练权重加载（用于 quick/fast 或作为 HMC 初始化）====
def _maybe_load_pretrained(nets, model_dir):
    """在 model_dir 下自动寻找可用的 *.pth/*.pt 权重并加载到 nets[0]。
    优先级（若 fast 且存在对应缓存，会在上游直接 return，不会进到这里）：
      1) posterior_mean_state_dict.pth （state_dict）
      2) 其它 *.pth/*.pt （尝试从 {'state_dict':...} 或直接 state_dict 加载）
    返回 (loaded: bool, used_path: Path|None, missing, unexpected)。
    """
    md = Path(model_dir)
    if not md.exists():
        print(f"[CKPT] model_dir 不存在：{md}")
        return False, None, (), ()
    # 排除 posterior_samples（那是样本，不是纯 state_dict）
    black = {'posterior_samples.pth'}
    # 优先 posterior_mean_state_dict.pth
    candidates = []
    pmean = md / 'posterior_mean_state_dict.pth'
    if pmean.exists(): candidates.append(pmean)
    # 其它 ckpt
    for p in sorted(md.glob('*.pt')) + sorted(md.glob('*.pth')):
        if p.name in black or p == pmean: 
            continue
        candidates.append(p)
    if not candidates:
        print(f"[CKPT] {md} 下未找到可加载的 *.pth/*.pt（排除 posterior_samples.pth）。") 
        return False, None, (), ()
    ckpt = candidates[-1]  # 最近/最后一个
    try:
        state = torch.load(ckpt, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        net = nets[0]
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"[CKPT] 已加载 '{ckpt.name}'  missing={list(missing)}  unexpected={list(unexpected)}")
        return True, ckpt, missing, unexpected
    except Exception as e:
        print(f"[CKPT] 加载 {ckpt} 失败：{e}")
        return False, ckpt, (), ()


# ===== 强制以相对路径加载本地 hamiltorch/util.py（functional 封装用） =====
ROOT = Path(__file__).resolve().parent
# Project root: if this file is inside 'src', use its parent; otherwise use current folder.
PROJECT_ROOT = ROOT.parent if ROOT.name.lower() == 'src' else ROOT
UTIL_PY = ROOT / "hamiltorch" / "util.py"
if not UTIL_PY.exists():
    raise FileNotFoundError(f"[路径不存在] {UTIL_PY}，请确认 src/hamiltorch/util.py 存在。")
_spec = importlib.util.spec_from_file_location("local_hamiltorch_util", UTIL_PY)
_local_util = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_local_util)
util = _local_util  # 统一引用名

def _resolve_rel(pathlike):
    p = Path(pathlike)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# ===========================================
# 1) functional 封装：优先 util.make_functional，失败则“手写”
# ===========================================
def _make_functional_nets(nets) -> Tuple[List, List, List]:
    """返回 (fmodels, params_templates, kinds)
    - fmodels[i](x, params=params_i) -> y
    - params_templates[i]: 该网络的参数列表（张量顺序固定）
    - kinds[i]: 'util' | 'manual'，表示来源
    """
    fmodels, templates, kinds = [], [], []
    for net in nets:
        # 1. 优先尝试 util.make_functional
        try:
            out = util.make_functional(net)
            if isinstance(out, (list, tuple)) and len(out) >= 2 and callable(out[0]):
                _fmodel, _params = out[0], out[1]
                _params = [p.detach().clone().to(DTYPE).to(device) for p in _params]
                fmodels.append(_fmodel)
                templates.append(_params)
                kinds.append('util')
                continue
        except Exception as e:
            print(f"[Warn] util.make_functional 失败，转用手写：{e}")

        # 2. 回退：手写 functional（假设 Net3D 结构：四层全连接 + activation）
        def fmodel(x: torch.Tensor, params: List[torch.Tensor]):
            w1,b1,w2,b2,w3,b3,w4,b4 = params
            x = activation(F.linear(x, w1, b1))
            x = activation(F.linear(x, w2, b2))
            x = activation(F.linear(x, w3, b3))
            x = F.linear(x, w4, b4)
            return x

        # 根据当前 net 的 state_dict 构建模板
        sd = net.state_dict()
        tpl = [
            sd['l1.weight'].detach().clone().to(DTYPE).to(device),
            sd['l1.bias'].detach().clone().to(DTYPE).to(device),
            sd['l2.weight'].detach().clone().to(DTYPE).to(device),
            sd['l2.bias'].detach().clone().to(DTYPE).to(device),
            sd['l3.weight'].detach().clone().to(DTYPE).to(device),
            sd['l3.bias'].detach().clone().to(DTYPE).to(device),
            sd['l4.weight'].detach().clone().to(DTYPE).to(device),
            sd['l4.bias'].detach().clone().to(DTYPE).to(device),
        ]
        fmodels.append(fmodel)
        templates.append(tpl)
        kinds.append('manual')
    return fmodels, templates, kinds

# ===========================================
# 2) 展平/还原与形状工具
# ===========================================
def _numel_of_shape(shape): 
    m = 1
    for s in shape: m *= int(s)
    return m

def _build_shapes_and_groups(params_templates: List[List[torch.Tensor]], n_params_single: int):
    shapes = []
    for i in range(n_params_single):
        shapes.append((1,))  # 单参数 θ_i
    for ps in params_templates:
        for t in ps:
            shapes.append(tuple(t.shape))
    group_lens = [1] * n_params_single + [len(ps) for ps in params_templates]
    return shapes, group_lens

def _flatten_init_vector(params_templates: List[List[torch.Tensor]], n_params_single: int) -> torch.Tensor:
    thetas = [torch.zeros(1, dtype=DTYPE, device=device, requires_grad=True) for _ in range(n_params_single)]
    flat = []
    for t in thetas: flat.append(t.reshape(-1))
    for ps in params_templates:
        for p in ps: flat.append(p.reshape(-1))
    vec = torch.cat(flat, dim=0)
    vec.requires_grad_(True)
    return vec

def _unflatten_vector(vec: torch.Tensor, shapes, group_lens, n_params_single: int):
    assert vec.ndim == 1
    tensors, off = [], 0
    for shp in shapes:
        m = _numel_of_shape(shp)
        t = vec[off:off+m].reshape(shp)
        t.requires_grad_(True)
        tensors.append(t); off += m
    params_single = tensors[:n_params_single]
    rest = tensors[n_params_single:]
    per_net_params = []
    ptr = 0
    for gl in group_lens[n_params_single:]:
        per = rest[ptr:ptr+gl]; ptr += gl
        per_net_params.append(per)
    return params_single, per_net_params

def _log_prior_flat(vec: torch.Tensor, tau_priors: float) -> torch.Tensor:
    return -0.5 * tau_priors * (vec * vec).sum()

# ===========================================
# 3) HMC 采样
# ===========================================
def sample_with_hmc(nets, data_dict, *, n_params_single, tau_priors, tau_likes,
                    num_samples, L, step_size, burn):
    fmodels, params_templates, kinds = _make_functional_nets(nets)
    shapes, group_lens = _build_shapes_and_groups(params_templates, n_params_single)
    params_init_vec = _flatten_init_vector(params_templates, n_params_single)

    def log_prob_fn(vec: torch.Tensor) -> torch.Tensor:
        params_single, per_net_params = _unflatten_vector(vec, shapes, group_lens, n_params_single)
        ll, _ = model_loss(
            data_dict, fmodels,
            params_unflattened=per_net_params,
            tau_likes=tau_likes, gradients=None,
            params_single=params_single
        )
        lp = ll + _log_prior_flat(vec, tau_priors)
        return lp if lp.ndim == 0 else lp.sum()

    hamiltorch.set_random_seed(123)
    samples = hamiltorch.sample(
        log_prob_func=log_prob_fn,
        params_init=params_init_vec,
        num_samples=num_samples,
        step_size=step_size,
        num_steps_per_sample=L,
        burn=burn,
        sampler=hamiltorch.Sampler.HMC,
        integrator=hamiltorch.Integrator.IMPLICIT,
        debug=False,
    )
    if isinstance(samples, list):
        samples = torch.stack(samples, dim=0)
    return samples, fmodels, params_templates, shapes, group_lens, kinds

# ===========================================
# 4) 验证集预测
# ===========================================
def predict_from_samples(samples_flat: torch.Tensor, fmodels, params_templates, shapes, group_lens,
                         data_val, *, n_params_single, tau_priors, tau_likes):
    pred_u_list, pred_f_list, logps = [], [], []

    def log_prob_on(vec: torch.Tensor) -> torch.Tensor:
        params_single, per_net_params = _unflatten_vector(vec, shapes, group_lens, n_params_single)
        ll, _ = model_loss(
            data_val, fmodels,
            params_unflattened=per_net_params,
            tau_likes=tau_likes, gradients=None,
            params_single=params_single
        )
        lp = ll + _log_prior_flat(vec, tau_priors)
        return lp if lp.ndim == 0 else lp.sum()

    for i in range(samples_flat.shape[0]):
        vec = samples_flat[i]
        params_single, per_net_params = _unflatten_vector(vec, shapes, group_lens, n_params_single)
        ll_val, outputs = model_loss(
            data_val, fmodels,
            params_unflattened=per_net_params,
            tau_likes=tau_likes, gradients=None,
            params_single=params_single
        )
        pred_u, pred_f = outputs
        pred_u_list.append(pred_u.detach())
        pred_f_list.append(pred_f.detach())
        logps.append(log_prob_on(vec).detach())

    pred_u_stack = torch.stack(pred_u_list, dim=0)  # (S, N, 1)
    pred_f_stack = torch.stack(pred_f_list, dim=0)  # (S, N, 1)
    return [pred_u_stack, pred_f_stack], logps

# ===========================================
# 5) 保存 .pth 工件（后验样本 + 可选 posterior_mean state_dict）
# ===========================================
def save_pth_artifacts(samples_flat: torch.Tensor,
                       nets, params_templates, shapes, group_lens, kinds,
                       model_dir: Path = None):
    if model_dir is None:
        model_dir = PROJECT_ROOT / 'model'
    model_dir = _resolve_rel(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "samples_flat": samples_flat.detach().cpu(),
        "shapes": shapes,
        "group_lens": group_lens,
        "n_params_single": n_params_single,
        "layer_sizes": layer_sizes,
        "kinds": kinds,
    }
    samples_path = model_dir / "posterior_samples.pth"
    torch.save(meta, samples_path)
    print(f"[Save] 已保存后验样本到: {samples_path}")

    can_save_state = all(k == "manual" for k in kinds)
    if can_save_state:
        vec_mean = samples_flat.mean(dim=0)
        _, per_net_params = _unflatten_vector(vec_mean, shapes, group_lens, n_params_single)
        net = nets[0]; pp = per_net_params[0]
        if len(pp) == 8:
            with torch.no_grad():
                net.l1.weight.copy_(pp[0]); net.l1.bias.copy_(pp[1])
                net.l2.weight.copy_(pp[2]); net.l2.bias.copy_(pp[3])
                net.l3.weight.copy_(pp[4]); net.l3.bias.copy_(pp[5])
                net.l4.weight.copy_(pp[6]); net.l4.bias.copy_(pp[7])
            state_path = model_dir / "posterior_mean_state_dict.pth"
            torch.save(net.state_dict(), state_path)
            print(f"[Save] 已保存 posterior_mean state_dict 到: {state_path}")
        else:
            print("[Warn] 模板参数数量与手写映射不符，跳过 state_dict 保存。")
    else:
        print("[Info] 本次运行使用了 util.make_functional（或混合），跳过 state_dict 保存。")

# ===========================================
# 6) 从 data/ 目录加载训练数据（file 模式）
# ===========================================
def load_train_from_dir(data_dir: Path):
    """支持两种格式（二选一）：
      1) data/train.npz ，包含键：x_u,y_u,x_f,y_f
      2) 四个 CSV：data/{x_u,y_u,x_f,y_f}.csv
         - x_*: N×3，y_*: N×1 或 N
    """
    data_dir = Path(data_dir)
    npz_path = data_dir / "train.npz"
    if npz_path.exists():
        arrs = np.load(npz_path, allow_pickle=False)
        required = ["x_u", "y_u", "x_f", "y_f"]
        for k in required:
            if k not in arrs:
                raise KeyError(f"{npz_path} 缺少键 '{k}'")
        x_u = torch.as_tensor(arrs["x_u"], dtype=DTYPE, device=device)
        y_u = torch.as_tensor(arrs["y_u"], dtype=DTYPE, device=device).reshape(-1, 1)
        x_f = torch.as_tensor(arrs["x_f"], dtype=DTYPE, device=device)
        y_f = torch.as_tensor(arrs["y_f"], dtype=DTYPE, device=device).reshape(-1, 1)
        return {"x_u": x_u, "y_u": y_u, "x_f": x_f, "y_f": y_f}

    # 否则尝试四个 CSV
    def _csv(name: str) -> torch.Tensor:
        p = data_dir / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(f"未找到 {p}")
        arr = np.loadtxt(p, delimiter=",")
        return torch.as_tensor(arr, dtype=DTYPE, device=device)

    x_u = _csv("x_u"); y_u = _csv("y_u").reshape(-1, 1)
    x_f = _csv("x_f"); y_f = _csv("y_f").reshape(-1, 1)

    if x_u.ndim != 2 or x_u.shape[1] != 3:
        raise ValueError(f"x_u.csv 期望形状 N×3，实际 {tuple(x_u.shape)}")
    if x_f.ndim != 2 or x_f.shape[1] != 3:
        raise ValueError(f"x_f.csv 期望形状 N×3，实际 {tuple(x_f.shape)}")
    if y_u.ndim != 2 or y_u.shape[1] != 1:
        raise ValueError(f"y_u.csv 期望形状 N×1，实际 {tuple(y_u.shape)}")
    if y_f.ndim != 2 or y_f.shape[1] != 1:
        raise ValueError(f"y_f.csv 期望形状 N×1，实际 {tuple(y_f.shape)}")
    return {"x_u": x_u, "y_u": y_u, "x_f": x_f, "y_f": y_f}

# ===========================================
# 7) 高层封装：一次性跑完（供 main.py 调用）
# ===========================================
def run_hmc_pipeline(*, mode: str, data_dir: Path = 'data', outdir: Path = 'result',
                     num_samples: int = CFG_NS, burn: int = CFG_BURN, L: int = CFG_L, step_size: float = CFG_STEP_SIZE,
                     N_tr_u: int = CFG_NTRU, N_tr_f: int = CFG_NTRF, N_val_1d: int = CFG_NVAL,
                     lb: float = CFG_LB, ub: float = CFG_UB, no_plots: bool = False,
                     fast: bool = False, model_dir: Path = None):
    # resolve directories relative to project root
    if model_dir is None:
        model_dir = PROJECT_ROOT / 'model'
    model_dir = _resolve_rel(model_dir)
    outdir = _resolve_rel(outdir)
    data_dir = _resolve_rel(data_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Paths] 输出目录: {outdir}")
    print(f"[Paths] 模型目录: {model_dir}")

    # 数据
    if mode == "synthetic":
        print("[Data] 使用合成数据")
        data_train = build_train_data_synth(N_tr_u=N_tr_u, N_tr_f=N_tr_f, lb=lb, ub=ub)
        data_val = build_val_data(N_val_1d=N_val_1d, lb=lb, ub=ub)
    elif mode == "file":
        print(f"[Data] 从 {data_dir} 读取训练数据")
        data_train = load_train_from_dir(Path(data_dir))
        data_val = build_val_data(N_val_1d=N_val_1d, lb=lb, ub=ub)
    else:
        raise ValueError(f"未知 mode: {mode}")

    # 网络
    # 快速模式：如果检测到缓存的 pth，直接使用
    if fast:
        
        has_samples = (model_dir / 'posterior_samples.pth').exists()
        has_state   = (model_dir / 'posterior_mean_state_dict.pth').exists()
        if has_samples or has_state:
            print('[Fast] 检测到缓存模型，跳过 HMC 采样。')
            nets = build_nets()
    _ = _maybe_load_pretrained(nets, model_dir)
            # 使用后验样本优先
            if has_samples:
                samples_flat, shapes, group_lens, nps = load_posterior_samples(model_dir)
                fmodels, params_templates, kinds = _make_functional_nets(nets)
                pred_list, log_prob_list = predict_from_samples(
                    samples_flat, fmodels, params_templates, shapes, group_lens,
                    data_val, n_params_single=nps, tau_priors=tau_priors, tau_likes=tau_likes
                )
                u_samps, f_samps = pred_list
                if u_samps.ndim == 3: u_samps = u_samps[..., 0]
                if f_samps.ndim == 3: f_samps = f_samps[..., 0]
                _save_npy(outdir / 'u_mean.npy', u_samps.mean(axis=0))
                _save_npy(outdir / 'u_std.npy', u_samps.std(axis=0))
                _save_npy(outdir / 'f_mean.npy', f_samps.mean(axis=0))
                _save_npy(outdir / 'f_std.npy', f_samps.std(axis=0))
                print(f"[Fast] 已基于 posterior_samples.pth 生成 {outdir}/u_*.npy 与 f_*.npy")
                if mode == 'synthetic' and (not no_plots):
                    save_posterior_slices(pred_list, N_val_1d=N_val_1d, lb=lb, ub=ub, outdir=str(outdir))
                return
            # 否则使用 posterior_mean_state_dict（无不确定性）
            else:
                net = load_posterior_mean_model(model_dir)
                # 构造一个“单样本”向量：theta 全 0（等价 alpha=1），权重来自 net
                fmodels, params_templates, kinds = _make_functional_nets(nets)
                shapes, group_lens = _build_shapes_and_groups(params_templates, n_params_single)
                # 展平 params_templates 再替换为当前 net 的 state_dict
                vec0 = _flatten_init_vector(params_templates, n_params_single)
                # 用 state_dict 写回 vec0 的网络参数区段
                sd = net.state_dict()
                # 重新构建 per_net_params 并替换
                params_single, per_net_params = _unflatten_vector(vec0, shapes, group_lens, n_params_single)
                pp = per_net_params[0]
                with torch.no_grad():
                    pp[0].copy_(sd['l1.weight']); pp[1].copy_(sd['l1.bias'])
                    pp[2].copy_(sd['l2.weight']); pp[3].copy_(sd['l2.bias'])
                    pp[4].copy_(sd['l3.weight']); pp[5].copy_(sd['l3.bias'])
                    pp[6].copy_(sd['l4.weight']); pp[7].copy_(sd['l4.bias'])
                # 重新拼回单样本 flat 向量
                # 把 params_single 设为 0（意味着 alpha=1），仅做 point 预测
                for t in params_single: t.data.zero_()
                # 手动拼回扁平向量
                flat_list = []
                for t in params_single: flat_list.append(t.reshape(-1))
                for ps in per_net_params:
                    for q in ps: flat_list.append(q.reshape(-1))
                single_vec = torch.cat(flat_list, dim=0).detach()
                samples_flat = single_vec.unsqueeze(0)
                pred_list, _ = predict_from_samples(
                    samples_flat, fmodels, params_templates, shapes, group_lens,
                    data_val, n_params_single=n_params_single, tau_priors=tau_priors, tau_likes=tau_likes
                )
                u_samps, f_samps = pred_list
                if u_samps.ndim == 3: u_samps = u_samps[..., 0]
                if f_samps.ndim == 3: f_samps = f_samps[..., 0]
                _save_npy(outdir / 'u_mean.npy', u_samps.mean(axis=0))
                _save_npy(outdir / 'u_std.npy', np.zeros_like(u_samps.mean(axis=0)))
                _save_npy(outdir / 'f_mean.npy', f_samps.mean(axis=0))
                _save_npy(outdir / 'f_std.npy', np.zeros_like(f_samps.mean(axis=0)))
                print(f"[Fast] 已基于 posterior_mean_state_dict.pth 生成 point 预测（std=0）：{outdir}/u_*.npy 与 f_*.npy")
                return
        else:
            print('[Fast] 未发现缓存 pth，继续执行完整 HMC。')

    nets = build_nets()

    # HMC 采样
    samples_flat, fmodels, params_templates, shapes, group_lens, kinds = sample_with_hmc(
        nets, data_train,
        n_params_single=n_params_single,
        tau_priors=tau_priors, tau_likes=tau_likes,
        num_samples=num_samples, L=L, step_size=step_size, burn=burn
    )
    print(f"[HMC] samples_flat shape: {tuple(samples_flat.shape)}")

    # 验证预测与对数概率
    pred_list, log_prob_list = predict_from_samples(
        samples_flat, fmodels, params_templates, shapes, group_lens,
        data_val, n_params_single=n_params_single, tau_priors=tau_priors, tau_likes=tau_likes
    )
    print("\nExpected validation log probability: {:.3f}".format(torch.stack(log_prob_list).mean()))
    print("\nThe exact values of single parameters:", exact_single)

    # α 统计
    theta_samples = samples_flat[:, :n_params_single].detach().cpu().numpy().ravel()
    alpha_samples = np.exp(theta_samples)
    print("alpha mean :", alpha_samples.mean())
    print("alpha std  :", alpha_samples.std())
    print("alpha var  :", alpha_samples.var())
    ci_lo, ci_hi = np.percentile(alpha_samples, [2.5, 97.5])
    print("alpha 95% CI:", (ci_lo, ci_hi))

    # 保存 NPY/图
    u_samps, f_samps = pred_list
    if u_samps.ndim == 3: u_samps = u_samps[..., 0]
    if f_samps.ndim == 3: f_samps = f_samps[..., 0]
    _save_npy(outdir / "u_mean.npy", u_samps.mean(axis=0))
    _save_npy(outdir / "u_std.npy", u_samps.std(axis=0))
    _save_npy(outdir / "f_mean.npy", f_samps.mean(axis=0))
    _save_npy(outdir / "f_std.npy", f_samps.std(axis=0))
    print(f"[Save] 已保存 NPY 到 {outdir}/ (u_mean/u_std/f_mean/f_std)")

    if (mode == "synthetic") and (not no_plots):
        save_posterior_slices(pred_list, N_val_1d=N_val_1d, lb=lb, ub=ub, outdir=str(outdir))

    # 保存 .pth 工件到项目级 model/ 目录
    save_pth_artifacts(samples_flat, nets, params_templates, shapes, group_lens, kinds, model_dir)

    print("\n[Done] 训练与预测完成。")


# ===========================================
# 8) CLI（可选）：当作脚本直接运行
# ===========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="B-PINN 3D HMC Runner")
    parser.add_argument("--mode", choices=["synthetic", "file"], default="synthetic")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--model-dir", type=str, default="model")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--num-samples", type=int, default=CFG_NS)
    parser.add_argument("--burn", type=int, default=CFG_BURN)
    parser.add_argument("--L", type=int, default=CFG_L, dest="num_steps_per_sample")
    parser.add_argument("--step-size", type=float, default=CFG_STEP_SIZE)
    parser.add_argument("--N-tr-u", type=int, default=CFG_NTRU, dest="N_tr_u")
    parser.add_argument("--N-tr-f", type=int, default=CFG_NTRF, dest="N_tr_f")
    parser.add_argument("--N-val-1d", type=int, default=CFG_NVAL, dest="N_val_1d")
    parser.add_argument("--lb", type=float, default=CFG_LB)
    parser.add_argument("--ub", type=float, default=CFG_UB)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_hmc_pipeline(
        mode=args.mode,
        data_dir=args.data_dir,
        outdir=args.outdir,
        num_samples=args.num_samples, burn=args.burn, L=args.num_steps_per_sample, step_size=args.step_size,
        N_tr_u=args.N_tr_u, N_tr_f=args.N_tr_f, N_val_1d=args.N_val_1d,
        lb=args.lb, ub=args.ub, no_plots=args.no_plots,
        fast=args.fast, model_dir=args.model_dir,
    )
# ===========================================
# 9) 载入已训练模型/后验样本（从 model/ 目录）
# ===========================================
def load_posterior_samples(model_dir: Path = ROOT / "model"):
    """
    加载保存的后验样本与元信息。
    返回: samples_flat(torch.Tensor), shapes(list[tuple]), group_lens(list[int]), n_params_single(int)
    """
    model_dir = Path(model_dir)
    meta_path = model_dir / "posterior_samples.pth"
    if not meta_path.exists():
        raise FileNotFoundError(f"未找到 {meta_path}，请先运行 HMC 管线以生成该文件。")
    meta = torch.load(meta_path, map_location=device)
    samples_flat = meta["samples_flat"].to(device=device, dtype=DTYPE)
    shapes = meta["shapes"]
    group_lens = meta["group_lens"]
    nps = int(meta.get("n_params_single", n_params_single))
    return samples_flat, shapes, group_lens, nps


def load_posterior_mean_model(model_dir: Path = ROOT / "model"):
    """
    返回加载了“后验均值权重”的网络（仅返回第一个 Net）。
    优先使用 posterior_mean_state_dict.pth；若无则从 posterior_samples.pth 计算均值并回填权重。
    """
    model_dir = Path(model_dir)
    # 1) 如果直接保存了 state_dict，优先加载
    state_path = model_dir / "posterior_mean_state_dict.pth"
    if state_path.exists():
        nets = build_nets()
        net = nets[0].to(device=device, dtype=DTYPE)
        sd = torch.load(state_path, map_location=device)
        net.load_state_dict(sd)
        net.eval()
        print(f"[Load] 已加载 {state_path}")
        return net

    # 2) 否则退化为：从 posterior_samples 取均值并回填
    samples_flat, shapes, group_lens, nps = load_posterior_samples(model_dir)
    vec_mean = samples_flat.mean(dim=0)
    # 重新构建一个网络并写回参数
    nets = build_nets()
    net = nets[0].to(device=device, dtype=DTYPE)
    _, per_net_params = _unflatten_vector(vec_mean, shapes, group_lens, nps)
    if len(per_net_params) == 0:
        raise RuntimeError("后验样本中未包含网络参数（仅 θ?）。无法回填到网络。")
    pp = per_net_params[0]

    # 兼容“手写 functional”的 8 张量顺序
    ok = hasattr(net, "l1") and hasattr(net, "l2") and hasattr(net, "l3") and hasattr(net, "l4") and (len(pp) == 8)
    if not ok:
        raise RuntimeError("当前网络结构或样本模板与手写映射不匹配，无法自动回填。")

    with torch.no_grad():
        net.l1.weight.copy_(pp[0]); net.l1.bias.copy_(pp[1])
        net.l2.weight.copy_(pp[2]); net.l2.bias.copy_(pp[3])
        net.l3.weight.copy_(pp[4]); net.l3.bias.copy_(pp[5])
        net.l4.weight.copy_(pp[6]); net.l4.bias.copy_(pp[7])
    net.eval()
    print("[Load] 从 posterior_samples.pth 计算均值并已回填网络权重。")
    return net
