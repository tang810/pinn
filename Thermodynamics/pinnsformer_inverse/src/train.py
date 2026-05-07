import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
import matplotlib.pyplot as plt
from . import data_loader
from .physics import ThermoElasticPINNLoss
from .model import PINNsformer


def _plot_k_comparison(true_k, inferred_k, save_path):
    """Plot a balanced comparison chart that highlights closeness without visual exaggeration."""
    delta = inferred_k - true_k
    rel_err = abs(delta) / (abs(true_k) + 1e-12) * 100.0
    agreement = max(0.0, 100.0 - rel_err)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # Left: absolute k values with full-scale baseline (avoids exaggeration).
    labels = ['True k', 'Inferred k']
    values = [true_k, inferred_k]
    bars = ax1.bar(labels, values, color=['#1f77b4', '#2ca02c'], width=0.55, edgecolor='black', linewidth=1.2)
    ax1.set_title('Absolute Value Comparison')
    ax1.set_ylabel('k')
    ax1.set_ylim(0.0, max(values) * 1.25)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{v:.6f}', ha='center', va='bottom')

    # Right: relative error against tolerance bands.
    ax2.axhspan(0, 5, color='#d9f2d9', alpha=0.8, label='Excellent (<=5%)')
    ax2.axhspan(5, 10, color='#fff4cc', alpha=0.8, label='Good (5%-10%)')
    ax2.axhspan(10, 20, color='#fde2e2', alpha=0.8, label='Fair (10%-20%)')
    ax2.bar(['Relative Error'], [rel_err], color='#ff7f0e', width=0.45, edgecolor='black', linewidth=1.2)
    ax2.set_title('Relative Error View')
    ax2.set_ylabel('Error (%)')
    ax2.set_ylim(0, max(20.0, rel_err * 1.8))
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.text(
        0.0,
        rel_err + max(0.4, rel_err * 0.06),
        f'{rel_err:.2f}%\nAgreement: {agreement:.2f}%',
        ha='center',
        va='bottom',
        fontweight='bold'
    )
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.95)

    fig.suptitle('True vs Final Inferred k', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.01, f'Abs error = {abs(delta):.6f} | Delta k = {delta:+.6f}', ha='center')
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据准备
    print("Preparing training data points using LHS...")
    data = data_loader.get_pinn_domain_points(args)
    res_pts = data["res"]
    ic_pts = data["ic"]
    bc_tensors_dict = data["bc"]

    print("Loading observed temperature data...")
    x_star, y_star, z_star, t_star, T_star = data_loader.get_evaluation_data(args)
    data_pts = torch.cat([x_star, y_star, z_star, t_star], dim=-1)
    T_data = torch.tensor(T_star, dtype=torch.float32).view(-1, 1)

    print(f"res_pts: {res_pts.shape}")
    print(f"ic_pts: {ic_pts.shape}")
    for name, tensor in bc_tensors_dict.items():
        print(f"bc_{name}: {tensor.shape}")
    print(f"data_pts: {data_pts.shape}")

    # Move to device
    res_pts = res_pts.to(device)
    ic_pts = ic_pts.to(device)
    for name in bc_tensors_dict:
        bc_tensors_dict[name] = bc_tensors_dict[name].to(device)
    data_pts = data_pts.to(device)
    T_data = T_data.to(device)

    # 温度归一化
    T_mean = T_data.mean()
    T_std = T_data.std()
    T_data = (T_data - T_mean) / T_std
    args.T_mean = T_mean.item()
    args.T_std = T_std.item()
    print(f"Temperature normalized: mean={T_mean:.2f}, std={T_std:.2f}")

    # 模型
    model = PINNsformer(
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        N=args.n_layers,
        heads=args.n_heads,
        T0=args.t_0
    ).to(device)

    # ---------- 加载预训练模型（如果指定） ----------
    pretrained_path = getattr(args, 'pretrained_model_path', None)
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        pretrained_ckpt = torch.load(pretrained_path, map_location=device)
        if "model_state_dict" in pretrained_ckpt:
            model.load_state_dict(pretrained_ckpt["model_state_dict"])
        else:
            model.load_state_dict(pretrained_ckpt)
        print("Pretrained model loaded successfully.")
    else:
        print("No pretrained model specified, training from scratch.")

    # ---------- 反演参数处理 ----------
    inverse_mode = getattr(args, 'inverse_mode', False)
    infer_params = getattr(args, 'infer_params', []) if inverse_mode else []
    loss_computer = ThermoElasticPINNLoss(args, trainable_params=infer_params, T_mean=T_mean, T_std=T_std).to(device)

    # ---------- 优化器：为模型和反演参数设置不同学习率 ----------
    if inverse_mode and infer_params:
        optimizer = AdamW([
            {'params': model.parameters()},
            {'params': loss_computer.parameters(), 'lr': args.learning_rate * 0.1}
        ], lr=args.learning_rate, weight_decay=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    batch_size = args.batch_size
    data_batch_size = args.data_batch_size

    log_file = open("training_log.txt", "a")
    log_file.write("\n" + "="*60 + "\n")
    log_file.write(f"New training run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if inverse_mode:
        log_file.write(f"Inverse mode: inferring {infer_params}\n")
    log_file.write("="*60 + "\n")

    # =========================
    # 分阶段训练设置
    # =========================
    phase1_epochs = 0   # 直接进入全物理训练，因为已有预训练模型
    # 保存所有损失权重的原始值
    original_weights = {
        'lambda_pde': args.lambda_pde,
        'lambda_elastic': args.lambda_elastic,
        'lambda_ic': args.lambda_ic,
        'lambda_ic_disp': args.lambda_ic_disp,
        'lambda_bc': args.lambda_bc,
        'lambda_bc_disp': args.lambda_bc_disp,
        'lambda_data': args.lambda_data,
        'lambda_rad': args.lambda_rad,
        'lambda_heatflux': args.lambda_heatflux,
    }

    # =========================
    # 断点续训
    # =========================
    start_epoch = 0
    resume_epoch = getattr(args, 'resume_epoch', 0)
    if resume_epoch > 0:
        checkpoint_path = args.model_save_path.replace(".pt", f"_epoch{resume_epoch}.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            loss_computer.load_state_dict(checkpoint['loss_computer_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Checkpoint {checkpoint_path} not found, starting from scratch.")

    # 根据恢复的epoch设置损失权重
    if start_epoch >= phase1_epochs:
        # Phase 2：恢复所有原始权重
        loss_computer.lambda_pde = original_weights['lambda_pde']
        loss_computer.lambda_elastic = original_weights['lambda_elastic']
        loss_computer.lambda_ic = original_weights['lambda_ic']
        loss_computer.lambda_ic_disp = original_weights['lambda_ic_disp']
        loss_computer.lambda_bc = original_weights['lambda_bc']
        loss_computer.lambda_bc_disp = original_weights['lambda_bc_disp']
        loss_computer.lambda_data = original_weights['lambda_data']
        loss_computer.lambda_rad = original_weights['lambda_rad']
        loss_computer.lambda_heatflux = original_weights['lambda_heatflux']
        print("Starting from phase 2 (full physics).")
    else:
        # Phase 1：所有物理项权重置零，仅保留数据项
        loss_computer.lambda_pde = 0.0
        loss_computer.lambda_elastic = 0.0
        loss_computer.lambda_ic = 0.0
        loss_computer.lambda_ic_disp = 0.0
        loss_computer.lambda_bc = 0.0
        loss_computer.lambda_bc_disp = 0.0
        loss_computer.lambda_data = 1.0
        loss_computer.lambda_rad = 0.0
        loss_computer.lambda_heatflux = 0.0
        print("Starting from phase 1 (data only).")

    print("\nStart training...\n")

    # 创建结果目录用于保存损失图
    results_dir = os.path.dirname(args.animation_save_path)
    os.makedirs(results_dir, exist_ok=True)

    # 用于记录损失历史的列表
    loss_history = {'total': [], 'data': [], 'heat': [], 'heatflux': [], 'rad': []}
    param_history = {'k': []} if 'k' in infer_params else {}

    # ===============================================================
    # Training Loop
    # ===============================================================
    for epoch in range(start_epoch, args.epochs):
        # 阶段切换：从 phase1 进入 phase2
        if epoch == phase1_epochs and start_epoch < phase1_epochs:
            loss_computer.lambda_pde = original_weights['lambda_pde']
            loss_computer.lambda_elastic = original_weights['lambda_elastic']
            loss_computer.lambda_ic = original_weights['lambda_ic']
            loss_computer.lambda_ic_disp = original_weights['lambda_ic_disp']
            loss_computer.lambda_bc = original_weights['lambda_bc']
            loss_computer.lambda_bc_disp = original_weights['lambda_bc_disp']
            loss_computer.lambda_data = original_weights['lambda_data']
            loss_computer.lambda_rad = original_weights['lambda_rad']
            loss_computer.lambda_heatflux = original_weights['lambda_heatflux']
            print(f"\n>>> Switching to phase 2 (full physics) at epoch {epoch+1} <<<\n")

        model.train()
        loss_computer.train()
        epoch_start = time.time()

        # ========== 根据当前阶段选择不同的数据采样策略 ==========
        if epoch < phase1_epochs:
            # ---------- Phase 1: 纯数据训练 ----------
            data_perm = torch.randperm(data_pts.size(0), device=device)
            num_batches = (data_pts.size(0) + data_batch_size - 1) // data_batch_size

            total_loss_epoch = 0.0
            valid_batches = 0
            total_grad_norm = 0.0
            loss_components = {
                "heat": 0.0, "elastic": 0.0, "ic": 0.0, "ic_disp": 0.0,
                "bc_temp": 0.0, "bc_disp": 0.0, "data": 0.0, "heatflux": 0.0,
                "rad": 0.0
            }

            for batch_idx in range(num_batches):
                optimizer.zero_grad(set_to_none=True)

                start = batch_idx * data_batch_size
                end = min(start + data_batch_size, data_pts.size(0))
                idx_data = data_perm[start:end]
                batch_data_pts = data_pts[idx_data]
                batch_T = T_data[idx_data]

                batch_res = None
                batch_ic = None
                batch_bc_dict = None

                losses = loss_computer(
                    model,
                    res_tensor=batch_res,
                    ic_tensor=batch_ic,
                    bc_tensors=batch_bc_dict,
                    data_tensor=batch_data_pts,
                    T_data=batch_T
                )
                total_loss = losses["total"]

                if not torch.isfinite(total_loss):
                    print(f"⚠️ NaN detected at Epoch {epoch+1}, Batch {batch_idx} (loss={total_loss.item():.2e})")
                    continue

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
                optimizer.step()

                total_loss_epoch += total_loss.item()
                total_grad_norm += grad_norm.item()
                valid_batches += 1
                for key in loss_components:
                    if key in losses:
                        loss_components[key] += losses[key].item()
                    else:
                        loss_components[key] += 0.0

        else:
            # ---------- Phase 2: 全物理训练 ----------
            res_perm = torch.randperm(res_pts.size(0), device=device)
            ic_perm = torch.randperm(ic_pts.size(0), device=device)
            data_perm = torch.randperm(data_pts.size(0), device=device)

            bc_perms = {}
            for name, tensor in bc_tensors_dict.items():
                bc_perms[name] = torch.randperm(tensor.size(0), device=device)

            num_res_batches = (res_pts.size(0) + batch_size - 1) // batch_size
            num_ic_batches = (ic_pts.size(0) + batch_size - 1) // batch_size
            num_data_batches = (data_pts.size(0) + data_batch_size - 1) // data_batch_size
            num_bc_batches = {}
            for name, tensor in bc_tensors_dict.items():
                num_bc_batches[name] = (tensor.size(0) + batch_size - 1) // batch_size
            num_batches = max(num_res_batches, num_ic_batches, num_data_batches, *num_bc_batches.values())

            total_loss_epoch = 0.0
            valid_batches = 0
            total_grad_norm = 0.0
            loss_components = {
                "heat": 0.0, "elastic": 0.0, "ic": 0.0, "ic_disp": 0.0,
                "bc_temp": 0.0, "bc_disp": 0.0, "data": 0.0, "heatflux": 0.0,
                "rad": 0.0
            }

            def get_batch_indices(perm, size, batch_sz, idx):
                start = (idx * batch_sz) % size
                end = min(start + batch_sz, size)
                return perm[start:end]

            for batch_idx in range(num_batches):
                optimizer.zero_grad(set_to_none=True)

                idx_res = get_batch_indices(res_perm, res_pts.size(0), batch_size, batch_idx)
                batch_res = res_pts[idx_res].clone().detach().requires_grad_(True) if len(idx_res) > 0 else None

                idx_ic = get_batch_indices(ic_perm, ic_pts.size(0), batch_size, batch_idx)
                batch_ic = ic_pts[idx_ic].clone().detach().requires_grad_(True) if len(idx_ic) > 0 else None

                batch_bc_dict = {}
                for name, tensor in bc_tensors_dict.items():
                    perm = bc_perms[name]
                    idx = get_batch_indices(perm, tensor.size(0), batch_size, batch_idx)
                    if len(idx) > 0:
                        batch_bc_dict[name] = tensor[idx].clone().detach().requires_grad_(True)

                idx_data = get_batch_indices(data_perm, data_pts.size(0), data_batch_size, batch_idx)
                batch_data_pts = data_pts[idx_data] if len(idx_data) > 0 else None
                batch_T = T_data[idx_data] if len(idx_data) > 0 else None

                if (batch_res is None and batch_ic is None and not batch_bc_dict
                        and batch_data_pts is None):
                    continue

                losses = loss_computer(
                    model,
                    res_tensor=batch_res,
                    ic_tensor=batch_ic,
                    bc_tensors=batch_bc_dict if batch_bc_dict else None,
                    data_tensor=batch_data_pts,
                    T_data=batch_T
                )
                total_loss = losses["total"]

                if not torch.isfinite(total_loss):
                    print(f"⚠️ NaN detected at Epoch {epoch+1}, Batch {batch_idx} (loss={total_loss.item():.2e})")
                    continue

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
                optimizer.step()

                total_loss_epoch += total_loss.item()
                total_grad_norm += grad_norm.item()
                valid_batches += 1
                for key in loss_components:
                    if key in losses:
                        loss_components[key] += losses[key].item()
                    else:
                        loss_components[key] += 0.0

        # 统计并输出
        if valid_batches == 0:
            print("❌ All batches invalid. Training stopped.")
            break

        avg_loss = total_loss_epoch / valid_batches
        avg_grad_norm = total_grad_norm / valid_batches
        avg_components = {k: v / valid_batches for k, v in loss_components.items()}

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # 基础日志
        log_msg = (
            f"Epoch {epoch+1:4d} | "
            f"Time: {epoch_time:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Loss: {avg_loss:.3e} | "
            f"Grad: {avg_grad_norm:.3e} | "
            f"Heat: {avg_components['heat']:.3e} | "
            f"Elastic: {avg_components['elastic']:.3e} | "
            f"Data: {avg_components['data']:.3e} | "
            f"HeatFlux: {avg_components['heatflux']:.3e}"
        )
        print(log_msg)

        # 记录损失历史
        loss_history['total'].append(avg_loss)
        loss_history['data'].append(avg_components['data'])
        loss_history['heat'].append(avg_components['heat'])
        loss_history['heatflux'].append(avg_components['heatflux'])
        loss_history['rad'].append(avg_components['rad'])

        # 记录反演参数
        if inverse_mode and infer_params:
            param_str = []
            for name in infer_params:
                if hasattr(loss_computer, name):
                    val = getattr(loss_computer, name).item()
                    param_str.append(f"{name}: {val:.6f}")
                    if name in param_history:
                        param_history[name].append(val)
            if param_str:
                print("Inferred: " + ", ".join(param_str))

        log_file.write(
            f"Epoch {epoch+1}, Loss={avg_loss:.6e}, Grad={avg_grad_norm:.6e}, "
            f"Heat={avg_components['heat']:.6e}, Elastic={avg_components['elastic']:.6e}, "
            f"Data={avg_components['data']:.6e}, HeatFlux={avg_components['heatflux']:.6e}, "
            f"LR={current_lr:.6e}\n"
        )
        log_file.flush()

        # 每 10 个 epoch 保存一次损失曲线图
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            # 总损失
            axes[0, 0].semilogy(range(1, len(loss_history['total'])+1), loss_history['total'], 'b-')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].grid(True)
            # 数据损失
            axes[0, 1].semilogy(range(1, len(loss_history['data'])+1), loss_history['data'], 'g-')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Data Loss')
            axes[0, 1].grid(True)
            # 热方程损失
            axes[1, 0].semilogy(range(1, len(loss_history['heat'])+1), loss_history['heat'], 'r-')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Heat Equation Loss')
            axes[1, 0].grid(True)
            # 反演参数 k 的曲线（如果有）
            if 'k' in param_history:
                axes[1, 1].plot(range(1, len(param_history['k'])+1), param_history['k'], 'm-')
                axes[1, 1].axhline(y=args.k, color='k', linestyle='--', linewidth=1.2, label=f'True k={args.k:.4f}')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('k')
                axes[1, 1].set_title('Inferred k (with True k)')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            else:
                # 如果没有反演参数，显示热流或辐射损失
                axes[1, 1].semilogy(range(1, len(loss_history['heatflux'])+1), loss_history['heatflux'], 'c-', label='HeatFlux')
                axes[1, 1].semilogy(range(1, len(loss_history['rad'])+1), loss_history['rad'], 'y-', label='Radiation')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].set_title('Boundary Losses')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'loss_curve_epoch{epoch+1}.png'), dpi=150)
            plt.close(fig)
            print(f"✓ Loss curve saved at epoch {epoch+1}")

        # 保存 checkpoint
        if (epoch + 1) % 50 == 0:
            os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss_computer_state_dict": loss_computer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss
            }, args.model_save_path.replace(".pt", f"_epoch{epoch+1}.pt"))
            print(f"Checkpoint saved at epoch {epoch+1}")

    log_file.close()
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "loss_computer_state_dict": loss_computer.state_dict() if inverse_mode else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, args.model_save_path)
    print(f"\nFinal model saved at {args.model_save_path}")

    # 最后保存一次完整的损失图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].semilogy(range(1, len(loss_history['total'])+1), loss_history['total'], 'b-')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True)
    axes[0, 1].semilogy(range(1, len(loss_history['data'])+1), loss_history['data'], 'g-')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Data Loss')
    axes[0, 1].grid(True)
    axes[1, 0].semilogy(range(1, len(loss_history['heat'])+1), loss_history['heat'], 'r-')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Heat Equation Loss')
    axes[1, 0].grid(True)
    if 'k' in param_history:
        axes[1, 1].plot(range(1, len(param_history['k'])+1), param_history['k'], 'm-')
        axes[1, 1].axhline(y=args.k, color='k', linestyle='--', linewidth=1.2, label=f'True k={args.k:.4f}')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('k')
        axes[1, 1].set_title('Inferred k (with True k)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].semilogy(range(1, len(loss_history['heatflux'])+1), loss_history['heatflux'], 'c-', label='HeatFlux')
        axes[1, 1].semilogy(range(1, len(loss_history['rad'])+1), loss_history['rad'], 'y-', label='Radiation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Boundary Losses')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_curve_final.png'), dpi=150)
    plt.close()
    print(f"✓ Final loss curve saved to {os.path.join(results_dir, 'loss_curve_final.png')}")

    # Save true-vs-inferred comparison for k (final value)
    if inverse_mode and hasattr(loss_computer, 'k'):
        final_k = loss_computer.k.item()
        k_compare_path = os.path.join(results_dir, 'k_true_vs_inferred.png')
        _plot_k_comparison(args.k, final_k, k_compare_path)
        print(f"✓ k comparison plot saved to {k_compare_path} (true={args.k:.6f}, inferred={final_k:.6f})")
