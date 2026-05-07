
# ================================
# File: src/viz.py
# ================================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib import animation

# ---- 可视化（严格按你提供的实现整合，并将保存路径落到 results 下） ----
def plot_loss_curve(loss_history,save_path=None):
    epochs = np.arange(1, len(loss_history.get('total', [])) + 1)
    plt.figure(figsize=(8, 4))
    if len(epochs) > 0:
        plt.plot(epochs, loss_history['total'], label='Total Loss')
        plt.plot(epochs, loss_history['data'], label='Data Loss')
        plt.plot(epochs, loss_history['pde'], label='PDE Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        #print(f"✅ Loss curve saved to {save_path}")
    plt.close()
    
def plot_error_heatmap(model, dataloader, scalers, device='cpu', inverse=False, bins=40, sigma=1.0, save_path=None):
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            X = batch['X'].to(device)
            Y_true = batch['Y_target'].cpu().numpy()
            Y_pred = model(X).cpu().numpy()
            all_true.append(Y_true)
            all_pred.append(Y_pred)

    true_all = np.concatenate(all_true, axis=0)
    pred_all = np.concatenate(all_pred, axis=0)

    if inverse:
        true_wt = scalers['wt'].inverse_transform(true_all[:, 0].reshape(-1, 1)).flatten()
        true_lt = scalers['lt'].inverse_transform(true_all[:, 1].reshape(-1, 1)).flatten()
        pred_wt = scalers['wt'].inverse_transform(pred_all[:, 0].reshape(-1, 1)).flatten()
        pred_lt = scalers['lt'].inverse_transform(pred_all[:, 1].reshape(-1, 1)).flatten()
    else:
        true_wt, true_lt = true_all[:, 0], true_all[:, 1]
        pred_wt, pred_lt = pred_all[:, 0], pred_all[:, 1]

    error = np.sqrt((pred_wt - true_wt)**2 + (pred_lt - true_lt)**2)

    heatmap, xedges, yedges = np.histogram2d(true_wt, true_lt, bins=bins, weights=error)
    counts, _, _ = np.histogram2d(true_wt, true_lt, bins=[xedges, yedges])
    avg_error = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts > 0)

    smoothed_error = gaussian_filter(avg_error.T, sigma=sigma)

    plt.figure(figsize=(7, 5))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(smoothed_error, extent=extent, aspect='auto', origin='lower', cmap='magma_r')
    plt.colorbar(label='Smoothed Error')
    plt.xlabel('True w_t')
    plt.ylabel('True l_t')
    plt.title('Smoothed Error Heatmap (w_t & l_t)')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        #print(f"✅ Heatmap saved to {save_path}")
    plt.close()

def plot_distribution_comparison(model, dataloader, scalers, device='cpu', inverse=True, save_path=None):
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            X = batch['X'].to(device)
            Y_true = batch['Y_target'].cpu().numpy()
            Y_pred = model(X).cpu().numpy()
            all_true.append(Y_true)
            all_pred.append(Y_pred)

    true_all = np.concatenate(all_true, axis=0)
    pred_all = np.concatenate(all_pred, axis=0)

    if inverse:
        true_wt = scalers['wt'].inverse_transform(true_all[:, 0].reshape(-1, 1)).flatten() * 1e6
        true_lt = scalers['lt'].inverse_transform(true_all[:, 1].reshape(-1, 1)).flatten() * 1e6
        pred_wt = scalers['wt'].inverse_transform(pred_all[:, 0].reshape(-1, 1)).flatten() * 1e6
        pred_lt = scalers['lt'].inverse_transform(pred_all[:, 1].reshape(-1, 1)).flatten() * 1e6
    else:
        true_wt, true_lt = true_all[:, 0], true_all[:, 1]
        pred_wt, pred_lt = pred_all[:, 0], pred_all[:, 1]

    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(true_wt, color='black', label='True w_t', kde=True, stat='density', ax=axes[0])
    sns.histplot(pred_wt, color='blue', label='Predicted w_t', kde=True, stat='density', ax=axes[0], alpha=0.6)
    axes[0].set_title('w_t Distribution (μm)')
    axes[0].legend()

    sns.histplot(true_lt, color='black', label='True l_t', kde=True, stat='density', ax=axes[1])
    sns.histplot(pred_lt, color='blue', label='Predicted l_t', kde=True, stat='density', ax=axes[1], alpha=0.6)
    axes[1].set_title('l_t Distribution (μm)')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        #print(f"✅ Saved parameter distribution comparison to {save_path}")
    plt.close()

def extract_monotonic_samples_exhaustive(model, dataloader, scalers, device='cpu'):
    model.eval()

    wt_all, lt_all, wt_pred_all, lt_pred_all = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            X = batch['X'].to(device)
            Y_true = batch['Y_target'].cpu().numpy()
            Y_pred = model(X).cpu().numpy()

            true_wt = scalers['wt'].inverse_transform(Y_true[:, 0].reshape(-1, 1)).flatten()
            true_lt = scalers['lt'].inverse_transform(Y_true[:, 1].reshape(-1, 1)).flatten()
            pred_wt = scalers['wt'].inverse_transform(Y_pred[:, 0].reshape(-1, 1)).flatten()
            pred_lt = scalers['lt'].inverse_transform(Y_pred[:, 1].reshape(-1, 1)).flatten()

            wt_all.extend(true_wt)
            lt_all.extend(true_lt)
            wt_pred_all.extend(pred_wt)
            lt_pred_all.extend(pred_lt)

    all_samples = list(zip(wt_all, lt_all, wt_pred_all, lt_pred_all))
    all_samples_sorted = sorted(set(all_samples), key=lambda x: (x[0], x[1]))

    max_wt = max(wt_all)
    max_lt = max(lt_all)
    selected = []
    last_wt, last_lt = -np.inf, -np.inf

    for w, l, wp, lp in all_samples_sorted:
        if np.isclose(last_wt, max_wt) and np.isclose(last_lt, max_lt):
            break
        if last_wt < max_wt and last_lt < max_lt:
            if w > last_wt and l > last_lt:
                selected.append((w, l, wp, lp))
                last_wt, last_lt = w, l
        elif np.isclose(last_wt, max_wt) and last_lt < max_lt:
            if np.isclose(w, last_wt) and l > last_lt:
                selected.append((w, l, wp, lp))
                last_wt, last_lt = w, l
        elif np.isclose(last_lt, max_lt) and last_wt < max_wt:
            if w > last_wt and np.isclose(l, last_lt):
                selected.append((w, l, wp, lp))
                last_wt, last_lt = w, l

    if len(selected) == 0:
        raise ValueError("❌ 没有找到可递增采样数据")

    wt_list, lt_list, wt_pred_list, lt_pred_list = zip(*selected)
    return list(wt_list), list(lt_list), list(wt_pred_list), list(lt_pred_list)

def create_beam_animation(wt, lt, wt_pred, lt_pred,
                          d_fixed=26e-6,
                          save_gif_path=None,
                          show=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    wt = np.array(wt)
    lt = np.array(lt)
    wt_pred = np.array(wt_pred)
    lt_pred = np.array(lt_pred)

    area_true = wt * lt
    area_pred = wt_pred * lt_pred
    error = (np.abs(area_pred - area_true) / area_true) * 100  # %

    cmap = plt.get_cmap('Spectral')
    norm = LogNorm(vmin=max(error.min(), 1e-6), vmax=error.max() + 1e-6)
    sm = ScalarMappable(cmap=cmap, norm=norm)

    def plot_beam(ax, w, l, d, color='gray', alpha=1.0):
        y = [0, w, w, 0, 0]
        x = [0, 0, l, l, 0] 
        z = [0] * 5
        ax.plot3D(x, y, z, color=color, alpha=alpha)
        ax.plot3D(x, y, [d] * 5, color=color, alpha=alpha)
        for i in range(4):
            ax.plot3D([x[i], x[i]], [y[i], y[i]], [0, d], color=color, alpha=alpha)

    def update(i):
        ax.cla()
        w_t = wt[i] * 1e6
        l_t = lt[i] * 1e6
        w_tp = wt_pred[i] * 1e6
        l_tp = lt_pred[i] * 1e6
        thick = d_fixed * 1e6

        margin = 0.2
        ax.set_xlim(0, max(l_t, l_tp) * (1 + margin))
        ax.set_ylim(0, max(w_t, w_tp) * (1 + margin))
        ax.set_zlim(0, thick * (1 + margin))
        ax.set_box_aspect([100, 2, 2.5])
        ax.set_axis_off()

        azim = 45 + i * 1.5
        elev = 20
        ax.view_init(elev=elev, azim=azim)

        ax.set_title(
            f"Sample {i+1} | Error = {error[i]:.1f}%\n"
            f"{'Parameter':<10} {'True':>8}    {'Pred':>8}\n"
            f"{'wt (μm)':<10} {w_t:>8.1f}  {w_tp:>8.1f}\n"
            f"{'lt (μm)':<10} {l_t:>8.1f} {l_tp:>8.1f}",
        )

        plot_beam(ax, w_t, l_t, thick, color='gray', alpha=0.8)
        color_pred = cmap(norm(error[i]))
        plot_beam(ax, w_tp, l_tp, thick, color=color_pred, alpha=0.9)

    ani = animation.FuncAnimation(fig, update, frames=len(wt), interval=200)

    if save_gif_path:
        os.makedirs(os.path.dirname(save_gif_path) or ".", exist_ok=True)
        ani.save(save_gif_path, writer='pillow', fps=15, dpi=200)
        #print(f"✅ Saved animation to: {save_gif_path}")

    if show:
        from IPython.display import HTML, display
        display(HTML(ani.to_jshtml()))

    plt.close()

def output_viz_data(model, test_loader, loss_history, scalers, folder_path='./results'):
    os.makedirs(folder_path, exist_ok=True)
    plot_loss_curve(loss_history, save_path=folder_path + '/1_loss_curve_data.png')
    plot_error_heatmap(model, test_loader, scalers, device='cpu', bins=40, sigma=1.2, save_path=folder_path + '/2_error_heatmap.png')
    plot_distribution_comparison(
        model,
        test_loader,
        scalers,
        device='cpu',
        save_path=folder_path + '/3_distribution_comparison.png'
    )
    wt_list, lt_list, wt_pred_list, lt_pred_list = extract_monotonic_samples_exhaustive(
        model, test_loader, scalers, device='cpu'
    )
    create_beam_animation(
        wt_list, lt_list, wt_pred_list, lt_pred_list,
        d_fixed=25e-6, save_gif_path=folder_path+'/4_beam_animation.gif', show=False
    )
    # 另一个聚焦动画（可选）
    # create_beam_object_focus_animation(...)  # 如需另存可打开