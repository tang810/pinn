import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(X_star, u_star, u_pred, case_tag, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t_max = np.max(X_star[:, 3])
    tol = 1e-4
    idx_t = np.where(np.abs(X_star[:, 3] - t_max) < tol)[0]

    if len(idx_t) > 0:
        x_plot = X_star[idx_t, 0]
        y_plot = X_star[idx_t, 1]
        z_plot = X_star[idx_t, 2]
        u_exact_plot = u_star[idx_t, 0]
        u_pred_plot = u_pred[idx_t, 0]
        suffix = f"(t = {t_max:.2f})"
    else:
        x_plot = X_star[:, 0]
        y_plot = X_star[:, 1]
        z_plot = X_star[:, 2]
        u_exact_plot = u_star[:, 0]
        u_pred_plot = u_pred[:, 0]
        suffix = "(All times)"

    fig1 = plt.figure(figsize=(16, 7))
    ax1 = fig1.add_subplot(121, projection="3d")
    sc1 = ax1.scatter(x_plot, y_plot, z_plot, c=u_exact_plot, cmap="jet", s=10, alpha=0.9)
    ax1.set_title(f"Exact Surface Temp {suffix}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    fig1.colorbar(sc1, ax=ax1, pad=0.1, fraction=0.03, label="Temperature (K)")

    ax2 = fig1.add_subplot(122, projection="3d")
    sc2 = ax2.scatter(x_plot, y_plot, z_plot, c=u_pred_plot, cmap="jet", s=10, alpha=0.9, vmin=u_exact_plot.min(), vmax=u_exact_plot.max())
    ax2.set_title(f"Predicted Surface Temp {suffix}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    fig1.colorbar(sc2, ax=ax2, pad=0.1, fraction=0.03, label="Temperature (K)")

    plt.tight_layout()
    fig1.savefig(os.path.join(out_dir, f"fig1_3d_surfaces_{case_tag}.png"), dpi=250, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plt.figure(figsize=(8, 7))
    ax3 = fig2.add_subplot(111)
    ax3.scatter(u_star, u_pred, c="royalblue", s=2, alpha=0.3)
    min_val = np.min([u_star.min(), u_pred.min()])
    max_val = np.max([u_star.max(), u_pred.max()])
    ax3.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y=x)")
    ax3.set_title("Prediction vs Exact")
    ax3.set_xlabel("Exact Temperature (K)")
    ax3.set_ylabel("Predicted Temperature (K)")
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.6)
    fig2.savefig(os.path.join(out_dir, f"fig2_parity_{case_tag}.png"), dpi=250, bbox_inches="tight")
    plt.close(fig2)

    fig3 = plt.figure(figsize=(8, 6))
    ax4 = fig3.add_subplot(111)
    errors = (u_pred - u_star).flatten()
    ax4.hist(errors, bins=50, color="coral", edgecolor="black", alpha=0.8)
    ax4.set_title("Error Histogram (Predicted - Exact)")
    ax4.set_xlabel("Temperature Error (K)")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, linestyle=":", alpha=0.6)
    fig3.savefig(os.path.join(out_dir, f"fig3_error_hist_{case_tag}.png"), dpi=250, bbox_inches="tight")
    plt.close(fig3)


def save_matrix_csv(all_results, out_path):
    header = (
        "n_sensors,n_time_steps,ax,aperp,err_ax,err_aperp,error_u,"
        "ax_noisy,aperp_noisy,err_ax_noisy,err_aperp_noisy,error_u_noisy,elapsed_sec"
    )
    rows = []
    for r in all_results:
        rows.append([
            r["n_sensors"],
            r["n_time_steps"],
            r["ax"],
            r["aperp"],
            r["err_ax"],
            r["err_aperp"],
            r["error_u"],
            r["ax_noisy"],
            r["aperp_noisy"],
            r["err_ax_noisy"],
            r["err_aperp_noisy"],
            r["error_u_noisy"],
            r["elapsed_sec"],
        ])
    np.savetxt(out_path, np.array(rows), delimiter=",", header=header, comments="", fmt="%.10e")
