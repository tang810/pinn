import os
import sys
sys.path.append(".")
sys.path.append("..")
print(__file__)
sys.path.append(os.path.dirname(__file__))
sys.path.append("../scripts")


import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
import torch  # noqa
from torch.utils.data import DataLoader  # noqa
from dit4science.model.dit_diffusion import create_diffusion  # noqa
from download import find_model  # noqa
from dit4science.model.dit_models import DiT_models  # noqa
from dit4science.utils.bench import get_error  # noqa

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_fiels(data, ground):
    """
    data is of shape: (batch, channel, H, W).
    """
    # vmin = -0.1
    # vmax = 1
    bs, c, H, W = data.shape
    data = data.detach().cpu().numpy()
    ground = ground.detach().cpu().numpy()
    error = np.abs(data - ground)
    reso = 5
    fig, axs = plt.subplots(c, bs*3, figsize=(reso*bs*3, reso*c))
    for k in range(3):
        for i in range(c):
            for j in range(bs):
                # axs[i][j+1].axis("off")
                # axs[i][j+2].axis("off")
                # axs[i][j].axis("off")
                if (k == 0):
                    # axs[i][j*3].imshow(data[j, i, :, :], cmap="jet", interpolation="catrom", vmin=vmin, vmax=vmax)  # noqa
                    axs[i][j*3].imshow(
                        data[j, i, :, :], cmap="jet", interpolation="catrom")
                if (k == 1):
                    # axs[i][j*3+1].imshow(ground[j, i, :, :], cmap="jet", interpolation="catrom", vmin=vmin, vmax=vmax)  # noqa
                    axs[i][j*3+1].imshow(
                        ground[j, i, :, :], cmap="jet", interpolation="catrom")
                if (k == 2):
                    axs[i][j*3+2].imshow(error[j, i, :, :])
    return fig, axs


def inference(
        dataset, model, vae_cond, vae_res, plot_tar=True,
        latent_size=16, batch_size=1, num_sampling_steps=250,flag=False):
    loader = DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    total_error_p, total_error_ux, total_error_uy = 0, 0, 0
    total_samples = 0
    
    for idx, (x, ground) in enumerate(loader):
        x = x.to(device)
        ground = ground.to(device)

        diffusion = create_diffusion(str(num_sampling_steps))
        y = x.to(device)
        y = vae_cond.encode(y).sample()

        n = batch_size
        z = torch.randn(n, 4, latent_size, latent_size, device=device)

        model_kwargs = dict(y=y)
        model.eval()

        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )

        pred = vae_res.decode(samples)

        error_p = get_error(pred[:, 0, :, :], ground[:, 0, :, :])
        error_ux = get_error(pred[:, 1, :, :], ground[:, 1, :, :])
        error_uy = get_error(pred[:, 2, :, :], ground[:, 2, :, :])
        total_error_p += error_p* pred.shape[0]
        total_error_ux += error_ux* pred.shape[0]
        total_error_uy += error_uy* pred.shape[0]
        total_samples += pred.shape[0]
        
        idx = 0
        pred = pred.detach().cpu() if device != "cpu" else pred.detach()
        if not plot_tar:  # Plot prediction only
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            cmap = 'jet'
            for i in range(3):
                pcm = axs[i].imshow(np.rot90(pred[idx, i, :, :]), cmap=cmap)
                fig.colorbar(pcm, ax=axs[i], shrink=0.715)

            axs[0].set_title("Pressure Field $P$")
            axs[1].set_title("Velocity Field $U_{x}$")
            axs[2].set_title("Velocity Field $U_{y}$")

            axs[0].text(-0.3, 0.5, 'Prediction', va='center', rotation='vertical',
                        transform=axs[0].transAxes, fontsize=18, fontweight="bold")
        else: 
            fig, axs = plt.subplots(3, 3, figsize=(15, 15))
            cmap = 'jet'
            for i in range(3):
                pcm = axs[0][i].imshow(np.rot90(x[idx, i, :, :].cpu().numpy()), cmap=cmap)
                fig.colorbar(pcm, ax=axs[0][i], shrink=0.76)
            for i in range(3):
                pcm = axs[1][i].imshow(np.rot90(pred[idx, i, :, :]), cmap=cmap)
                fig.colorbar(pcm, ax=axs[1][i], shrink=0.76)
            for i in range(3):
                pcm = axs[2][i].imshow(np.rot90(ground.detach().cpu()[idx, i, :, :]), cmap=cmap)
                fig.colorbar(pcm, ax=axs[2][i], shrink=0.76)
            axs[0][0].set_title("Airfoil geometry")
            axs[0][1].set_title("Initial Horizontal Velocity")
            axs[0][2].set_title("Initial Parallel Velocity")
            
            axs[1][0].set_title("Pressure Field $P$")
            axs[1][1].set_title("Velocity Field $U_{x}$")
            axs[1][2].set_title("Velocity Field $U_{y}$")

            axs[2][0].set_title("Pressure Field $P$")
            axs[2][1].set_title("Velocity Field $U_{x}$")
            axs[2][2].set_title("Velocity Field $U_{y}$")
            
            axs[0][0].text(
                -0.3, 0.5, 'Input', va='center', rotation='vertical',
                transform=axs[0][0].transAxes, fontsize=18, fontweight="bold")
            
            axs[1][0].text(
                -0.3, 0.5, 'Prediction', va='center', rotation='vertical',
                transform=axs[1][0].transAxes, fontsize=18, fontweight="bold")
            axs[2][0].text(
                -0.3, 0.5, 'Ground Truth', va='center', rotation='vertical',
                transform=axs[2][0].transAxes, fontsize=18, fontweight="bold")

        fig.savefig(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "sample.png"
            ))
    avg_error_p = total_error_p / total_samples
    avg_error_ux = total_error_ux / total_samples
    avg_error_uy = total_error_uy / total_samples
    
    print(f"Average error_p:{avg_error_p:.4f}, error_ux{avg_error_ux:4f}, error_uy{avg_error_uy:4f} ")
    print(f"{avg_error_p:.4f}\n{avg_error_ux:4f}\n{avg_error_uy:4f} ")
    print("Inference finished")


if __name__ == "__main__":
    inference()
