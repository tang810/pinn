import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LightSource


def topozeko(bed, sur, output_path, d2=False, **kwargs):
    axes = kwargs.get("axes", "off")
    bed_colors = kwargs.get("bed_colors", 128)
    bed_colormap = kwargs.get("bed_colormap", "copper")
    bed_colormap_flipud = kwargs.get("bed_colormap_flipud", "on")
    bed_trans = kwargs.get("bed_trans", 1)
    cbar_colors = kwargs.get("cbar_colors", 128)
    d4_colormap = kwargs.get("D4_colormap", "jet")
    d4_colormap_flipud = kwargs.get("D4_colormap_flipud", "off")
    extra_dimension = kwargs.get("extra_dimension", "")
    light_orientation = kwargs.get("light_orientation", [-90, 45])
    sur_color = kwargs.get("sur_color", [1, 1, 1])
    sur_trans = kwargs.get("sur_trans", 1)
    tick = kwargs.get("tick", "on")
    tick_size = kwargs.get("tick_size", 18)
    title = kwargs.get("title", "")
    title_size = kwargs.get("title_size", 22)
    vertical_scaling = kwargs.get("vertical_scaling", 1)
    view_orientation = kwargs.get("view_orientation", [0, 45])
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    zlabel = kwargs.get("zlabel", "")

    if bed.shape != sur.shape:
        raise ValueError("BED and SUR must share the same shape.")

    if extra_dimension == "" or extra_dimension == "on":
        thi = sur - bed
    else:
        thi = extra_dimension

    if torch.isnan(thi).all():
        thi = torch.zeros_like(thi)

    thi_min = torch.min(thi)
    thi_max = torch.max(thi)
    bed_min = torch.min(bed)
    bed_max = torch.max(bed)

    if extra_dimension == "" or extra_dimension == "on":
        i = torch.where(thi < 0)
        if len(i[0]) > 0:
            raise ValueError("BED exceeds SUR at some points.")

    if d2:
        fig2, ax2 = plt.subplots()
        pcm = ax2.pcolormesh(thi.cpu().numpy())
        fig2.colorbar(pcm, ax=ax2)
        ax2.set_title(title or "2D Thickness")
        ax2.set_xlabel(xlabel or "X")
        ax2.set_ylabel(ylabel or "Y")
        plt.tight_layout()
        d2_path = output_path.replace(".png", "_2d.png")
        fig2.savefig(d2_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(view_orientation[1], view_orientation[0])
    ax.grid(True)

    a = bed.shape
    x1, x2 = 1, a[1]
    y1, y2 = 1, a[0]
    x, y = torch.meshgrid(
        torch.linspace(x1, x2, a[1], device=bed.device),
        torch.linspace(y1, y2, a[0], device=bed.device),
        indexing="ij",
    )

    if extra_dimension == "":
        cmap = plt.get_cmap(bed_colormap)
        if bed_colormap_flipud == "off":
            colors = [sur_color] + [cmap(i) for i in range(bed_colors)]
        else:
            colors = [sur_color] + [cmap(i) for i in range(bed_colors - 1, -1, -1)]
        cmap = plt.cm.colors.ListedColormap(colors)
        h1 = ax.plot_surface(x.cpu().numpy(), y.cpu().numpy(), bed.T.cpu().numpy(), alpha=bed_trans, cmap=cmap)
        h1.set_clim((bed_min - (1 / bed_colors) * (bed_max - bed_min)).item(), bed_max.item())
        ax.plot_surface(x.cpu().numpy(), y.cpu().numpy(), sur.T.cpu().numpy(), color=sur_color, alpha=sur_trans)

        if 0 < vertical_scaling <= 1:
            ax.set_zlim(bed_min.item(), (bed_min + (torch.max(sur) - bed_min) / vertical_scaling).item())
        else:
            raise ValueError("vertical_scaling must be in (0, 1].")
    else:
        if d4_colormap_flipud == "off":
            cmap1 = plt.get_cmap(d4_colormap)
            colors1 = [cmap1(i) for i in range(cbar_colors)]
        else:
            cmap1 = plt.get_cmap(d4_colormap)
            colors1 = [cmap1(i) for i in range(cbar_colors - 1, -1, -1)]
        cmap2 = plt.get_cmap(bed_colormap)
        colors2 = [cmap2(i) for i in range(bed_colors)]
        cmap = plt.cm.colors.ListedColormap(colors1 + colors2)
        zbed = bed_colors + ((bed.T - bed_min) / (bed_max - bed_min + 1e-12)) * bed_colors
        h1 = ax.plot_surface(x.cpu().numpy(), y.cpu().numpy(), zbed.cpu().numpy(), alpha=bed_trans, cmap=cmap)
        h1.set_clim(0, cbar_colors + bed_colors)
        fig.colorbar(h1, ax=ax)

    ax.tick_params(axis="both", labelsize=tick_size)
    ax.set_title(title, fontweight="bold", fontsize=title_size)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_zlabel(zlabel, fontweight="bold")

    ls = LightSource(azdeg=light_orientation[0], altdeg=light_orientation[1])
    rgb = ls.shade(bed.T.cpu().numpy(), cmap=plt.get_cmap(bed_colormap))
    ax.plot_surface(x.cpu().numpy(), y.cpu().numpy(), bed.T.cpu().numpy(), facecolors=rgb, alpha=bed_trans)

    ax.set_axisbelow(True)
    if axes == "off":
        ax.set_axis_off()
    else:
        ax.set_axis_on()
    if tick == "off":
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
