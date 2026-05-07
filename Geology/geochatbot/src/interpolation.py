import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _prepare(df, element):
    element_data = df[["longitude", "latitude", element]].dropna()
    x = element_data[["longitude", "latitude"]].values
    y = element_data[element].values
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    return x, y, x_min, x_max, y_min, y_max


def _to_map(base_values, xx, yy, x, y, values, element):
    try:
        import branca.colormap as cm
        import folium
        from folium.plugins import Draw, LocateControl, MeasureControl
    except Exception:
        # Fallback: no folium stack, return None and let caller save PNG instead.
        return None

    m = folium.Map(location=[x[:, 1].mean(), x[:, 0].mean()], zoom_start=10)
    cmap = cm.LinearColormap(
        ["blue", "green", "yellow", "red"],
        vmin=float(values.min()),
        vmax=float(values.max()),
        caption=element,
    )

    plt.figure(figsize=(6, 4))
    plt.imshow(
        base_values,
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        origin="lower",
        cmap="plasma",
        alpha=0.8,
    )
    png_path = os.path.join(os.getcwd(), "_tmp_interp.png")
    plt.axis("off")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    folium.raster_layers.ImageOverlay(
        image=png_path,
        bounds=[[yy.min(), xx.min()], [yy.max(), xx.max()]],
        opacity=0.7,
    ).add_to(m)

    for lat, lon, val in zip(x[:, 1], x[:, 0], values):
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup=f"{element}: {val}",
        ).add_to(m)

    MeasureControl().add_to(m)
    LocateControl().add_to(m)
    Draw().add_to(m)
    cmap.add_to(m)

    try:
        os.remove(png_path)
    except OSError:
        pass
    return m


def create_idw_map(df, element, grid_size=100):
    x, y, x_min, x_max, y_min, y_max = _prepare(df, element)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    points = np.c_[xx.ravel(), yy.ravel()]
    try:
        from sklearn.neighbors import KNeighborsRegressor

        idw = KNeighborsRegressor(n_neighbors=4, weights="distance")
        idw.fit(x, y)
        vals = idw.predict(points).reshape(xx.shape)
    except Exception:
        vals = np.zeros(points.shape[0], dtype=float)
        p = 2.0
        eps = 1e-12
        for i, pt in enumerate(points):
            d = np.sqrt(((x - pt) ** 2).sum(axis=1)) + eps
            w = 1.0 / (d**p)
            vals[i] = np.sum(w * y) / np.sum(w)
        vals = vals.reshape(xx.shape)
    return _to_map(vals, xx, yy, x, y, y, element), vals


def create_kriging_like_map(df, element, grid_size=100):
    x, y, x_min, x_max, y_min, y_max = _prepare(df, element)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    try:
        from scipy.interpolate import Rbf

        rbfi = Rbf(x[:, 0], x[:, 1], y, function="gaussian")
        vals = rbfi(xx, yy)
    except Exception:
        _, vals = create_idw_map(df, element, grid_size=grid_size)
    return _to_map(vals, xx, yy, x, y, y, element), vals


def save_summary(summary, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_heatmap_png(vals, output_path, title):
    plt.figure(figsize=(6, 4))
    plt.imshow(vals, cmap="plasma", origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
