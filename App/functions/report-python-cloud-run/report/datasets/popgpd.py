# Packages for loading data
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Packages for plotting
from resilientplotterclass import rpc

from report.datasets.datasetcontent import DatasetContent
from report.datasets.utils import plot_to_base64
from report.utils.gentext import describe_data

matplotlib.use("Agg")
plt.rcParams["svg.fonttype"] = "none"
world = gpd.read_file(
    Path(__file__).parent.parent.parent / "data" / "world_administrative.zip"
)


def get_world_pop_content(xarr: xr.Dataset) -> DatasetContent:
    """Get content for the dataset"""
    dataset_id = "world_pop"
    title = "The Population"
    text = "Here we generate some content based on the dataset"
    text = describe_data(xarr, dataset_id)

    image_base64 = create_world_pop_plot(xarr)
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def create_world_pop_plot(xarr):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    base = world.boundary.plot(
        ax=ax, edgecolor="grey", facecolor="grey", alpha=0.1, zorder=0
    )

    p = rpc.scatter(
        xarr,
        ax=ax,
        data_type="data",
        x="lon",
        y="lat",
        s=xarr["pop_tot"].values / 100,
        hue="pop_tot",
        edgecolor="none",
        cmap="RdYlGn",
        add_colorbar=True,
        cbar_kwargs={"label": "Population"},
    )

    lonmin = min(xarr.lon.values)
    lonmax = max(xarr.lon.values)
    latmin = min(xarr.lat.values)
    latmax = max(xarr.lat.values)

    xlim = [lonmin - 0.1, lonmax + 0.1]
    ylim = [latmin - 0.1, latmax + 0.1]

    ax.set(
        xlim=xlim,
        ylim=ylim,
    )

    ax.set_aspect(1 / np.cos(np.mean(ylim) * np.pi / 180))
    ax.grid(False)

    fig.tight_layout()

    return plot_to_base64(fig)
