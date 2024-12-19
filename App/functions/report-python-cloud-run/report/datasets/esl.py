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

matplotlib.use("Agg")
plt.rcParams["svg.fonttype"] = "none"
# from matplotlib import colors ##TODO

world = gpd.read_file(
    Path(__file__).parent.parent.parent / "data" / "world_administrative.zip"
)


def get_esl_content(xarr: xr.Dataset) -> list[DatasetContent]:
    dataset_contents_list = []
    dataset_contents_list.append(get_esl26_content(xarr))
    dataset_contents_list.append(get_esl45_content(xarr))
    dataset_contents_list.append(get_esl85_content(xarr))

    return dataset_contents_list


def get_esl26_content(xarr: xr.Dataset) -> DatasetContent:
    """Get content for ESL dataset"""
    dataset_id = "esl_RCP26"
    title = "Extreme Sea Level"
    text = "Here we generate some content based on the ESL dataset"

    image_base64 = create_esl_plot(xarr, "RCP26")
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def get_esl45_content(xarr: xr.Dataset) -> DatasetContent:
    """Get content for ESL dataset"""
    dataset_id = "esl_RCP45"
    title = "Extreme Sea Level"
    text = "Here we generate some content based on the ESL dataset"

    image_base64 = create_esl_plot(xarr, "RCP45")
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def get_esl85_content(xarr: xr.Dataset) -> DatasetContent:
    """Get content for ESL dataset"""
    dataset_id = "esl_RCP85"
    title = "Extreme Sea Level"
    text = "Here we generate some content based on the ESL dataset"

    image_base64 = create_esl_plot(xarr, "RCP85")
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def create_esl_plot(xarr, scenario):
    match scenario:
        case "RCP26":
            GWL = 1.5
            GWLs = "1.5 ℃"
        case "RCP45":
            GWL = 3
            GWLs = "3 ℃"
        case "RCP85":
            GWL = 5
            GWLs = "5 ℃"

    # ens = 50 # look at ds.ensemble.values for options
    rp = 50.0  # look at ds.rp.values for options

    xarr = xarr.sel(gwl=GWL, rp=rp)  # filter the other params

    lonmin = min(xarr.lon.values)
    lonmax = max(xarr.lon.values)
    latmin = min(xarr.lat.values)
    latmax = max(xarr.lat.values)

    xlim = [lonmin - 0.1, lonmax + 0.1]
    ylim = [latmin - 0.1, latmax + 0.1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    base = world.boundary.plot(
        ax=ax, edgecolor="grey", facecolor="grey", alpha=0.1, zorder=0
    )

    rpc.scatter(
        xarr.sel(ensemble=5),
        data_type="data",
        ax=ax,
        x="lon",
        y="lat",
        hue="esl",
        cmap="RdYlGn_r",
        add_colorbar=False,
    )

    xarr["lat"].values = xarr["lat"].values + 0.01

    rpc.scatter(
        xarr.sel(ensemble=50),
        data_type="data",
        ax=ax,
        x="lon",
        y="lat",
        hue="esl",
        cmap="RdYlGn_r",
        add_colorbar=False,
    )

    xarr["lat"].values = xarr["lat"].values + 0.01

    rpc.scatter(
        xarr.sel(ensemble=95),
        data_type="data",
        ax=ax,
        x="lon",
        y="lat",
        vmin=np.nanmin(xarr.sel(ensemble=5).esl.values),
        vmax=np.nanmax(xarr.sel(ensemble=95).esl.values),
        hue="esl",
        cmap="RdYlGn_r",
        add_colorbar=True,
        cbar_kwargs={"label": "ESL [m]"},
    )

    ax.set_title("%s-year extreme sea level for %s global warming level" % (rp, GWLs))

    ax.set(
        xlim=xlim,
        ylim=ylim,
    )

    ax.set_aspect(1 / np.cos(np.mean(ylim) * np.pi / 180))
    ax.grid(False)

    fig.tight_layout()

    return plot_to_base64(fig)
