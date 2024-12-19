# Packages for loading data
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Packages for plotting
from resilientplotterclass import rpc
from shapely import Polygon  # type: ignore

from report.datasets.datasetcontent import DatasetContent
from report.datasets.utils import plot_to_base64
from report.utils.gentext import describe_overview

matplotlib.use("Agg")
plt.rcParams["svg.fonttype"] = "none"
world = gpd.read_file(
    Path(__file__).parent.parent.parent / "data" / "world_administrative.zip"
)


def get_overview(polygon: Polygon, dataset_contents: DatasetContent) -> DatasetContent:
    """Get overview"""
    dataset_id = "overview"
    title = "Overview"
    text = "Here we generate some content based on all datasets"
    text = describe_overview(polygon, dataset_contents)

    image_base64 = create_overview_img(polygon)
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def cal_xylims(gdf_aoi, buffer):
    bounds = gdf_aoi.bounds.values[0]
    center = gdf_aoi.centroid
    length = max([bounds[2] - bounds[0], bounds[3] - bounds[1]])
    ylims = [center.y[0] - length / 2 - buffer, center.y[0] + length / 2 + buffer]
    ylength = (ylims[1] - ylims[0]) / np.cos(np.radians(center.y[0]))
    xlims = [center.x[0] - ylength / 2, center.x[0] + ylength / 2]

    return xlims, ylims


def create_overview_img(polygon: Polygon):
    gdf_aoi = gpd.GeoDataFrame(
        {"Name": ["Custom"], "geometry": [polygon]}, crs="EPSG:4326"
    )
    center = gdf_aoi.centroid

    fig, ax = plt.subplots(1, 2, figsize=(20, 16), width_ratios=[1, 1])

    xlims, ylims = cal_xylims(gdf_aoi, 2.5)

    ax[0].scatter(center.x[0], center.y[0], color="r", marker="o")
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    rpc.basemap(
        crs="EPSG:4326", map_type="satellite", ax=ax[0], source="CartoDB.Voyager"
    )

    worldax = inset_axes(ax[0], width=2.5, height=2, loc="upper left")
    worldax.scatter(center.x[0], center.y[0], color="r", marker="o")
    xlims, ylims = cal_xylims(gdf_aoi, 18)
    worldax.set_xlim(xlims)
    worldax.set_ylim(ylims)
    rpc.basemap(
        crs=gdf_aoi.crs, ax=worldax, map_type="satellite", source="CartoDB.Positron"
    )
    worldax.set_xticklabels([])
    worldax.set_yticklabels([])
    worldax.set_xlabel(None)
    worldax.set_ylabel(None)
    worldax.tick_params(axis="both", which="both", length=0)

    xlims, ylims = cal_xylims(gdf_aoi, 0.01)

    rpc.geometries(gdf_aoi, ax=ax[1], facecolor="none", edgecolor="white", linewidth=1)
    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)
    rpc.basemap(crs=gdf_aoi.crs, map_type="satellite", ax=ax[1])

    return plot_to_base64(fig)
