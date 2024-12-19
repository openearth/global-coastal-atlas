# Packages for loading data
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

# Packages for plotting
from resilientplotterclass import rpc
from pathlib import Path
import matplotlib
import numpy as np
from shapely import Polygon
import rioxarray as rio
from rioxarray.merge import merge_arrays

from report.datasets.utils import plot_to_base64
from utils.stac import STACClientGCA
from report.datasets.datasetcontent import DatasetContent

matplotlib.use("Agg")
plt.rcParams["svg.fonttype"] = "none"

world = gpd.read_file(
    Path(__file__).parent.parent.parent / "data" / "world_administrative.zip"
)

# def get_sub_threat_content(xarr: xr.Dataset) -> DatasetContent:
#     """Get content for the dataset"""
#     dataset_id = "land_sub"
#     title = "Land Subsidence"
#     text = "Here we generate some content based on the dataset" ##TODO

#     image_base64 = create_sub_treat_plot(xarr)
#     return DatasetContent(
#         dataset_id=dataset_id,
#         title=title,
#         text=text,
#         image_base64=image_base64,
#     )


def get_landsub_content(polygon: Polygon) -> list[DatasetContent]:
    dataset_contents_list = []
    dataset_contents_list.append(get_landsub2010_content(polygon))
    dataset_contents_list.append(get_landsub2040_content(polygon))

    return dataset_contents_list


def get_landsub2040_content(polygon: Polygon) -> DatasetContent:
    rasteridlist = find_extent(polygon)
    combinedraster = get_raster("Haz-Land_Sub_2040_COGs", rasteridlist)
    clippedraster = clip_raster(polygon, combinedraster)

    """Get content for the dataset"""
    dataset_id = "landsub2040"
    title = "Land Subsidence in 2040"
    text = "Here we generate some content based on the dataset"  ##TODO

    image_base64 = create_landsub_plot(polygon, clippedraster)
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def get_landsub2010_content(polygon: Polygon) -> DatasetContent:
    rasteridlist = find_extent(polygon)
    combinedraster = get_raster("Haz-Land_Sub_2010_COGs", rasteridlist)
    clippedraster = clip_raster(polygon, combinedraster)

    """Get content for the dataset"""
    dataset_id = "landsub2010"
    title = "Land Subsidence in 2010"
    text = "Here we generate some content based on the dataset"  ##TODO

    image_base64 = create_landsub_plot(polygon, clippedraster)
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def create_sub_treat_plot(xarr: xr.Dataset):
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
        hue="epsi",
        edgecolor="none",
        cmap="RdYlGn",
        add_colorbar=True,
        cbar_kwargs={"label": "Land Subsidence Index"},
    )

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.25, pad=0.10)

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


def find_extent(polygon: Polygon) -> list:
    polyx = polygon.exterior.xy[0]
    polyy = polygon.exterior.xy[1]

    minx = min(polyx)
    miny = min(polyy)
    maxx = max(polyx)
    maxy = max(polyy)

    extentx = [divmod(minx, 1)[0] // 3 * 3, (divmod(maxx, 1)[0] // 3 + 1) * 3]
    extenty = [np.round(miny, 0) // 3 * 3 - 1, (np.round(maxy, 0) // 3 + 1) * 3 - 1]

    xlist = np.arange(
        divmod(minx, 1)[0] // 3 * 3, (divmod(maxx, 1)[0] // 3 + 2) * 3, 3
    )  ##TODO: now hard coded to have additional one tile -> to be improved
    ylist = np.arange(
        divmod(miny, 1)[0] // 3 * 3 - 1, (divmod(maxy, 1)[0] // 3 + 2) * 3 - 1, 3
    )

    rasteridlist = []

    for y in ylist:
        for x in xlist:
            idname = "B01_x" + str(x) + "_" + "y" + str(y)
            rasteridlist.append(idname)

    return rasteridlist


def get_raster(collectionid: str, rasteridlist: list):
    stac = (
        "https://storage.googleapis.com/dgds-data-public/gca/SOTC/gca-sotc/catalog.json"
    )

    rasterlist = []

    for rasterid in rasteridlist:
        gca_client = STACClientGCA.open(stac)
        href = (
            gca_client.get_child(collectionid)
            .get_item(rasterid)
            .assets["band_data"]
            .href
        )
        raster = rio.open_rasterio(href, masked=True)
        rasterlist.append(raster)

    combinedraster = merge_arrays(dataarrays=rasterlist, method="sum")

    return combinedraster


def clip_raster(polygon: Polygon, raster):
    # clip = raster.rio.clip_box(*polygon.bounds) ##TODO: clip_box can only clip to a box
    clip = raster.rio.clip([polygon])
    return clip


def create_landsub_plot(polygon: Polygon, clip):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    base = world.boundary.plot(
        ax=ax, edgecolor="grey", facecolor="grey", alpha=0.1, zorder=0
    )

    clip.plot()

    lonmin = min(polygon.exterior.xy[0])
    lonmax = max(polygon.exterior.xy[0])
    latmin = min(polygon.exterior.xy[1])
    latmax = max(polygon.exterior.xy[1])

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
