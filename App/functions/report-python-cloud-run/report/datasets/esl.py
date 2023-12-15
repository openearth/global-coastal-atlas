import base64
from io import BytesIO, StringIO
import matplotlib

matplotlib.use("Agg")
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

import geopandas as gpd

from .datasetcontent import DatasetContent


def get_esl_content(xarr: xr.Dataset) -> DatasetContent:
    """Get content for ESL dataset"""
    dataset_id = "esl"
    title = "Extreme Sea Level"
    text = "Here we generate some content based on the ESL dataset"

    image_base64 = create_esl_plot(xarr)
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def create_esl_plot(xarr):
    GWL = 0  # look at ds.gwl.values for options
    GWLs = "present-day"
    # ens = 50 # look at ds.ensemble.values for options
    rp = 50.0  # look at ds.rp.values for options
    world = gpd.read_file(
        """https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/world-administrative-boundaries/exports/shp?lang=en&timezone=Europe%2FBerlin"""
    )
    cmap = matplotlib.cm.RdYlGn_r
    norm = colors.BoundaryNorm(np.arange(0, 7.5, 0.5), cmap.N)
    ds_fil = xarr.sel(gwl=GWL, rp=rp)  # filter the other params
    lonmin = min(ds_fil.lon.values)
    lonmax = max(ds_fil.lon.values)
    latmin = min(ds_fil.lat.values)
    latmax = max(ds_fil.lat.values)
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 8)  # fig.set_size_inches(15, 20)
    base = world.boundary.plot(
        ax=ax, edgecolor="grey", facecolor="grey", alpha=0.1, zorder=0
    )
    im1 = ax.scatter(
        ds_fil.lon.values,
        ds_fil.lat.values,
        10 * ds_fil.sel(ensemble=5).esl.values,
        ds_fil.sel(ensemble=5).esl.values,
        cmap=cmap,
        norm=norm,
        zorder=1,
    )
    # plt.set_clim(0,5)
    im2 = ax.scatter(
        ds_fil.lon.values,
        ds_fil.lat.values + 0.1,
        10 * ds_fil.sel(ensemble=50).esl.values,
        ds_fil.sel(ensemble=50).esl.values,
        cmap=cmap,
        norm=norm,
        zorder=1,
    )
    im3 = ax.scatter(
        ds_fil.lon.values,
        ds_fil.lat.values + 0.2,
        10 * ds_fil.sel(ensemble=95).esl.values,
        ds_fil.sel(ensemble=95).esl.values,
        cmap=cmap,
        norm=norm,
        zorder=1,
    )
    ax.set_title("%s-year extreme sea level for %s global warming level" % (rp, GWLs))
    ax.axis("square")
    ax.set(
        xlabel="lon",
        ylabel="lat",
        xlim=[lonmin - 2, lonmax + 2],
        ylim=[latmin - 2, latmax + 2],
    )
    # fig.colorbar(im1, ax=ax)
    im1.set_clim(0, 7)

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )  # to give colorbar own axes
    plt.colorbar(im1, cax=cax)  # Similar to fig.colorbar(im, cax = cax)
    cax.set_title("ESL in meters")
    #
    imgdata = BytesIO()
    fig.savefig(imgdata, format="png")

    return base64.b64encode(imgdata.getbuffer()).decode("ascii")
