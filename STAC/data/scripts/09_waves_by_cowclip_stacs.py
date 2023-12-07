# %%
import os
import pathlib
import sys
import json
import xarray as xr
import numpy as np
import datetime
import rasterio
import shapely
import pandas as pd
from posixpath import join as urljoin

import pystac
from coclicodata.drive_config import p_drive
from coclicodata.etl.cloud_utils import dataset_from_google_cloud,load_google_credentials, dir_to_google_cloud
from coclicodata.etl.extract import get_mapbox_url, zero_terminated_bytes_as_str
from pystac import Catalog, CatalogType, Collection, Summaries
from coclicodata.coclico_stac.io import CoCliCoStacIO
from coclicodata.coclico_stac.layouts import CoCliCoCOGLayout
from coclicodata.coclico_stac.templates import (
    extend_links,
    gen_default_collection_props,
    gen_default_item,
    gen_default_item_props,
    gen_default_summaries,
    gen_mapbox_asset,
    gen_zarr_asset,
    get_template_collection,
)
from coclicodata.coclico_stac.extension import CoclicoExtension
from coclicodata.coclico_stac.datacube import add_datacube
from coclicodata.coclico_stac.utils import (
    get_dimension_dot_product,
    get_dimension_values,
    get_mapbox_item_id,
    rm_special_characters,
)

# TODO: move itemize to ETL or stac.blueprint when generalized
def itemize(
    da,
    item: pystac.Item,
    blob_name: str,
    asset_roles: "List[str] | None" = None,  # "" enables Python 3.8 development not to crash: https://github.com/tiangolo/typer/issues/371
    asset_media_type=pystac.MediaType.COG,
) -> pystac.Item:
    """ """
    import rioxarray  # noqa

    item = item.clone()
    dst_crs = rasterio.crs.CRS.from_epsg(4326)

    bbox = rasterio.warp.transform_bounds(da.rio.crs, dst_crs, *da.rio.bounds())
    geometry = shapely.geometry.mapping(shapely.geometry.box(*bbox))

    item.id = blob_name
    item.geometry = geometry
    item.bbox = bbox
    item.datetime = pd.Timestamp(da["time"].item()).to_pydatetime()  # dataset specific
    # item.datetime = cftime_to_pdts(da["time"].item()).to_pydatetime() # dataset specific

    ext = pystac.extensions.projection.ProjectionExtension.ext(
        item, add_if_missing=True
    )
    ext.bbox = da.rio.bounds()
    ext.shape = da.shape[-2:]
    ext.epsg = da.rio.crs.to_epsg()
    ext.geometry = shapely.geometry.mapping(shapely.geometry.box(*ext.bbox))
    ext.transform = list(da.rio.transform())[:6]
    ext.add_to(item)

    roles = asset_roles or ["data"]

    href = os.path.join(
        GCS_PROTOCOL,
        BUCKET_NAME,
        BUCKET_PROJ,
        metadata["TITLE_ABBREVIATION"],
        blob_name,
    )

    # TODO: We need to generalize this `href` somewhat.
    asset = pystac.Asset(
        href=href,
        media_type=asset_media_type,
        roles=roles,
    )

    item.add_asset("data", asset)

    return item

if __name__ == "__main__":
    # hard-coded input params at project level
    GCS_PROTOCOL = "https://storage.googleapis.com"
    GCS_PROJECT = "DGDS - I1000482-002"
    BUCKET_NAME = "dgds-data-public"
    BUCKET_PROJ = "gca"

    STAC_DIR = "data/current"
    TEMPLATE_COLLECTION = "template"  # stac template for dataset collection
    COLLECTION_ID = "waves_by_cowclip"  # name of stac collection

    # hard-coded input params which differ per dataset
    METADATA = "metadata_waves_by_cowclip.json"
    DATASET_DIR = "output_NetCDF"
    CF_FILE = "waves_by_cowclip_cf_merged.nc"

    # these are added at collection level, determine dashboard graph layout using all items
    PLOT_SERIES = "rpc"
    PLOT_X_AXIS = "time"
    PLOT_TYPE = "line"
    MIN = 0
    MAX = 3
    LINEAR_GRADIENT = [
        {"color": "hsl(110,90%,80%)", "offset": "0.000%", "opacity": 100},
        {"color": "hsla(55,88%,53%,0.5)", "offset": "50.000%", "opacity": 100},
        {"color": "hsl(0,90%,70%)", "offset": "100.000%", "opacity": 100},
    ]

    # define local directories
    home = pathlib.Path().home()
    tmp_dir = home.joinpath("data", "tmp")
    coclico_data_dir = p_drive.joinpath(
        "11209199-climate-resilient-ports",
        "00_general",
        "02_COWCLIP_Harvest_Morim_et_al"
    )  # remote p drive

    # use local or remote data dir
    use_local_data = False

    if use_local_data:
        ds_dir = tmp_dir.joinpath(DATASET_DIR)
    else:
        ds_dir = coclico_data_dir.joinpath(DATASET_DIR)

    if not ds_dir.exists():
        raise FileNotFoundError(f"Data dir does not exist, {str(ds_dir)}")

    # directory to export result
    cog_dirs = ds_dir.joinpath("COG")

    # load metadata template
    metadata_fp = ds_dir.joinpath(METADATA)
    with open(metadata_fp, "r") as f:
        metadata = json.load(f)

    catalog = Catalog.from_file(os.path.join(pathlib.Path(__file__).parent.parent.parent, STAC_DIR, "catalog.json"))

    template_fp = os.path.join(
        pathlib.Path(__file__).parent.parent.parent, STAC_DIR, TEMPLATE_COLLECTION, "collection.json"
    )

    # generate collection for dataset
    collection = get_template_collection(
        template_fp=template_fp,
        collection_id=COLLECTION_ID,
        title=metadata["TITLE"],
        description=metadata["SHORT_DESCRIPTION"],
        keywords=metadata["KEYWORDS"],
        # license=metadata["LICENSE"],
        # spatial_extent=metadata["SPATIAL_EXTENT"],
        # temporal_extent=metadata["TEMPORAL_EXTENT"],
        # providers=metadata["PROVIDERS"],
    )

    layout = CoCliCoCOGLayout()

    # do for all CoGs (CF compliant)
    # open the dataset
    ds_fp = ds_dir.joinpath(CF_FILE)
    ds = xr.open_dataset(ds_fp)

    # extract list of data variables
    for i, time in enumerate(ds.time.values):
        for j, rcp in enumerate(ds.rcp.values):
            for k, ens_perc in enumerate(ds.ens_perc.values):
                for l, nvar_stat in enumerate(ds.nvar_stat.values):
                    var_stat = ds.var_stat.values[l].decode("utf-8")
                    for m, var in enumerate(list(set(ds.variables) - set(ds.dims) - set(ds.coords))):
                        # make array 2d and fix spatial dimensions and crs
                        ds_ = ds.copy()
                        ds_ = ds_.isel({"time": i, "rcp": j, "ens_perc": k, "nvar_stat": l})
                        ds_ = ds_[var]
                        #ds_ = ds_.drop_vars(["rcp", "ens_perc", 'time', 'var_stat'])

                        # Skip if all data is nan
                        if np.isnan(ds_.values).all():
                            print('skipped')

                        # Set dimensions
                        ds_.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                        if not ds_.rio.crs:
                            ds_ = ds_.rio.write_crs("EPSG:4326")

                        # Get file path
                        blob_name = pathlib.Path(os.path.join(f'time={str(time)[:4]}', f'rcp={rcp}', f'ens_perc={ens_perc}', f'var_stat={var_stat}', f'{var}.tif'))
                        
                        outpath = cog_dirs.joinpath(blob_name)
                        template_item = pystac.Item(
                            "id", None, None, datetime.datetime(2000, 1, 1), {}
                        )

                        item = itemize(ds_, template_item, blob_name=str(blob_name))
                        collection.add_item(item, strategy=layout)

    # TODO: use gen_default_summaries() from blueprint.py after making it frontend compliant.
    collection.summaries = Summaries({})

    # this calls CollectionCoclicoExtension since stac_obj==pystac.Collection
    coclico_ext = CoclicoExtension.ext(collection, add_if_missing=True)

    # Add frontend properties defined above to collection extension properties. The
    # properties attribute of this extension is linked to the extra_fields attribute of
    # the stac collection.
    coclico_ext.units = metadata["UNITS"]
    coclico_ext.plot_series = PLOT_SERIES
    coclico_ext.plot_x_axis = PLOT_X_AXIS
    coclico_ext.plot_type = PLOT_TYPE
    coclico_ext.min_ = MIN
    coclico_ext.max_ = MAX
    coclico_ext.linear_gradient = LINEAR_GRADIENT

    '''
    # Add thumbnail
    collection.add_asset(
        "thumbnail",
        pystac.Asset(
            "https://storage.googleapis.com/dgds-data-public/coclico/assets/thumbnails/" + COLLECTION_ID + ".png",  # noqa: E501,  # noqa: E501
            title="Thumbnail",
            media_type=pystac.MediaType.PNG,
        ),
    )
    '''

    # add collection to catalog
    catalog.add_child(collection)

    # normalize the paths
    collection.normalize_hrefs(
        os.path.join(pathlib.Path(__file__).parent.parent.parent, STAC_DIR, COLLECTION_ID), strategy=layout
    )

    # save updated catalog to local drive
    catalog.save(
        catalog_type=CatalogType.SELF_CONTAINED,
        dest_href=os.path.join(pathlib.Path(__file__).parent.parent.parent, STAC_DIR),
        # dest_href=str(tmp_dir),
        stac_io=CoCliCoStacIO(),
    )
    print("Done!")

    # upload directory with cogs to google cloud
    file_path_credentials = pathlib.Path('P:/11205479-coclico/FASTTRACK_DATA')
    load_google_credentials(
        google_token_fp=file_path_credentials.joinpath("google_credentials.json")
    )

    dir_to_google_cloud(
        dir_path=str(cog_dirs),
        gcs_project=GCS_PROJECT,
        bucket_name=BUCKET_NAME,
        bucket_proj=BUCKET_PROJ,
        dir_name=metadata["TITLE_ABBREVIATION"],
    )

# %%
