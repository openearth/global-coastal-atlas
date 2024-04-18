import datetime
import os
import pathlib
import sys
from re import S, template
from tqdm import tqdm

# make modules importable when running this file as script
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import datetime
import itertools
import operator
import os
from typing import List, Mapping, Optional
import json
from pystac.extensions import eo, raster
import fsspec
from typing import Any, Dict, List, Optional, Tuple, Union
import dask

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import rasterio.warp
import shapely.geometry
import xarray as xr


import pathlib
import gcsfs
from posixpath import join as urljoin
from google.cloud import storage
from coclicodata.etl.cloud_utils import (
    p_drive,
    dataset_to_google_cloud,
    dataset_from_google_cloud,
    geojson_to_mapbox,
    load_env_variables,
    load_google_credentials,
)

# from stac.blueprint import IO, Layout, get_template_collection



import pystac
from coclicodata.drive_config import p_drive
from coclicodata.etl.cloud_utils import dataset_from_google_cloud
from coclicodata.etl.extract import get_mapbox_url, zero_terminated_bytes_as_str
from pystac import Catalog, CatalogType, Collection, Summaries
from coclicodata.coclico_stac.io import CoCliCoStacIO
from coclicodata.coclico_stac.layouts import  CoCliCoCOGLayout
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
sys.path.append(r'C:\Users\rowe\Documents\GitHub\coastmonitor\src')
from coastmonitor.io.utils import name_block


# import cftime
# import numpy as np
import pandas as pd
import pystac
import rasterio

# import rioxarray as rio
import shapely
import xarray as xr
import math
import dask
from posixpath import join as urljoin
from pystac.extensions import eo, raster
from stactools.core.utils import antimeridian

# from datacube.utils.cog import write_cog
from coclicodata.drive_config import p_drive

# from pystac import Catalog, CatalogType, Collection, Summaries
from coclicodata.etl.cloud_utils import load_google_credentials, dir_to_google_cloud
from coclicodata.coclico_stac.io import CoCliCoStacIO
from coclicodata.coclico_stac.layouts import CoCliCoCOGLayout
from coclicodata.coclico_stac.extension import (
    CoclicoExtension,
)  # self built stac extension

# from coastmonitor.io.cloud import (
#     to_https_url,
#     to_storage_location,
#     to_uri_protocol,
#     write_block,
# )
from coastmonitor.io.utils import name_block
from rasterio import logging





__version__ = "0.0.1"


def create_item(block, item_id, antimeridian_strategy=antimeridian.Strategy.SPLIT):
    dst_crs = rasterio.crs.CRS.from_epsg(4326)

    # when the data spans a range, it's common practice to use the middle time as the datetime provided
    # in the STAC item. So then you have to infer the start_datetime, end_datetime and get the middle
    # from those.
    # start_datetime, end_datetime = ...
    # middle_datetime = start_datetime + (end_datetime - start_datetime) / 2

    # the bbox of the STAC item is provided in 4326
    bbox = rasterio.warp.transform_bounds(block.rio.crs, dst_crs, *block.rio.bounds())
    geometry = shapely.geometry.mapping(shapely.make_valid(shapely.geometry.box(*bbox)))
    bbox = shapely.make_valid(shapely.box(*bbox)).bounds

    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=pd.Timestamp(block["time"].item()),
        properties={},
    )

    # useful for global datasets that cross the antimerdian E-W line
    antimeridian.fix_item(item, antimeridian_strategy)

    # use this when the data spans a certain time range
    # item.common_metadata.start_datetime = start_datetime
    # item.common_metadata.end_datetime = end_datetime

    item.common_metadata.created = datetime.datetime.utcnow()

    ext = pystac.extensions.projection.ProjectionExtension.ext(
        item, add_if_missing=True
    )
    ext.bbox = block.rio.bounds()  # these are provided in the crs of the data
    ext.shape = tuple(v for k, v in block.sizes.items() if k in ["y", "x"])
    ext.epsg = block.rio.crs.to_epsg()
    ext.geometry = shapely.geometry.mapping(shapely.geometry.box(*ext.bbox))
    ext.transform = list(block.rio.transform())[:6]
    ext.add_to(item)

    # add CoCliCo frontend properties to visualize it in the web portal
    # TODO: This is just example. We first need to decide which properties frontend needs for COG visualization
    coclico_ext = CoclicoExtension.ext(item, add_if_missing=True)
    coclico_ext.item_key = item_id
    coclico_ext.add_to(item)

    # add more functions to describe the data at item level, for example the frontend properties to visualize it
    ...

    return item

def create_collection(
    description: str | None = None, extra_fields: dict[str, Any] | None = None
) -> pystac.Collection:
    providers = [
        pystac.Provider(
            name="Deltares",
            roles=[
                pystac.provider.ProviderRole.PROCESSOR,
                pystac.provider.ProviderRole.HOST,
            ],
            url="https://deltares.nl",
        ),
        pystac.Provider(
            "Global Climate Forum",
            roles=[
                pystac.provider.ProviderRole.PRODUCER,
            ],
            url="https://globalclimateforum.org",
        ),
    ]

    start_datetime = START_DATETIME

    extent = pystac.Extent(
        pystac.SpatialExtent([[-180.0, 90.0, 180.0, -90.0]]),
        pystac.TemporalExtent([[start_datetime, None]]),
    )

    links = [
        pystac.Link(
            rel=pystac.RelType.LICENSE,
            target="https://coclicoservices.eu/legal/",
            media_type="text/html",
            title="ODbL-1.0 License",
        )
    ]

    keywords = KEYWORDS# [
    #     "Coast",
    #     "Coastal Mask",
    #     "Coastal Change",
    #     "Coastal Hazards",
    #     "Flood Risk",
    #     "CoCliCo",
    #     "Deltares",
    #     "Cloud Optimized GeoTIFF",
    # ]

    if description is None:
        description = DATASET_DESCRIPTION #(
        #     "Coastal mask that is derived from Copernicus elevation data combined with"
        #     " a maximum distanceto coastal water bodies."
        # )

    collection = pystac.Collection(
        id=COLLECTION_ID,
        title=COLLECTION_TITLE,
        description=description,  # noqa: E502
        license="ODbL-1.0",
        providers=providers,
        extent=extent,
        catalog_type=pystac.CatalogType.RELATIVE_PUBLISHED,
    )

    collection.add_asset(
        "thumbnail",
        pystac.Asset(
            # "https://coclico.blob.core.windows.net/assets/thumbnails/coastal-mask-thumbnail.png",
            THUMBNAIL_PHOTO,  # noqa: E501
            title="Thumbnail",
            media_type=pystac.MediaType.PNG,
        ),
    )
    collection.links = links
    collection.keywords = keywords

    pystac.extensions.item_assets.ItemAssetsExtension.add_to(collection)

    ASSET_EXTRA_FIELDS = {
        "xarray:storage_options": {"token": "google_default"},
    }

    collection.extra_fields["item_assets"] = {
        "data": {
            "type": pystac.MediaType.COG,
            "title": COLLECTION_TITLE,
            "roles": ["data"],
            "description": DATASET_DESCRIPTION,
            **ASSET_EXTRA_FIELDS,
        }
    }

    if extra_fields:
        collection.extra_fields.update(extra_fields)

    pystac.extensions.scientific.ScientificExtension.add_to(collection)
    collection.extra_fields["sci:citation"] = CITATION

    # add coclico frontend properties to collection
    coclico_ext = CoclicoExtension.ext(collection, add_if_missing=True)
    coclico_ext.units = "bool"
    coclico_ext.plot_type = "raster"
    coclico_ext.min = 0
    coclico_ext.max = 1

    return collection



def write_block(
    block: xr.DataArray,
    prefix: str = "",
    href_prefix: str = "",
    x_dim: str = "x",
    y_dim: str = "y",
    storage_options: Optional[Mapping[str, str]] = None,
):
    """
    Write a block of a DataArray to disk.

    Parameters
    ----------
    block : xarray.DataArray
        A singly-chunked DataArray
    prefix : str, default ""
        The prefix to use when writing to disk. This might be just a path prefix
        like "path/to/dir", in which case the data will be written to disk. Or it
        might be an fsspec-uri, in which case the file will be written to that
        file system (e.g. Azure Blob Storage, S3, GCS)
    x_dim : str, default "x"
        The name of the x dimension / coordinate.
    y_dim : str, default "y"
        The name of the y dimension / coordinate.
    storage_options : mapping, optional
        A mapping of additional keyword arguments to pass through to the fsspec
        filesystem class derived from the protocol in `prefix`.

    Returns
    -------
    xarray.DataArray
        A size-1 DataArray with the :class:pystac.Item for that block.

    Examples
    --------
    >>> import xarray as xr

    """
    # this is specific to azure blob storage. We could generalize to accept an fsspec URL.
    import rioxarray  # noqa

    storage_options = storage_options or {}
    blob_name = name_block(block, prefix=prefix, x_dim=x_dim, y_dim=y_dim)

    # map the blob name (str) to serializable filesystem (could also be cloud bucket)
    fs, _, paths = fsspec.get_fs_token_paths(blob_name, storage_options=storage_options)
    if len(paths) > 1:
        raise ValueError("too many paths", paths)
    path = paths[0]
    memfs = fsspec.filesystem("memory")

    with memfs.open("data", "wb") as buffer:
        block.squeeze().rio.to_raster(buffer, driver="COG")
        buffer.seek(0)

        if fs.protocol == "file":
            # can't write a MemoryFile to an io.BytesIO
            # fsspec should arguably handle this
            buffer = buffer.getvalue()
        fs.pipe_file(path, buffer)
        nbytes = len(buffer)

    # make a template for storing the results, probably usefuil for Dask to ensure correct
    # datatypes are returned without making complex meta arguments.
    result = (
        block.isel(**{k: slice(1) for k in block.dims}).astype(object).compute().copy()
    )
    template_item = pystac.Item("id", None, None, datetime.datetime(2000, 1, 1), {})
    item = itemize(
        block,
        template_item,
        nbytes=nbytes,
        x_dim=x_dim,
        y_dim=y_dim,
        prefix=prefix,
        href_prefix=href_prefix,
    )

    # indexing first entry along all dimensions and store pystac item as data value
    result[(0,) * block.ndim] = item
    return result


def itemize(
    block,
    item: pystac.Item,
    nbytes: int,
    *,
    asset_roles: list[str] | None = None,
    asset_media_type=pystac.MediaType.COG,
    prefix: str = "",
    href_prefix: str = "",
    time_dim="time",
    x_dim="x",
    y_dim="y",
) -> pystac.Item:
    """
    Generate a pystac.Item for an xarray DataArray

    The following properties will be be set on the output item using data derived
    from the DataArray:

        * id
        * geometry
        * datetime
        * bbox
        * proj:bbox
        * proj:shape
        * proj:geometry
        * proj:transform

    The Item will have a single asset. The asset will have the following properties set:

        * file:size

    Parameters
    ----------
    block : xarray.DataArray
        A singly-chunked DataArray
    item : pystac.Item
        A template pystac.Item to use to construct.
    asset_roles:
        The roles to assign to the item's asset.
    prefix : str, default ""
        The prefix to use when writing to disk. This might be just a path prefix
        like "path/to/dir", in which case the data will be written to disk. Or it
        might be an fsspec-uri, in which case the file will be written to that
        file system (e.g. Azure Blob Storage, S3, GCS)
    time_dim : str, default "time"
        The name of the time dimension / coordinate.
    x_dim : str, default "x"
        The name of the x dimension / coordinate.
    y_dim : str, default "y"
        The name of the y dimension / coordinate.
    storage_options : mapping, optional
        A mapping of additional keyword arguments to pass through to the fsspec
        filesystem class derived from the protocol in `prefix`.

    Returns
    -------
    xarray.DataArray
        A size-1 DataArray with the :class:pystac.Item for that block.
    """
    import rioxarray  # noqa

    item = item.clone()
    dst_crs = rasterio.crs.CRS.from_epsg(4326)

    bbox = rasterio.warp.transform_bounds(block.rio.crs, dst_crs, *block.rio.bounds())
    geometry = shapely.geometry.mapping(shapely.geometry.box(*bbox))

    # feature = gen_default_item(f"{var}-mapbox-{item_id}")
    # feature.add_asset("mapbox", gen_mapbox_asset(mapbox_url))

    name = pathlib.Path(name_block(block, x_dim=x_dim, y_dim=y_dim)).stem
    item.id = f"{COLLECTION_ID}-{name}"
    # item.id = pathlib.Path(name_block(block, x_dim=x_dim, y_dim=y_dim)).stem
    item.geometry = geometry
    item.bbox = bbox
    # item.datetime = pd.Timestamp(block.coords[time_dim].item()).to_pydatetime()

    ext = pystac.extensions.projection.ProjectionExtension.ext(
        item, add_if_missing=True
    )
    ext.bbox = block.rio.bounds()
    ext.shape = block.shape[-2:]
    ext.epsg = block.rio.crs.to_epsg()
    ext.geometry = shapely.geometry.mapping(shapely.geometry.box(*ext.bbox))
    ext.transform = list(block.rio.transform())[:6]
    ext.add_to(item)

    roles = asset_roles or ["data"]

    # TODO: We need to generalize this `href` somewhat.
    asset = pystac.Asset(
        href=name_block(block, x_dim=x_dim, y_dim=y_dim, prefix=href_prefix),
        media_type=asset_media_type,
        roles=roles,
    )
    asset.extra_fields["file:size"] = nbytes
    item.add_asset("cm", asset)
    # item.add_asset(str(block.band[0].item()), asset)

    return item
def create_asset(
    item, asset_title, asset_href, nodata, resolution, data_type, nbytes=None
):
    asset = pystac.Asset(
        href=asset_href,
        media_type=pystac.MediaType.COG,
        title=asset_title,
        roles=["data"],
    )

    item.add_asset(asset_title, asset)

    pystac.extensions.file.FileExtension.ext(asset, add_if_missing=True)

    if nbytes:
        asset.extra_fields["file:size"] = nbytes

    raster.RasterExtension.ext(asset, add_if_missing=True).bands = [
        raster.RasterBand.create(
            nodata=nodata,
            spatial_resolution=resolution,
            data_type=data_type,  # e.g., raster.DataType.INT8
        )
    ]

    eo.EOExtension.ext(asset, add_if_missing=True).bands = [
        eo.Band.create(
            name=asset_title,
            # common_name=asset_title, # Iff in <eo#common-band-names>`
            description= COLLECTION_TITLE + " for this region.",
        )
    ]
    ...
    return item

def make_template(data: xr.DataArray) -> xr.DataArray:
    """DataArray template for xarray to infer datatypes returned by function.

    Comparable to the dask meta argument.
    """
    offsets = dict(
        zip(
            data.dims,
            [
                np.hstack(
                    [
                        np.array(
                            0,
                        ),
                        np.cumsum(x)[:-1],
                    ]
                )
                for x in data.chunks
            ],
        )
    )
    template = data.isel(**offsets).astype(object)
    return template

def process_block(
    block: xr.DataArray,
    data_type: raster.DataType,  # Make sure to have raster.DataType properly imported
    resolution: int,
    storage_prefix: str = "",
    name_prefix: str = "",
    include_band: str = "",
    time_dim: str = "",
    x_dim: str = "x",
    y_dim: str = "y",
    profile_options: Dict[str, Union[str, int]] = {},
    storage_options: Dict[str, str] = {},
) -> "pystac.Item":
    """
    Process a data block, save it, and return a placeholder STAC item.

    Args:
    - block: The data block.
    - storage_prefix: The storage prefix.
    ... [other parameters]

    Returns:

    - pystac.Item: Placeholder STAC item.
    """

    # Date when Lincke et al. sent Deltares this data
    block = block.assign_coords(time=BLOCK_TIME_COORD)
    # item_name = name_block(
    #     block,
    #     storage_prefix="",
    #     name_prefix=name_prefix,
    #     include_band=None,
    #     time_dim=time_dim,
    #     x_dim=x_dim,
    #     y_dim=y_dim,
    # )


    # item_id = pathlib.Path(item_name).stem
    item_id= ITEM_ID
    item = create_item(block, item_id=item_id)

    for var in block:
        da = block[var]
        # it's more efficient to save the data as unsigned integer, so replace the -9999 nodata values by 0
        da = (
            da.where(da != -9999, 0)
            .astype("uint8")
            .rio.write_nodata(0)
            .rio.set_spatial_dims(x_dim="x", y_dim="y")
        )

        # href = name_block(
        #     da,
        #     storage_prefix=storage_prefix,
        #     name_prefix=name_prefix,
        #     # include_band=da.name,
        #     include_band="",
        #     time_dim=time_dim,
        #     x_dim=x_dim,
        #     y_dim=y_dim,
        # )
        href=HREF

        # uri = to_uri_protocol(href, protocol="gs")

        # TODO: include this file checking
        # if not file_exists(file, str(storage_destination), existing_blobs):
        # nbytes = write_block(da, uri, storage_options, profile_options, overwrite=True)

        memfs = fsspec.filesystem("memory")

        with memfs.open("data", "wb") as buffer:
            da.squeeze().rio.to_raster(buffer, **profile_options)
            buffer.seek(0)

        nbytes = len(buffer.getvalue())

        item = create_asset(
            item,
            asset_title=da.name,
            asset_href=href,
            nodata=da.rio.nodata.item(),  # use item() as this converts np dtype to python dtype
            resolution=resolution,
            data_type=raster.DataType.UINT8,  # should be same as how data is written
            nbytes=nbytes,
        )

    return item

def to_cog_and_stac(
    data: xr.DataArray, prefix="", storage_options=None
) -> List[pystac.Item]:
    template = make_template(data)
    storage_options = storage_options or {}

    r = data.map_blocks(
        write_block,
        kwargs=dict(prefix=prefix, storage_options=storage_options),
        template=template,
    )
    result = r.compute()
    new_items = collate(result)
    return new_items


def collate(items: xr.DataArray) -> List[pystac.Item]:
    """
    Collate many items by id, gathering together assets with the same item ID.
    """
    # flat list with stac items from datarray
    items2 = items.data.ravel().tolist()

    new_items = []
    key = operator.attrgetter("id")

    # TODO: group by osm slippy mapbox box?
    for _, group in itertools.groupby(sorted(items2, key=key), key=key):
        items = list(group)
        item = items[0].clone()
        new_items.append(item)
        # todo: check
        for other_item in items:
            other_item = other_item.clone()
            for k, asset in other_item.assets.items():
                item.add_asset(k, asset)
    return new_items


# rename or swap dimension names, the latter in case the name already exists as coordinate
if __name__ == "__main__":
    # hard-coded input params at project level
    GCS_PROTOCOL = "https://storage.googleapis.com"
    GCS_PROJECT = "DGDS - I1000482-002"
    BUCKET_NAME = "dgds-data-public"
    BUCKET_PROJ = "gca/SOTC"

   # hard-coded input params at project level
    gca_data_dir = pathlib.Path(
        p_drive,
        r"11209197-018-global-coastal-atlas",
        r"MSc_students\ClenmarRowe\Data\All_Datasets",
        r"Orig_Datasets",
    )
    dataset_dir = gca_data_dir.joinpath(r"02_Exposure\Elevation_DEM\DELTADEM")
    DATASET_FILENAME = "Delta_DTM.tif"  # sample from source data

    # opening metadata
    metadata_fp = dataset_dir.joinpath("DeltaDTM_metadata.json")
    with open(metadata_fp, "r") as f:
        metadata = json.load(f)

    # STAC configs
    STAC_DIR = "current"
    TEMPLATE_COLLECTION = "template"  # stac template for dataset collection
    COLLECTION_TITLE = metadata["TITLE"]
    DATASET_DESCRIPTION = metadata["DESCRIPTION"]
    KEYWORDS=metadata["KEYWORDS"]
    # hard-coded input params which differ per dataset
    COLLECTION_ID = "Exp-Delta_DEM_COGs"  # name of stac collection
    TEMPLATE_COLLECTION = "template"  # stac template for dataset collection
    CITATION=metadata["CITATION"]
    START_DATETIME=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    BLOCK_TIME_COORD=pd.Timestamp(2024, 1, 1).isoformat()
    HREF_PREFIX = "https://storage.googleapis.com/dgds-data-public/gca/SOTC/Exp-Delta_DEM_COGs/"
    THUMBNAIL_PHOTO="https://storage.googleapis.com/dgds-data-public/gca/SOTC/Exp-Delta_DEM_COGs/Delta_DTM-thumbnail.png"



    # data configurations
    # DATASET_FILENAME = "Global_merit_coastal_mask_landwards.tif"  # source data
    
    # HOME = pathlib.Path().home()
    # DATA_DIR = HOME.joinpath("data", "src")
    # OUTDIR = pathlib.Path.home() / "data" / "tmp" / "cogs_test"
    # OUTDIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR = dataset_dir

   
    # USE_LOCAL_DATA = True  # can be used when data is also stored locally
    

    # TODO: check what can be customized with layout.
    layout = CoCliCoCOGLayout()

    # if USE_LOCAL_DATA:
    #     DATASET_DIR = (  # overwrite dataset directory if dirname is diferent on local
    #         "coastal_mask"
    #     )
    #     ds_dir = DATA_DIR.joinpath(DATASET_DIR)
    # else:
    ds_dir = gca_data_dir.joinpath(DATASET_DIR)

    if not ds_dir.exists():
        raise FileNotFoundError(f"Data dir does not exist, {str(ds_dir)}")

    # directory to store results

    GCS_PROJECT = "DGDS - I1000482-002"  # deltares cloud
    BUCKET_NAME = "dgds-data-public"  # deltares bucket folder
    BUCKET_PROJ = "gca/SOTC"  # deltares bucket project
    prefix = HREF_PREFIX.replace("https://storage.googleapis.com/dgds-data-public/","")  # Prefix for the folder
    




    def list_tiff_files(bucket_name, prefix=''):
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

        tiff_files = []
        for blob in blobs:
            if blob.name.endswith('.tif') or blob.name.endswith('.tiff'):
                tiff_files.append(blob.name)

        return tiff_files
    



    cred_dir = pathlib.Path(p_drive, "11207608-coclico", "FASTTRACK_DATA")
    load_google_credentials(
        google_token_fp=cred_dir.joinpath("google_credentials.json")
    )

    # aDD with the name of your bucket

    tiff_files = list_tiff_files(BUCKET_NAME, prefix)


    stac_io = CoCliCoStacIO()
    layout = CoCliCoCOGLayout()


    rel_root=pathlib.Path(__file__).parent.parent.parent.parent.parent



    template_fp = os.path.join(
        rel_root, STAC_DIR, TEMPLATE_COLLECTION, "collection.json"
    )


    # collection = get_template_collection(
    #     template_fp=template_fp,
    #     collection_id=COLLECTION_ID,
    #     title=COLLECTION_TITLE,
    #     description=DATASET_DESCRIPTION,
    #     keywords=KEYWORDS,
    # )

    collection = create_collection()

    profile_options = {
    "driver": "COG",
    "dtype": "uint8",
    "compress": "DEFLATE",
    # "interleave": "band",
    # "ZLEVEL": 9,
    # "predictor": 1,
    }
    storage_options = {"token": "google_default"}

    # Print the names of all TIFF files
    # for tiff_file in tqdm(tiff_files[0:10],desc="items added to collection"):
    for tiff_file in tqdm(tiff_files,desc="items added to collection"):
        # print((tiff_file.replace(prefix,""))) DeltaDTM_v1_0_N00E006.tif
        HREF=urljoin( HREF_PREFIX,tiff_file.replace(prefix,""))

        chunk = xr.open_dataset(
        HREF, engine="rasterio", mask_and_scale=False
        )
        chunk = chunk.assign_coords(time=BLOCK_TIME_COORD)


        FULL_NAME=tiff_file[-25:]
        ITEM_ID=FULL_NAME[:-4]

        item = process_block(
            # item = process_block(
            chunk,
            resolution=30,
            data_type=raster.DataType.UINT8,
            storage_prefix=HREF_PREFIX,
            name_prefix="DeltaDTM_v1_0_",
            include_band=False,
            time_dim=False,
            x_dim="x",
            y_dim="y",
            profile_options=profile_options,
            storage_options=storage_options,
        )


        # item = dask.compute(delayed_item)


        collection.add_item(item)
    print(rel_root)
    print(collection.id)
    
    collection.update_extent_from_items()

    catalog = pystac.Catalog.from_file(os.path.join(rel_root, STAC_DIR, "catalog.json"))
    # catalog = pystac.Catalog.from_file(str(STAC_DIR + "/" +"catalog.json"))

    if catalog.get_child(collection.id):
        catalog.remove_child(collection.id)
        print(f"Removed child: {collection.id}.")

    catalog.add_child(collection)

    collection.normalize_hrefs(os.path.join(rel_root,STAC_DIR, collection.id), strategy=layout)
    # collection.normalize_hrefs(str(STAC_DIR +"/" +collection.id), strategy=layout)

    catalog.save(
        catalog_type=pystac.CatalogType.SELF_CONTAINED,
        dest_href=os.path.join(rel_root, STAC_DIR),
        stac_io=CoCliCoStacIO(),
    )
    print("Catalog Updated")
    print(catalog)
    print(collection)
    # %%
    # TODO: # check coastal_mask_stacs.py validate funcs with coclico_new..
    print("validating collections")
    # collection.validate_all()

    # # %%
    print("validating catalog")
    # catalog.validate_all()
    print("done")
