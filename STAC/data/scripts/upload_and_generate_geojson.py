# %%
import pathlib
import sys
from importlib.resources import path
import os

# make modules importable when running this file as script
# sys.path.append(str(pathlib.Path(__file__).parent.parent))

import geojson
import xarray as xr
from coclicodata.etl.cloud_utils import (
    p_drive,
    dataset_to_google_cloud,
    dataset_from_google_cloud,
    geojson_to_mapbox,
    load_env_variables,
    load_google_credentials,
)
from coclicodata.etl.extract import (
    clear_zarr_information,
    get_geojson,
    get_mapbox_url,
    zero_terminated_bytes_as_str,
)
from coclicodata.coclico_stac.utils import (
    get_dimension_dot_product,
    get_dimension_values,
    get_mapbox_item_id,
    rm_special_characters,
)

if __name__ == "__main__":
    # hard-coded input params
    GCS_PROJECT = "DGDS - I1000482-002"  # deltares cloud
    BUCKET_NAME = "dgds-data-public"  # deltares bucket folder
    BUCKET_PROJ = "gca"  # deltares bucket project
    MAPBOX_PROJ = "global-data-viewer"  # mapbox project

    # hard-coded input params at project level
    gca_data_dir = pathlib.Path(
        p_drive,
        "11208003-latedeo2022",
        "020_InternationalDeltaPortfolio",
        "datasets",
    )
    dataset_dir = gca_data_dir.joinpath(r"00_mapping_global_threat_of_land_subsidence")
    cred_dir = pathlib.Path(p_drive, "11205479-coclico", "FASTTRACK_DATA")
    IN_FILENAME = "Global_TLS.zarr"  # original filename as on P drive
    OUT_FILENAME = "Global_TLS.zarr"  # file name in the cloud and on MapBox
    VARIABLES = [
        "eapa",
        "egdp",
        "epsi",
    ]  # what variable(s) do you want to show as marker color?
    # dimensions to include, i.e. what are the dimensions that you want to use as to affect the marker color (never include stations). These will be the drop down menu's. Note down without n.. in front.
    ADDITIONAL_DIMENSIONS = ["time"]
    # use these to reduce dimension like {ensemble: "mean", "time": [1995, 2020, 2100]}, i.e. which of the dimensions do you want to use. Also specify the subsets (if there are a lot maybe make a selection). These will be the values in the drop down menu's. If only one (like mean), specify a value without a list to squeeze the dataset. Needs to span the entire dim space (except for (n)stations).
    MAP_SELECTION_DIMS = {
        "time": [2010, 2040],
    }
    # which dimensions to ignore (if n... in front of dim, it goes searching in additional_dimension for dim without n in front (ntime -> time). Except for nstations, just specify station in this case). This spans up the remainder of the dimension space.
    DIMENSIONS_TO_IGNORE = ["stations"]  # dimensions to ignore

    # TODO: safe cloud creds in password client
    load_env_variables(env_var_keys=["MAPBOX_ACCESS_TOKEN"])
    load_google_credentials(
        google_token_fp=cred_dir.joinpath("google_credentials.json")
    )

    # TODO: come up with checks for data

    # upload data to gcs from local drive
    source_data_fp = dataset_dir.joinpath(IN_FILENAME)

    # UNCOMMENT TO UPDATE GOOGLE CLOUD
    dataset_to_google_cloud(
        ds=source_data_fp,
        gcs_project=GCS_PROJECT,
        bucket_name=BUCKET_NAME,
        bucket_proj=BUCKET_PROJ,
        zarr_filename=OUT_FILENAME,
    )

    # read data from gcs
    ds = dataset_from_google_cloud(
        bucket_name=BUCKET_NAME, bucket_proj=BUCKET_PROJ, zarr_filename=OUT_FILENAME
    )

    # # read data from local source
    # fpath = pathlib.Path.home().joinpath(
    #     "data", "tmp", "shoreline_change_projections.zarr"
    # )
    # fpath = dataset_dir.joinpath(IN_FILENAME)
    # ds = xr.open_zarr(fpath)

    ds = zero_terminated_bytes_as_str(ds)

    # remove characters that cause problems in the frontend.

    ds = rm_special_characters(
        ds=ds, dimensions_to_check=ADDITIONAL_DIMENSIONS, characters=["%"]
    )

    # This dataset has quite some dimensions, so if we would parse all information the end-user
    # would be overwhelmed by all options. So for the stac items that we generate for the frontend
    # visualizations a subset of the data is selected. Of course, this operation is dataset specific.
    for k, v in MAP_SELECTION_DIMS.items():
        if k in ds.dims and ds.coords:
            ds = ds.sel({k: v})
        else:
            try:
                # assume that coordinates with strings always have same dim name but with n
                ds = ds.sel({"n" + k: k == v})
            except:
                raise ValueError(f"Cannot find {k}")

    if len(ADDITIONAL_DIMENSIONS) > 0:
        dimvals = get_dimension_values(ds, dimensions_to_ignore=DIMENSIONS_TO_IGNORE)
        dimcombs = get_dimension_dot_product(dimvals)
    else:
        dimcombs = []

    for var in VARIABLES:
        collection = get_geojson(
            ds,
            variable=var,
            dimension_combinations=dimcombs,
            stations_dim=(  # note, make nstations if stations has string, else make stations
                "stations"
            ),
        )

        # save feature collection as geojson in tempdir and upload to cloud
        with dataset_dir.joinpath("platform") as outdir:
            # with tempfile.TemporaryDirectory() as outdir:
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            # TODO: put this in a function because this is also used in generate_stac scripts?
            mapbox_url = get_mapbox_url(
                MAPBOX_PROJ, OUT_FILENAME, var, add_mapbox_protocol=False
            )

            fn = mapbox_url.split(".")[1]

            fp = pathlib.Path(outdir, fn).with_suffix(".geojson")

            with open(fp, "w") as f:
                # load
                print(f"Writing data to {fp}")
                geojson.dump(collection, f)
            print("Done!")

            # Note, if mapbox cli raises an util collection error, this should be monkey
            # patched. Instructions are in documentation of the function.
            geojson_to_mapbox(source_fpath=fp, mapbox_url=mapbox_url)

# %%
