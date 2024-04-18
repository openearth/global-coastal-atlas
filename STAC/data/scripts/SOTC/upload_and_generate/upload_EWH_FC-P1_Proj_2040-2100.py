# %%
import pathlib
import sys
from importlib.resources import path
import os
import pandas as pd
from dask.diagnostics import ProgressBar
from tqdm import tqdm

# make modules importable when running this file as script

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
    BUCKET_PROJ = "gca/SOTC/Haz-Offshore_Waves/Future_Climate/2040-2100_Phase1_WW3"  # deltares bucket project
    MAPBOX_PROJ = "global-data-viewer"  # mapbox project

    # hard-coded input params at project level
    gca_data_dir = pathlib.Path(
        p_drive,
        r"11209197-018-global-coastal-atlas",
        r"MSc_students\ClenmarRowe\Data\All_Datasets",
        r"Orig_Datasets",
    )
    # dataset_dir = gca_data_dir.joinpath(r"02_Future\Extreme_Wave_Height\GCM_All_8_CMIP6_Hindcast_1993-2014\ACCESS-CM2")
    cred_dir = pathlib.Path(p_drive, "11207608-coclico", "FASTTRACK_DATA")

    VARIABLES = [
        "hs_1_26","hs_5_85"]

    GCM_all=["EC-EARTH3","ACCESS-CM2"]

    for i,GCM in tqdm(enumerate(GCM_all),desc="GCM completed"):# Project paths & files (manual input)
 
        dataset_dir = gca_data_dir.joinpath(r"01_Hazards\02_Future\Extreme_Wave_Height",f"GCM_{GCM}_Projection_2040-2100")
     
        IN_FILENAME = f"{GCM}_WW3_3Hourly_2040-2100.zarr"
        OUT_FILENAME = f"Haz-{GCM}_Projected_40-100.zarr"

          # what variable(s) do you want to show as marker color?
        # dimensions to include, i.e. what are the dimensions that you want to use as to affect the marker color (never include stations). These will be the drop down menu's. Note down without n.. in front.
        ADDITIONAL_DIMENSIONS = ["time"]
        # use these to reduce dimension like {ensemble: "mean", "time": [1995, 2020, 2100]}, i.e. which of the dimensions do you want to use. Also specify the subsets (if there are a lot maybe make a selection). These will be the values in the drop down menu's. If only one (like mean), specify a value without a list to squeeze the dataset. Needs to span the entire dim space (except for (n)stations).
        MAP_SELECTION_DIMS = {
            # "time": pd.date_range(start='2004-01-01', end='2014-01-01', freq='AS').to_numpy(),  # TODO: fix bug. If I put ["ssp126", "ssp245", "ssp585"] if doesn't work properly as nscenarios and scenarios are disconnected in the variable. If I put nscenarios = [0,1,2] (like for ensembles) I get an error because of the disconnection
            # "Time_Horizon": [2000, 2050, 2100]
            "time": ['2040-01-01', '2100-12-31']      
        }
        # which dimensions to ignore (if n... in front of dim, it goes searching in additional_dimension for dim without n in front (ntime -> time). Except for nstations, just specify station in this case). This spans up the remainder of the dimension space.
        DIMENSIONS_TO_IGNORE = ["latitude","longitude"]  # dimensions to ignore

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
    


# %%
