# -*- coding: utf-8 -*-
# ===============================================
# Panos Athanasiou
# 05/04/2020
# extract indicators for the coastal points
# ===============================================
import xarray as xr
import os
import geopandas as gpd
import time
import glob
import numpy as np
import pandas as pd
from coast_info.geo_utils import dir_average
##INPUTS###############################################################################################################
path = r'2_Analysis'
vars_paths = ['Hs', 'Tp', 'MWD']
vars = ['swh', 'pp1d', 'mwd']
#######################################################################################################################
inds = {}

for i, var in enumerate(vars):
    ds = xr.open_dataset(os.path.join(path, 'era5_coastal_{}_all_years.nc'.format(vars_paths[i])),
                         engine="netcdf4", chunks={'points': 200})
    if var != 'mwd':
        quan = ds[var].quantile([0.5, 0.95], dim='time')
        inds['{}_p50'.format(var)] = quan.sel(quantile=0.5).values
        inds['{}_p95'.format(var)] = quan.sel(quantile=0.95).values

Hs = xr.open_dataset(os.path.join(path, 'era5_coastal_Hs_all_years.nc'), engine="netcdf4")
MWD = xr.open_dataset(os.path.join(path, 'era5_coastal_MWD_all_years.nc'), engine="netcdf4")
# Tp = xr.open_dataset(os.path.join(path, 'era5_coastal_Tp_all_years.nc'), engine="netcdf4")

MWD_p95 = []
for point in range(len(Hs.points)):
    mask = Hs.swh[:, point] >= inds['swh_p95'][point]
    MWD_p95.append(dir_average(MWD.mwd[mask, point]))
inds['mwd_p95'] = MWD_p95
inds['lon'] = ds.longitude.data
inds['lat'] = ds.latitude.data

df = pd.DataFrame(inds)
df.to_csv(os.path.join(path, 'era5_coastal_indices.csv'))





