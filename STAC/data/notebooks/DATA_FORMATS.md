# Choosing a cloud-optimized format for preprocessing

Guide to pick the output format when preparing a dataset for the STAC catalog.
Examples only reference the numbered notebooks (`NN_*.ipynb`) in this folder and in the `coclicodata` repository.

## Decision table

| Data type | Output format | What it is | When to use | Examples |
|-----------|---------------|------------|-------------|----------|
| 2D raster, one layer (single image) | **COG** (single Cloud Optimized GeoTIFF) | A GeoTIFF with internal tiling and overviews, so clients read only the needed window over HTTP. | Regular grid, one variable, file size manageable (up to a few GB). | `10_Land_subsidence_prediction_maps_updated` (`ds.rio.to_raster(..., driver="COG", compress="DEFLATE")`), `14_groundwater_exploration` (`rio_cogeo.cog_translate` with the `deflate` profile and overviews), `17_seabed_litho` (categorical int16, nearest-neighbour overviews), `20_groundwater_recharge` (two 0.1° float32 model variants, one COG each, average overviews, published as two items of one collection) |
| 2D raster, several scenarios/years/probabilities | **COG per layer** (one COG per scenario/year, organised in folders) | Same COG as above, one file per layer; the STAC item/asset points at each file. | Time or scenario dimension is small and layers are consumed independently. | `11a_Salinity_preprocessing` (XYZ to COG, naming `{probability}_{year}.tif`), `11b_Salinity`, `11b_Salinity_increase` |
| 2D raster, very large extent or very high resolution | **COG tiles** (a grid of COGs, e.g. one per tile/region, plus a STAC item per tile) | A large raster split spatially into several COGs so no single file becomes unwieldy. | Global or continental bbox at high resolution where a single COG would be too big to produce or handle. | coclicodata `25_cfhp` (flood maps, EU extent: array chunked in blocks of `2**15` px with `x_slice`/`y_slice`, each block written with `rio.to_raster(..., compress="DEFLATE", driver="COG")`, per return period/scenario/time), `21_gebco` (global 15 arc-second bathymetry, 4 int16 COG tiles of 21600 px with average overviews, one STAC item per tile, WMS tile layers grouped in one GeoServer layer group) |
| 3D/4D gridded data (x, y + depth/time/ensemble) | **Zarr** (chunked n-D array store) | Chunked, compressed multi-dimensional arrays with CF metadata, readable lazily with xarray. | Datacubes where several dimensions are sliced independently. | `15_resistivity`, `16_chloride` (x, y, z datacubes, `to_zarr(..., zarr_format=2)`) |
| Time series at stations/points (variable x time x station) | **Zarr** | Stations or transects as a dimension, time as a CF `time` coordinate. | Per-location time series or scenario tables. | `12_crop_production` (`time`, `station`), `09_Subsidence` (point time series), `01_shorelinemonitor`, `02_shorelinemonitor_highres`, `03_shorelinemonitor_future` |
| Large point tables (millions of rows, flattened with an `index` dimension) | **Zarr** | Tabular records stored as 1D arrays along `index`, with CF attributes. | Very large point/transect datasets where each column is a variable. | `04_beachsediment`, `05_worldpop`, `06_gdp`, `07_drivers`, `08_ESLbyGWL` |
| Vector data (polygons/lines/points with attributes) | **GeoParquet** | Columnar, compressed vector format with geometry column, readable by geopandas/DuckDB. | Features with attributes, e.g. administrative zones or shapefile replacements. | `13_crop_productivity_correction` (CSV joined to shapefile, `gdf.to_parquet`), `14_groundwater_exploration` (`.shp` to `.parquet`), `18_hydrolakes` (1.4 M lake polygons, `.shp` to a single zstd GeoParquet with bbox covering column), `19_country_boundaries` (258 Natural Earth polygons, `.shp` to a single GeoParquet) |

## Additional examples from coclicodata

Notebooks in `coclicodata\notebooks` (numbered ones only, plus `FASTTRACK`).

| Data type | Output format | Examples |
|-----------|---------------|----------|
| 2D raster, single COG | COG | `FASTTRACK\19_coastal_mask` (TIF to CF compliant COG), `26_pp` (population projections, single-COG test first with `driver="COG"`, then loop over all TIFs) |
| 2D raster per time step / scenario | COG per layer | `13_slp` (sea level projections, one COG per year, `compress="DEFLATE"`), `17_slp_ar6`, `18_slp_ar5` (`write_cog` from datacube, one COG per scenario/time/quantile) |
| 2D raster, very large extent | COG tiles | `25_cfhp` (see tile row above) |
| Gridded or multi-dimensional NetCDF (time, ensemble, return period) | Zarr | `01_storm_surge`, `02_wave_energy`, `03_sea_level`, `04_shoreline_evolution`, `05_shoreline_change` (NetCDF to CF compliant Zarr) |
| Points/stations with many locations | Zarr | `19_ss_wc` (51010 locations), `20_twl` (51010 locations), `FASTTRACK\13_slp` (stations) |
| Zarr plus GeoJSON for map display | Zarr + GeoJSON | `06_coastal_adaptation`, `07_flood_risk` |
| Vector from GeoPackage/shapefile/CSV | GeoParquet | `21_cet` (GeoPackage to parquet), `33_cba`, `33_cba_hr` (CSV to parquet), `99_LAU_NUTS` (shapefile to parquet), `28_ceed_LAU` |
| Statistics/derived vector layers from rasters | GeoParquet (+ GPKG) | `25_cfhp_stats`, `25_cfhp_stats_pre`, `26_pp_stats`, `27_bc_stats`, `27_be_stats` |

## Rules of thumb

- One variable on a regular 2D grid: **COG**. Always write CRS, nodata and metadata (see `11b_Salinity`).
- Add overviews and compression (`DEFLATE`); keep the original `.tif` extension.
- Extra dimension beyond x/y (depth, time, ensemble): **Zarr**, unless the number of layers is small enough to publish as separate COGs.
- Vector with attributes: **GeoParquet**, reprojected to EPSG:4326 for STAC.
- Split into **COG tiles** only if a single COG is too large. Prefer a single COG otherwise.
- STAC bboxes must be in EPSG:4326, whatever the native CRS (see `15_resistivity`).
