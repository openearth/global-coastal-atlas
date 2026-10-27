# Combined_Preprocessing_AFRICA_VH – documentation

Notebook: `Combined_Preprocessing_AFRICA_VH.ipynb`

STAC version (reads its inputs from the STAC catalog and from local files): `Combined_Preprocessing_AFRICA_VH_STAC.ipynb`. Same 8 steps; see section 0 of that notebook.

## 0. Environment to run the pipeline

Both notebooks run in the conda environment `gca-exploration`, defined in `STAC/data/scripts/exploration/environment_exploration.yml` (Python 3.11, numpy, pandas, matplotlib, rasterio, geopandas, pyarrow, rasterstats, fsspec, pystac-client, jupyterlab, ipykernel and others).

Install it with miniforge (or conda), from the repository root:

```
mamba env create -f STAC/data/scripts/exploration/environment_exploration.yml
conda activate gca-exploration
python -m ipykernel install --user --name gca-exploration
```

If the environment already exists, update it instead (needed to get `rasterstats` for step 8):

```
mamba env update -n gca-exploration -f STAC/data/scripts/exploration/environment_exploration.yml
```

Then open the notebook in VS Code or JupyterLab (`jupyter lab`) and select the kernel `gca-exploration`.

Before running the STAC version, edit the setup cell (section 0 of the notebook):
- `MODEL_DIR`: output folder.
- `DOMAIN_PATH`, `INLAND_PATH`, `ARIDITY_PATH`, `UNCONSOLIDATED_PATH`: local files that are not in the STAC. An empty or wrong path skips only the outputs that need that file.

Internet access is needed to read the STAC (public bucket `gca-data-public`); no credentials are required. Reading the Africa area takes some minutes and a few GB of memory.

Purpose: prepare the input data of the continental-scale groundwater flow model of Africa (variable-density flow and salt transport, iMOD/MODFLOW). The notebook turns about 20 global/continental datasets into a stack of rasters on one common model grid, plus a MODFLOW RIV package.

Status of this document: written from reading the notebook code. Nothing was executed. File names are those hard-coded in the notebook.

## 1. Model grid (used by every step)

| Item | Value |
|---|---|
| CRS | EPSG:3857 |
| Extent (xmin, ymin, xmax, ymax) | -2 145 000, -4 440 000, 5 965 000, 4 645 000 m |
| Resolution | 10 km for the model (811 columns × 908 rows); 5 km for the intermediate geology rasters |
| NoData | -9999 (intermediate), -999 (final continuous rasters), 255 (masks), NaN (layer rasters of step 7) |

## 2. Overview: inputs → transformations → outputs

| Step | Name | Main inputs | Transformation | Main outputs |
|---|---|---|---|---|
| 1 | Soil hydraulic conductivity | SoilGrids clay and sand, 6 depth intervals, 5 km | Reproject to the grid, thickness-weighted mean over depth, silt = 100 − sand − clay, Cosby pedotransfer function | `clay/sand/silt_weighted_5km_3857.tif`, `hk_soil_cosby_5km.tif` |
| 2 | Global sediment thickness (GST) | Bird & Mooney sediment thickness as XYZ points (shapefile) | Points → raster at 0.125° | `GST_python_epsg4326.tif` |
| 3 | Harmonization | GEBCO, soil depth, recharge, aridity, unconsolidated thickness, GST, HK soil, domain, inland, HydroLAKES, GLHYMPS | Reproject to the 10 km grid, clean, mask, convert units, rasterize masks | `*_10km.tif` rasters, `GLHYMPS_Africa_3857.gpkg` |
| 4 | HK consolidated | GLiM | Lithology class (first 4 characters of `Litho`) → HK lookup → rasterize at 5 km | `hk_consolidated_5km_xxyy.tif` |
| 5 | HK unconsolidated | GUM, GLiM (`su**` classes), consolidated HK, inland mask | GUM class → HK; GLiM fills gaps; default HK 0.1 on land without other data | `hk_unconsolidated_5km_GUM.tif`, `hk_uncon_5km_GLIM.tif`, `hk_unconsolidated_5km.tif` |
| 6 | Porosity | SoilGrids fractions, GLHYMPS, GLiM | Soil: texture classes. Unconsolidated: GLHYMPS `Porosity_x`. Consolidated: GLiM lookup. Then 5 → 10 km | `porosity_layer1/2/3_5km.tif` and `_10km.tif` |
| 7 | Numerical layers | Thicknesses, HK and porosity per geological unit, seabed lithology, inland mask | 6 numerical layers, dominant geology per layer, VK = 0.1·HK | `hk/vk/porosity_layer1..6.tif`, `*_seabed.tif`, `dominant_geology_layer1..6.tif` |
| 8 | RIV package | HydroRIVERS, GEBCO, soil thickness, VK layer 1, aridity, inland mask, domain | River geometry per reach, CRIV per 10 km cell, stage and bed elevation | `riv_package_10x10_c5.0.txt` + 2 diagnostics CSV |

## 3. Step by step

### Step 1 – Soil HK from SoilGrids (Cosby)

- Input: `clay_5km_<depth>.tif`, `sand_5km_<depth>.tif` for `0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm, 100-200cm`.
- Reproject to EPSG:3857 / 5 km (`Resampling.average`). Nodata → NaN.
- Weighted mean over depth with weights 0.05, 0.10, 0.15, 0.30, 0.40, 1.00 (the interval thicknesses in m).
- Silt = 100 − sand − clay, clipped to 0–100.
- Cosby: `K[cm/s] = 0.0070556 · 10^(−0.884 + 0.0153·sand%)`; converted to m/day (`·0.01·86400`); clipped to 1e-5 … 100.
- Output: `clay/sand/silt_weighted_5km_3857.tif` (% , float32) and `hk_soil_cosby_5km.tif` (m/day).

### Step 2 – GST points → raster

- Input: shapefile `gst_points_epsg4326_small.shp`, value field `meter` (sediment thickness, m).
- Each point is written into a 0.125° cell (extent snapped to the grid). Empty cells stay NaN.
- Output: `GST_python_epsg4326.tif` (EPSG:4326, nodata -9999).
- STAC alternative: collection `gst` (`gst.tif`) holds the same data as a global COG (`31_gst` notebooks). Each point is a cell centre, so the raster origin is (-180.0625, 90.0625). Reading it replaces this step.

### Step 3 – Harmonization to the 10 km grid

Inputs and resampling:

| Input | Resampling | Treatment after reprojection |
|---|---|---|
| GEBCO (`Africa_GEBCO.tif`) | bilinear | NaN → 0, not masked |
| Soil thickness (`Africa_SOIL.tif`, cm) | bilinear | cm → m, NaN → 0, negatives → 0, set to 0 outside the inland mask |
| Recharge (`RF_RK_recharge_01.tif`, mm/year) | bilinear | NaN → 0, set to 0 outside the inland mask, mm/year → m/day (`/1000/365`) |
| GST (`GST_python_epsg4326.tif`) | nearest | NaN → 0, negatives → 0, set to 0 outside the domain |
| Unconsolidated thickness (`..._WORLD_SU_thk_est_mean_bestfit...tif`) | bilinear | NaN → 0, negatives → 0, set to 0 outside the domain |
| HK soil (`hk_soil_cosby_5km.tif`) | average | NaN kept, set to NaN outside the inland mask and where soil thickness is 0 |
| Aridity index (`ai_v31_yr.tif`) | bilinear | Domain mask, NaN → 0, divided by 10 000 |

Vectors:

| Input | Treatment |
|---|---|
| Domain polygon (`Africa_domain_epsg3857.shp`) | Rasterized: 1 inside, 0 outside → `domain_10km.tif` |
| Inland polygon (`Africa_inland.shp`) | Rasterized → `inland_10km.tif` (land = 1, ocean = 0) |
| HydroLAKES polygons | Intersected with the domain, rasterized → `lakes_10km.tif` |
| GLHYMPS 2.0 polygons | Reprojected, bounding-box crop to the grid → `GLHYMPS_Africa_3857.gpkg` |

Final rasters in `OUT_DIR` (`...\IMOD\Inputs\10_x_10_km`): `gebco_10km.tif` (m), `soil_thickness_10km.tif` (m), `recharge_10km.tif` (m/day), `gst_10km.tif` (m), `unconsolidated_10km.tif` (m), `hk_soil_10km.tif` (m/day), `aridity_index_10km.tif` (–), `domain_10km.tif`, `inland_10km.tif`, `lakes_10km.tif`. The `*_raw.tif` files are intermediates. All are checked against the grid (CRS, size, transform).

### Step 4 – HK of consolidated sediments (GLiM)

- Input: GLiM clipped to Africa, column `Litho`.
- Class = first four letters of `Litho` in lower case (`xxyy`).
- HK lookup (m/day): 0.01 for evaporites, metamorphic, plutonic and volcanic; 0.1 for fine or pyroclastic sedimentary classes; 0.2 for pure carbonate; 0.5 for mixed; 1 for coarse. `su**` (unconsolidated), `wb` (water) and `nd` (no data) get the bedrock value 0.01, because bedrock lies below them. Unknown classes get 0.01.
- Output: `hk_consolidated_5km_xxyy.tif` (m/day, nodata -9999).

### Step 5 – HK of unconsolidated sediments (GUM + GLiM)

- GUM: column `XX` (class code such as `Au`, `Ed`) → HK lookup (0 to 10 m/day). Water classes `Wl`, `Wu` are removed. Rasterized at 5 km (`hk_unconsolidated_5km_GUM.tif`).
- GLiM backup: only `su**` classes, lookup 0.1 to 10 m/day, default 0.001 (`hk_uncon_5km_GLIM.tif`; the code uses `res = 10000` although the name says 5 km, the merge resamples it).
- Merge: GUM first, GLiM where GUM is empty. Land cells with neither unconsolidated nor consolidated data get HK 0.1.
- Output: `hk_unconsolidated_5km.tif`.

### Step 6 – Effective porosity per geological unit

| Layer | Source | Rule |
|---|---|---|
| 1 Soil | Weighted sand, silt and clay | sand ≥ 70 % → 0.40; silt ≥ 50 % → 0.30; clay ≥ 40 % → 0.10; otherwise 0.25 |
| 2 Unconsolidated | GLHYMPS `Porosity_x` | divided by 100; values < 0 or > 1 removed; minimum 0.01 |
| 3 Consolidated | GLiM `Litho` (first four letters) | lookup 0.01–0.25; `su**` 0.25; `wb` 0.1; `nd` and unknown 0.05 |

All three are rasterized at 5 km and then resampled to 10 km with bilinear interpolation: `porosity_layer1_10km.tif`, `porosity_layer2_10km.tif`, `porosity_layer3_10km.tif` (units: fraction).

### Step 7 – Numerical layers (6 layers)

Inputs (from the `10_x_10_km\6_ml` folder): `inland_10km`, `soil_thickness_10km`, `unconsolidated_10km`, `gst_10km`, `hk_soil_10km`, `hk_unconsolidated_10km`, `hk_consolidated_10km_xxyy`, `porosity_soil_10km`, `porosity_uncon_10km`, `porosity_cons_10km`, `seabed_litho_epsg3857` (13 seabed sediment classes).

Transformation:

- Consolidated thickness = GST − (soil + unconsolidated), minimum 0.
- VK = 0.1 · HK for each unit.
- Numerical layer thickness: L1 = max(soil, 10 m); L2 and L3 = half each of max(unconsolidated, 20 m); L4 and L5 = half each of max(consolidated, 20 m); L6 = 100 m (basal layer, constant salinity reservoir).
- For each layer, the geological unit with the largest vertical overlap is the dominant unit and gives HK, VK and porosity. Layers below the mapped geology get bedrock (HK 0.05, VK 0.005, porosity 0.01).
- Offshore (inland = 0): layers 1–5 default HK 0.1, VK 0.05, porosity 0.1; layer 6 bedrock. Layers 1 and 2 are overwritten by seabed lithology properties (13 classes, HK 0.05–10 m/day, porosity 0.35–0.70, VK = 0.1·HK).

Outputs (float32, NaN nodata): `hk_layer1..6.tif`, `vk_layer1..6.tif`, `porosity_layer1..6.tif`, `hk_seabed.tif`, `vk_seabed.tif`, `porosity_seabed.tif`, `dominant_geology_layer1..6.tif` (0 outside, 1 soil, 2 unconsolidated, 3 consolidated, 4 bedrock, 5 seabed sediment).

### Step 8 – RIV package (HydroRIVERS)

- Inputs: HydroRIVERS Africa (`DIS_AV_CMS` = mean discharge), `gebco_10km.tif`, GEBCO 1 km in EPSG:3857, `soil_thickness_10km.tif`, `vk_layer1.tif`, `inland_10km.tif`, aridity raster, domain polygon.
- Rivers are clipped to the domain, kept when discharge > 0 and when they lie on land. A 10 km grid is intersected with the rivers to get river length per cell.
- Width `W = 7.2·Q^0.5·f_w` (1–700 m) and depth `D = 0.27·Q^0.4·f_d` (0.5–50 m). The factors depend on elevation class: mountain (> 1000 m) 0.4 / 2.5, transition (300–1000 m) 0.7 / 1.5, lowland 1 / 1.
- Riverbed thickness 0.1 / 10 / 50 m and riverbed K 0.1 / 2 / 5 m/day (mountain / transition / lowland), K limited to VK of layer 1.
- Cell values: length = maximum over reaches (capped at 10 km), width = root-sum-square, depth, K and thickness = mean.
- Conductance: `C_geom = K·L·W / thickness`; `C_climate = cell_area · river_fraction(aridity) / c(aridity)`; the larger of the two is used. River cells in the ocean are removed.
- Stage = 5th percentile of the 1 km DEM in the cell, limited to between DEM − 50 m and DEM − 0.1 m. Bed elevation = stage − depth.
- Output: `riv_package_10x10_c5.0.txt` with one line per cell: `layer row col stage conductance bed_elev` (layer 1, row/col 0-based). Diagnostics: `river_conductance_diagnostics.csv`, `river_conductance_summary.csv`.

## 4. Are the input datasets in the STAC catalog?

Catalog checked: `https://storage.googleapis.com/gca-data-public/gca/gca-stac-7/catalog.json`.

| Input in the notebook | Step | In STAC | STAC collection | Remarks |
|---|---|---|---|---|
| SoilGrids clay, 6 depths (`clay_5km_<depth>.tif`) | 1, 6 | **Yes** | `soil_clay_cog` (also `soil_clay_zarr`) | Same item names (`clay_5km_0-5cm` …) |
| SoilGrids sand, 6 depths | 1, 6 | **Yes** | `soil_sand_cog` (also `soil_sand_zarr`) | |
| Soil thickness / depth to bedrock (`Africa_SOIL.tif`) | 3 | **Yes** | `soil_depth` | Global; the notebook uses an Africa clip in cm |
| GEBCO (`Africa_GEBCO.tif`, 1 km) | 3, 8 | **Yes** | `gebco` | Global; clip to Africa |
| Groundwater recharge (`RF_RK_recharge_01.tif`) | 3 | **Yes** | `groundwater_recharge` | RF and RF+residual kriging (RF_RK) |
| HydroLAKES | 3 | **Yes** | `hydrolakes` | |
| GLHYMPS 2.0 | 3, 6 | **Yes** | `glhymps` | Column `Porosity_x` present |
| GLiM | 4, 5, 6 | **Yes** | `glim` | Column `Litho` present |
| GUM | 5 | **Yes** | `gum` | The notebook reads column `XX`; the STAC layer also has `XX` (class code), `YY` and `ZZ`, so it can be used directly |
| HydroRIVERS | 8 | **Yes** | `hydrorivers` | Column `DIS_AV_CMS` present |
| Seabed lithology (`seabed_litho_epsg3857.tif`) | 7 | **Yes** | `seabed_lithology` | Rasterization to EPSG:3857 is done outside the notebook |
| Global sediment thickness (Bird & Mooney, `gst_points...shp`) | 2, 3, 7 | **Yes** | `gst` | Stored as a global 0.125° COG (`gst.tif`, nodata -9999), not as points. Step 2 can be replaced by reading it and clipping to Africa; step 3 resamples it to 10 km (nearest) |
| Unconsolidated sediment thickness (`..._WORLD_SU_thk_est_mean_bestfit_thk_mWORLD_BESTFIT.tif`) | 3, 7 | **No** | – | |
| Aridity index (`ai_v31_yr.tif`, Global Aridity Index v3.1) | 3, 8 | **No** | – | |
| Model domain polygon (`Africa_domain*.shp`) | 3, 8 | **No** | – | Project-specific; `country_boundaries` could be a starting point |
| Inland polygon (`Africa_inland.shp`) | 3, 7, 8 | **No** | – | Project-specific land/ocean mask |

Summary: 12 of 16 inputs are in the STAC. Missing: unconsolidated thickness, aridity index, and the two project polygons (domain, inland).

Note: STAC holds global data. The notebook uses Africa clips, so the data must be clipped to the model extent after reading.

## 5. Points to check in the notebook

- All paths are absolute (`C:\GROUNDWATCH\...`). Two different output roots are used: `...\Modeling\AFRICA\IMOD\Inputs\...` (steps 3–5) and `...\Modeling\AFRICA\Inputs\...` (step 8).
- Step 7 reads files that this notebook does not produce under those names: `hk_unconsolidated_10km`, `hk_consolidated_10km_xxyy` (steps 4–5 write 5 km files), `porosity_soil_10km`, `porosity_uncon_10km`, `porosity_cons_10km` (step 6 writes `porosity_layer1/2/3_10km`), and `seabed_litho_epsg3857`. The 5 → 10 km resampling of the HK rasters and the renaming/copying into `6_ml` happen outside the notebook.
- Step 8 reads `aridity_index_clipped_3857.tif`, not the `aridity_index_10km.tif` written in step 3.
- The `RF_RK` recharge variant is used (the STAC collection also holds the plain RF variant).
