# Dataset metadata files

Every dataset that gets published to the Global Coastal Atlas STAC catalog needs a
`metadata_<dataset_name>.json` file. This file is:

1. **Written** by a preprocessing notebook in `STAC/data/notebooks/` (e.g.
   `13_crop_productivity_correction.ipynb`), right after the data itself
   (GeoParquet / NetCDF / Zarr / COG) has been produced. The notebook fills in most
   values automatically (e.g. spatial/temporal extent derived from the data) and
   writes the JSON next to the processed data.
2. **Read** by the matching STAC-builder script/notebook in `STAC/data/scripts/`
   (e.g. `13_crop_productivity_correction.ipynb`), which loads the JSON with
   `json.load()` and uses its keys to populate the `pystac.Collection` / `pystac.Item`
   (title, description, license, providers, extent, units, keywords, etc.), and to
   name the collection (`COLLECTION_ID` / `TITLE_ABBREVIATION`) and the WMS layer
   (`WMS_DATASET`).

Use the files in this folder as a starting point for a new dataset:

- `metadata_template.json` — empty template with every recognised key. Copy it to
  `metadata_<your_dataset_name>.json` and fill it in (or let your preprocessing
  notebook fill/refresh it, as `13_crop_productivity_correction.ipynb` does).
- `metadata_crop_productivity_correction.json` — a real, filled-in example.

## Required vs. optional fields

The data provider should fill in **as much of the template as possible** — the
more context they give (title, description, licence, citation, units,
provenance, etc.), the less guesswork is needed later. A field is only marked
**Optional** below when it can be *derived without extra input*, either because:

- it can be computed directly by inspecting the processed file (CRS, bounding
  box, resolution, dtype, no-data value, column/dimension names — see
  `STAC/data/notebooks/14_groundwater_exploration.ipynb` for an example of how
  this inspection is done for rasters/shapefiles), or
- it is a Global Coastal Atlas cataloguing convention (an id, abbreviation, or
  media type) that the person who uploads the dataset to the STAC catalog
  assigns while publishing it, and does not depend on information only the data
  provider would know.

Everything else is **Required** and should come from the data provider (or be
confirmed with them), because it captures information that cannot be
reconstructed from the file alone (title, description, licence, citation,
authorship, units, provenance, keywords).

## Field reference

| Key | Meaning | Required |
|---|---|---|
| `TITLE` | Human-readable dataset title, used as the STAC collection/item title. Should be provided by the data provider. | Required |
| `TITLE_ABBREVIATION` | Short code/slug for the dataset. Used by several publish scripts as `COLLECTION_ID` (`metadata["TITLE_ABBREVIATION"]`), i.e. the STAC collection id. Keep it short, lowercase/CamelCase, no spaces. Should be provided by the data provider if they have a preferred short name; otherwise assigned by the person who uploads the dataset to the STAC catalog. | Optional |
| `DESCRIPTION` | Full description of the dataset, used as the STAC collection/item description. Should be provided by the data provider. | Required |
| `SHORT_DESCRIPTION` | One-line summary, e.g. for cards/tooltips in the frontend. Should be provided by the data provider; can otherwise be shortened from `DESCRIPTION` by the person who uploads the dataset. | Required |
| `INSTITUTION` | Institution associated with the dataset (CF-style global attribute), e.g. `"Deltares"`. Should be provided by the data provider. | Required |
| `PROVIDERS` | Object with `name`, `url`, `roles`, `description` of the data provider. Mapped to `pystac.Provider` entries on the collection. Should be provided by the data provider (name/url/description of who produced the data). | Required |
| `HISTORY` | List of processing steps/provenance notes (CF-style `history` attribute), e.g. `["Deltares", "Food-Security salinity_correction"]`. Should be provided by the data provider (original provenance); the person uploading the dataset appends their own processing steps. | Required |
| `MEDIA_TYPE` | STAC/IANA media type of the data asset, e.g. `application/vnd.apache.parquet` for GeoParquet, or the COG/Zarr/NetCDF media type. Determined by the person who uploads the dataset, based on the output format they chose. | Optional |
| `DATA_MODEL` | Internal label describing the data structure, e.g. `geoparquet_table`, `raster_cog`, `zarr_datacube`. Used to decide how the publish script should build items/assets. Assigned by the person who uploads the dataset. | Optional |
| `DIMENSIONS` | List of the dataset's logical dimensions (e.g. `["year", "area_map_name"]`, or `["lon", "lat", "time"]`). Can be derived by inspecting the processed data's columns/dimensions. | Optional |
| `SPATIAL_EXTENT` | `[west, south, east, north]` bounding box in `CRS`. Can be computed directly from the processed data's geometry/bounds (see the "Refresh temporal/spatial extent" cell in `13_crop_productivity_correction.ipynb`). | Optional |
| `TEMPORAL_EXTENT` | `[start_iso, end_iso]` ISO-8601 datetimes. Can be computed directly from the data (e.g. min/max `year` column), same as `SPATIAL_EXTENT`. | Optional |
| `LICENSE` | Licence text/name, e.g. `"Creative Commons Attribution 4.0"`. Must match the licence the source data was released under — should be provided by the data provider. | Required |
| `AUTHOR` | Author(s) of the dataset/derived product. Should be provided by the data provider. | Required |
| `KEYWORDS` | List of search keywords shown on the STAC collection (`pystac.Collection.keywords`). Should be provided by the data provider, since they best know the relevant domain terms. | Required |
| `TAGS` | Additional free-form tags (not necessarily surfaced in STAC, used for internal search/filtering). Should be provided by the data provider. | Required |
| `CITATION` | How the dataset should be cited. Should be provided by the data provider (use the citation they request). | Required |
| `DOI` | DOI of the source dataset/paper, if any. Should be provided by the data provider; use an empty string if none exists. | Required |
| `LONG_NAME` | CF-style descriptive name of the main variable, e.g. `"Corrected crop yield"`. Should be provided by the data provider. | Required |
| `UNITS` | Units of the main variable(s), e.g. `"m"`, `"t; ha; FTE"` for multiple columns. Mapped to the Coclico STAC extension `units` property. Should be provided by the data provider (native units). | Required |
| `COMMENT` | Free-text notes on processing/joins/sources, e.g. which shapefile/CSV was joined and how. Should be provided by the data provider, and/or completed by the person who uploads the dataset with processing notes. | Required |
| `CRS` | Coordinate reference system of the processed data, e.g. `"EPSG:4326"`. Can be read directly from the processed file. | Optional |
| `ITEM_BBOX_CRS` | CRS used for the STAC item bounding boxes (usually `EPSG:4326`, as required by the STAC spec). Set by convention by the person who uploads the dataset (normally always `EPSG:4326`). | Optional |
| `SPATIAL_RESOLUTION` | Native spatial resolution of raster data (e.g. in metres/degrees). Can be read directly from the raster file; `null` for non-raster/table data. | Optional |
| `NODATA` | No-data value used in raster data. Can be read directly from the raster file; `null` if not applicable (e.g. tabular/GeoParquet data). | Optional |
| `DATA_TYPE` | Data type of the main variable, e.g. `"float64"`, `"float32"`, `"int16"`. Can be read directly from the processed data's dtype. For raster publish scripts this is mapped to a `raster.DataType` enum, so it must be one of the supported types (see `DATA_TYPE_MAP` in `11_salinity.ipynb`/`11_salinity_increase.ipynb`). | Optional |
| `COLLECTION_ID` | Machine id of the STAC collection (folder name under `STAC/data/current/`, used to build the storage href). Some scripts use `TITLE_ABBREVIATION` instead — keep both consistent. Assigned by the person who uploads the dataset to the STAC catalog. | Optional |
| `WMS_DATASET` | Name of the WMS/GeoServer dataset/layer used to build the local WMS URL for map visualisation. Only relevant for raster datasets served via GeoServer. Should be provided by the person who uploads the dataset to the STAC catalog. | Optional |

## Notes on some special values

- Set `SPATIAL_RESOLUTION`, `NODATA` to `null` for tabular/vector datasets
  (GeoParquet, CSV-derived tables) where they don't apply.
- `PROVIDERS.roles` in the JSON is a simple string for readability; the publish
  script always adds Deltares/GCA as an additional `processor`/`host` provider in
  code, so you don't need to encode every role here.
- Keep `TITLE_ABBREVIATION`/`COLLECTION_ID`/`WMS_DATASET` short, lowercase (or
  snake_case) and consistent with the folder name used under
  `STAC/data/current/<collection_id>/` so the publish scripts and frontend can
  find the data.

## Workflow

1. Copy `metadata_template.json` to `metadata_<your_dataset>.json`.
2. Fill in the values you already know (title, description, provider, licence,
   citation, units, etc.).
3. In your preprocessing notebook (`STAC/data/notebooks/xx_....ipynb`), load this
   JSON, refresh `SPATIAL_EXTENT`/`TEMPORAL_EXTENT` from the processed data, and
   write the final file next to the processed data output (see the "Metadata JSON
   for STAC publish script" cell in `13_crop_productivity_correction.ipynb` for a
   worked example).
4. In the matching publish script/notebook (`STAC/data/scripts/xx_....ipynb`),
   load the JSON and build the STAC collection/items from it.
