# Global Coastal Atlas viewer

Web map for exploring the datasets of the Global Coastal Atlas STAC catalog. Each layer can be switched on and
off, has an opacity slider, a legend (where available) and a click query. The area of interest is set with a
bounding box (default: Egypt).

## Contents

- [Layers](#layers)
- [Features](#features)
- [Architecture](#architecture)
- [Run with Docker Compose](#run-with-docker-compose)
- [Configuration](#configuration)
- [API](#api)
- [Project structure](#project-structure)

## Layers

| Dataset | STAC collection | Display |
|---|---|---|
| GEBCO 2025 bathymetry / elevation | `gebco` | WMS |
| Seabed lithology | `seabed_lithology` | WMS |
| Groundwater recharge (RF, RF_RK) | `groundwater_recharge` | WMS |
| Absolute depth to bedrock | `soil_depth` | WMS |
| Global sediment thickness (GST-1) | `gst` | WMS |
| Soil clay content, six depth intervals | `soil_clay_cog` | WMS |
| Soil sand content, six depth intervals | `soil_sand_cog` | WMS |
| Soil clay content, depth cube | `soil_clay_zarr` | Zarr, rendered by the backend |
| Soil sand content, depth cube | `soil_sand_zarr` | Zarr, rendered by the backend |
| HydroLAKES | `hydrolakes` | GeoParquet, drawn as GeoJSON |
| Country boundaries | `country_boundaries` | GeoParquet, drawn as GeoJSON |
| GLHYMPS 2.0 hydrogeology (Africa) | `glhymps` | GeoParquet, drawn as GeoJSON |
| GLiM lithological map (Africa) | `glim` | GeoParquet, drawn as GeoJSON |
| GUM unconsolidated sediments (Africa) | `gum` | GeoParquet, drawn as GeoJSON |
| HydroRIVERS (Africa) | `hydrorivers` | GeoParquet, drawn as GeoJSON |

Layer titles, descriptions, licences and extents are read from the STAC catalog.

## Features

- Layer list grouped by dataset, with visibility toggle, opacity slider and legend.
- Click query: feature attributes for vector layers, pixel values for WMS layers.
- Bounding box that can be typed, drawn on the map or reset to the default. *Apply* zooms to the box and reloads the vector layers for it.
- Soil depth cubes (Zarr): depth interval selector, depth profile at a clicked point, and a depth section along a drawn transect (distance versus depth).

## Architecture

```
STAC catalog (Google Cloud Storage, public)
        │  pystac: title, description, licence, extent, asset links
FastAPI backend (main.py)
  ├─ /api/layers          layer list with STAC metadata
  ├─ /api/vector/{id}     GeoParquet from the bucket → features inside the bbox → GeoJSON
  ├─ /api/wms             proxy to the GeoServer WMS
  ├─ /api/zarr/{id}/...   Zarr cubes read over HTTPS → map, depth profile, transect
  └─ /api/config          viewer settings
        │
OpenLayers page (static/index.html)
```

- **WMS layers.** The GeoServer is on an internal network with a self-signed certificate, so the browser does not
  contact it directly. The backend proxies GetMap, GetFeatureInfo and GetLegendGraphic requests for the layers
  registered in `main.py`. The WMS URL is taken from the `visual` asset of the STAC items and can be overridden
  with `WMS_URL`. The machine running the viewer must be able to reach the GeoServer (Deltares network or VPN).
- **Vector layers.** Large GeoParquet files (HydroLAKES is about 1 GB) are never loaded entirely. Row groups are
  skipped using their `bbox` statistics and only rows inside the requested bounding box are read. At most
  `MAX_FEATURES` features (the largest ones) are returned, simplified to the scale of the bounding box.
- **Zarr cubes.** The backend reads the cubes over HTTPS and renders the requested depth interval, profile or
  transect.

## Run with Docker Compose

### Requirements

- Docker Desktop (Windows, macOS) or Docker Engine with the Compose plugin. Check with `docker compose version`.
- Internet access to the STAC catalog.
- For the WMS layers: access to the Deltares network or VPN.

### Start

From the `viewer` folder:

```bash
docker compose up -d --build
```

The first build takes a few minutes while the Python packages are installed. Check the status, which should
show `healthy` after a few seconds:

```bash
docker compose ps
```

Open **<http://localhost:8183>**. A quick check without the browser is <http://localhost:8183/api/layers>.

### Commands

| Action | Command |
|---|---|
| Follow the logs | `docker compose logs -f` |
| Stop and remove the container | `docker compose down` |
| Rebuild after changing `main.py` or `static/` | `docker compose up -d --build` |
| Change the port (for example to 9000) | set `ports` in `docker-compose.yml` to `"9000:8000"` |

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `port is already allocated` | Port 8183 is in use. Change the port as shown above. |
| The map opens but WMS layers are empty | The machine cannot reach the GeoServer. Connect to the Deltares network or VPN. |
| `docker compose` is not recognised | Update Docker Desktop, or use the legacy `docker-compose` command. |

## Configuration

Settings are environment variables. Set them in the shell or in a `.env` file next to `docker-compose.yml`, then
run `docker compose up -d` again.

| Variable | Default | Description |
|---|---|---|
| `STAC_CATALOG_URL` | `https://storage.googleapis.com/gca-data-public/gca/gca-stac-7/catalog.json` | STAC catalog to read |
| `DEFAULT_BBOX` | `24.0,21.0,37.5,32.0` | Initial bounding box `minx,miny,maxx,maxy` (Egypt) |
| `WMS_URL` | from the STAC `visual` asset | GeoServer WMS endpoint |
| `MAX_FEATURES` | `20000` | Maximum number of vector features per request |

Example `.env`:

```
DEFAULT_BBOX=-20.0,-5.0,10.0,20.0
WMS_URL=https://<geoserver>/geoserver/wms
```

## API

| Endpoint | Description |
|---|---|
| `GET /api/health` | Health check |
| `GET /api/config` | Viewer settings (default bounding box) |
| `GET /api/layers` | Layer list with STAC metadata |
| `GET /api/vector/{id}?bbox=minx,miny,maxx,maxy` | Vector layer as GeoJSON for the bounding box |
| `GET /api/wms` | WMS proxy (GetMap, GetFeatureInfo, GetLegendGraphic) |
| `GET /api/zarr/{id}/map` | Zarr cube rendered for a depth interval |
| `GET /api/zarr/{id}/profile` | Depth profile at a point |
| `GET /api/zarr/{id}/transect` | Depth section along a transect |

## Project structure

```
viewer/
├── main.py              FastAPI backend
├── static/index.html    OpenLayers front end
├── requirements.txt     Python dependencies
├── Dockerfile           Container image
└── docker-compose.yml   Service definition
```
