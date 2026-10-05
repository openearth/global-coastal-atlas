# Local GeoServer for GCA salinity WMS (optional)

## What this is
A Docker GeoServer used by notebook step **10) Publish layer to Geoserver WMS locally**.
It serves local COGs produced by `11b_Salinity.ipynb` / `11b_Salinity_increase.ipynb`.

## Start
1. Edit `.env` and set `STAC_FOLDER_HOST_PATH` to your `stac_folder` parent
   (must contain `salinity/cogs` and/or `salinity_increase/cogs`).
   Prefer a local disk path (e.g. `C:/Ocean/.../stac_folder`); network/`P:` mounts often fail in Docker.
2. From this directory:

```powershell
docker compose up -d
```

3. Wait until healthy, then open http://localhost:8085/geoserver  
   Login: `admin` / `geoserver`

4. Run section 10 in `11_salinity.ipynb` or `11_salinity_increase.ipynb`.

## Stop
```powershell
docker compose down
```

## Paths inside the container
| Host | Container |
|------|-----------|
| `%STAC_FOLDER_HOST_PATH%/salinity/cogs/...` | `/data/stac_folder/salinity/cogs/...` |
| `%STAC_FOLDER_HOST_PATH%/salinity_increase/cogs/...` | `/data/stac_folder/salinity_increase/cogs/...` |

## WMS example
```text
http://localhost:8085/geoserver/salinity/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities
```
