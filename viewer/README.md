# GCA viewer

Small web map that shows the datasets published in the Global Coastal Atlas STAC catalog from the
`17_` to `21_` notebooks (`STAC/data/scripts`), for an area of interest (default: Egypt).
Layers can be switched on and off, have an opacity slider, a legend (WMS) and a click query.

| Layer | STAC collection | How it is shown |
|---|---|---|
| GEBCO 2025 bathymetry / elevation | `gebco` | GeoServer WMS layer group `gebco:gebco` |
| Seabed lithology | `seabed_lithology` | GeoServer WMS `seabed_lithology:seabed_litho` |
| Groundwater recharge (RF, RF_RK) | `groundwater_recharge` | GeoServer WMS `groundwater_recharge:<item>` |
| HydroLAKES | `hydrolakes` | GeoParquet read for the bbox, drawn as GeoJSON |
| Country boundaries | `country_boundaries` | GeoParquet read for the bbox, drawn as GeoJSON |

None of the layers has a time dimension.

## How it works
```
STAC catalog (gs://gca-data-public/gca/gca-stac-7/catalog.json)
        │ pystac: title, description, licence, extent, data/visual asset hrefs
FastAPI (main.py)
  /api/layers               layer list with STAC metadata
  /api/vector/{id}?bbox=    GeoParquet from the bucket -> features that intersect the bbox -> GeoJSON
  /api/wms                  proxy to the GeoServer WMS (GetMap, GetFeatureInfo, GetLegendGraphic)
        │
OpenLayers page (static/index.html)
```
- **WMS**: the GeoServer is on the internal network with a self-signed certificate, so the browser never talks
  to it directly: the backend proxy does (only the layers listed in `main.py` are allowed). The WMS URL is taken
  from the `visual` asset of the STAC items (`WMS_URL` overrides it). The pod/computer running the viewer
  must reach the GeoServer (Deltares network or VPN).
- **Vector layers**: HydroLAKES is 1 GB, so the viewer never loads it entirely. The parquet row groups are skipped
  from their `bbox` statistics and only the rows inside the bbox are read (about 6 s for Egypt). Up to
  `MAX_FEATURES` (20000, the biggest ones) are sent, simplified to the bbox size.
- **Bounding box**: the box in the side panel can be typed, drawn on the map or reset to the default. *Apply* zooms
  to it and re-reads the vector layers for it.
- Seabed lithology and the others use the GeoServer default (grey) style until SLD styles exist.

## Settings (environment variables)
| Variable | Default |
|---|---|
| `STAC_CATALOG_URL` | `https://storage.googleapis.com/gca-data-public/gca/gca-stac-7/catalog.json` |
| `DEFAULT_BBOX` | `24.0,21.0,37.5,32.0` (Egypt) |
| `WMS_URL` | from the STAC `visual` asset |
| `MAX_FEATURES` | `20000` |

## Run on Kubernetes (Docker Desktop / kind)
Same steps as `freshem_stac`: a local registry on `localhost:5000`, one Deployment and one ClusterIP Service.
```bash
docker start local-registry          # once, or: docker run -d -p 5000:5000 --restart=always --name local-registry registry:2
cd viewer
docker build -t gca-viewer:local .
docker tag gca-viewer:local localhost:5000/gca-viewer:local
docker push localhost:5000/gca-viewer:local
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl rollout status deploy/gca-viewer
kubectl port-forward svc/gca-viewer 8183:8000
```
Open <http://localhost:8183>. After changing the code: build, tag, push and `kubectl rollout restart deploy/gca-viewer`.

## Run with Docker only
```bash
docker run --rm -p 8183:8000 gca-viewer:local
```
