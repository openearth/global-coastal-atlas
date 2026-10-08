import json
import logging
import os
import re
import threading
from functools import lru_cache

import fsspec
import numpy as np
import pyarrow.parquet as pq
import pystac
import requests
import shapely
import urllib3
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("gca-viewer")
logging.basicConfig(level=logging.INFO)

# Published GlobalCoastalAtlas STAC static catalog (public GCS bucket)
STAC_CATALOG_URL = os.environ.get(
    "STAC_CATALOG_URL",
    "https://storage.googleapis.com/gca-data-public/gca/gca-stac-7/catalog.json",
)
# GeoServer WMS endpoint. Default: taken from the "visual" asset of the STAC items.
WMS_URL_OVERRIDE = os.environ.get("WMS_URL")
# The internal GeoServer uses a self-signed certificate
WMS_VERIFY_SSL = os.environ.get("WMS_VERIFY_SSL", "false").lower() == "true"
# Area shown when the viewer opens: minx,miny,maxx,maxy (lon/lat). Default: Egypt and its coasts.
DEFAULT_BBOX = [float(v) for v in os.environ.get("DEFAULT_BBOX", "24.0,21.0,37.5,32.0").split(",")]
# Vector layers are read for the bbox only; above this number of features the smallest ones are dropped
MAX_FEATURES = int(os.environ.get("MAX_FEATURES", "20000"))

if not WMS_VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# One entry per map layer. Title, description, extent and licence come from the STAC collection;
# the GeoServer layer names are the ones published by the STAC/data/scripts/17_ ... 21_ notebooks.
LAYERS = [
    {"id": "gebco", "collection": "gebco", "type": "wms", "wms_layers": "gebco:gebco", "title": "GEBCO 2025 bathymetry / elevation", "default": False},
    {"id": "seabed_lithology", "collection": "seabed_lithology", "type": "wms", "wms_layers": "seabed_lithology:seabed_litho", "title": "Seabed lithology", "default": True},
    {"id": "recharge_rf", "collection": "groundwater_recharge", "item": "RF_recharge_01", "type": "wms", "wms_layers": "groundwater_recharge:RF_recharge_01", "title": "Groundwater recharge, random forest (RF)", "default": False},
    {"id": "recharge_rf_rk", "collection": "groundwater_recharge", "item": "RF_RK_recharge_01", "type": "wms", "wms_layers": "groundwater_recharge:RF_RK_recharge_01", "title": "Groundwater recharge, RF + residual kriging (RF_RK)", "default": False},
    {"id": "hydrolakes", "collection": "hydrolakes", "type": "vector", "title": "HydroLAKES", "default": False},
    {"id": "country_boundaries", "collection": "country_boundaries", "type": "vector", "title": "Country boundaries", "default": True},
]
LAYER_BY_ID = {layer["id"]: layer for layer in LAYERS}
ALLOWED_WMS_LAYERS = {layer["wms_layers"] for layer in LAYERS if layer["type"] == "wms"}

# Columns sent to the browser for the vector layers (those missing in the parquet are skipped)
VECTOR_COLUMNS = {
    "hydrolakes": ["Hylak_id", "Lake_name", "Country", "Continent", "Lake_area", "Depth_avg", "Vol_total", "Elevation"],
    "country_boundaries": ["NAME", "ADMIN", "ISO_A3", "CONTINENT", "SUBREGION", "POP_EST"],
}
# Column used to keep the biggest features when a bbox holds more than MAX_FEATURES
VECTOR_SIZE_COLUMN = {"hydrolakes": "Lake_area"}

_catalog = None
_catalog_lock = threading.Lock()


def get_catalog() -> pystac.Catalog:
    global _catalog
    with _catalog_lock:
        if _catalog is None:
            _catalog = pystac.Catalog.from_file(STAC_CATALOG_URL)
        return _catalog


def get_item(layer: dict):
    """STAC collection and item of a layer (the item named in the config, else the first one)."""
    collection = get_catalog().get_child(layer["collection"])
    if collection is None:
        raise LookupError(f"Collection {layer['collection']} not found in {STAC_CATALOG_URL}")
    items = list(collection.get_items())
    item = next((i for i in items if i.id == layer.get("item")), None) if layer.get("item") else None
    return collection, item or (items[0] if items else None)


def wms_base_url() -> str:
    if WMS_URL_OVERRIDE:
        return WMS_URL_OVERRIDE
    for layer in LAYERS:
        if layer["type"] == "wms":
            _, item = get_item(layer)
            visual = item.assets.get("visual") if item else None
            if visual:
                return re.sub(r"/wms/[^/]+/?$", "/wms", visual.href)  # .../geoserver/wms/<workspace> -> .../geoserver/wms
    raise LookupError("No WMS asset found in the STAC items")


def layer_metadata(layer: dict) -> dict:
    out = {k: layer[k] for k in ("id", "type", "title", "default")}
    out["wms_layers"] = layer.get("wms_layers")
    try:
        collection, item = get_item(layer)
        out.update(
            group=collection.title or collection.id,
            description=collection.description,
            license=collection.license,
            bbox=collection.extent.spatial.bboxes[0],
            collection=collection.id,
            data_href=item.assets["data"].href if item else None,
        )
    except Exception as exc:  # the layer stays usable without its STAC metadata
        log.warning("STAC metadata for %s not available: %s", layer["id"], exc)
        out.update(group=layer["collection"], description="", license=None, bbox=None, collection=layer["collection"], data_href=None)
    return out


def read_vector(layer_id: str, bbox: tuple) -> bytes:
    """GeoJSON of the features of a GeoParquet (read from the bucket) that intersect the bbox.

    Row groups are skipped from their bbox statistics, so only the needed part of the file is downloaded
    (HydroLAKES is 1 GB, but a country-sized bbox touches one row group).
    """
    minx, miny, maxx, maxy = bbox
    _, item = get_item(LAYER_BY_ID[layer_id])
    href = item.assets["data"].href
    log.info("Reading %s for bbox %s from %s", layer_id, bbox, href)
    size_col = VECTOR_SIZE_COLUMN.get(layer_id)
    rows = []  # (size, wkb, properties)
    with fsspec.open(href).open() as f:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        columns = [c for c in VECTOR_COLUMNS[layer_id] if c in names]
        has_bbox = "bbox" in names
        meta = pf.metadata
        col_index = {meta.schema.column(i).path: i for i in range(meta.num_columns)}
        for rg in range(pf.num_row_groups):
            if has_bbox and not row_group_overlaps(meta.row_group(rg), col_index, bbox):
                continue
            if has_bbox:
                b = pf.read_row_group(rg, columns=["bbox"]).column("bbox").combine_chunks()
                mask = (
                    (b.field("xmax").to_numpy(zero_copy_only=False) >= minx)
                    & (b.field("xmin").to_numpy(zero_copy_only=False) <= maxx)
                    & (b.field("ymax").to_numpy(zero_copy_only=False) >= miny)
                    & (b.field("ymin").to_numpy(zero_copy_only=False) <= maxy)
                )
                keep = np.flatnonzero(mask)
                if keep.size == 0:
                    continue
                table = pf.read_row_group(rg, columns=columns + ["geometry"]).take(keep)
            else:  # no covering column: test the geometries themselves
                table = pf.read_row_group(rg, columns=columns + ["geometry"])
            geoms = shapely.from_wkb(table.column("geometry").to_pylist())
            hit = shapely.intersects(geoms, shapely.box(minx, miny, maxx, maxy))
            props = table.drop_columns(["geometry"]).to_pylist()
            wkbs = table.column("geometry").to_pylist()
            for i in np.flatnonzero(hit):
                rows.append((props[i].get(size_col) or 0 if size_col else 0, wkbs[i], props[i]))

    if len(rows) > MAX_FEATURES:
        rows = sorted(rows, key=lambda r: r[0], reverse=True)[:MAX_FEATURES]
    tolerance = max(maxx - minx, maxy - miny) / 3000  # simplify to roughly one screen pixel of the whole bbox
    features = []
    for _, wkb, prop in rows:
        geom = shapely.set_precision(shapely.simplify(shapely.from_wkb(wkb), tolerance), tolerance / 10)
        if geom.is_empty:
            continue
        features.append('{"type":"Feature","geometry":%s,"properties":%s}' % (shapely.to_geojson(geom), json.dumps(prop, default=str)))
    log.info("%s: %d features", layer_id, len(features))
    return ('{"type":"FeatureCollection","features":[' + ",".join(features) + "]}").encode()


def row_group_overlaps(rg_meta, col_index: dict, bbox: tuple) -> bool:
    """False only when the row group statistics prove that no row can intersect the bbox."""
    minx, miny, maxx, maxy = bbox
    try:
        def stat(name):
            return rg_meta.column(col_index[name]).statistics
        return (
            stat("bbox.xmax").max >= minx and stat("bbox.xmin").min <= maxx
            and stat("bbox.ymax").max >= miny and stat("bbox.ymin").min <= maxy
        )
    except Exception:  # statistics not written: read the row group
        return True


@lru_cache(maxsize=32)
def cached_vector(layer_id: str, bbox: tuple) -> bytes:
    return read_vector(layer_id, bbox)


def parse_bbox(text: str) -> tuple:
    try:
        minx, miny, maxx, maxy = (round(float(v), 4) for v in text.split(","))
    except ValueError:
        raise HTTPException(400, "bbox must be minx,miny,maxx,maxy")
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise HTTPException(400, "bbox must be in lon/lat degrees (minx < maxx, miny < maxy)")
    return minx, miny, maxx, maxy


app = FastAPI(title="Global Coastal Atlas viewer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {"default_bbox": DEFAULT_BBOX, "max_features": MAX_FEATURES}


@app.get("/api/layers")
def layers():
    return [layer_metadata(layer) for layer in LAYERS]


@app.get("/api/vector/{layer_id}")
def vector(layer_id: str, bbox: str):
    if layer_id not in LAYER_BY_ID or LAYER_BY_ID[layer_id]["type"] != "vector":
        raise HTTPException(404, "Unknown vector layer")
    try:
        data = cached_vector(layer_id, parse_bbox(bbox))
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Reading %s failed", layer_id)
        raise HTTPException(502, f"Could not read {layer_id}: {exc}")
    return Response(data, media_type="application/geo+json", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/wms")
def wms(request: Request):
    """Proxy to the GeoServer WMS (internal network, self-signed certificate): GetMap, GetFeatureInfo, GetLegendGraphic."""
    params = dict(request.query_params)
    for key, value in params.items():
        if key.upper() in ("LAYERS", "LAYER", "QUERY_LAYERS"):
            if any(name not in ALLOWED_WMS_LAYERS for name in value.split(",")):
                raise HTTPException(403, f"Layer not allowed: {value}")
    try:
        upstream = requests.get(wms_base_url(), params=params, verify=WMS_VERIFY_SSL, timeout=120)
    except requests.RequestException as exc:
        raise HTTPException(502, f"GeoServer not reachable: {exc}")
    return Response(
        upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("Content-Type", "application/octet-stream"),
        headers={"Cache-Control": "public, max-age=300"},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
