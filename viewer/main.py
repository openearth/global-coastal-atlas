import json
import logging
import math
import os
import re
import struct
import threading
import zlib
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
# the GeoServer layer names are the ones published by the STAC/data/scripts/17_ ... 21_ notebooks (22_ ... 25_ are GeoParquet vector layers).
LAYERS = [
    {"id": "gebco", "collection": "gebco", "type": "wms", "wms_layers": "gebco:gebco", "title": "GEBCO 2025 bathymetry / elevation", "default": False},
    {"id": "seabed_lithology", "collection": "seabed_lithology", "type": "wms", "wms_layers": "seabed_lithology:seabed_litho", "title": "Seabed lithology", "default": True},
    {"id": "recharge_rf", "collection": "groundwater_recharge", "item": "RF_recharge_01", "type": "wms", "wms_layers": "groundwater_recharge:RF_recharge_01", "title": "Groundwater recharge, random forest (RF)", "default": False},
    {"id": "recharge_rf_rk", "collection": "groundwater_recharge", "item": "RF_RK_recharge_01", "type": "wms", "wms_layers": "groundwater_recharge:RF_RK_recharge_01", "title": "Groundwater recharge, RF + residual kriging (RF_RK)", "default": False},
    {"id": "hydrolakes", "collection": "hydrolakes", "type": "vector", "title": "HydroLAKES", "default": False},
    {"id": "country_boundaries", "collection": "country_boundaries", "type": "vector", "title": "Country boundaries", "default": True},
    {"id": "glhymps", "collection": "glhymps", "type": "vector", "title": "GLHYMPS 2.0 hydrogeology (Africa)", "default": False},
    {"id": "glim", "collection": "glim", "type": "vector", "title": "GLiM lithological map (Africa)", "default": False},
    {"id": "gum", "collection": "gum", "type": "vector", "title": "GUM unconsolidated sediments (Africa)", "default": False},
    {"id": "hydrorivers", "collection": "hydrorivers", "type": "vector", "title": "HydroRIVERS (Africa)", "default": False},
    {"id": "soil_depth", "collection": "soil_depth", "type": "wms", "wms_layers": "soil_depth:soil_depth", "title": "Absolute depth to bedrock (cm)", "default": False},
    {"id": "gst", "collection": "gst", "type": "wms", "wms_layers": "gst:gst", "title": "Global sediment thickness GST-1 (m)", "default": False},
]
# Soil clay / sand (26_ ... 30_ notebooks): one COG layer per depth interval on GeoServer, plus one Zarr depth cube
# per variable (drawn by this backend; supports depth profiles and transects through the depth dimension)
SOIL_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
SOIL_VARIABLES = {"clay": ("Clay", (0.0, 60.0)), "sand": ("Sand", (0.0, 100.0))}  # name: (label, colour range in %)
for _var, (_label, _) in SOIL_VARIABLES.items():
    for _d in SOIL_DEPTHS:
        LAYERS.append({
            "id": f"soil_{_var}_{_d}", "collection": f"soil_{_var}_cog", "item": f"{_var}_5km_{_d}", "type": "wms",
            "wms_layers": f"soil_{_var}_cog:{_var}_5km_{_d}", "title": f"{_label} content {_d.replace('cm', ' cm')}",
            "group": f"Soil {_var} content, one layer per depth (COG, GeoServer)", "default": False,
        })
for _var, (_label, _range) in SOIL_VARIABLES.items():
    LAYERS.append({
        "id": f"soil_{_var}_zarr", "collection": f"soil_{_var}_zarr", "type": "zarr", "variable": _var, "range": _range,
        "title": f"{_label} content, depth cube (Zarr)", "group": f"Soil {_var} content, depth cube (Zarr)", "default": False,
    })
LAYER_BY_ID = {layer["id"]: layer for layer in LAYERS}
ALLOWED_WMS_LAYERS = {layer["wms_layers"] for layer in LAYERS if layer["type"] == "wms"}

# Columns sent to the browser for the vector layers (those missing in the parquet are skipped)
VECTOR_COLUMNS = {
    "hydrolakes": ["Hylak_id", "Lake_name", "Country", "Continent", "Lake_area", "Depth_avg", "Vol_total", "Elevation"],
    "country_boundaries": ["NAME", "ADMIN", "ISO_A3", "CONTINENT", "SUBREGION", "POP_EST"],
    "glhymps": ["OBJECTID_1", "Descriptio", "Porosity_x", "logK_Ice_x", "Transmissi", "Shape_Area"],
    "glim": ["OBJECTID", "Litho", "xx", "Shape_Area"],
    "gum": ["OBJECTID", "Symbol", "Descriptio", "Shape_Area"],
    "hydrorivers": ["HYRIV_ID", "LENGTH_KM", "UPLAND_SKM", "DIS_AV_CMS", "ORD_STRA", "ORD_FLOW"],
}
# Column used to keep the biggest features when a bbox holds more than MAX_FEATURES
VECTOR_SIZE_COLUMN = {
    "hydrolakes": "Lake_area", "glhymps": "Shape_Area", "glim": "Shape_Area", "gum": "Shape_Area", "hydrorivers": "DIS_AV_CMS",
}

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
            group=layer.get("group") or collection.title or collection.id,
            description=collection.description,
            license=collection.license,
            bbox=collection.extent.spatial.bboxes[0],
            collection=collection.id,
            data_href=item.assets["data"].href if item else None,
        )
    except Exception as exc:  # the layer stays usable without its STAC metadata
        log.warning("STAC metadata for %s not available: %s", layer["id"], exc)
        out.update(group=layer.get("group") or layer["collection"], description="", license=None, bbox=None, collection=layer["collection"], data_href=None)
    if layer["type"] == "zarr":
        try:
            cube = get_cube(layer["id"])
            out["zarr"] = {
                "depths": [{"top": t, "bottom": b} for t, b in zip(cube.top, cube.bottom)],
                "vmin": layer["range"][0], "vmax": layer["range"][1], "units": cube.units, "variable": layer["variable"],
            }
        except Exception as exc:
            log.warning("Zarr cube %s not available: %s", layer["id"], exc)
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
        geom = shapely.from_wkb(wkb)
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
        geom = shapely.set_precision(shapely.simplify(geom, tolerance), tolerance / 10)
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


class Cube:
    """A (depth, y, x) Zarr datacube on the public bucket, read with zarr directly (no xarray needed)."""

    def __init__(self, href: str, variable: str):
        import zarr
        from zarr.storage import FsspecStore

        store = FsspecStore.from_url(href, read_only=True, storage_options={"asynchronous": True})
        group = zarr.open_group(store=store, mode="r", zarr_format=2)  # no consolidated metadata in these stores
        self.arr = group[variable]
        self.x = group["x"][:]
        self.y = group["y"][:]
        self.top = [int(v) for v in group["depth_top"][:]]
        self.bottom = [int(v) for v in group["depth_bottom"][:]]
        self.units = str(self.arr.attrs.get("units", ""))
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])  # negative: y runs north to south

    def index(self, lon, lat):
        """Nearest-cell indices and a mask of the points that fall inside the grid."""
        ix = np.rint((np.asarray(lon) - self.x[0]) / self.dx).astype(int)
        iy = np.rint((np.asarray(lat) - self.y[0]) / self.dy).astype(int)
        ok = (ix >= 0) & (ix < len(self.x)) & (iy >= 0) & (iy < len(self.y))
        return ix, iy, ok


_cubes = {}
_cubes_lock = threading.Lock()


def get_cube(layer_id: str) -> Cube:
    layer = LAYER_BY_ID.get(layer_id)
    if not layer or layer["type"] != "zarr":
        raise HTTPException(404, "Unknown Zarr layer")
    with _cubes_lock:
        if layer_id not in _cubes:
            _, item = get_item(layer)
            _cubes[layer_id] = Cube(item.assets["data"].href, layer["variable"])
        return _cubes[layer_id]


@lru_cache(maxsize=12)
def cube_slice(layer_id: str, depth: int) -> np.ndarray:
    """One whole depth level (about 14 MB), kept in memory so that panning the map does not hit the bucket again."""
    return np.asarray(get_cube(layer_id).arr[depth, :, :])


_STOPS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=float)  # viridis
LUT = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_STOPS)), _STOPS[:, c]) for c in range(3)], axis=1).astype(np.uint8)


def encode_png(rgba: np.ndarray) -> bytes:
    h, w, _ = rgba.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgba.reshape(h, w * 4)], axis=1).tobytes()

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b"")


def clean(values) -> list:
    """JSON-safe list: NaN becomes null."""
    return [round(float(v), 2) if math.isfinite(v) else None for v in values]


def haversine_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


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


@app.get("/api/zarr/{layer_id}/map")
def zarr_map(layer_id: str, request: Request):
    """WMS-like GetMap for a Zarr cube: BBOX in EPSG:3857, WIDTH, HEIGHT and DEPTH (index of the depth interval) -> PNG."""
    q = {k.upper(): v for k, v in request.query_params.items()}
    try:
        minx, miny, maxx, maxy = (float(v) for v in q["BBOX"].split(","))
        w, h, depth = int(q["WIDTH"]), int(q["HEIGHT"]), int(q.get("DEPTH", 0))
    except (KeyError, ValueError):
        raise HTTPException(400, "BBOX (EPSG:3857), WIDTH, HEIGHT and DEPTH are required")
    cube = get_cube(layer_id)
    if not (0 < w <= 4096 and 0 < h <= 4096 and 0 <= depth < len(cube.top)):
        raise HTTPException(400, "Invalid size or depth index")
    r = 6378137.0
    lon = np.degrees((minx + (np.arange(w) + 0.5) * (maxx - minx) / w) / r)
    lat = np.degrees(np.arctan(np.sinh((maxy - (np.arange(h) + 0.5) * (maxy - miny) / h) / r)))
    ix = np.rint((lon - cube.x[0]) / cube.dx).astype(int)
    iy = np.rint((lat - cube.y[0]) / cube.dy).astype(int)
    okx, oky = (ix >= 0) & (ix < len(cube.x)), (iy >= 0) & (iy < len(cube.y))
    img = np.zeros((h, w, 4), np.uint8)
    if okx.any() and oky.any():
        values = cube_slice(layer_id, depth)[np.clip(iy, 0, len(cube.y) - 1)[:, None], np.clip(ix, 0, len(cube.x) - 1)[None, :]]
        valid = oky[:, None] & okx[None, :] & np.isfinite(values)
        vmin, vmax = LAYER_BY_ID[layer_id]["range"]
        t = np.clip((np.nan_to_num(values) - vmin) / (vmax - vmin), 0, 1)
        img[..., :3] = LUT[(t * 255).astype(np.uint8)]
        img[..., 3] = np.where(valid, 255, 0)
    return Response(encode_png(img), media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/zarr/{layer_id}/profile")
def zarr_profile(layer_id: str, lon: float, lat: float):
    """Values of every depth interval at one point (the nearest grid cell)."""
    cube = get_cube(layer_id)
    ix, iy, ok = cube.index([lon], [lat])
    if not ok[0]:
        raise HTTPException(404, "Point outside the cube")
    values = [float(cube_slice(layer_id, d)[iy[0], ix[0]]) for d in range(len(cube.top))]
    return {"lon": round(float(cube.x[ix[0]]), 4), "lat": round(float(cube.y[iy[0]]), 4), "values": clean(values),
            "top": cube.top, "bottom": cube.bottom, "units": cube.units}


@app.get("/api/zarr/{layer_id}/transect")
def zarr_transect(layer_id: str, coords: str, n: int = 200):
    """Depth cross-section along a line: coords = 'lon,lat;lon,lat;...' (at least two vertices), n samples along it."""
    try:
        pts = np.array([[float(v) for v in p.split(",")] for p in coords.split(";")])
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2 or pts.shape[0] > 200:
            raise ValueError
    except ValueError:
        raise HTTPException(400, "coords must be lon,lat;lon,lat;... (2 to 200 vertices)")
    n = max(2, min(int(n), 500))
    cube = get_cube(layer_id)
    seg = haversine_km(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        raise HTTPException(400, "The line has no length")
    dist = np.linspace(0, cum[-1], n)
    lon = np.interp(dist, cum, pts[:, 0])
    lat = np.interp(dist, cum, pts[:, 1])
    ix, iy, ok = cube.index(lon, lat)
    ix, iy = np.clip(ix, 0, len(cube.x) - 1), np.clip(iy, 0, len(cube.y) - 1)
    rows = []
    for d in range(len(cube.top)):
        values = np.where(ok, cube_slice(layer_id, d)[iy, ix], np.nan)
        rows.append(clean(values))
    vmin, vmax = LAYER_BY_ID[layer_id]["range"]
    return {"distance_km": [round(float(v), 3) for v in dist], "lon": [round(float(v), 4) for v in lon], "lat": [round(float(v), 4) for v in lat],
            "values": rows, "top": cube.top, "bottom": cube.bottom, "units": cube.units, "vmin": vmin, "vmax": vmax}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
