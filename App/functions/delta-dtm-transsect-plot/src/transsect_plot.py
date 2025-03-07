import asyncio

import fsspec
import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
import xarray
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rioxarray.merge import merge_arrays
from shapely import LineString

BLOB_SERVICE_ACCOUNT_URL = "https://coclico.blob.core.windows.net"

# Read the parquet file from the blob client in top-level code
# so it is only read once for repeated requests
parquet_gdf = gpd.read_parquet(
    f"{BLOB_SERVICE_ACCOUNT_URL}/items/deltares-delta-dtm.parquet",
    filesystem=fsspec.filesystem("https"),
)


def get_intersecting_item_ids(line: LineString) -> list[str]:
    intersecting_items = parquet_gdf[parquet_gdf.intersects(line)]
    return intersecting_items.id.values.tolist()


def get_item_hrefs(item_ids: list[str]) -> list[str]:
    return [
        f"{BLOB_SERVICE_ACCOUNT_URL}/deltares-delta-dtm/v1.1/{item_id}.tif"
        for item_id in item_ids
    ]


async def get_datasets(item_hrefs: list[str]) -> list[xarray.DataArray]:
    async def fetch_dataset(href: str) -> xarray.DataArray:
        # Open the raster and assign a name based on the index
        data_array = rioxarray.open_rasterio(rasterio.open(href))
        return data_array

    tasks = [fetch_dataset(href) for href in item_hrefs]
    arrays = await asyncio.gather(*tasks)
    return arrays


def merge_and_mask_datasets(datasets: list[xarray.DataArray]) -> xarray.DataArray:
    # Convert each DataArray to a Dataset before merging
    merged = merge_arrays(datasets)

    # Mask non-valid values
    masked_merged = merged.where(merged != -9999.0)
    return masked_merged.squeeze()


def transsect(
    merged_dataset: xarray.DataArray, line: LineString, n_samples=256
) -> np.ndarray:
    x, y = line.xy
    distances = np.linspace(0, line.length, n_samples)
    points = [line.interpolate(distance) for distance in distances]
    tgt_x = xarray.DataArray(
        [point.x for point in points],
        dims="points",
    )
    tgt_y = xarray.DataArray(
        [point.y for point in points],
        dims="points",
    )
    values: np.ndarray = merged_dataset.sel(x=tgt_x, y=tgt_y, method="nearest").data

    return values


def get_distances(line: LineString, n_samples: int = 256) -> np.ndarray:
    # Calculate actual distance along the line in meters
    x, y = line.xy
    distances: list[float] = [0.0]  # Start with 0
    total_distance: float = 0.0

    # Calculate cumulative distance along the line
    for i in range(1, len(x)):
        # Calculate distance between consecutive points (in degrees)
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]

        # Convert to approximate distance in meters
        # Using simple approximation: 1 degree latitude ≈ 111 km, 1 degree longitude varies with latitude
        # For more accuracy, a proper geodesic calculation would be better
        lat_avg = (y[i] + y[i - 1]) / 2  # Average latitude
        meters_per_lon_degree = 111320 * np.cos(
            np.radians(lat_avg)
        )  # Approximate meters per degree longitude at this latitude
        meters_per_lat_degree = 111320  # Approximate meters per degree latitude

        distance_meters = np.sqrt(
            (dx * meters_per_lon_degree) ** 2 + (dy * meters_per_lat_degree) ** 2
        )
        total_distance += distance_meters
        distances.append(total_distance)

    # Interpolate distances for all sample points
    sample_distances = np.linspace(start=0.0, stop=total_distance, num=n_samples)

    return sample_distances


def create_profile_plot(values: np.ndarray, sample_distances: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot the elevation profile with actual distances
    ax.plot(sample_distances, values, "b-", linewidth=2)

    # Fill the area under the curve
    ax.fill_between(
        sample_distances, np.min(values) - 0.5, values, alpha=0.3, color="skyblue"
    )

    # Add grid and labels
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlabel("Distance along transect (meters)", fontsize=12)
    ax.set_ylabel("Elevation (m)", fontsize=12)
    ax.set_title("Elevation Profile Along Transect", fontsize=14)

    # Add some padding to the y-axis
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # Tight layout for better spacing
    fig.tight_layout()

    return fig


async def plot_transsect(line: LineString) -> Figure:
    item_ids = get_intersecting_item_ids(line)
    if not item_ids:
        # Handle case where no items intersect with the line
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            "No data available for the specified transect line",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_xlabel("Distance along transect (meters)", fontsize=12)
        ax.set_ylabel("Elevation (m)", fontsize=12)
        ax.set_title("Elevation Profile Along Transect", fontsize=14)
        fig.tight_layout()
        return fig

    item_hrefs = get_item_hrefs(item_ids)
    datasets = await get_datasets(item_hrefs)
    merged_dataset = merge_and_mask_datasets(datasets)
    elevation_values = transsect(merged_dataset, line, n_samples=1000)
    sample_distances = get_distances(line, len(elevation_values))
    return create_profile_plot(elevation_values, sample_distances)
