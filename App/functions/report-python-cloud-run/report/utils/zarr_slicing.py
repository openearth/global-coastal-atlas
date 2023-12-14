from enum import Enum
import shapely  # type: ignore
import xarray as xr
import numpy as np


class DatasetType(Enum):
    RASTER = "raster"
    POINT = "point"


class ZarrSlicer:
    @staticmethod
    def get_sliced_dataset(geojson_str: str, zarr_uri: str) -> xr.Dataset:
        """Fetch Zarr from remote store and slice with geojson polygon

        Args:
            geojson_str (str): String containing geojson polygon
            zarr_uri (str): String containing zarr uri

        Returns:
            xr.Dataset: sliced lazy loaded zarr dataset
        """
        polygon_shape = ZarrSlicer._create_shape_from_geojson(geojson_str)
        zarr = ZarrSlicer._get_dataset_from_zarr_url(zarr_uri)
        sliced_zarr = ZarrSlicer.slice_xarr_with_polygon(zarr, polygon_shape)
        return sliced_zarr

    @staticmethod
    def slice_xarr_with_polygon(
        xarr: xr.Dataset, polygon: shapely.Polygon
    ) -> xr.Dataset:
        """Slice xarray dataset with geojson polygon

        Args:
            xarr (xr.Dataset): xarray dataset
            polygon (Polygon): geojson polygon

        Returns:
            xr.Dataset: sliced xarray dataset
        """
        dataset_type = ZarrSlicer._get_dataset_type(xarr)

        if dataset_type == DatasetType.RASTER:
            spatial_dims = ZarrSlicer._get_spatial_dimensions(xarr)
            indexer = ZarrSlicer._get_indexer_from_raster(xarr, polygon, spatial_dims)
        elif dataset_type == DatasetType.POINT:
            points = ZarrSlicer._create_points_from_xarr(xarr)
            boolean_mask = ZarrSlicer._get_boolean_mask_from_points(points, polygon)
            spatial_dims = ZarrSlicer._get_spatial_dimensions(xarr)

            indexer = {spatial_dims[0]: boolean_mask}
        else:
            raise ValueError("Dataset type not supported")

        sliced_xarr = xarr.sel(indexer)
        return sliced_xarr

    @staticmethod
    def check_xarr_contains_data(xarr: xr.Dataset) -> bool:
        """Check if xarray dataset contains data

        Args:
            xarr (xr.Dataset): xarray dataset

        Returns:
            bool: True if xarray dataset contains data
        """
        return not np.isnan(xarr).all()

    @staticmethod
    def _get_dataset_type(xarr: xr.Dataset) -> DatasetType:
        """Get dataset type from xarray dataset. We differentiate between
        raster and point datasets"""
        # if lat and lon are dimensions, we assume it is a raster dataset
        if "lat" in xarr.dims and "lon" in xarr.dims:
            return DatasetType.RASTER
        else:
            return DatasetType.POINT

    @staticmethod
    def _create_points_from_xarr(xarr: xr.Dataset) -> shapely.MultiPoint:
        """Create shapely multipoint from xarray dataset"""
        lats = xarr.coords["lat"].values
        lons = xarr.coords["lon"].values
        points = shapely.points(lons, lats)
        return points

    @staticmethod
    def _get_spatial_dimensions(xarr: xr.Dataset) -> list[str]:
        """Get spatial dimension from xarray dataset"""
        dims = {xarr.lat.dims[0], xarr.lon.dims[0]}
        return list(dims)

    @staticmethod
    def _get_boolean_mask_from_points(
        points: shapely.MultiPoint, polygon: shapely.Polygon
    ) -> [bool]:
        """Get boolean mask from points and polygon"""
        return shapely.within(points, polygon)

    @staticmethod
    def _get_indexer_from_raster(
        raster: xr.Dataset, polygon: shapely.Polygon, spatial_dims: list[str]
    ) -> [bool]:
        """Get boolean mask from raster and polygon"""
        spatial_dim_size = {dim: len(raster[dim].values) for dim in spatial_dims}

        coords = np.stack(
            np.meshgrid(raster[spatial_dims[1]].values, raster[spatial_dims[0]].values),
            -1,
        ).reshape(
            spatial_dim_size[spatial_dims[0]], spatial_dim_size[spatial_dims[1]], 2
        )

        raster_points = shapely.points(coords)

        mask = shapely.within(raster_points, polygon)

        # Reduce mask to square shape
        # TODO: create point wise indexing for DataSet;
        indexer = {"lat": mask.any(axis=1), "lon": mask.any(axis=0)}
        return indexer

    @staticmethod
    def _create_shape_from_geojson(geojson: str) -> shapely.Polygon:
        """Create shapely polygon from geojson polygon"""
        return shapely.from_geojson(geojson)

    @staticmethod
    def _get_dataset_from_zarr_url(url: str) -> xr.Dataset:
        """Get zarr store from url"""
        return xr.open_zarr(url)
