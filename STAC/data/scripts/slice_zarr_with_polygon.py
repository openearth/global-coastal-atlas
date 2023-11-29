import shapely  # type: ignore
import xarray as xr


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
        points = ZarrSlicer._create_points_from_xarr(xarr)
        boolean_mask = ZarrSlicer._get_boolean_mask_from_points(points, polygon)
        coordinate_dim = ZarrSlicer._get_spatial_dimension(xarr)
        sliced_xarr = xarr.sel({coordinate_dim: boolean_mask})

        return sliced_xarr

    @staticmethod
    def _create_points_from_xarr(xarr: xr.Dataset) -> shapely.MultiPoint:
        """Create shapely multipoint from xarray dataset"""
        lats = xarr.coords["lat"].values
        lons = xarr.coords["lon"].values
        points = shapely.points(lons, lats)
        return points

    @staticmethod
    def _get_spatial_dimension(xarr: xr.Dataset) -> str:
        """Get spatial dimension from xarray dataset"""
        return xarr.lat.dims[0]

    @staticmethod
    def _get_boolean_mask_from_points(
        points: shapely.MultiPoint, polygon: shapely.Polygon
    ) -> [bool]:
        """Get boolean mask from points and polygon"""
        return shapely.within(points, polygon)

    @staticmethod
    def _create_shape_from_geojson(geojson: str) -> shapely.Polygon:
        """Create shapely polygon from geojson polygon"""
        return shapely.from_geojson(geojson)

    @staticmethod
    def _get_dataset_from_zarr_url(url: str) -> xr.Dataset:
        """Get zarr store from url"""
        return xr.open_zarr(url)


if __name__ == "__main__":
    # create polygon from bangladesh
    geojson_str = """
 {
        "coordinates": [
          [
            [
              90.89270322181983,
              24.89144849850402
            ],
            [
              87.51739497682678,
              23.95385040469374
            ],
            [
              87.23135190521862,
              20.297974055822536
            ],
            [
              93.35267363766218,
              19.544987025644943
            ],
            [
              95.01172345299642,
              23.219849978261337
            ],
            [
              90.89270322181983,
              24.89144849850402
            ]
          ]
        ],
        "type": "Polygon"
      }
    """

    polygon_shape: shapely.Polygon = ZarrSlicer._create_shape_from_geojson(geojson_str)

    zarr_url = "https://storage.googleapis.com/dgds-data-public/gca/ESLbyGWL.zarr"
    ds = ZarrSlicer._get_dataset_from_zarr_url(zarr_url)

    print(f"zarr size: {ds.nbytes / 1e9} GB")

    sliced_ds = ZarrSlicer.get_sliced_dataset(geojson_str, zarr_url)

    print(f"sliced zarr size: {sliced_ds.nbytes / 1e9} GB")
