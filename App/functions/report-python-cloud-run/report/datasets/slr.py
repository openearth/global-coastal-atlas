# Packages for loading data
import matplotlib.pyplot as plt

# Packages for plotting
import matplotlib

from shapely import Polygon  # type: ignore
import pystac_client
import rioxarray as rio

from report.datasets.utils import plot_to_base64
from report.datasets.datasetcontent import DatasetContent
from report.utils.gentext import describe_data

matplotlib.use("Agg")
plt.rcParams["svg.fonttype"] = "none"


def get_slr_content(polygon: Polygon) -> DatasetContent:
    slps = get_slps_data(polygon)
    """Get content for the dataset"""
    dataset_id = "slr"
    title = "Sea Level Rise Projection"
    text = "Here we generate some content based on the dataset"
    text = describe_data(slps, dataset_id)

    image_base64 = create_slr_plot(slps)
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


# def get_slr_content(polygon: Polygon) -> list[DatasetContent]:
#     dataset_contents_list = []
#     dataset_contents_list.append(get_slr26_content(polygon))
#     dataset_contents_list.append(get_slr45_content(polygon))
#     dataset_contents_list.append(get_slr85_content(polygon))

#     return dataset_contents_list

# def get_slr26_content(polygon: Polygon) -> DatasetContent:
#     slps = get_slps_data(polygon, 'RCP26')
#     """Get content for the dataset"""
#     dataset_id = "slr_RCP26"
#     title = "Sea Level Rise Projection"
#     text = "Here we generate some content based on the dataset"
#     text = describe_data(slps, dataset_id)

#     image_base64 = create_slr_plot(slps, 'RCP26')
#     return DatasetContent(
#         dataset_id=dataset_id,
#         title=title,
#         text=text,
#         image_base64=image_base64,
#     )

# def get_slr45_content(polygon: Polygon) -> DatasetContent:
#     slps = get_slps_data(polygon, 'RCP45')
#     """Get content for the dataset"""
#     dataset_id = "slr_RCP45"
#     title = "Sea Level Rise Projection"
#     text = "Here we generate some content based on the dataset"
#     text = describe_data(slps, dataset_id)

#     image_base64 = create_slr_plot(slps, 'RCP45')
#     return DatasetContent(
#         dataset_id=dataset_id,
#         title=title,
#         text=text,
#         image_base64=image_base64,
#     )

# def get_slr85_content(polygon: Polygon) -> DatasetContent:
#     slps = get_slps_data(polygon, 'RCP85')
#     """Get content for the dataset"""
#     dataset_id = "slr_RCP85"
#     title = "Sea Level Rise Projection"
#     text = "Here we generate some content based on the dataset"
#     text = describe_data(slps, dataset_id)

#     image_base64 = create_slr_plot(slps, 'RCP85')
#     return DatasetContent(
#         dataset_id=dataset_id,
#         title=title,
#         text=text,
#         image_base64=image_base64,
#     )


def get_slps_data(polygon: Polygon) -> dict:
    # match scenario:
    #     case 'RCP26':
    #         ssps = ['ssp126']
    #     case 'RCP45':
    #         ssps = ['ssp245']
    #     case 'RCP85':
    #         ssps = ['ssp585']

    # Set stac URL
    STAC_url = "https://raw.githubusercontent.com/openearth/coclicodata/main/current/catalog.json"

    # Open the catalog
    catalog = pystac_client.Client.open(STAC_url)

    # Get the AR6 collection
    collection = catalog.get_child("slp")

    ssps = ["high_end", "ssp126", "ssp245", "ssp585"]
    msls = ["msl_m"]
    years = [
        "2031",
        "2041",
        "2051",
        "2061",
        "2071",
        "2081",
        "2091",
        "2101",
        "2111",
        "2121",
        "2131",
        "2141",
        "2151",
    ]

    slps = []

    # Iterate over all ssps, ens and years
    for ssp in ssps:
        for msl in msls:
            for year in years:
                # Get the item
                item = collection.get_item(f"{ssp}\{msl}\{year}.tif")

                # Get href
                href = item.assets["data"].href

                # Load tif into xarray
                ds = rio.open_rasterio(href, masked=True)

                # First clip to bounding box of polygon
                ds_clip = ds.rio.clip_box(
                    *polygon.bounds, allow_one_dimensional_raster=True
                )

                ds_point = ds_clip.sel(
                    x=polygon.centroid.x, y=polygon.centroid.y, method="nearest"
                )

                # Check if nearest pixel to centroid has data, if not use max of ds_clip
                if not ds_point.notnull().values:
                    # Find the maximum value in the dataset
                    max_value = ds_clip.max()

                    # Find the coordinates of the max value
                    max_coords = ds_clip.where(ds == max_value, drop=True)

                    # Retrieve the coordinates (x, y) for the maximum value
                    max_x = max_coords.coords["x"].values
                    max_y = max_coords.coords["y"].values

                    # Select the dataset at the coordinates of the maximum value
                    ds_point = ds_clip.sel(x=max_x, y=max_y)

                # Retrieve the point value
                value = ds_point.values.item()

                # Append the result as a dictionary
                slps.append({"ssp": ssp, "msl": msl, "year": year, "value": value})

    return slps


# def create_slr_plot(slps: dict, scenario: str):

#     match scenario:
#         case 'RCP26':
#             ssp = ['ssp126']
#             color = 'g'
#         case 'RCP45':
#             ssp = ['ssp245']
#             color = 'b'
#         case 'RCP85':
#             ssp = ['ssp585']
#             color = 'r'

#     fig, ax = plt.subplots(1,1, figsize=(5,5))

#     # Filter the data for the current ssp
#     ssp_values_m = [slp['value'] for slp in slps if (slp['msl'] == 'msl_m')]
#     ssp_values_l = [slp['value'] for slp in slps if (slp['msl'] == 'msl_l')]
#     ssp_values_h = [slp['value'] for slp in slps if (slp['msl'] == 'msl_h')]
#     ssp_years = ["2031","2041","2051", "2061", "2071", "2081", "2091", "2101","2111","2121","2131","2141","2151"]

#     # Line plot
#     ax.plot(ssp_years, ssp_values_m, label=ssp, marker='o', color=color)
#     ax.plot(ssp_years, ssp_values_l, label=ssp, color=color, alpha=0)
#     ax.plot(ssp_years, ssp_values_h, label=ssp, color=color, alpha=0)

#     ax.fill_between(ssp_years, ssp_values_l, ssp_values_h, color=color, alpha=0.2)

#     ax.set_xlabel("Year")
#     ax.set_ylabel("Sea Level Rise [mm]")
#     ax.set_xticks(ssp_years, ssp_years, rotation=45)
#     ax.legend(['Medium Confidence', '_', '_', 'Low to High Confidence Range'])

#     return plot_to_base64(fig)


def create_slr_plot(slps: dict):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ssps = ["high_end", "ssp126", "ssp245", "ssp585"]

    for ssp in ssps:
        # Filter the data for the current ssp
        ssp_values = [slp["value"] for slp in slps if slp["ssp"] == ssp]
        ssp_years = [slp["year"] for slp in slps if slp["ssp"] == ssp]

        # Plot the data for this ssp
        ax.plot(ssp_years, ssp_values, label=ssp, marker="o")

    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Level Rise [mm]")
    ax.set_xticks(ssp_years, ssp_years, rotation=45)
    ax.legend()

    return plot_to_base64(fig)
