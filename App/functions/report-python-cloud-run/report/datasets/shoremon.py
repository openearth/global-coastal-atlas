# Packages for loading data
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

# Packages for plotting
from resilientplotterclass import rpc
from pathlib import Path
import matplotlib
import numpy as np

from report.datasets.utils import plot_to_base64
from report.datasets.datasetcontent import DatasetContent
from report.utils.gentext import describe_data

matplotlib.use("Agg")
plt.rcParams["svg.fonttype"] = "none"

world = gpd.read_file(
    Path(__file__).parent.parent.parent / "data" / "world_administrative.zip"
)


def get_sedclass_content(xarr: xr.Dataset) -> DatasetContent:
    """Get content for the dataset"""
    dataset_id = "sediment_class"
    title = "Coastal Types"
    text = "Here we generate some content based on the the dataset"
    text = describe_data(xarr, dataset_id)

    image_base64 = create_sedclass_plot(xarr)
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def get_shoremon_content(xarr: xr.Dataset) -> DatasetContent:
    """Get content for the dataset"""
    dataset_id = "shoreline_change"
    title = "Historical Shoreline Change (1984-2021)"
    text = "Here we generate some content based on the the dataset"
    text = describe_data(xarr, dataset_id)

    image_base64 = create_shoremon_plot(xarr)
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def get_shoremon_fut_content(xarr: xr.Dataset) -> list[DatasetContent]:
    dataset_contents_list = []
    dataset_contents_list.append(get_shoremon_fut2050_content(xarr))
    dataset_contents_list.append(get_shoremon_fut2100_content(xarr))

    return dataset_contents_list


def get_shoremon_fut2050_content(xarr: xr.Dataset) -> DatasetContent:
    ### remove outlier ###      ##TODO##
    """Get content for the dataset"""
    dataset_id = "future_shoreline_change_2050"
    title = "Future Shoreline Projections in 2050"
    text = "Here we generate some content based on the the dataset"
    text = describe_data(xarr, dataset_id)

    image_base64 = create_shoremon_fut_plot(xarr, 2050)
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


def get_shoremon_fut2100_content(xarr: xr.Dataset) -> DatasetContent:
    """Get content for the dataset"""
    dataset_id = "future_shoreline_change_2100"
    title = "Future Shoreline Projections in 2100"
    text = "Here we generate some content based on the the dataset"
    text = describe_data(xarr, dataset_id)

    image_base64 = create_shoremon_fut_plot(xarr, 2100)
    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=image_base64,
    )


# def get_shoremon_fut_content(xarr: xr.Dataset) -> list[DatasetContent]:
#     dataset_contents_list = []
#     dataset_contents_list.append(get_shoremon_fut45_content(xarr))
#     dataset_contents_list.append(get_shoremon_fut85_content(xarr))

#     return dataset_contents_list


# def get_shoremon_fut45_content(xarr: xr.Dataset) -> DatasetContent:
#     """Get content for the dataset"""
#     dataset_id = "future_shoreline_change_RCP45"
#     title = "Future Shoreline Projections"
#     text = "Here we generate some content based on the the dataset" ##TODO: to be filled in by LLM
#     text = describe_data(xarr, dataset_id)

#     image_base64 = create_shoremon_fut_plot(xarr, 'RCP45')
#     return DatasetContent(
#         dataset_id=dataset_id,
#         title=title,
#         text=text,
#         image_base64=image_base64,
#     )

# def get_shoremon_fut85_content(xarr: xr.Dataset) -> DatasetContent:
#     """Get content for the dataset"""
#     dataset_id = "future_shoreline_change_RCP85"
#     title = "Future Shoreline Projections"
#     text = "Here we generate some content based on the the dataset" ##TODO: to be filled in by LLM
#     text = describe_data(xarr, dataset_id)

#     image_base64 = create_shoremon_fut_plot(xarr, 'RCP85')
#     return DatasetContent(
#         dataset_id=dataset_id,
#         title=title,
#         text=text,
#         image_base64=image_base64,
#     )


def create_sedclass_plot(xarr):
    sediment_classes_dict = {
        0: "sand",
        1: "mud",
        2: "coastal cliff",
        3: "vegetated",
        4: "other",
    }
    color_dict = {0: "yellow", 1: "brown", 2: "blue", 3: "green", 4: "gray"}

    from matplotlib.colors import ListedColormap, Normalize
    import matplotlib as mpl

    existing_values = np.unique(xarr["sediment_label"])
    existing_class = [sediment_classes_dict[val] for val in existing_values]
    existing_color = [color_dict[val] for val in existing_values]

    sediment_classes = [sediment_classes_dict[i] for i in xarr["sediment_label"].values]
    xarr["sediment_class"] = xr.DataArray(sediment_classes, dims="stations")

    cmap = ListedColormap(existing_color)
    norm = Normalize(vmin=0, vmax=cmap.N)
    cb = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    portion = []

    for type in existing_class:
        port = len(np.where(xarr["sediment_class"] == type)[0]) / len(
            xarr["sediment_class"]
        )
        portion.append(port)

    # Plot the data
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[1, 1])

    base = world.boundary.plot(
        ax=ax[0], edgecolor="grey", facecolor="grey", alpha=0.1, zorder=0
    )

    aspect = len(existing_class) / 0.8
    p = rpc.scatter(
        xarr,
        ax=ax[0],
        x="lon",
        y="lat",
        hue="sediment_label",
        edgecolor="none",
        cmap=cmap,
        add_colorbar=False,
    )

    cbar = plt.colorbar(
        cb,
        ax=ax[0],
        **{
            "label": "Sediment classes",
            "pad": 0.01,
            "fraction": 0.05,
            "aspect": aspect,
        },
    )
    cbar.set_ticks(ticks=np.arange(0.5, len(existing_color), 1), labels=existing_class)

    ax[1].pie(portion, labels=existing_class, autopct="%1.1f%%", colors=existing_color)

    lonmin = min(xarr.lon.values)
    lonmax = max(xarr.lon.values)
    latmin = min(xarr.lat.values)
    latmax = max(xarr.lat.values)

    xlim = [lonmin - 0.1, lonmax + 0.1]
    ylim = [latmin - 0.1, latmax + 0.1]

    ax[0].set(
        xlim=xlim,
        ylim=ylim,
    )

    ax[0].set_aspect(1 / np.cos(np.mean(ylim) * np.pi / 180))
    ax[0].grid(False)

    fig.tight_layout()

    return plot_to_base64(fig)


def create_shoremon_plot(xarr):
    lonmin = min(xarr.lon.values)
    lonmax = max(xarr.lon.values)
    latmin = min(xarr.lat.values)
    latmax = max(xarr.lat.values)

    xlim = [lonmin - 0.1, lonmax + 0.1]
    ylim = [latmin - 0.1, latmax + 0.1]

    # Get pie chart data
    labels = [
        "Extreme\nerosion",
        "Severe\nerosion",
        "Intense\nerosion",
        "Erosion",
        "Stable",
        "Accretion",
        "Intense\naccretion",
        "Severe\naccretion",
        "Extreme\naccretion",
    ]
    colors = [matplotlib.cm.RdYlGn(i) for i in np.linspace(0.05, 0.95, len(labels))]
    bins = [-np.inf, -5, -3, -1, -0.5, 0.5, 1, 3, 5, np.inf]
    data = np.histogram(xarr["changerate"].values, bins=bins)[0]

    # Plot data
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), width_ratios=[1.2, 0.8, 0.3])
    base = world.boundary.plot(
        ax=axs[0], edgecolor="grey", facecolor="grey", alpha=0.1, zorder=0
    )
    rpc.scatter(
        xarr,
        data_type="data",
        ax=axs[0],
        x="lon",
        y="lat",
        hue="changerate",
        vmin=-5,
        vmax=5,
        cmap="RdYlGn",
        add_colorbar=True,
        cbar_kwargs={"label": "Erosion/Accretion [m/yr]"},
    )
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[0].set_aspect(1 / np.cos(np.mean(ylim) * np.pi / 180))

    # Add a pie chart showing the distribution of the classes
    axs[1].pie(
        data,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        counterclock=False,
    )
    axs[1].axis("equal")

    # Add legend
    labels2 = [
        "Extreme erosion (<-5 m/yr)",
        "Severe erosion (-5 to -3 m/yr)",
        "Intense erosion (-3 to -1 m/yr)",
        "Erosion (-1 to -0.5 m/yr)",
        "Stable (-0.5 to 0.5 m/yr)",
        "Accretion (0.5 to 1 m/yr)",
        "Intense accretion (1 to 3 m/yr)",
        "Severe accretion (3 to 5 m/yr)",
        "Extreme accretion (>5 m/yr)",
    ]
    colors2 = [matplotlib.cm.RdYlGn(i) for i in np.linspace(0.05, 0.95, len(labels2))]
    axs[2].legend(
        handles=[
            matplotlib.patches.Patch(color=colors2[i], label=labels2[i])
            for i in range(len(labels2))
        ],
        loc="center left",
        frameon=False,
    )
    axs[2].axis("off")

    fig.tight_layout()

    return plot_to_base64(fig)


# def create_shoremon_fut_plot(xarr):
#     scenariolist = ['sp_rcp45_p50', 'sp_rcp85_p50']
#     yearlist = [2021, 2050, 2100]


#     fig, ax = plt.subplots(2, 2, figsize=(15, 15))


#     for yr in range(0,2):
#         for scenario in scenariolist:
#             if scenario == 'sp_rcp45_p50':
#                 scenarioname = 'RCP4.5'
#                 jj = 0
#             elif scenario == 'sp_rcp85_p50':
#                 scenarioname = 'RCP8.5'
#                 jj = 1

#             diff = xarr.diff('time', 1).sel(time=str(yearlist[yr + 1]))

#             base = world.boundary.plot(
#                     ax=ax[yr, jj], edgecolor="grey", facecolor="grey", alpha=0.1, zorder=0
#                 )

#             p = rpc.scatter(diff / (yearlist[yr + 1] - yearlist[yr]),
#                             ax=ax[yr, jj], data_type='data',
#                             x='lon', y='lat',
#                             vmin=-5, vmax=5,
#                             hue=scenario,
#                             edgecolor='none', cmap='RdYlGn',
#                             add_colorbar=True, cbar_kwargs={'label': 'Average Shoreline Change Rate [m/yr]'})

#             lonmin = min(xarr.lon.values)
#             lonmax = max(xarr.lon.values)
#             latmin = min(xarr.lat.values)
#             latmax = max(xarr.lat.values)

#             xlim = [lonmin - 0.1, lonmax + 0.1]
#             ylim = [latmin - 0.1, latmax + 0.1]

#             ax[yr, jj].set(
#                 xlim=xlim,
#                 ylim=ylim,
#             )

#             ax[yr, jj].set_aspect(1/np.cos(np.mean(ylim)*np.pi/180))
#             ax[yr, jj].grid(False)
#             ax[yr, jj].set_title('50%ile Future Prediction between Year {}'.format(yearlist[yr+1]) + ' and Year {}'.format(yearlist[yr]) + '\n' + '- Scenario {}'.format(scenarioname))

#     fig.tight_layout()

#     return plot_to_base64(fig)


def create_shoremon_fut_plot(xarr, year):
    scenariolist = ["sp_rcp45_p50", "sp_rcp85_p50"]
    yearlist = [2021, 2050, 2100]
    yr = yearlist.index(year) - 1

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), width_ratios=[6, 4])

    for nn in range(len(scenariolist)):
        match scenariolist[nn]:
            case "sp_rcp45_p50":
                var = "sp_rcp45_p50"
                scenarioname = "RCP4.5"
            case "sp_rcp85_p50":
                var = "sp_rcp85_p50"
                scenarioname = "RCP8.5"

        diff = xarr.diff("time", 1).sel(time=str(yearlist[yr + 1]))
        rate = diff / (yearlist[yr + 1] - yearlist[yr])

        base = world.boundary.plot(
            ax=ax[nn, 0], edgecolor="grey", facecolor="grey", alpha=0.1, zorder=0
        )

        p = rpc.scatter(
            rate,
            ax=ax[nn, 0],
            data_type="data",
            x="lon",
            y="lat",
            vmin=-5,
            vmax=5,
            hue=var,
            edgecolor="none",
            cmap="RdYlGn",
            add_colorbar=True,
            cbar_kwargs={"label": "Average Shoreline Change Rate [m/yr]"},
        )

        lonmin = min(xarr.lon.values)
        lonmax = max(xarr.lon.values)
        latmin = min(xarr.lat.values)
        latmax = max(xarr.lat.values)

        xlim = [lonmin - 0.1, lonmax + 0.1]
        ylim = [latmin - 0.1, latmax + 0.1]

        ax[nn, 0].set(
            xlim=xlim,
            ylim=ylim,
        )

        ax[nn, 0].set_aspect(1 / np.cos(np.mean(ylim) * np.pi / 180))
        ax[nn, 0].grid(False)
        ax[nn, 0].set_title(
            "50%ile Future Prediction between Year {}".format(yearlist[yr + 1])
            + " and Year {}".format(yearlist[yr])
            + "\n"
            + "- Scenario {}".format(scenarioname)
        )

        # Get pie chart data
        labels = [
            "Extreme\nerosion",
            "Severe\nerosion",
            "Intense\nerosion",
            "Erosion",
            "Stable",
            "Accretion",
            "Intense\naccretion",
            "Severe\naccretion",
            "Extreme\naccretion",
        ]
        colors = [matplotlib.cm.RdYlGn(i) for i in np.linspace(0.05, 0.95, len(labels))]
        bins = [-np.inf, -5, -3, -1, -0.5, 0.5, 1, 3, 5, np.inf]
        data = np.histogram(rate[var].values, bins=bins)[0]

        # Add a pie chart showing the distribution of the classes
        ax[nn, 1].pie(
            data,
            labels=labels,
            colors=colors,
            autopct="%1.0f%%",
            startangle=90,
            counterclock=False,
        )
        ax[nn, 1].axis("equal")

    fig.tight_layout()

    return plot_to_base64(fig)


# def create_shoremon_fut_plot(xarr, scenario):

#     yearlist = [2021, 2050, 2100]

#     match scenario:
#         case 'RCP45':
#             var = 'sp_rcp45_p50'
#             scenarioname = 'RCP4.5'

#         case 'RCP85':
#             var = 'sp_rcp85_p50'
#             scenarioname = 'RCP8.5'

#     fig, ax = plt.subplots(2, 2, figsize=(10, 10), width_ratios=[6,4])

#     for yr in range(0,len(yearlist) - 1):
#             diff = xarr.diff('time', 1).sel(time=str(yearlist[yr + 1]))
#             rate = diff / (yearlist[yr + 1] - yearlist[yr])

#             base = world.boundary.plot(
#                     ax=ax[yr, 0], edgecolor="grey", facecolor="grey", alpha=0.1, zorder=0
#                 )

#             p = rpc.scatter(rate,
#                             ax=ax[yr, 0], data_type='data',
#                             x='lon', y='lat',
#                             vmin=-5, vmax=5,
#                             hue=var,
#                             edgecolor='none', cmap='RdYlGn',
#                             add_colorbar=True, cbar_kwargs={'label': 'Average Shoreline Change Rate [m/yr]'})

#             lonmin = min(xarr.lon.values)
#             lonmax = max(xarr.lon.values)
#             latmin = min(xarr.lat.values)
#             latmax = max(xarr.lat.values)

#             xlim = [lonmin - 0.1, lonmax + 0.1]
#             ylim = [latmin - 0.1, latmax + 0.1]

#             ax[yr, 0].set(
#                 xlim=xlim,
#                 ylim=ylim,
#             )

#             ax[yr, 0].set_aspect(1/np.cos(np.mean(ylim)*np.pi/180))
#             ax[yr, 0].grid(False)
#             ax[yr, 0].set_title('50%ile Future Prediction between Year {}'.format(yearlist[yr+1]) + ' and Year {}'.format(yearlist[yr]) + '\n' + '- Scenario {}'.format(scenarioname))

#             # Get pie chart data
#             labels = ['Extreme\nerosion', 'Severe\nerosion', 'Intense\nerosion', 'Erosion', 'Stable', 'Accretion', 'Intense\naccretion', 'Severe\naccretion', 'Extreme\naccretion']
#             colors = [matplotlib.cm.RdYlGn(i) for i in np.linspace(0.05, 0.95, len(labels))]
#             bins = [-np.inf, -5, -3, -1, -0.5, 0.5 , 1, 3, 5, np.inf]
#             data = np.histogram(rate[var].values, bins=bins)[0]

#             # Add a pie chart showing the distribution of the classes
#             ax[yr, 1].pie(data, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90, counterclock=False)
#             ax[yr, 1].axis('equal')


#     fig.tight_layout()

#     return plot_to_base64(fig)
