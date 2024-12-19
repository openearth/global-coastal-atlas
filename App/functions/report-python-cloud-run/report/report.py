# %%
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path

# import fitz  # type: ignore
from jinja2 import Environment, FileSystemLoader
from shapely import Polygon  # type: ignore
import weasyprint

from report.utils.stac import STACClientGCA, ZarrDataset
from report.utils.zarr_slicing import ZarrSlicer
from report.datasets.datasetcontent import DatasetContent
from report.datasets.base_dataset import get_dataset_content
from report.datasets.overview import get_overview
from report.datasets.slr import get_slr_content

# from datasets.subtreat import get_landsub_content
from datetime import datetime

POLYGON_DEFAULT = """{"coordinates":[[[2.3915028831735015,51.7360381463356],[5.071438932343227,50.89406012060684],[6.955992986278972,51.49577449585874],[7.316959036046541,53.18700330195111],[6.636226617140238,53.961350092621075],[3.8631377106468676,54.14643052276938],[2.1218958391276317,53.490771261555096],[2.3915028831735015,51.7360381463356]]],"type":"Polygon"}"""
# Get stac catalog location from environment variable if available otherwise use default
# this is used to use local catalog instead of remote versions in deployed versions
STAC_ROOT_DEFAULT = os.getenv(
    "STAC_ROOT_DEFAULT",
    "https://raw.githubusercontent.com/openearth/global-coastal-atlas/subsidence_etienne/STAC/data/current/catalog.json",
)
STAC_COCLICO = os.getenv(
    "STAC_COCLICO",
    "https://raw.githubusercontent.com/openearth/coclicodata/main/current/catalog.json",
)


@dataclass
class ReportContent:
    datasets: list[DatasetContent]


def create_report_html(polygon: Polygon, stac_root: str) -> str:
    env = Environment(loader=FileSystemLoader(Path(__file__).parent))
    # htmlpath = Path(__file__).parent / Path("template.html.jinja")
    csspath = Path(__file__).parent / Path("template.css")

    template = env.get_template("template.html.jinja")

    data = generate_report_content(polygon=polygon)
    css: str = csspath.read_bytes().decode()
    html = template.render(data=data, css=css)

    return html


def create_report_pdf(page_content: str) -> BytesIO:  ##TODO
    in_memory_pdf = BytesIO()
    weasyprint.HTML(string=page_content, base_url=".").write_pdf(in_memory_pdf)

    return in_memory_pdf


def generate_report_content(polygon: Polygon) -> ReportContent:
    start = datetime.now()

    dataset_contents: list[DatasetContent] = []
    final_dataset_contents: list[DatasetContent] = []

    ### getting gca datasets ###
    time = datetime.now()
    print("start retrieving gca dataset {}".format(time - start))

    gca_client = STACClientGCA.open(STAC_ROOT_DEFAULT)
    zarr_datasets: list[ZarrDataset] = gca_client.get_all_zarr_uris()

    for zarr_dataset in zarr_datasets:
        xarr = ZarrSlicer._get_dataset_from_zarr_url(zarr_dataset.zarr_uri)
        sliced_xarr = ZarrSlicer.slice_xarr_with_polygon(xarr, polygon)
        sliced_xarr = sliced_xarr.rio.write_crs("EPSG:4326")
        if ZarrSlicer.check_xarr_contains_data(sliced_xarr):
            dataset_content = get_dataset_content(zarr_dataset.dataset_id, sliced_xarr)
            if dataset_content:
                if isinstance(dataset_content, list):
                    dataset_contents.extend(dataset_content)
                else:
                    dataset_contents.append(dataset_content)

    time = datetime.now()
    print("finished retrieving gca dataset {}".format(time - start))

    ## getting SLR ###
    time = datetime.now()
    print("start retrieving slr dataset {}".format(time - start))
    dataset_content = get_slr_content(polygon)
    if dataset_content:
        if isinstance(dataset_content, list):
            dataset_contents.extend(dataset_content)
        else:
            dataset_contents.append(dataset_content)

    time = datetime.now()
    print("finished retrieving slr dataset {}".format(time - start))

    # ### getting land subsidence ###
    # time = datetime.now()
    # print('start retrieving landsub dataset {}'.format(time - start))
    # dataset_content = get_landsub_content(polygon)
    # if dataset_content:
    #     if isinstance(dataset_content,list):
    #         dataset_contents.extend(dataset_content)
    #     else:
    #         dataset_contents.append(dataset_content)
    # time = datetime.now()
    # print('finished retrieving landsub dataset {}'.format(time - start))

    # ### getting DTM ### ##TODO

    ### generating overview ###
    print("start making overview {}".format(time - start))
    dataset_content = get_overview(polygon, dataset_contents)
    dataset_contents.append(dataset_content)
    print("finished making overview {}".format(time - start))

    ### re-arranging datasets ###
    collection_dict = [
        "overview",
        "dtm",
        "sediment_class",
        "world_pop",
        "flooding",
        "shoreline_change",
        "landsub2010",
        "landsub2040",
        "slr",
        "esl",
        "future_shoreline_change_2050",
        "future_shoreline_change_2100",
    ]

    existing_collection = [
        dataset_contents[ind].dataset_id for ind in range(len(dataset_contents))
    ]

    for item in collection_dict:
        if item in existing_collection:
            final_dataset_contents.append(
                dataset_contents[existing_collection.index(item)]
            )
        else:
            None

    return ReportContent(datasets=final_dataset_contents)


# %%for testing only

### Test Case 1: Terschelling; Case 2: Aveiro, Portugal; Case 3: Po Delta, Italy ###
name = ["Terschelling", "Aveiro", "PoDelta"]
polylist = [
    Polygon(
        [
            (5.0713, 53.3602),
            (5.2567, 53.44),
            (5.5281, 53.4646),
            (5.6351, 53.4086),
            (5.1669, 53.3079),
            (5.0713, 53.3602),
        ]
    ),
    Polygon(
        [
            (-8.8754, 40.7728),
            (-8.6110, 40.7749),
            (-8.6067, 40.5479),
            (-8.8719, 40.5444),
            (-8.8754, 40.7728),
        ]
    ),
    Polygon(
        [
            (12.1953, 45.1659),
            (12.6634, 45.1680),
            (12.6733, 44.7430),
            (12.2017, 44.7437),
            (12.1953, 45.1659),
        ]
    ),
]

for nn in range(1):
    if __name__ == "__main__":
        html = create_report_html(polygon=polylist[nn], stac_root=STAC_ROOT_DEFAULT)
        print(html)
        pdf = create_report_pdf(html)
        print(pdf.getvalue())

        # Write pdf to file
        with open("report-" + name[nn] + ".html", "w") as f:
            f.write(html)
        with open("report-" + name[nn] + ".pdf", "wb") as f:
            f.write(pdf.getvalue())
# %%
