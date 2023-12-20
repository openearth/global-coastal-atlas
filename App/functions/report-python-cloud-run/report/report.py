from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import fitz  # type: ignore
import jinja2
from shapely import Polygon  # type: ignore

from report.utils.stac import STACClientGCA, ZarrDataset
from report.utils.zarr_slicing import ZarrSlicer
from report.datasets.datasetcontent import DatasetContent
from report.datasets.base_dataset import get_dataset_content

POLYGON_DEFAULT = """{"coordinates":[[[2.3915028831735015,51.7360381463356],[5.071438932343227,50.89406012060684],[6.955992986278972,51.49577449585874],[7.316959036046541,53.18700330195111],[6.636226617140238,53.961350092621075],[3.8631377106468676,54.14643052276938],[2.1218958391276317,53.490771261555096],[2.3915028831735015,51.7360381463356]]],"type":"Polygon"}"""
STAC_ROOT_DEFAULT = "https://raw.githubusercontent.com/openearth/global-coastal-atlas/subsidence_etienne/STAC/data/current/catalog.json"


@dataclass
class ReportContent:
    datasets: list[DatasetContent]


def create_report_html(polygon: Polygon, stac_root: str) -> str:
    htmlpath = Path(__file__).parent / Path("template.html.jinja")
    csspath = Path(__file__).parent / Path("template.css")

    with htmlpath.open() as f:
        template = jinja2.Template(f.read())

    data = generate_report_content(polygon=polygon, stac_root=stac_root)
    css: str = csspath.read_bytes().decode()
    html = template.render(data=data, css=css)

    return html


def create_report_pdf(page_content: str) -> BytesIO:
    story = fitz.Story(html=page_content)

    MEDIABOX = fitz.paper_rect("A4")  # output page format: Letter
    WHERE = MEDIABOX + (36, 36, -36, -36)  # leave borders of 0.5 inches
    in_memory_pdf = BytesIO()
    writer = fitz.DocumentWriter(in_memory_pdf)

    with fitz.DocumentWriter(in_memory_pdf) as writer:
        more = 1
        while more:
            device = writer.begin_page(MEDIABOX)
            more, _ = story.place(WHERE)
            story.draw(device)
            writer.end_page()

    return in_memory_pdf


def generate_report_content(polygon: Polygon, stac_root: str) -> ReportContent:
    gca_client = STACClientGCA.open(stac_root)
    zarr_datasets: list[ZarrDataset] = gca_client.get_all_zarr_uris()  # type: ignore

    dataset_contents: list[DatasetContent] = []
    for zarr_dataset in zarr_datasets:
        xarr = ZarrSlicer._get_dataset_from_zarr_url(zarr_dataset.zarr_uri)
        sliced_xarr = ZarrSlicer.slice_xarr_with_polygon(xarr, polygon)
        if ZarrSlicer.check_xarr_contains_data(sliced_xarr):
            dataset_content = get_dataset_content(zarr_dataset.dataset_id, sliced_xarr)
            if dataset_content:
                dataset_contents.append(dataset_content)

    return ReportContent(datasets=dataset_contents)


if __name__ == "__main__":
    polygon = Polygon(
        [
            [2.3915028831735015, 51.7360381463356],
            [5.071438932343227, 50.89406012060684],
            [6.955992986278972, 51.49577449585874],
            [7.316959036046541, 53.18700330195111],
            [6.636226617140238, 53.961350092621075],
            [3.8631377106468676, 54.14643052276938],
            [2.1218958391276317, 53.490771261555096],
            [2.3915028831735015, 51.7360381463356],
        ]
    )
    html = create_report_html(polygon=polygon, stac_root=STAC_ROOT_DEFAULT)
    print(html)
    pdf = create_report_pdf(html)
    print(pdf.getvalue())
