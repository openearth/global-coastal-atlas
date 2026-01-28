# Packages for loading data
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt

# Packages for plotting
from shapely import Polygon  # type: ignore

from report.datasets.datasetcontent import DatasetContent
from report.utils.gentext import describe_rag

matplotlib.use("Agg")
plt.rcParams["svg.fonttype"] = "none"
world = gpd.read_file(Path(__file__).parent.parent.parent / "data" / "world_administrative.zip")


def get_rag_overview(polygon: Polygon, dataset_contents: DatasetContent) -> DatasetContent:
    """Get overview"""
    dataset_id = "rag"
    title = "Retrieval-Augmented Generation (RAG)"
    text = "Here we generate some text based on relevant documents from our knowledge base."
    text = describe_rag(polygon, dataset_contents)

    return DatasetContent(
        dataset_id=dataset_id,
        title=title,
        text=text,
        image_base64=None,
    )
