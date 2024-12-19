from typing import Optional
import xarray as xr

from report.datasets.datasetcontent import DatasetContent

from report.datasets.shoremon import (
    get_sedclass_content,
    get_shoremon_content,
    get_shoremon_fut_content,
)
from report.datasets.popgpd import get_world_pop_content
# from .subtreat import get_sub_threat_content


def get_dataset_content(dataset_id: str, xarr: xr.Dataset) -> Optional[DatasetContent]:
    match dataset_id:
        # case "esl_gwl":
        #     return get_esl_content(xarr)
        case "sed_class":
            return get_sedclass_content(xarr)
        case "shore_mon":
            return get_shoremon_content(xarr)
        case "shore_mon_fut":
            return get_shoremon_fut_content(xarr)
        case "world_pop":
            return get_world_pop_content(xarr)
        # case "sub_threat":
        #     return None
        case _:
            return None
