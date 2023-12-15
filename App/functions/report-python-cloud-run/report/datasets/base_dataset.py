from typing import Optional
import xarray as xr

from .datasetcontent import DatasetContent
from .esl import get_esl_content


def get_dataset_content(dataset_id: str, xarr: xr.Dataset) -> Optional[DatasetContent]:
    match dataset_id:
        case "esl_gwl":
            return get_esl_content(xarr)
        case _:
            return None
