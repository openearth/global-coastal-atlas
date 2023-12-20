from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetContent:
    dataset_id: str
    title: str
    text: str
    image_base64: Optional[str] = None
    image_svg: Optional[str] = None
