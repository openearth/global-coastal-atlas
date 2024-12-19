from dataclasses import dataclass
from pystac_client import Client


@dataclass
class ZarrDataset:
    dataset_id: str
    zarr_uri: str


class STACClientGCA(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_all_zarr_uris(self) -> list[ZarrDataset]:
        collections = self.get_collections()
        zarr_datasets = []

        for collection in collections:
            # we only look at collections that have a child links
            #if collection.get_item_links(): ##TODO: to be removed
                zarr_datasets.append(
                    ZarrDataset(
                        dataset_id=collection.id,
                        zarr_uri=collection.assets["data"].href,
                    )
                )
        return zarr_datasets
