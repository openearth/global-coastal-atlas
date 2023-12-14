from collections import namedtuple
from pystac_client import Client

Zarr_dataset = namedtuple("Zarr_dataset", ["dataset_id", "zarr_uri"])


class STACClientGCA(Client):
    def get_all_zarr_uris(self) -> list[Zarr_dataset]:
        collections = self.get_collections()
        zarr_datasets = []

        for collection in collections:
            # we only look at collections that have a child links
            if collection.get_child_links():
                zarr_datasets.append(
                    Zarr_dataset(collection.id, collection.assets["data"].href)
                )
        return zarr_datasets
