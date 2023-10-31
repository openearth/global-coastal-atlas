import pystac_client
from pathlib import Path

def test_stac_catalog():
    stac_folder = Path(__file__).parent.parent / "data" / "current"
    catalog = pystac_client.Client.open(stac_folder / "catalog.json")
    
    used_uris = set()

    used_uris.update(catalog.validate())

    for collection in catalog.get_collections():
        used_uris.update(collection.validate())

        for item in collection.get_items():
            used_uris.update(item.validate())
    
    print(used_uris)



