import pystac_client
from pathlib import Path

def test_stac_catalog():
    stac_folder = Path(__file__).parent.parent / "data" / "current"
    catalog = pystac_client.Client.open(stac_folder / "catalog.json")
    
    used_uris = set()
    import jsonschema
    import jsonschema.exceptions
    import jsonschema.validators
    from referencing import Registry, Resource

    from pystac.validation.local_validator import get_local_schema_cache

    used_uris.update(catalog.validate())

    for collection in catalog.get_collections():
        used_uris.update(collection.validate())

        for item in collection.get_items():
            used_uris.update(item.validate())
    
    print(used_uris)



