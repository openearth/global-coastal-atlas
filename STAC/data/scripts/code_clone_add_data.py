# created by Etiënne Kras, dd 24-04-2024
# Script uses inspiration from https://github.com/stac-utils/stac-fastapi-elasticsearch-opensearch/blob/main/data_loader.py, also to align with the STAC URL from STAC-fastAPI.
# Note, this script needs a dataset to be formatted into a collection & item structure already..
# Note, ..

import json
from geojson import Feature, Point, FeatureCollection, Polygon
import os

import click
import requests


def load_data(data_dir, filename):
    """Load json data from a file within the specified data directory."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        click.secho(f"File not found: {filepath}", fg="red", err=True)
        raise click.Abort()
    with open(filepath) as file:
        return json.load(file)


def load_collection(base_url, collection_id, data_dir):
    """Load a STAC collection into the database."""
    collection = load_data(data_dir, "collection.json")
    collection["id"] = collection_id
    try:
        resp = requests.post(f"{base_url}/collections", json=collection)
        if resp.status_code == 200:
            click.echo(f"Status code: {resp.status_code}")
            click.echo(f"Added collection: {collection['id']}")
        elif resp.status_code == 409:
            click.echo(f"Status code: {resp.status_code}")
            click.echo(f"Collection: {collection['id']} already exists")
    except requests.ConnectionError:
        click.secho("Failed to connect", fg="red", err=True)


# Adjusted: added to remove collection from the catalog
def load_items(base_url, collection_id, use_bulk, data_dir):
    """Load STAC items into the database based on the method selected."""
    # # Attempt to dynamically find a suitable feature collection file
    # feature_files = [
    #     file
    #     for file in os.listdir(data_dir)
    #     if file.endswith(".json") and file != "collection.json"
    # ]
    # if not feature_files:
    #     click.secho(
    #         "No feature collection files found in the specified directory.",
    #         fg="red",
    #         err=True,
    #     )
    # #     raise click.Abort()
    # feature_collection_file = feature_files[
    #     0
    # ]  # Use the first found feature collection file
    featurecol = []
    for dir, fol, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".json") and filename != "collection.json":
                filepath = os.path.join(data_dir, dir, filename)
                with open(filepath) as file:
                    data = json.load(file)
                    del data["properties"][
                        "deltares:paint"
                    ]  # remove the paint property, will be mapbox (redundant)
                    featurecol.append(data)
    feature_collection = FeatureCollection(featurecol)
    # feature_collection = load_data(data_dir, feature_collection_file)

    load_collection(base_url, collection_id, data_dir)
    if use_bulk:
        load_items_bulk_insert(base_url, collection_id, feature_collection, data_dir)
    else:
        load_items_one_by_one(base_url, collection_id, feature_collection, data_dir)


def load_items_one_by_one(base_url, collection_id, feature_collection, data_dir):
    """Load STAC items into the database one by one."""
    for feature in feature_collection["features"]:
        try:
            feature["collection"] = collection_id
            resp = requests.post(
                f"{base_url}/collections/{collection_id}/items", json=feature
            )
            if resp.status_code == 200:
                click.echo(f"Status code: {resp.status_code}")
                click.echo(f"Added item: {feature['id']}")
            elif resp.status_code == 409:
                click.echo(f"Status code: {resp.status_code}")
                click.echo(f"Item: {feature['id']} already exists")
        except requests.ConnectionError:
            click.secho("Failed to connect", fg="red", err=True)


def load_items_bulk_insert(base_url, collection_id, feature_collection, data_dir):
    """Load STAC items into the database via bulk insert."""
    try:
        for i, _ in enumerate(feature_collection["features"]):
            feature_collection["features"][i]["collection"] = collection_id
        resp = requests.post(
            f"{base_url}/collections/{collection_id}/items", json=feature_collection
        )
        if resp.status_code == 200:
            click.echo(f"Status code: {resp.status_code}")
            click.echo("Bulk inserted items successfully.")
        elif resp.status_code == 204:
            click.echo(f"Status code: {resp.status_code}")
            click.echo("Bulk update successful, no content returned.")
        elif resp.status_code == 409:
            click.echo(f"Status code: {resp.status_code}")
            click.echo("Conflict detected, some items might already exist.")
    except requests.ConnectionError:
        click.secho("Failed to connect", fg="red", err=True)


# Adjusted: added to remove collection from the catalog
def remove_collection(base_url, collection_id):
    """Delete a STAC collection from the database."""
    try:
        resp = requests.delete(f"{base_url}/collections/{collection_id}")
    except requests.ConnectionError:
        click.secho("Failed to connect", fg="red", err=True)


# @click.command()
# @click.option("--base-url", required=True, help="Base URL of the STAC API")
# @click.option(
#     "--collection-id",
#     default="test-collection",
#     help="ID of the collection to which items are added",
# )
# @click.option("--use-bulk", is_flag=True, help="Use bulk insert method for items")
# @click.option(
#     "--data-dir",
#     type=click.Path(exists=True),
#     default="sample_data/",
#     help="Directory containing collection.json and feature collection file",
# )
# def main(base_url, collection_id, use_bulk, data_dir):
#     """Load STAC items into the database."""
#     load_items(base_url, collection_id, use_bulk, data_dir)


if __name__ == "__main__":
    """Load STAC items into the database."""
    base_url = "http://34.91.70.6:8080"  # URL of the (fast)STAC API
    collection_id = "example-collection"
    use_bulk = False
    # data_dir = r"C:\Users\kras\Documents\GitHub\coclicodata\current\ssl"
    data_dir = r"C:\Users\kras\Documents\GitHub\coclicodata\current\ssl"
    # data_dir = r"C:\Users\kras\Documents\GitHub\stac-fastapi-elasticsearch-opensearch\sample_data"
    load_items(base_url, collection_id, use_bulk, data_dir)

    # remove_collection(base_url, collection_id)
