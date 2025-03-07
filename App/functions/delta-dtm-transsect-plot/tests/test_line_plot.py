from fastapi.testclient import TestClient
from shapely import LineString
from src.main import LineStringModel


def test_line_plot_endpoint(client: TestClient):
    """Test that the line-plot endpoint returns a PNG image."""
    # Create a simple LineString for testing
    # These coordinates are in the Netherlands (roughly Amsterdam to Utrecht)
    line = LineString([(4.9, 52.37), (5.12, 52.09)])

    # Create the GeoJSON payload
    payload = LineStringModel(line=line).to_geojson_model().model_dump()

    # Make the request to the endpoint
    response = client.post("/line-plot", json=payload)

    # Check that the response is successful
    assert response.status_code == 200

    # Check that the response is a PNG image
    assert response.headers["content-type"] == "image/png"

    # Check that the content is not empty
    assert len(response.content) > 0
