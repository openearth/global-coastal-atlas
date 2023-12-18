import os
import pytest

import main  # type: ignore


@pytest.fixture
def client():
    main.app.testing = True
    return main.app.test_client()


# def test_handler_no_env_variable(client):
#     r = client.get("/")

#     assert r.data.decode() == "Hello World!"
#     assert r.status_code == 200


def test_main_handler(client):
    r = client.get(
        "/?polygon=%7B%22coordinates%22%3A%5B%5B%5B4.4548747899694945%2C53.1816853031429%5D%2C%5B4.156603653116093%2C52.553604613949375%5D%2C%5B3.2409806283587557%2C51.753524628396576%5D%2C%5B3.442140232282071%2C51.404367273499105%5D%2C%5B4.343890180908346%2C51.274368422568756%5D%2C%5B4.732336312623886%2C52.09148048735997%5D%2C%5B5.197084363068569%2C52.86878161448763%5D%2C%5B5.238703591466617%2C53.11095934419029%5D%2C%5B4.4548747899694945%2C53.1816853031429%5D%5D%5D%2C%22type%22%3A%22Polygon%22%7D"
    )

    # assert r.data.decode() == "Hello World!"
    assert r.status_code == 200


# def test_handler_with_env_variable(client):
#     os.environ["NAME"] = "Foo"
#     r = client.get("/")

#     assert r.data.decode() == "Hello Foo!"
#     assert r.status_code == 200
