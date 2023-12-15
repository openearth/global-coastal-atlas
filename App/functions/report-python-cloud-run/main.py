from io import BytesIO
import json
import os

from shapely import Polygon  # type: ignore
from shapely.geometry import shape  # type: ignore
from flask import Flask, make_response, request

from report.report import (
    create_report_html,
    create_report_pdf,
    POLYGON_DEFAULT,
    STAC_ROOT_DEFAULT,
)

app = Flask(__name__)


@app.route("/")
def return_report():
    polygon_str = POLYGON_DEFAULT
    geo: dict = json.loads(polygon_str)
    polygon: Polygon = shape(geo)
    web_page_content = create_report_html(polygon=polygon, stac_root=STAC_ROOT_DEFAULT)
    pdf_object = create_report_pdf(web_page_content)

    response = make_response(pdf_object.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "inline; filename=coastal_report.pdf"
    return response


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
