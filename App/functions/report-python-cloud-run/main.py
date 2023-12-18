import json
import os

from shapely import Polygon  # type: ignore
from shapely.geometry import shape  # type: ignore
from flask import Flask, make_response, request, render_template

from report.report import (
    create_report_html,
    create_report_pdf,
    POLYGON_DEFAULT,
    STAC_ROOT_DEFAULT,
)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def return_report():
    """Return a report for the given polygon"""
    polygon_str = request.args.get("polygon")

    if not polygon_str:
        polygon_str = POLYGON_DEFAULT

    origin = request.headers.get("Referer")
    print(f"detected origin: {origin}")

    # For now we pin the stac_root on a default because we
    # don't have a way to pass it in from the client and cant handle the password
    # protected preview deployments
    stac_root = STAC_ROOT_DEFAULT

    polygon = shape(json.loads(polygon_str))
    if not isinstance(polygon, Polygon):
        raise ValueError("Invalid polygon")

    web_page_content = create_report_html(polygon=polygon, stac_root=stac_root)
    pdf_object = create_report_pdf(web_page_content)

    response = make_response(pdf_object.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "inline; filename=coastal_report.pdf"
    return response


@app.route("/html")
def return_html():
    """Return a report for the given polygon"""
    polygon_str = request.args.get("polygon")

    if not polygon_str:
        polygon_str = POLYGON_DEFAULT

    origin = request.headers.get("Referer")
    print(f"detected origin: {origin}")

    # For now we pin the stac_root on a default because we
    # don't have a way to pass it in from the client and cant handle the password
    # protected preview deployments
    stac_root = STAC_ROOT_DEFAULT

    polygon = shape(json.loads(polygon_str))
    if not isinstance(polygon, Polygon):
        raise ValueError("Invalid polygon")

    web_page_content = create_report_html(polygon=polygon, stac_root=stac_root)

    return render_template(web_page_content)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
