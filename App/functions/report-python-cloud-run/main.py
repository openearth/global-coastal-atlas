from io import BytesIO
import os

from flask import Flask, make_response

from report.report import create_report_pdf

app = Flask(__name__)


@app.route("/")
def return_report():
    pdf_object: BytesIO = create_report_pdf()

    response = make_response(pdf_object.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "inline; filename=coastal_report.pdf"
    return response


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
