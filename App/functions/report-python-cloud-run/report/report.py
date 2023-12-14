from io import BytesIO
from pathlib import Path
import fitz
from shapely import Polygon


def create_report_html():
    pass


def create_report_pdf() -> BytesIO:
    story = fitz.Story()
    htmlpath = Path(__file__).parent / Path("template.html")
    csspath = Path(__file__).parent / Path("template.css")

    HTML = htmlpath.read_bytes().decode()
    CSS = csspath.read_bytes().decode()

    story = fitz.Story(html=HTML, user_css=CSS)

    MEDIABOX = fitz.paper_rect("A4")  # output page format: Letter
    WHERE = MEDIABOX + (36, 36, -36, -36)  # leave borders of 0.5 inches
    in_memory_pdf = BytesIO()
    writer = fitz.DocumentWriter(in_memory_pdf)

    with fitz.DocumentWriter(in_memory_pdf) as writer:
        more = 1
        while more:
            device = writer.begin_page(MEDIABOX)
            more, _ = story.place(WHERE)
            story.draw(device)
            writer.end_page()

    return in_memory_pdf


if __name__ == "__main__":
    pdf = create_report_pdf()
    print(pdf.getvalue())
