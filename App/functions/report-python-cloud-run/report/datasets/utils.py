import base64
from io import BytesIO
from matplotlib import pyplot as plt


def plot_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to base64"""
    img = BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")

    return base64.b64encode(img.getbuffer()).decode("ascii")


def plot_to_svg(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to svg"""
    img = BytesIO()
    fig.savefig(img, format="svg", bbox_inches="tight")

    return img.getvalue().decode("ascii")
