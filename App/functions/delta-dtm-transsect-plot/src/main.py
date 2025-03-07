import io
import traceback
from typing import Annotated

import fastapi
import matplotlib

matplotlib.use("AGG")

import matplotlib.pyplot as plt
import uvicorn
from fastapi import BackgroundTasks, Body
from fastapi.responses import Response
from pydantic_shapely import FeatureBaseModel, GeometryField
from shapely import LineString
from src.transsect_plot import plot_transsect

app = fastapi.FastAPI(
    title="Delta DTM Line Plot API",
    description="API for generating elevation profile plots along a transect line",
    version="1.0.0",
)


class LineStringModel(FeatureBaseModel, geometry_field="line"):
    line: Annotated[LineString, GeometryField()]


@app.post(
    "/line-plot",
    response_class=Response,
    summary="Generate an elevation profile plot along a transect line",
    description="Takes a LineString geometry and returns a plot of the elevation profile along that line",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "A PNG image of the elevation profile plot",
        },
        500: {
            "description": "Internal server error",
        },
    },
)
async def line_plot(
    line_string: Annotated[
        LineStringModel.GeoJsonDataModel,  # type: ignore
        Body(
            example={
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[4.9, 52.37], [5.12, 52.09]],
                },
                "properties": {},
            }
        ),
    ],
    background_tasks: BackgroundTasks,
) -> Response:
    try:
        # Use the existing plot_transsect function to generate the figure
        line: LineString = line_string.geometry.to_shapely()
        fig = await plot_transsect(line)

        # Create an in-memory bytes buffer
        buf = io.BytesIO()

        # Save the figure to the in-memory buffer
        fig.savefig(buf, format="png")

        # Close the figure to free up resources
        plt.close(fig)

        # Seek to the beginning of the buffer
        buf.seek(0)

        # Return the image as a response with appropriate headers
        background_tasks.add_task(buf.close)
        background_tasks.add_task(plt.close, fig)
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="transect_plot.png"'},
        )
    except Exception as e:
        # Log the error
        print("Error generating plot:")
        print(traceback.format_exc())
        # Raise an HTTP exception
        raise fastapi.HTTPException(
            status_code=500,
            detail=f"Error generating plot: {str(e)}",
        )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to the Delta DTM Line Plot API. Use /line-plot endpoint to generate elevation profile plots."
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
