# Minimal Jupyter environment for the Freshem data-prep notebooks
# (STAC/data/notebooks/15_resistivity.ipynb and 16_chloride.ipynb).
#
# Just enough to read the Freshem Zarr v3 stores, reproject extents and
# write the enriched Zarr + metadata JSON. Lighter/faster than the repo's
# full "coclico" conda env (no GDAL/pystac/coclicodata). The STAC-builder
# notebooks in STAC/data/scripts/ still need the coclico env for now.
FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token="]
