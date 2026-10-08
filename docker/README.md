# Docker environment for the Freshem data-prep notebooks

Minimal, standalone JupyterLab container (plain `pip`, no conda) to run:

- `STAC/data/notebooks/15_resistivity.ipynb`
- `STAC/data/notebooks/16_chloride.ipynb`

It exists because the Freshem Zarr stores (`freshem_stac/processed/*.zarr`)
are written in **Zarr v3 format** (`zarr.json`), which the repo's existing
`coclico`/`globalcoastalatlas` conda env can't read (it's pinned to
zarr-python 2.x). Rather than touching that shared conda env, this is a
lightweight image with just `xarray`, `zarr>=3`, `pandas`, `numpy`, `pyproj`
and `jupyterlab`.

The STAC-builder notebooks (`STAC/data/scripts/15_resistivity.ipynb`,
`16_chloride.ipynb`) still need `pystac`/`coclicodata`/`xstac`, so for now
keep running those in the existing `coclico` conda env (we'll revisit whether
they also need containerizing once the data-prep step is confirmed working).

## Run

```powershell
cd docker
docker compose up --build
```

Open http://localhost:8888 (no token). Inside JupyterLab, the repo is mounted
at `/workspace/global-coastal-atlas` and the Freshem outputs at
`/workspace/freshem_stac`, matching the `STAC_DATA_REPO` / `FRESHEM_REPO`
environment variables the notebooks read.

Run `STAC/data/notebooks/15_resistivity.ipynb` first, then
`16_chloride.ipynb`. Each writes a copy of its Zarr store plus
`metadata_<dataset>.json` to `freshem_stac/processed/stac_folder/<dataset>/`.

## Notes

- Nothing here touches git — it's just a local runtime. `docker compose down`
  removes the container; your edits to the mounted repo/data folders persist
  on the host as usual.
- Cloud upload cells in the `scripts/` notebooks stay disabled (`RUN_UPLOAD =
  False`) until bucket/credentials are confirmed.
- To run the same notebooks directly on the host instead (e.g. in the
  existing `coclico` conda env), just upgrade zarr there: `pip install -U
  "zarr>=3"`. The env-var path overrides are only used when set, so nothing
  else changes.
