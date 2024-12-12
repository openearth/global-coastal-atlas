# Developing the report-python-cloud-run Function

## Prerequisites

Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Testing

Run the report function locally:

```bash
python report.py
```

## Deploying

Deploying to Cloud run is done using github actions. The workflow is defined in `.github/workflows/deploy_function.yml`. The workflow is triggered on push to the `main` branch.