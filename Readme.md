# Global Coastal Atlas

This repository contains code for the entire Global Coastal Atlas project.
The repository is structured as follows:

- STAC --> A directory with all the 'Back-End' components
  - data --> Directory with the components required for filling the STAC; notebooks, scripts & current
    - current --> **relative** STAC catalog for development purposes
    - notebooks --> Single notebooks for transforming a dataset into a cloud-native CF compliant version
    - scripts --> Python scripts for converting and uploading a dataset to MapBox and for merging all information in a STAC
  - docs --> Directory with workflow visualizations followed within the data directory
  - visualization --> Directory with notebooks for visualizing a single dataset (could be used as input for the Front-End)
  - workbench --> Directory with notebooks for testing / exploring different dataset combination or analyses (expert users)
- App --> A directory with all the 'Front-End components'

## Way of working

When adding a dataset, please make sure to open a branch and follow the entire data workflow as described in `STAC/docs`.

Check whether your changes to the STAC are OK by running `python -m pytest` in the root folder of this branch in the (Miniforge) prompt.
Note, make sure you activated the correct environment ("globalcoastalatlas" in this case). You can also do this in VS Code 'testing' by adding the branch. 
Only when all components are correctly ingested and visualized, open a pull request to main in order to update the platform.

# local development guide Horizon VM

- install windows-nvm: https://github.com/coreybutler/nvm-windows/releases
- restart terminal or VSCode if these were opened
- In VSCode open a cmd terminal
- Install Node 18: `nvm install 18`
- Set to use Node 18: `nvm use 18`
- Add required variables to `app\.env`, look in `app\example.env` for an example
  - NUXT_PUBLIC_MAPBOX_TOKEN
  - NUXT_STAC_ROOT
- make sure you are in the `app` folder: `cd app`
- Install required dependencies: `npm install`
- Start the development server: `npm run dev`