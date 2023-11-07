Global Coastal Atlas


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
 - Start the development server: `npm run dev`


