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
Only when all components are correctly ingested and visualized, open a pull request to main in order to update the platform. 

## Controlled vocabulary
The table below contains the controlled vocabulary used for the datasets in the Global Coastal Atlas stac.

[comment]: <vocab table>

 | name                  | long_name                                            | units         | dtype   | stucture_type   |   ncollections |
|:----------------------|:-----------------------------------------------------|:--------------|:--------|:----------------|---------------:|
| lat                   | Latitude                                             |               |         | dim             |              7 |
| lon                   | Longitude                                            |               |         | dim             |              7 |
| gwl                   | global warming level                                 |               |         | dim             |              1 |
| lat                   | latitude                                             |               |         | dim             |              1 |
| lon                   | longitude                                            |               |         | dim             |              1 |
| rp                    | return period                                        |               |         | dim             |              1 |
| lat                   | Latitude                                             | degrees_north |         | var             |              7 |
| lon                   | Longitude                                            | degrees_east  |         | var             |              7 |
| transect_geom         |                                                      |               |         | var             |              7 |
| transect_id           |                                                      |               |         | var             |              7 |
| continent             |                                                      |               |         | var             |              6 |
| country               |                                                      |               |         | var             |              6 |
| changerate            | Changerate                                           | m/yr          |         | var             |              4 |
| country_id            |                                                      |               |         | var             |              3 |
| intercept             | Intercept                                            | m             |         | var             |              3 |
| hotspot_id            |                                                      |               |         | var             |              2 |
| outliers              | Outliers detection method 1                          |               |         | var             |              2 |
| sp                    | Shoreline Position                                   | m             |         | var             |              2 |
| changerate_unc        | Changerate Uncertainty                               | m/yr          |         | var             |              1 |
| coastline_idint       | coastline_idint                                      |               |         | var             |              1 |
| date_nourishment      | Nourishment Date(s)                                  | yr            |         | var             |              1 |
| err_changerate        | Error Changerate                                     | m/yr          |         | var             |              1 |
| err_timespan          | err_timespan                                         | yr            |         | var             |              1 |
| esl                   | extreme sea level                                    | m             |         | var             |              1 |
| gdp                   | GDP per capita                                       |               |         | var             |              1 |
| intercept_unc         | Intercept Uncertainty                                | m             |         | var             |              1 |
| lat                   | latitude                                             | degrees_north |         | var             |              1 |
| ldb_type              | Littoral Drift Barrier Type                          |               |         | var             |              1 |
| littoraldb_id_conf    | Littoral Drift Barrier Identification Confidentially |               |         | var             |              1 |
| lon                   | longitude                                            | degrees_east  |         | var             |              1 |
| low_detect_shlined    | low_detect_shlined                                   |               |         | var             |              1 |
| no_sedcomp            | no_sedcomp                                           |               |         | var             |              1 |
| no_shorelines         | Number Of Shorelines                                 |               |         | var             |              1 |
| nourishment_id_conf   | Nourishment Identification Confidentially            |               |         | var             |              1 |
| pop_10_m              | Population below 10 m MSL                            |               |         | var             |              1 |
| pop_1_m               | Population below 1 m MSL                             |               |         | var             |              1 |
| pop_5_m               | Population below 5 m MSL                             |               |         | var             |              1 |
| pop_tot               | Total population                                     |               |         | var             |              1 |
| reclamation_id_conf   | Reclamation Identification Confidentially            |               |         | var             |              1 |
| rmse                  | Root Mean Squared Error                              | m             |         | var             |              1 |
| sandy                 | Sandy                                                |               |         | var             |              1 |
| seasonal_displacement | Seasonal Displacement                                | m             |         | var             |              1 |
| seasonal_id_conf      | Seasonality Identification Confidentially            |               |         | var             |              1 |
| sediment_label        | Sediment Label                                       |               |         | var             |              1 |
| sp_ambient            | Ambient Shoreline Position                           | m             |         | var             |              1 |
| sp_rcp45_p5           | RCP4.5 5th percentile Shoreline Position             | m             |         | var             |              1 |
| sp_rcp45_p50          | RCP4.5 50th percentile Shoreline Position            | m             |         | var             |              1 |
| sp_rcp45_p95          | RCP4.5 95th percentile Shoreline Position            | m             |         | var             |              1 |
| sp_rcp85_p5           | RCP8.5 5th percentile Shoreline Position             | m             |         | var             |              1 |
| sp_rcp85_p50          | RCP8.5 50th percentile Shoreline Position            | m             |         | var             |              1 |
| sp_rcp85_p95          | RCP8.5 95th percentile Shoreline Position            | m             |         | var             |              1 |
| stations              |                                                      |               |         | var             |              1 |
| t_max_seasonal_sp     | Time of Maximum Seasonal Shoreline Position          |               |         | var             |              1 |
| t_min_seasonal_sp     | Time of Minimum Seasonal Shoreline Position          |               |         | var             |              1 |
| t_recl_construction   | Time of Reclamation Construction                     |               |         | var             |              1 |
| timespan              | Timespand                                            | yr            |         | var             |              1 | 

[comment]: <vocab table>



