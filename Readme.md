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

 | name                  | long_name                                            | units         | dtype   | stucture_type   |   ncollections | collections                          |
|:----------------------|:-----------------------------------------------------|:--------------|:--------|:----------------|---------------:|:-------------------------------------|
| lat                   | Latitude                                             |               |         | dim             |              7 | shore_mon_hr, shore_mon_drivers, ... |
| lon                   | Longitude                                            |               |         | dim             |              7 | shore_mon_hr, shore_mon_drivers, ... |
| gwl                   | global warming level                                 |               |         | dim             |              1 | esl_gwl                              |
| lat                   | latitude                                             |               |         | dim             |              1 | esl_gwl                              |
| lon                   | longitude                                            |               |         | dim             |              1 | esl_gwl                              |
| rp                    | return period                                        |               |         | dim             |              1 | esl_gwl                              |
| lat                   | Latitude                                             | degrees_north |         | var             |              7 | shore_mon_hr, shore_mon_drivers, ... |
| lon                   | Longitude                                            | degrees_east  |         | var             |              7 | shore_mon_hr, shore_mon_drivers, ... |
| transect_geom         |                                                      |               |         | var             |              7 | shore_mon_hr, shore_mon_drivers, ... |
| transect_id           |                                                      |               |         | var             |              7 | shore_mon_hr, shore_mon_drivers, ... |
| continent             |                                                      |               |         | var             |              6 | shore_mon_drivers, shore_mon, ...    |
| country               |                                                      |               |         | var             |              6 | shore_mon_drivers, shore_mon, ...    |
| changerate            | Changerate                                           | m/yr          |         | var             |              4 | shore_mon_fut, shore_mon, ...        |
| country_id            |                                                      |               |         | var             |              3 | shore_mon_fut, shore_mon, ...        |
| intercept             | Intercept                                            | m             |         | var             |              3 | shore_mon_fut, shore_mon, ...        |
| hotspot_id            |                                                      |               |         | var             |              2 | shore_mon_hr, shore_mon_drivers      |
| outliers              | Outliers detection method 1                          |               |         | var             |              2 | shore_mon_hr, shore_mon              |
| sp                    | Shoreline Position                                   | m             |         | var             |              2 | shore_mon_hr, shore_mon              |
| changerate_unc        | Changerate Uncertainty                               | m/yr          |         | var             |              1 | shore_mon                            |
| coastline_idint       | coastline_idint                                      |               |         | var             |              1 | shore_mon                            |
| date_nourishment      | Nourishment Date(s)                                  | yr            |         | var             |              1 | shore_mon_drivers                    |
| err_changerate        | Error Changerate                                     | m/yr          |         | var             |              1 | shore_mon                            |
| err_timespan          | err_timespan                                         | yr            |         | var             |              1 | shore_mon                            |
| esl                   | extreme sea level                                    | m             |         | var             |              1 | esl_gwl                              |
| gdp                   | GDP per capita                                       |               |         | var             |              1 | world_gdp                            |
| intercept_unc         | Intercept Uncertainty                                | m             |         | var             |              1 | shore_mon                            |
| lat                   | latitude                                             | degrees_north |         | var             |              1 | esl_gwl                              |
| ldb_type              | Littoral Drift Barrier Type                          |               |         | var             |              1 | shore_mon_drivers                    |
| littoraldb_id_conf    | Littoral Drift Barrier Identification Confidentially |               |         | var             |              1 | shore_mon_drivers                    |
| lon                   | longitude                                            | degrees_east  |         | var             |              1 | esl_gwl                              |
| low_detect_shlined    | low_detect_shlined                                   |               |         | var             |              1 | shore_mon                            |
| no_sedcomp            | no_sedcomp                                           |               |         | var             |              1 | shore_mon                            |
| no_shorelines         | Number Of Shorelines                                 |               |         | var             |              1 | shore_mon                            |
| nourishment_id_conf   | Nourishment Identification Confidentially            |               |         | var             |              1 | shore_mon_drivers                    |
| pop_10_m              | Population below 10 m MSL                            |               |         | var             |              1 | world_pop                            |
| pop_1_m               | Population below 1 m MSL                             |               |         | var             |              1 | world_pop                            |
| pop_5_m               | Population below 5 m MSL                             |               |         | var             |              1 | world_pop                            |
| pop_tot               | Total population                                     |               |         | var             |              1 | world_pop                            |
| reclamation_id_conf   | Reclamation Identification Confidentially            |               |         | var             |              1 | shore_mon_drivers                    |
| rmse                  | Root Mean Squared Error                              | m             |         | var             |              1 | shore_mon                            |
| sandy                 | Sandy                                                |               |         | var             |              1 | shore_mon                            |
| seasonal_displacement | Seasonal Displacement                                | m             |         | var             |              1 | shore_mon_drivers                    |
| seasonal_id_conf      | Seasonality Identification Confidentially            |               |         | var             |              1 | shore_mon_drivers                    |
| sediment_label        | Sediment Label                                       |               |         | var             |              1 | sed_class                            |
| sp_ambient            | Ambient Shoreline Position                           | m             |         | var             |              1 | shore_mon_fut                        |
| sp_rcp45_p5           | RCP4.5 5th percentile Shoreline Position             | m             |         | var             |              1 | shore_mon_fut                        |
| sp_rcp45_p50          | RCP4.5 50th percentile Shoreline Position            | m             |         | var             |              1 | shore_mon_fut                        |
| sp_rcp45_p95          | RCP4.5 95th percentile Shoreline Position            | m             |         | var             |              1 | shore_mon_fut                        |
| sp_rcp85_p5           | RCP8.5 5th percentile Shoreline Position             | m             |         | var             |              1 | shore_mon_fut                        |
| sp_rcp85_p50          | RCP8.5 50th percentile Shoreline Position            | m             |         | var             |              1 | shore_mon_fut                        |
| sp_rcp85_p95          | RCP8.5 95th percentile Shoreline Position            | m             |         | var             |              1 | shore_mon_fut                        |
| stations              |                                                      |               |         | var             |              1 | esl_gwl                              |
| t_max_seasonal_sp     | Time of Maximum Seasonal Shoreline Position          |               |         | var             |              1 | shore_mon_drivers                    |
| t_min_seasonal_sp     | Time of Minimum Seasonal Shoreline Position          |               |         | var             |              1 | shore_mon_drivers                    |
| t_recl_construction   | Time of Reclamation Construction                     |               |         | var             |              1 | shore_mon_drivers                    |
| timespan              | Timespand                                            | yr            |         | var             |              1 | shore_mon                            | 

[comment]: <vocab table>



