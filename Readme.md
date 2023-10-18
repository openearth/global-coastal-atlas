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

| group      | name                  | long_name                                            | units         | type           |   ncollections | duplicate   |
|:-----------|:----------------------|:-----------------------------------------------------|:--------------|:---------------|---------------:|:------------|
| dimension  | ensemble              | ensemble                                             | 1             | int32          |              1 |             |
| dimension  | gwl                   | global warming level                                 | degrees       | float64        |              1 |             |
| dimension  | rp                    | return period                                        | yr            | float64        |              1 |             |
| dimension  | time                  | Time                                                 |               | datetime64[ns] |              3 |             |
| coordinate | changerate            | Changerate                                           | m/yr          | float32        |              1 |             |
| coordinate | coastline_idint       | coastline_idint                                      |               | float64        |              1 |             |
| coordinate | continent             | Continent                                            |               | string         |              6 |             |
| coordinate | country               | Country                                              |               | string         |              6 |             |
| coordinate | country_id            | Country Identity                                     |               | string         |              3 |             |
| coordinate | err_changerate        | Error Changerate                                     | m/yr          | float64        |              1 |             |
| coordinate | err_timespan          | err_timespan                                         | yr            | float64        |              1 |             |
| coordinate | hotspot_id            | Hotspot Identity                                     |               | string         |              2 |             |
| coordinate | intercept             | Intercept                                            | m             | float32        |              1 |             |
| coordinate | intercept             | Intercept                                            | m             | float64        |              1 | X           |
| coordinate | intercept_unc         | Intercept Uncertainty                                | m             | float64        |              1 |             |
| coordinate | lat                   | Latitude                                             | degrees_north | float64        |              7 |             |
| coordinate | lat                   | latitude                                             | degrees_north | float64        |              1 | X           |
| coordinate | lon                   | Longitude                                            | degrees_east  | float64        |              7 |             |
| coordinate | lon                   | longitude                                            | degrees_east  | float64        |              1 | X           |
| coordinate | low_detect_shlined    | low_detect_shlined                                   |               | float64        |              1 |             |
| coordinate | no_sedcomp            | no_sedcomp                                           |               | float64        |              1 |             |
| coordinate | no_shorelines         | Number Of Shorelines                                 |               | float64        |              1 |             |
| coordinate | rmse                  | Root Mean Squared Error                              | m             | float64        |              1 |             |
| coordinate | stations              | stations                                             | 1             | string         |              1 |             |
| coordinate | timespan              | Timespand                                            | yr            | float64        |              1 |             |
| coordinate | transect_geom         | Transect Geometry                                    |               | string         |              7 |             |
| coordinate | transect_id           | Transect Identity                                    |               | string         |              7 |             |
| variable   | changerate            | Changerate                                           | m/yr          | float64        |              3 | X           |
| variable   | changerate_unc        | Changerate Uncertainty                               | m/yr          | float64        |              1 |             |
| variable   | date_nourishment      | Nourishment Date(s)                                  | yr            | string         |              1 |             |
| variable   | esl                   | extreme sea level                                    | m             | float64        |              1 |             |
| variable   | gdp                   | GDP per capita                                       |               | float64        |              1 |             |
| variable   | intercept             | Intercept                                            | m             | float64        |              1 | X           |
| variable   | ldb_type              | Littoral Drift Barrier Type                          |               | string         |              1 |             |
| variable   | littoraldb_id_conf    | Littoral Drift Barrier Identification Confidentially |               | string         |              1 |             |
| variable   | nourishment_id_conf   | Nourishment Identification Confidentially            |               | string         |              1 |             |
| variable   | outliers              | Outliers detection method 1                          |               | float32        |              2 |             |
| variable   | pop_10_m              | Population below 10 m MSL                            |               | float64        |              1 |             |
| variable   | pop_1_m               | Population below 1 m MSL                             |               | float64        |              1 |             |
| variable   | pop_5_m               | Population below 5 m MSL                             |               | float64        |              1 |             |
| variable   | pop_tot               | Total population                                     |               | float64        |              1 |             |
| variable   | reclamation_id_conf   | Reclamation Identification Confidentially            |               | string         |              1 |             |
| variable   | sandy                 | Sandy                                                |               | int8           |              1 |             |
| variable   | seasonal_displacement | Seasonal Displacement                                | m             | float64        |              1 |             |
| variable   | seasonal_id_conf      | Seasonality Identification Confidentially            |               | string         |              1 |             |
| variable   | sediment_label        | Sediment Label                                       |               | int32          |              1 |             |
| variable   | sp                    | Shoreline Position                                   | m             | float64        |              2 |             |
| variable   | sp_ambient            | Ambient Shoreline Position                           | m             | float64        |              1 |             |
| variable   | sp_rcp45_p5           | RCP4.5 5th percentile Shoreline Position             | m             | float64        |              1 |             |
| variable   | sp_rcp45_p50          | RCP4.5 50th percentile Shoreline Position            | m             | float64        |              1 |             |
| variable   | sp_rcp45_p95          | RCP4.5 95th percentile Shoreline Position            | m             | float64        |              1 |             |
| variable   | sp_rcp85_p5           | RCP8.5 5th percentile Shoreline Position             | m             | float64        |              1 |             |
| variable   | sp_rcp85_p50          | RCP8.5 50th percentile Shoreline Position            | m             | float64        |              1 |             |
| variable   | sp_rcp85_p95          | RCP8.5 95th percentile Shoreline Position            | m             | float64        |              1 |             |
| variable   | t_max_seasonal_sp     | Time of Maximum Seasonal Shoreline Position          |               | float64        |              1 |             |
| variable   | t_min_seasonal_sp     | Time of Minimum Seasonal Shoreline Position          |               | float64        |              1 |             |
| variable   | t_recl_construction   | Time of Reclamation Construction                     |               | float64        |              1 |             |

[comment]: <vocab table>



