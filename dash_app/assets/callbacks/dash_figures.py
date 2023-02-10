import pandas as pd
import numpy as np
import plotly.express as px
import json

# initialize figure
def empty_map():

    fig = px.scatter_mapbox(lat = [-90], lon= [-180], zoom= 7, height=1000, center = {'lat': 52, 'lon':5.3})
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black")
    fig.update_traces(hoverinfo='skip')
    return fig

def geo_figure(value, json_dict):
    df2_cop = pd.DataFrame.from_dict(json_dict, orient='columns')

    if value == 1:
        fig = px.scatter_mapbox(
                        df2_cop, lat="Intersect_lat", 
                        lon="Intersect_lon", 
                        hover_name="transect_id", 
                        color= 'color_cr',
                        color_discrete_map= {'red' : 'red', 'green' : 'green'},
                        hover_data = {'color_cr': False, 'country_name': True, 'changerate': True},
                        labels = {'country_name' : 'country', 'Intersect_lat': 'latitude', 'Intersect_lon': 'longitude'},
                        zoom= 7, height=1000, center = {'lat': 52, 'lon':5.3})
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black", showlegend=False)
        fig.update_traces(marker={'size': 5})

        fig_line = px.line_mapbox(
                                    df2_cop, lat="Intersect_lat",
                                    hover_name="transect_id",
                                    hover_data = {'color_line': False, 'country_name': True, 'changerate': True},
                                    labels = {'country_name' : 'country', 'Intersect_lat': 'latitude', 'Intersect_lon': 'longitude'},
                                    lon="Intersect_lon", line_group= 'hotspot', color = 'color_line',
                                    color_discrete_map= {'red' : 'red', 'green' : 'green'},
                                )
        fig.add_traces(
            fig_line.data
        )
        return fig
    
    if value == 2:
        
        if not df2_cop.empty:
            fig = px.scatter_mapbox(
                            df2_cop, lat="Intersect_lat", 
                            lon="Intersect_lon", 
                            hover_name="transect_id", 
                            color= 'prediction_label',
                            color_discrete_map= {'sand' : '#E3B225', 'mud': '#993307', 
                                                 'rocky': '#201E2C', 'vegetation': '#0C953B', 'other': '#A2A6A3'},
                            hover_data = {'color_sandy': False, 'country_name': True, 'prediction_label': True, 'changerate': True},
                            labels = {'country_name' : 'country', 'Intersect_lat': 'latitude', 'Intersect_lon': 'longitude', 'prediction_label': 'composition'},
                            zoom= 7, height=1000, center = {'lat': 52, 'lon':5.3})
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black", showlegend=False)
            fig.update_traces(marker={'size': 5})
        else:
            fig = empty_map()
    else:
        fig = None

    return fig

def shoreline_scatter(df, box, sdate= '1984-01-01', resolution = 'annual'):
    sdate = pd.to_datetime(sdate)
    
    if resolution == 'annual':
        dt, dist = json.loads(df[df['transect_id'] == box]['dt_annual'].values[0]), json.loads(df[df['transect_id'] == box]['dist_annual'].values[0])
        dist = np.array(dist) - df[df['transect_id'] == box]['intercept'].values[0]
        outl1, outl2 = json.loads(df[df['transect_id'] == box]['outliers_annual'].values[0]), json.loads(df[df['transect_id'] == box]['outliers_2'].values[0])
        outl = outl1
        dt = [(sdate + pd.DateOffset(int(x)*365))  for x in dt]
    else:
        dt, dist = json.loads(df[df['transect_id'] == box]['dt_monthly'].values[0]), json.loads(df[df['transect_id'] == box]['dist_monthly'].values[0])
        outl = json.loads(df[df['transect_id'] == box]['outliers_monthly'].values[0])
        dt = [(sdate + pd.DateOffset(x)) if (sdate + pd.DateOffset(x)).day == 1
                                        else (sdate + pd.DateOffset(x) + pd.offsets.MonthBegin(-1)) if (sdate + pd.DateOffset(x)).day < 15
                                        else (sdate + pd.DateOffset(x) + pd.offsets.MonthBegin(0)) for x in dt]
    dt, dist = np.array(dt), np.array(dist)
    dt_o, dist_o= dt[outl], dist[outl]
    dt, dist = np.delete(dt, outl), np.delete(dist, outl)
    fig = px.scatter(x = dt, y = dist, trendline= 'ols', trendline_color_override="blue" ,
                         title= box, color_discrete_sequence = ["#212529"])
    fig.update_layout(margin=dict(l=5,r=5,b=5,t=30), paper_bgcolor="#212529", plot_bgcolor='#9C9C9C', 
                        showlegend= False, font_color = 'white', font_size = 15,
                        xaxis_title = 'Time [years]', yaxis_title = 'Shoreline position [m]',
                        )
    if len(dt_o) > 0:
        fig2 = px.scatter(x = dt_o, y = dist_o, trendline_color_override="red" ,
                            title= box, color_discrete_sequence = ["red"])
        fig.add_trace(fig2.data[0])

    fig.for_each_trace(lambda trace: trace.update(marker= {'size': 10}))
    
    return fig