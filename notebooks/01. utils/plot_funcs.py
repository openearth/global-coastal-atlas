import pandas as pd
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def historical_shorelinepositions(df, box, sdate= '1984-01-01', resolution = 'annual'):
    sdate = pd.to_datetime(sdate)
    
    if resolution == 'annual':
        dt, dist = json.loads(df[df['transect_id'] == box]['dt_annual'].values[0]), json.loads(df[df['transect_id'] == box]['dist_annual'].values[0])
        dist = np.array(dist) - df[df['transect_id'] == box]['intercept'].values[0]
        outl= df[df['transect_id'] == box]['outliers_annual'].values[0]
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
    
    fig.show()

def future_shorelinepositions(df, box):

    date_range = pd.date_range(start = '1984-01-01', end = '2100-01-01', freq = 'AS')
    
    fig = go.Figure(go.Scatter(
            name='Ambient',
            x=date_range,
            y=df[df['transect_id'] == box]['Ambient'].values[0],
            mode='lines',
            line=dict(color='rgb(17, 17, 17)', dash = 'dash'),
        ))

        
    fig1 =   [
        go.Scatter(
            name='RCP4.5',
            x=date_range,
            y=df[df['transect_id'] == box]['RCP4.5_p50'].values[0],
            mode='lines',
            line=dict(color='rgb(15, 183, 245)'),
        ),
        go.Scatter(
            x=date_range,
            y=df[df['transect_id'] == box]['RCP4.5_p5'].values[0],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            x=date_range,
            y=df[df['transect_id'] == box]['RCP4.5_p95'].values[0],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(15, 183, 245, 0.3)',
            fill='tonexty',
            showlegend=False
        )]
    fig2 = [
        go.Scatter(
            name='RCP8.5',
            x=date_range,
            y=df[df['transect_id'] == box]['RCP8.5_p50'].values[0],
            mode='lines',
            line=dict(color='rgb(24, 24, 24)'),
        ),
        go.Scatter(
            x=date_range,
            y=df[df['transect_id'] == box]['RCP8.5_p5'].values[0],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            x=date_range,
            y=df[df['transect_id'] == box]['RCP8.5_p95'].values[0],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(24, 24, 24, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ]

    dt, dist = json.loads(df[df['transect_id'] == box]['dt_annual'].values[0]), json.loads(df[df['transect_id'] == box]['dist_annual'].values[0])
    dist = np.array(dist) - df[df['transect_id'] == box]['intercept'].values[0]
    outl1, outl2 = json.loads(df[df['transect_id'] == box]['outliers_1'].values[0]), json.loads(df[df['transect_id'] == box]['outliers_2'].values[0])
    outl2 = [int(float(m) + sum(1 for i, n in enumerate(outl1) if int(m) >= (n - i))) for m in outl2]
    outl = outl1 + outl2
    dt = [(date_range[0] + pd.DateOffset(int(x)*365))  for x in dt]

    dt, dist = np.array(dt), np.array(dist)
    dt_o, dist_o= dt[outl], dist[outl]
    dt, dist = np.delete(dt, outl), np.delete(dist, outl)
    fig3 = [go.Scatter(name = 'Historical', x = dt, y = dist, mode = 'markers', marker=dict(color='rgba(0, 0, 0)'))]

    new_fig = fig1 + fig2 + fig3
    for f in new_fig:
        fig.add_trace(f)

    if len(dt_o) > 0:
        fig_outl = go.Scatter(name = 'Outliers', x = dt_o, y = dist_o, mode = 'markers', marker=dict(color="#FF0000"))
        fig.add_trace(fig_outl)

    fig.add_vline(x=pd.to_datetime('2021-06-01'), line_width=1.5, line_dash="solid", line_color="red")
    fig.update_layout(margin=dict(l=5,r=5,b=5,t=30), font_size = 15,xaxis_title = 'Time [years]', yaxis_title = 'Shoreline position [m]', legend_traceorder = 'grouped')
    
    fig.show()

def seasonal_shorelinepositions(df, box):
    dt, dist = json.loads(df[df['transect_id'] == box]['dt_annual'].values[0]), json.loads(df[df['transect_id'] == box]['dist_annual'].values[0])
    dt = [(sdate + pd.DateOffset(x)) if (sdate + pd.DateOffset(x)).day == 1
                                        else (sdate + pd.DateOffset(x) + pd.offsets.MonthBegin(-1)) if (sdate + pd.DateOffset(x)).day < 15
                                        else (sdate + pd.DateOffset(x) + pd.offsets.MonthBegin(0)) for x in dt]
    
    stl = STL(y, period= 12, seasonal= 61, robust= True, seasonal_deg = 0).fit()