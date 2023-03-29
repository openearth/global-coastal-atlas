import pandas as pd
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from plotly.subplots import make_subplots

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
    fig.update_layout(margin=dict(l=5,r=5,b=5,t=50),
                        showlegend= False, font_size = 15,
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


def Oscillation_STL(df_STL):
    df_STL = df_STL[df_STL.index.year <= 2021]
    years = np.unique(df_STL.index.year.values)
    df_STL_year = [df_STL[df_STL.index.year == year_] for year_ in years]

    maxima, minima = [], []
    for df_STL_y in df_STL_year:
        y = df_STL_y.values

        max_ind = np.argmax(y)
        min_ind = np.argmin(y)

        maxima_y = y[max_ind]; maxima.append(maxima_y)

        # for local minima
        minima_y = y[min_ind]; minima.append(minima_y)

    mean_ampl = np.mean(maxima) - np.mean(minima)

    df_STL = df_STL.resample('D').interpolate(method= 'linear')

    y2, T2 = df_STL.values, df_STL.index

    cross = y2[y2 != 0]
    zero_cross = np.where(np.diff(np.sign(cross)))[0]

    y2, T2 = y2[zero_cross], T2[zero_cross]

    periods = []
    for i in range(0, len(T2)-2, 2):
        p = (T2[i+2] - T2[i]).days
        periods.append(p)

    return np.mean(periods), mean_ampl


def seasonal_shorelinepositions(df, box):
    dt, dist = json.loads(df[df['transect_id'] == box]['dt'].values[0]), json.loads(df[df['transect_id'] == box]['dist'].values[0])
    sdate = pd.to_datetime('1984-04-01')
    dt = [(sdate + pd.DateOffset(x)) if (sdate + pd.DateOffset(x)).day == 1
                                        else (sdate + pd.DateOffset(x) + pd.offsets.MonthBegin(-1)) if (sdate + pd.DateOffset(x)).day < 15
                                        else (sdate + pd.DateOffset(x) + pd.offsets.MonthBegin(0)) for x in dt]
    outl = json.loads(df[df['transect_id'] == box]['outliers'].values[0])
    dt, dist = np.delete(dt,outl), np.delete(dist,outl)
    df = pd.DataFrame(data= dist, index = dt, columns = [box]).sort_index()
    df = df[df.index >= pd.to_datetime('2013-03-01')]
    df.index = pd.to_datetime(df.index.date)

    df_filled = df.resample('MS').asfreq(np.nan).interpolate(method = 'spline', order = 3)
    #df_filled = df.assign(box=df[box].fillna(df[box].rolling(24,min_periods=1,).mean()))
    y = df_filled[box].values
    stl = STL(y, period= 12, seasonal= 61, robust= True, seasonal_deg = 0).fit()
    stl_seas_trend = stl.trend + stl.seasonal
    df_stl = pd.Series(stl.seasonal, index = df_filled.index).to_frame()
    Ps, Ds = Oscillation_STL(df_STL= df_stl)

    fig = make_subplots(rows=2, cols= 1, row_heights = [0.35, 0.65])
    fig.add_trace(go.Scatter(name = 'Seasonal component',  x = df.index, y = stl.seasonal, mode = 'lines', line=dict(color='green', dash = 'solid')), row=1, col=1)
    fig = fig.add_trace(go.Scatter(name = 'Shoreline positions',  x = df.index, y = df[box].values, mode = 'markers', marker=dict(color='rgba(0, 0, 0)')), row=2, col=1)
    fig.add_trace(go.Scatter(x = df.index, y = stl_seas_trend, mode = 'lines', line=dict(color='green', dash = 'dash'), showlegend= False), row=2, col=1)

    fig.update_layout(
    title=dict(text=f'{box}\n: period = {int(Ps)} days and displacement = {round(Ds,1)} m', font=dict(size=15), y=0.95), height = 800)
                              

    fig.show()