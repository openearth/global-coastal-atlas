import pandas as pd
import json
import numpy as np
import plotly.express as px

def shoreline_figure(dt, dist, outl, box, sdate= '1984-01-01', resolution = 'annual'):
    sdate = pd.to_datetime(sdate)
    if resolution == 'annual':
        dt = [(sdate + pd.DateOffset(int(x)*365))  for x in dt]
    else:
        dt = [(sdate + pd.DateOffset(x)) if (sdate + pd.DateOffset(x)).day == 1
                                        else (sdate + pd.DateOffset(x) + pd.offsets.MonthBegin(-1)) if (sdate + pd.DateOffset(x)).day < 15
                                        else (sdate + pd.DateOffset(x) + pd.offsets.MonthBegin(0)) for x in dt_d]
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