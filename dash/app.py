from dash import Dash, html, dcc, Input, Output, ALL, State, MATCH, ALLSMALLER
import dash_auth
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import numpy as np

### Username
VALID_USERNAME_PASSWORD_PAIRS = {
                                    'Deltares': 'Globalcoastalatlas'
                                }
# initialize dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

# load shoreline monitor dataframe
path_shorelineNL = r'P:\1000545-054-globalbeaches\15_GlobalCoastalAtlas\datasets\ShorelineMonitor\shoreline_NL.csv'
df = pd.read_csv(path_shorelineNL)
df['changerate'] = df['changerate'].round(2)
df['color_cr'] = np.where(df['changerate'] < 0, 'red', 'green')
df['color_sandy'] = np.where(df['flag_sandy'] == True, 'brown', 'lightgrey')

# Icons
test_png = 'dash/assets/deltares.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')


# initialize figure
fig = px.scatter_mapbox(lat = pd.Series([-190]), lon= pd.Series([-90]), zoom= 7, height=1000, center = {'lat': 52.6, 'lon':5})
fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black")

# create theme buttons
button_style = {'color': 'white', 'font-size': 20, 'width': '100%', 'border-color': 'dark', 'textAlign': 'left'}
button_group = dcc.RadioItems(
            [ 
                {"label": html.Div(["Shoreline Development"], style= button_style), "value": 1},
                {"label": html.Div(["Coastal Topography"], style= button_style), "value": 2},
                {"label": html.Div(["Sea Conditions"], style= button_style),"value": 3},
            ], 
            value= 0,
            id="choose_theme",
            style= {'width': '100%'},
            className="btn-group-vertical",
            inputClassName="btn-check",
            labelClassName="btn",
)

# Make initial app layout
app.layout = html.Div(children=[
                      dbc.Card(
                        dbc.CardBody([
                                dbc.Row([
                                        dbc.Row([
                                                dbc.Col(html.Div(id = 'Theme_container',
                                                                 children = [dbc.Row([
                                                                                        dbc.Col(html.Img(src = 'data:image/png;base64,{}'.format(test_base64), style= {'width': '60px'}), width= 'auto'),
                                                                                        dbc.Col(children='Dutch Coastal Atlas', style = {'color': 'white', 'font-size': 20,'textAlign': 'center'}, width = 'auto')
                                                                                    
                                                                                    ], align = 'center'),
                                                                            html.Br(),
                                                                            html.H2(children = 'Themes', style = {'color': 'white', 'textAlign': 'left'}),
                                                                            html.Br(),
                                                                            button_group
                                                                            ]
                                                                ), width = 2, style = {'background-color': 'dark'}
                                                        ),
                                                dbc.Col(html.Div(id = 'Graph_container', children =  dcc.Graph(id='mapbox',figure= fig)), width = 6, style = {'background-color' : 'dark'}),
                                                dbc.Col(html.Div(id = 'Data_container', children = []), width = 4, style = {'background-color': 'dark'})
                                                ])
                                        ])
                                    ])
                                , color = 'dark') 
                            ])       
                     

@app.callback(
    Output(component_id = 'Data_container', component_property = 'children'),
    Input(component_id = 'choose_theme', component_property = 'value')
)
def update_data_container(value):
    if value == 1:
        container = dbc.Col([
                            html.Br(), 
                            html.H2('Shoreline Development', style = {'color': 'white', 'textAlign': 'center'}),
                            html.Br(), 
                            html.Br(), 
                            dbc.Row([
                                    dbc.Col(
                                            dcc.Checklist(
                                                            [ 
                                                                {"label": "Changerate", "value": 1},
                                                                {"label": "Drivers", "value": 2},
                                                            ], 
                                                            value= [0],
                                                            id= {'type': 'button_shorelinedevelop', 'index': value},
                                                            labelStyle= {'color': 'white', 'font-size': 20, 'textAlign': 'left', 'display': 'block'},
                                                            inputStyle={"margin-left": "20px", "margin-right": "10px", 'margin-bottom': '50px'}
                                                            ), width = 4, style = {'margin-left': '10px'}  
                                            ),
                                    dbc.Col(
                                            dcc.RangeSlider(id= {'type': 'range_shorelinedevelop', 'index': value}, min= -50, max = 50, disabled = True,
                                                            allowCross=False, tooltip={"placement": "bottom", "always_visible": True}, marks={
                                                                                                                                                -50: '-50 m/yr',
                                                                                                                                                -25: '-25 m/yr',
                                                                                                                                                0: '0 m/yr',
                                                                                                                                                25: '25 m/yr',
                                                                                                                                                50: '50 m/yr',
                                                                                                                                            }
                                                            ), width = 7, style = {'margin-left': '10px', 'margin-top': '13px'}    
                                            ), 
                                     ], style={'verticalAlign': 'middle'})
                        ])
            
        return container
    
    if value == 2:
        container = dbc.Col([
                            html.Br(), 
                            dbc.Row(html.H2('Beach State', style = {'color': 'white', 'textAlign': 'center'})),
                            html.Br(), 
                            html.Br(), 
                            dbc.Row([
                                    dbc.Col(
                                            dcc.Checklist(options = [ 
                                                                        {"label": "Composition", "value": 1},
                                                                        {"label": 'Dn50', "value": 2},
                                                                      ], 
                                                            value= [0],
                                                            id= {'type': 'button_beachstate', 'index': value},
                                                            labelStyle= {'color': 'white', 'font-size': 20, 'textAlign': 'left', 'display': 'block'},
                                                            inputStyle={"margin-left": "20px", "margin-right": "10px", 'margin-bottom': '30px'}
                                                        )
                                            ), 
                                    html.Br(),
                                    dbc.Col([
                                            dbc.Row(dcc.RadioItems(options = [
                                                                        {'label': 'Sand', 'value': 'sand', 'disabled': True},
                                                                        {'label': 'Other', 'value': 'other', 'disabled': True}
                                                                    ],
                                                            id= {'type': 'radio_beachstate', 'index': value},
                                                            labelStyle= {'color': 'white', 'font-size': 20, 'textAlign': 'left'},
                                                            inputStyle={"margin-left": "10px", "margin-right": "10px"}
                                                          )
                                                    ),
                                            html.Br(),
                                            dbc.Row(dcc.RangeSlider(id= {'type': 'range_dn50', 'index': value}, min= 0, max = 5, disabled = True,
                                                                    allowCross=False, tooltip={"placement": "bottom", "always_visible": True}, marks={
                                                                                                                                                        0: '0 μm',
                                                                                                                                                        1: '1 μm',
                                                                                                                                                        2: '2 μm',
                                                                                                                                                        3: '3 μm',
                                                                                                                                                        4: '4 μm',
                                                                                                                                                        5: '5 μm'
                                                                                                                                                    }
                                                                    )
                                                    )        
                                            ])
                                    ])
                            ])
                    
        return container
    if value == 3:
        container = dbc.Col([
                                html.Br(), 
                                dbc.Row(html.H2('Sea Conditions', style = {'color': 'white', 'textAlign': 'center'}))
                            ])
                    
        return container
    else:
        raise PreventUpdate

@app.callback(
    Output(component_id = {'type': 'range_shorelinedevelop', 'index': ALL}, component_property = 'disabled'),
    Input(component_id = {'type': 'button_shorelinedevelop', 'index': ALL}, component_property = 'value')
)
def range_disability(value):
    if len(value) > 0: value = value[0]
    if 1 in value:
        return [False]
    else:
        return [True]


@app.callback(
    Output(component_id = 'mapbox', component_property = 'figure'),
    Input(component_id =  'choose_theme', component_property = 'value'),
    Input(component_id = {'type': 'range_shorelinedevelop', 'index': ALL}, component_property = 'value'),
    Input(component_id= {'type': 'radio_beachstate', 'index': ALL}, component_property = 'value')
)
def update_figure(value, rangeval, beachcomp):
    # print(value)
    # if len(value) > 0: value = value[0]
    df2_cop = df.copy()
    if value == 1:
        if len(rangeval) > 0 and not None in rangeval:
            minv, maxv = rangeval[0][0], rangeval[0][1]
            df2_cop = df2_cop[df2_cop['changerate'].between(minv, maxv)]
        else:
            df2_cop = df2_cop
        fig = px.scatter_mapbox(
                        df2_cop, lat="Intersect_lat", 
                        lon="Intersect_lon", 
                        hover_name="transect_id", 
                        color= 'color_cr',
                        color_discrete_map= {'red' : 'red', 'green' : 'green'},
                        hover_data = {'color_cr': False, 'country_name': True, 'changerate': True},
                        labels = {'country_name' : 'country', 'Intersect_lat': 'latitude', 'Intersect_lon': 'longitude'},
                        zoom= 7, height=1000, center = {'lat': 52.6, 'lon':5})
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black", showlegend=False)
        fig.update_traces(marker={'size': 5})
        return fig
    
    if value == 2:
        if beachcomp == 'sand':
            df2_cop = df2_cop[df2_cop['flag_sandy'] == True]
        if beachcomp == 'other':
            df2_cop = df2_cop[df2_cop['flag_sandy'] == True]

        fig = px.scatter_mapbox(zoom= 7, height=1000, center = {'lat': 40, 'lon':5})
        fig = px.scatter_mapbox(
                        df2_cop, lat="Intersect_lat", 
                        lon="Intersect_lon", 
                        hover_name="transect_id", 
                        color= 'color_sandy',
                        color_discrete_map= {'brown' : 'brown', 'lightgrey' : 'grey'},
                        hover_data = {'color_cr': False, 'country_name': True, 'changerate': True},
                        labels = {'country_name' : 'country', 'Intersect_lat': 'latitude', 'Intersect_lon': 'longitude'},
                        zoom= 7, height=1000, center = {'lat': 52.6, 'lon':5})
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black", showlegend=False)
        fig.update_traces(marker={'size': 5})
        return fig
    
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)