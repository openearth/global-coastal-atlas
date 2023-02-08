from dash import Dash, html, dcc, Input, Output, ALL, State, MATCH, ALLSMALLER
#import dash_auth
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import numpy as np
import json

from dash_funcs import shoreline_figure

VALID_USERNAME_PASSWORD_PAIRS = {
                                    'Deltares': 'Globalcoastalatlas'
                                }
# initialize dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )

# load shoreline monitor dataframe
#path_shorelineNL = r'P:\1000545-054-globalbeaches\15_GlobalCoastalAtlas\datasets\ShorelineMonitor\shoreline_NL.csv'
path_shorelineNL = r'assets/data/Shoreline_NL.csv'
df = pd.read_csv(path_shorelineNL)
df['changerate'] = df['changerate'].round(2)
df['color_cr'] = np.where(df['changerate'] < 0, 'red', 'green')
df['color_sandy'] = np.where(df['flag_sandy'] == True, 'brown', 'lightgrey')
df['flag_sandy'] = np.where(df['flag_sandy'] == True, 'sand', 'other')

# Icons
test_png = 'dash/assets/deltares.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')


# initialize figure
def empty_fig():

    fig = px.scatter_mapbox(lat = [-90], lon= [-180], zoom= 7, height=1000, center = {'lat': 52.6, 'lon':5})
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black")
    fig.update_traces(hoverinfo='skip')

    return fig

fig = px.scatter_mapbox(
                        df, lat="Intersect_lat", 
                        lon="Intersect_lon", 
                        zoom= 7,
                        color_discrete_sequence = ["#706B6B"],
                        height=1000, 
                        labels = {'Intersect_lat': 'latitude', 'Intersect_lon': 'longitude'},
                        center = {'lat': 52.6, 'lon':5})
fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black")
fig.update_traces(hoverinfo='skip')

# create theme buttons
button_style = {'color': 'white', 'font-size': 20, 'width': '100%', 'border-color': 'dark', 'textAlign': 'left'}
button_group = dcc.RadioItems(
            [ 
                {"label": html.Div(["Shoreline Development"], style= button_style), "value": 1},
                {"label": html.Div(["Coastal Topography"], style= button_style), "value": 2},
                {"label": html.Div(["Sea Conditions"], style= button_style),"value": 3},
                {'label': html.Div(["Socio-economic"], style= button_style),"value": 3}
            ], 
            value= 0,
            id="choose_theme",
            style= {'width': '100%'},
            className="btn-group-vertical",
            inputClassName="btn-check",
            labelClassName="btn",
)

# Make initial app layout
app.layout = html.Div(children=[dcc.Store(id = 'filter-df-store', data = df.to_dict()),
                                dcc.Store(id = 'store-clicked-datasets', data = {'button_SD': None, 'button_BS': None}),
                                dcc.Store(id = 'store-filter-vals', data = {'changerate_range': None, 'beach_comp': None}),
                      dbc.Card(
                        dbc.CardBody([
                                dbc.Row([
                                        dbc.Row([
                                                dbc.Col(html.Div(id = 'theme_container',
                                                                 children = [dbc.Row([
                                                                                        dbc.Col(html.A([html.Img(src = 'data:image/png;base64,{}'.format(test_base64), style= {'width': '60px'})], href = 'https://www.deltares.nl/en/'), width= 'auto'),
                                                                                        dbc.Col(children='Dutch Coastal Atlas', style = {'color': 'white', 'font-size': 20,'textAlign': 'center'}, width = 'auto')
                                                                                    
                                                                                    ], align = 'center'),
                                                                            html.Br(),
                                                                            html.H2(children = 'Themes', style = {'color': 'white', 'textAlign': 'left'}),
                                                                            html.Br(),
                                                                            button_group
                                                                            ]
                                                                ), width = 2, style = {'background-color': 'dark'}
                                                        ),
                                                dbc.Col(html.Div(id = 'geomap_container', children =  dcc.Graph(id='mapbox',figure= fig)), width = 6, style = {'background-color' : 'dark'}),
                                                dbc.Col([
                                                        dbc.Row(html.Div(id = 'data_container', children = [])),
                                                        dbc.Row(html.Div(id = 'plot_container', children = []))
                                                        ], width = 4, style = {'background-color': 'dark'})
                                                ])
                                        ])
                                    ])
                                , color = 'dark') 
                            ])       
                     

@app.callback(
    Output(component_id = 'data_container', component_property = 'children'),
    Input(component_id = 'choose_theme', component_property = 'value'),
    State(component_id = 'store-filter-vals', component_property = 'data'),
    State(component_id = 'store-clicked-datasets', component_property = 'data')
)
def update_data_container(value, filter_json, button_json):
    cr_range = filter_json['changerate_range']
    beachcomp = filter_json['beach_comp']

    SD_buttonval = button_json['button_SD']
    BS_buttonval = button_json['button_BS']

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
                                                            value = [] if SD_buttonval == None else SD_buttonval,
                                                            id= {'type': 'button_shorelinedevelop', 'index': value},
                                                            labelStyle= {'color': 'white', 'font-size': 20, 'textAlign': 'left', 'display': 'block'},
                                                            inputStyle={"margin-left": "20px", "margin-right": "10px", 'margin-bottom': '50px'}
                                                            ), width = 4, style = {'margin-left': '10px'}  
                                            ),
                                    dbc.Col(
                                            dcc.RangeSlider(id= {'type': 'range_shorelinedevelop', 'index': value}, min= -50, max = 50, 
                                                            disabled = True, value = [-50, 50] if cr_range == None else cr_range,
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
                                                            value= [] if BS_buttonval == None else BS_buttonval,
                                                            id= {'type': 'button_beachstate', 'index': value},
                                                            labelStyle= {'color': 'white', 'font-size': 20, 'textAlign': 'left', 'display': 'block'},
                                                            inputStyle={"margin-left": "20px", "margin-right": "10px", 'margin-bottom': '30px'}
                                                        )
                                            ), 
                                    html.Br(),
                                    dbc.Col([
                                            dbc.Row(dcc.Checklist(options = [
                                                                        {'label': 'Sand', 'value': 'sand', 'disabled': True},
                                                                        {'label': 'Other', 'value': 'other', 'disabled': True}
                                                                    ],
                                                            value = ['sand', 'other'] if beachcomp == None else beachcomp,
                                                            id= {'type': 'radio_composition', 'index': value},
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
        Output(component_id = 'store-filter-vals', component_property = 'data'),
        Input(component_id = {'type': 'range_shorelinedevelop', 'index': ALL}, component_property = 'value'),
        Input(component_id = {'type': 'radio_composition', 'index': ALL}, component_property = 'value'),
        State(component_id = 'store-filter-vals', component_property = 'data'),
)
def store_filter_values(cr_range, beachcomp, store_dict):
    if len(cr_range) > 0 and not None in cr_range:  
        cr_range = (cr_range[0][0], cr_range[0][1])
        store_dict['changerate_range'] = cr_range

    if len(beachcomp) > 0: 
        beachcomp = beachcomp[0]
        store_dict['beach_comp'] = beachcomp

    return store_dict


@app.callback(
        Output(component_id = 'filter-df-store', component_property = 'data'),
        Input(component_id = 'store-filter-vals', component_property = 'data'),
             )
def filter_dataframe(store_dict):
    df2_cop = df.copy()
    # first filter
    cr_range = store_dict['changerate_range']
    if cr_range != None: df2_cop = df2_cop[df2_cop['changerate'].between(cr_range[0], cr_range[1])]
    
    # second filter
    beachcomp = store_dict['beach_comp']
    if beachcomp != None: df2_cop = df2_cop[df2_cop['flag_sandy'].isin(beachcomp)]
    
    return df2_cop.to_dict()

@app.callback(
            Output(component_id = 'store-clicked-datasets', component_property = 'data'),
            Input(component_id = {'type': 'button_shorelinedevelop', 'index': ALL}, component_property = 'value'),
            Input(component_id = {'type': 'button_beachstate', 'index': ALL}, component_property = 'value'),
            State(component_id = 'store-clicked-datasets', component_property = 'data'),
            )
def store_datasets(button_SD, button_BS, button_dict):
    if button_SD != None and button_SD != []: button_dict['button_SD'] = button_SD[0]
    if button_BS != None and button_BS != []: button_dict['button_BS'] = button_BS[0]

    return button_dict

@app.callback(
    Output(component_id = {'type': 'range_shorelinedevelop', 'index': ALL}, component_property = 'disabled'),
    Input(component_id = {'type': 'button_shorelinedevelop', 'index': ALL}, component_property = 'value'),
)
def development_disability(value):
    if len(value) > 0: 
        value = value[0]
    if 1 in value:
        return [False]
    else:
        return [True]
    
@app.callback(
    Output(component_id = {'type': 'radio_composition', 'index': ALL}, component_property = 'options'),
    Output(component_id = {'type': 'range_dn50', 'index': ALL}, component_property = 'disabled'),
    Input(component_id = {'type': 'button_beachstate', 'index': ALL}, component_property = 'value'),
    State(component_id = {'type': 'radio_composition', 'index': ALL}, component_property = 'options')
            )
def topography_disability(beachval, comp_disabled):
    if len(beachval) > 0: beachval = beachval[0]
    disabilities = []
    if 1 in beachval: 
        new_lst = []
        for d in comp_disabled[0]:
            d['disabled'] = False
            new_lst.append(d)
        disabilities.append([new_lst])
    else:
        new_lst = []
        for d in comp_disabled[0]:
            d['disabled'] = True
            new_lst.append(d)
        disabilities.append([new_lst])

    if 2 in beachval: disabilities.append([False])
    else: disabilities.append([True])

    return tuple(disabilities)


@app.callback(
    Output(component_id = 'mapbox', component_property = 'figure'),
    Input(component_id =  'choose_theme', component_property = 'value'),
    Input(component_id = 'filter-df-store', component_property = 'data')
            )
def update_geofigure(value,  json_dict):

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
                        zoom= 7, height=1000, center = {'lat': 52.6, 'lon':5})
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black", showlegend=False)
        fig.update_traces(marker={'size': 5})
        return fig
    
    if value == 2:
        
        if not df2_cop.empty:
            fig = px.scatter_mapbox(
                            df2_cop, lat="Intersect_lat", 
                            lon="Intersect_lon", 
                            hover_name="transect_id", 
                            color= 'color_sandy',
                            color_discrete_map= {'brown' : 'brown', 'lightgrey' : 'grey'},
                            hover_data = {'color_sandy': False, 'country_name': True, 'changerate': True},
                            labels = {'country_name' : 'country', 'Intersect_lat': 'latitude', 'Intersect_lon': 'longitude'},
                            zoom= 7, height=1000, center = {'lat': 52.6, 'lon':5})
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black", showlegend=False)
            fig.update_traces(marker={'size': 5})
        else:
            fig = empty_fig()
        return fig
    
    else:
        raise PreventUpdate

@app.callback(
    Output(component_id = 'plot_container', component_property = 'children'),
    Input(component_id = 'mapbox', component_property= 'clickData'),
    Input(component_id = {'type': 'radio_tempres', 'index': ALL}, component_property = 'value'),
    Input(component_id =  'choose_theme', component_property = 'value')
)
def create_timeseries(clickdata, tempres, themeval):
    df_cop = df.copy()
    if themeval == 1 and clickdata != None:
        box = clickdata['points'][0]['hovertext']
        dt, dist = json.loads(df_cop[df_cop['transect_id'] == box]['dt'].values[0]), json.loads(df_cop[df_cop['transect_id'] == box]['dist'].values[0])
        outl1, outl2 = json.loads(df_cop[df_cop['transect_id'] == box]['outliers_1'].values[0]), json.loads(df_cop[df_cop['transect_id'] == box]['outliers_2'].values[0])
        outl = outl1
        fig = shoreline_figure(dt = dt, dist = dist, outl = outl, box = box, sdate= '1984-01-01', resolution = 'annual')

        child = [ 
                    dbc.Row(dcc.Graph(figure = fig)),
                    dbc.Row([
                            dbc.Col(html.Div('Temporal resolution:', style = {'color': 'white', 'font-size': 20, "textAlign": "center"})),
                            dbc.Col(dcc.RadioItems(options = [
                                                            {'label': 'Annual', 'value': 'annual', 'disabled': False},
                                                            {'label': 'Monthly', 'value': 'monthly', 'disabled': True}
                                                        ],
                                                value = 'annual',
                                                id= {'type': 'radio_tempres', 'index': themeval},
                                                labelStyle= {'color': 'white', 'font-size': 20, 'textAlign': 'left'},
                                                inputStyle={"margin-right": "5px", 'margin-left': '10px'}
                                            )     
                                    )
                            ], style = {'margin-top': '10px'})
                ]
        return child
    else:
        return []

if __name__ == '__main__':
    app.run_server(debug=True)