from dash import Dash, html, dcc, Input, Output, ALL, State, MATCH, ALLSMALLER
#import dash_auth
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import numpy as np
import json

# from dash_funcs import shoreline_figure, empty_fig

VALID_USERNAME_PASSWORD_PAIRS = {
                                    'Deltares': 'Globalcoastalatlas123'
                                }
# initialize dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dutch Coastal Atlas"
server = app.server
####
# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )
from assets.callbacks.data_container import data_container
from assets.callbacks.dash_figures import shoreline_scatter, geo_figure

# load shoreline monitor dataframe
df = pd.read_csv('assets/data/all_data.csv')
# Icons
delt_png = 'assets/images/deltares.png'
delt_base64 = base64.b64encode(open(delt_png, 'rb').read()).decode('ascii')

fig = px.scatter_mapbox(
                        df, lat="Intersect_lat", 
                        lon="Intersect_lon", 
                        zoom= 7,
                        color_discrete_sequence = ["#706B6B"],
                        height=1000, 
                        labels = {'Intersect_lat': 'latitude', 'Intersect_lon': 'longitude'},
                        center = {'lat': 52, 'lon':5.3})
fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="Black")
fig.update_traces(hoverinfo='skip')

# create theme buttons
button_style = {'color': 'white', 'font-size': 20, 'width': '100%', 'border-color': 'dark', 'textAlign': 'left'}
button_group = dcc.RadioItems(
            [ 
                {"label": html.Div(["Shoreline Development"], style= button_style), "value": 1},
                {"label": html.Div(["Coastal Topography"], style= button_style), "value": 2},
                {"label": html.Div(["Sea Conditions"], style= button_style),"value": 3},
                {'label': html.Div(["Socio-economic"], style= button_style),"value": 4}
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
                                                                                        dbc.Col(html.A([html.Img(src = 'data:image/png;base64,{}'.format(delt_base64), style= {'width': '60px'})], href = 'https://www.deltares.nl/en/'), width= 'auto'),
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
    container = data_container(value, filter_json, button_json)
    if container == None:
        raise PreventUpdate
    else:
        return container

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
    if beachcomp != None: df2_cop = df2_cop[df2_cop['prediction_label'].isin(beachcomp)]
    
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

    fig = geo_figure(value = value, json_dict = json_dict)

    if fig != None:
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
    
    # initialize the temporal resolution
    if tempres != []: tempres = tempres[0]
    else: tempres = 'annual'
    
    if themeval == 1 and clickdata != None:
        box = clickdata['points'][0]['hovertext']
        hotspot = df[df['transect_id'] == box]['hotspot'].values[0]
        disable, value = True, 'annual'
        # is the transect in hotspot? enable button and allow to automatic select monthly 
        # else tempres is annual
        if not pd.isna(hotspot): 
            disable = False
            value = tempres
        else:
            tempres = 'annual'
            
        fig = shoreline_scatter(df = df_cop, box = box, sdate= '1984-01-01', resolution = tempres)

        child = [ 
                    dbc.Row(dcc.Graph(figure = fig)),
                    dbc.Row([
                            dbc.Col(html.Div('Temporal resolution:', style = {'color': 'white', 'font-size': 20, "textAlign": "center"})),
                            dbc.Col(dcc.RadioItems(options = [
                                                            {'label': 'Annual', 'value': 'annual', 'disabled': False},
                                                            {'label': 'Monthly', 'value': 'monthly', 'disabled': disable}
                                                        ],
                                                value = value,
                                                id= {'type': 'radio_tempres', 'index': themeval},
                                                labelStyle= {'color': 'white', 'font-size': 20, 'textAlign': 'left'},
                                                inputStyle={"margin-right": "5px", 'margin-left': '10px'}
                                            )     
                                    ),
                            ], style = {'margin-top': '10px'}),
                    dbc.Row([
                            dbc.Col(html.Div('Future projection:', style = {'color': 'white', 'font-size': 20, "textAlign": "center"})),
                            dbc.Col(dcc.RadioItems(options = [
                                                            {'label': 'GHG emission', 'value': 'ghg'},
                                                            {'label': 'Machine learning', 'value': 'ml'}
                                                        ],
                                                value = value,
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
