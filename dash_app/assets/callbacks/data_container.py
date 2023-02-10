from dash import html, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px


def data_container(value, filter_json, button_json):
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
                                                                        {'label': 'Mud', 'value': 'mud', 'disabled': True},
                                                                        {'label': 'Cliff', 'value': 'rock', 'disabled': True},
                                                                        {'label': 'Vegetation', 'value': 'vegetation', 'disabled': True},
                                                                        {'label': 'Other', 'value': 'other', 'disabled': True},
                                                                    ],
                                                            value = ['sand', 'mud', 'rock', 'vegetation', 'other'] if beachcomp == None else beachcomp,
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
        return None