from app import app

import warnings
warnings.filterwarnings('ignore')

import os,sys

# Add parent dir and sub dirs to the python path for importing the modules from different directories
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.extend([rootdir, parentdir])

from dash import Dash, html, callback, Output, Input, State, dcc
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import dash
import pathlib

from predict import predict_price_old


#Reading the data
PATH = pathlib.Path(__file__).parent


info_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Prediction using normal Regression", className="card-title", style={"text-align": "left",
                                                                                         "color": "black", "padding":20}),
                html.P("For the prediction of the prices, this model is trained in simple regression algorithm."),
                html.P("This model does not contain any other regularizations and optimization techniques")
                # html.P("Berlin's long history as a European capital has left it with a rich legacy of museums. Amongst them is the Berlin Police Museum, which records the constant struggle between the Berlin Police Force and it's substantial criminal underworld."),
                # html.P("The exhibits in the museum make it clear that the wealth of Berlin is what attracts the criminal underclass of Germany and beyond. Unfortunately, most of the museum's contents were destroyed during the Allied bombing raids of 1945. The museum reopened in 1973, with historical documents found in the intervening years, as well as materials on the Post War Federal Republic."),
                # html.P("Today it attracts over 10,000 visitors a year. Among the most interesting exhibits is a collection of flags and other material on the criminal societies of '20's. These may not be as well known as their Chicago counterparts based on which they were modelled, but nonetheless, they posed an equal threat to police authority.")
            ],
        ),
    ],    
    className="h-100",
    color="light",)

# App layout
layout = dbc.Container([
        dbc.Row([
        dbc.Col(info_card, width=3, style={"height": "30%"}),
        dbc.Col(
        html.Div([
            dbc.Label("Year Built"),
            dbc.Col(dbc.Input(id="x_1", type="number", placeholder="Which year the car was built?")),
            dbc.Label("Total Mileage(kmpl)"),
            dbc.Col(dbc.Input(id='x_2', type="number", placeholder="What is the mileage of the car? (in kmpl)")),
            # dbc.Input(id="x_1", type="number", placeholder="Put a value for x_1"),
            dbc.Label("Engine (CC)"),
            dbc.Col(dbc.Input(id="x_3", type="number", placeholder="What is the capacity of the engine? (in HP)")),

             dbc.Label("Max Power"),
            dbc.Col(dbc.Input(id="x_4", type="number", placeholder="What is the max power of car?")),

            dbc.Col(dbc.Button(id="submit", children="Get Price Prediction", color="primary", className="me-1"), style={'padding':'10px', 'margin':'auto', 'width':'100px'}),
            # dbc.Label("The predicted price of the car is:", style={"font-weight": "bold", 'font_size': '30px', 'text-align': 'left'}),
            # html.Output(id="y", children='')
        html.H2(children="The predicted price of the car is according to old model is:", style={'font_size': '30px', 'text-align': 'center'}),
        html.H2(id="y", children='', style={'font_size': '30px', 'text-align': 'center'}),
        ],
        className="mb-3"), width=8, style={"height":"100%"}),
    ]
    ,justify="around", style={"height": "100vh"}),
    # dash.page_container,

] , fluid=True, style={'background-image': 'url("/assets/blurry-gray-background.avif")', 
                             'background-size': 'cover', 'background-repeat': 'no-repeat',
                             'background-position': 'center', 'height': '100vh'})

@callback(
    Output(component_id="y", component_property="children"),
    State(component_id="x_1", component_property="value"),
    State(component_id="x_2", component_property="value"), 
    State(component_id="x_3", component_property="value"),
    State(component_id="x_4", component_property="value"),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)
def calculate_y(x_1, x_2, x_3, x_4, submit):
    # return "Hello World"
    return predict_price_old(x_1, x_2, x_3, x_4)[0]
