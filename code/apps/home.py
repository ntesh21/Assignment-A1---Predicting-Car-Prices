import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output, State
import os,sys

# Add parent dir and sub dirs to the python path for importing the modules from different directories
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.extend([rootdir, parentdir])

import pandas as pd
from app import app



intro = '''
             This webapp helps in predicting the price of the car from the attributes of the car provides by the users.
        '''

how_it_works = '''
                    The prediction is done using the Machine Learning method. Linear Regression model which is trained using the 
                    car price dataset is used to predict the price of the car give the certain parameters from the users.

                    The model takes cars features such as engine, max power, year made and predicts the most likely price of the car with those very features.   
               '''

method = '''
                Please Choose the prediction model from the dropdown!!!
         '''

#Image Card for the home page
image_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4(intro, className="card-title", style={"padding":20, "text-align": "center"}),
                dbc.CardImg(src="../assets/cpp.png")
            ]
        ),
    ],
    className="h-100",
    color="light",
)

#Text body for the home page
graph_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Car Price Prediction", className="card-title", style={"text-align": "center",
                                                                                         "color": "black", "padding":20}),
                html.P(how_it_works),
                html.P(method)
                # html.P("Berlin's long history as a European capital has left it with a rich legacy of museums. Amongst them is the Berlin Police Museum, which records the constant struggle between the Berlin Police Force and it's substantial criminal underworld."),
                # html.P("The exhibits in the museum make it clear that the wealth of Berlin is what attracts the criminal underclass of Germany and beyond. Unfortunately, most of the museum's contents were destroyed during the Allied bombing raids of 1945. The museum reopened in 1973, with historical documents found in the intervening years, as well as materials on the Post War Federal Republic."),
                # html.P("Today it attracts over 10,000 visitors a year. Among the most interesting exhibits is a collection of flags and other material on the criminal societies of '20's. These may not be as well known as their Chicago counterparts based on which they were modelled, but nonetheless, they posed an equal threat to police authority.")
            ],
        ),
    ],
    className="h-100",
    color="light",
)

#Layout of home page
layout = html.Div([
                dbc.Row([dbc.Col(image_card, width=3, style={"height": "100%"}),
                         dbc.Col(graph_card, width=8, style={"height": "100%"})],
                        align="stretch", justify="around", style={"height": "100vh"})
                ])
