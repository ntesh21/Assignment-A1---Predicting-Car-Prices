from dash import Dash, html, callback, Output, Input, State, dcc
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import dash
import pathlib
from apps.lr import Lasso, LassoPenalty

from predict import predict_price_new


#Reading the data
PATH = pathlib.Path(__file__).parent


info_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Prediction using Regression with Regularization and Optimization", className="new-card-title", style={"text-align": "left",
                                                                                         "color": "black", "padding":20}),
                html.P("For the prediction of the prices, this model is trained in regression algorithm that is trainied with various other regularization and optimization parameters such as Lasso, Ridge, ElasticNet. The model is also tested with various learning rates, various optimizers such as stocashtic, mini batch, batch and also weight initialization method such as Xavier. The best model out of all the experiments is selected."),
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
            dbc.Col(dbc.Input(id="xn_1", type="number", placeholder="Which year the car was built?")),
            dbc.Label("Total Mileage(kmpl)"),
            dbc.Col(dbc.Input(id='xn_2', type="number", placeholder="What is the mileage of the car? (in kmpl)")),
            # dbc.Input(id="x_1", type="number", placeholder="Put a value for x_1"),
            dbc.Label("Engine (CC)"),
            dbc.Col(dbc.Input(id="xn_3", type="number", placeholder="What is the capacity of the engine? (in HP)")),

             dbc.Label("Max Power"),
            dbc.Col(dbc.Input(id="xn_4", type="number", placeholder="What is the max power of car?")),

            dbc.Col(dbc.Button(id="nsubmit", children="New Price Prediction", color="primary", className="me-1"), style={'padding':'10px', 'margin':'auto', 'width':'100px'}),
            # dbc.Label("The predicted price of the car is:", style={"font-weight": "bold", 'font_size': '30px', 'text-align': 'left'}),
            # html.Output(id="y", children='')
        html.H2(children="Price with new model is:", style={'font_size': '30px', 'text-align': 'center'}),
        html.H2(id="ny", children='', style={'font_size': '30px', 'text-align': 'center'}),
        ],
        className="mb-3"), width=8, style={"height":"100%"}),
    ]
    ,justify="around", style={"height": "100vh"}),
    # dash.page_container,

])

@callback(
    Output(component_id="ny", component_property="children"),
    State(component_id="xn_1", component_property="value"),
    State(component_id="xn_2", component_property="value"), 
    State(component_id="xn_3", component_property="value"),
    State(component_id="xn_4", component_property="value"),
    Input(component_id="nsubmit", component_property='n_clicks'),
    prevent_initial_call=True
)
def calculate_ny(xn_1, xn_2, xn_3, xn_4, submit):
    # return "Hello World"
    return predict_price_new(xn_1, xn_2, xn_3, xn_4)[0]
