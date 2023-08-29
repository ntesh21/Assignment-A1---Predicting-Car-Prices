# Import packages
from dash import Dash, html, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from predict import predict_price

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.SANDSTONE]
app = Dash(__name__, external_stylesheets=external_stylesheets, assets_folder='assets', assets_url_path='/assets/')

intro = '''
             This webapp helps in predicting the price of the car from the attributes of the car provides by the users.
        '''

how_it_works = '''
                    The prediction is done using the Machine Learning method. Linear Regression model which is trained using the 
                    car price dataset is used to predict the price of the car give the certain parameters from the users.

                    The model takes cars features such as engine, max power, year made and predicts the most likely price of the car with those very features.   
               '''

# App layout
app.layout = dbc.Container([
    html.A(html.H1(children='Car Price Prediction', style={'textAlign':'center', 'padding':'15px'}), href='#'),
    html.P(children=intro, style={'textAlign':'center', "font-weight": "bold", 'font_size': '26px',}),
    html.H3(children='How it works?', style={'margin-top':'20px', 'margin-left':'20px'}),
    html.P(children=how_it_works, style={'textAlign':'center', 'margin-left':'30px', 'margin-right':'30px'}),
    html.H5(children='Enter the car"s features to predict the price', style={'textAlign':'center'}),
    dbc.Row([
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
        ],
        className="mb-3")
    ], style={'margin': 'auto', 'width': '40%'}),
    html.H2(children="The predicted price of the car is:", style={'font_size': '30px', 'text-align': 'center'}),
    html.H2(id="y", children='', style={'font_size': '30px', 'text-align': 'center'})

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
    return predict_price(x_1, x_2, x_3, x_4)[0]


# Run the app
if __name__ == '__main__':
    app.run(debug=True)