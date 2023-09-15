import dash
import dash_bootstrap_components as dbc

#meta tags for making the apps mobile responsive
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX],
                suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

server = app.server
