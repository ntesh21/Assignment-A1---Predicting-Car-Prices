import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output
from predict import Lasso, LassoPenalty

#Connect to main app.py file
from app import app
from app import server

#Connect to your app pages
from apps import  new, prev, home

#Navbar
navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/apps/home")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Previous Method", href="/apps/prev"),
                    dbc.DropdownMenuItem("New Method", href="/apps/new"),
                ],
                nav=True,
                in_navbar=True,
                label="Predict Price",
            ),
            # dbc.NavItem(dbc.NavLink("About", href="#"))
        ],
    brand="Car Price Prediction",
    brand_href="/apps/home",
    style={"margin-bottom":5},
    color="primary",
    dark=True,
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content', children=[])
])

#Not making a 404 page, routing it directly to home
default_template = home.layout

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/prev':
        return prev.layout
    elif pathname == '/apps/new':
        return new.layout
    elif pathname == '/apps/home':
        return home.layout
    else:
        return default_template


if __name__ == '__main__':
    app.run_server(debug=False)
