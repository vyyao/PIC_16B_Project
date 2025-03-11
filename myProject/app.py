# import necessary libraries first
import numpy as np
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly import express as px
import plotly.graph_objs as go
from dash import Dash, html, Input, Output, Patch, dash_table
import dash_leaflet as dl
import json
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, no_update
import base64
import datetime
from datetime import date, datetime, timedelta
import io
from flask import Flask, jsonify
import asyncio
from threading import Thread
import aiohttp
import nest_asyncio
from dash.dash_table import DataTable, FormatTemplate
from dash.dash_table.Format import Format, Scheme, Trim
import dash_bootstrap_components as dbc
from myProject import *

# from API_calls import * 
# from weather_visualization import *
# from keras_weather_model import *

def load_app():
    app = Dash()
    app.title="California Weather"
    app.layout = html.Div([
        dcc.Tabs([
            # TAB 1 - DAILY WEATHER ---------------------------------------------------------------------------------
            dcc.Tab(label='Daily Weather', children=[ 
                dcc.Markdown('''
                    ## Directions: 
                    1. Pan and zoom in/out on the map to find your location.
                    2. Click your location on the map for current weather.
                        Press the "Submit" button to confirm your selection and your weather reuslts will load promptly.
                    3. Check out the "Weather Graphs" and "Weather Predictions" tabs for more weather information.'''),
                dcc.Markdown("### Zoom to City"),
                dcc.Dropdown(list(ca_city_dict.keys()), id="city"),
                html.Br(),
                dcc.Markdown("### Select Location:"),
                dl.Map( 
                    id='map',
                    n_clicks=0,
                    children=[
                        dl.TileLayer()
                    ],
                    center=[34, -118],
                    zoom=9,
                    style={'height': '50vh'}
                ),
                html.Div(id='coords'),
                dcc.Markdown("### Select Date:"),
                dcc.DatePickerSingle(
                    id='date-picker',
                    min_date_allowed=date(2000, 1, 1),
                    max_date_allowed=date(2024, 12, 31),
                    initial_visible_month=date(2024, 1, 1),
                ),
                html.Button('Submit',id='submit',n_clicks=0,className="button"), 
                dcc.Markdown('## Current Weather:'),
                dcc.Markdown(id='table-caption'),
                dash_table.DataTable(id='weather',
                    style_as_list_view=True,
                    style_cell={'padding': '2px'},
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    },
                    style_cell_conditional=[
                        {'textAlign': 'center'}
                    ],
                ),
                dcc.Store(id='latitude'), # stored latitude for other tabs/functions
                dcc.Store(id='longitude'), # stored longitude for other tabs/functions
                dcc.Store(id='date'), # stored date for other tabs/function
            ], className="tab",),

            # TAB 2 - WEATHER GRAPHS ---------------------------------------------------------------------------------
            dcc.Tab(label='Weather Graphs', children=[
                dcc.Markdown('''
                    ## Directions: 
                    1. Select a weather attribute you are interested in.
                    2. Select the type of graph you wish to see.
                    3. Click the "Create Graph" button to confirm your choice. 
                        Your graph will load promptly with weather information from your previously selected location.'''),
                dcc.Markdown("### Select an Attribute:"),
                dcc.Dropdown(visual_options,id='options'),
                dcc.Markdown("### Select a Graph:"),
                dcc.Dropdown(graph_options,id='graph_type'),
                html.Br(),
                html.Br(),
                html.Button('Create Graph', id='submit-graph', n_clicks=0, className="button"),
                html.Br(),
                dcc.Markdown(id='caption'),
                dcc.Graph(id='visual'),
            ], className="tab",),

            # TAB 3 - WEATHER PREDICTIONS ---------------------------------------------------------------------------------
            dcc.Tab(label='Weather Predictions', children=[
                dcc.Markdown('''
                    ## Directions: 
                    1. Select a weather attribute you are interested in.
                    2. Select the number of days into the future you want to be predicted.
                    3. Click the "Predict Weather" button to confirm your choice. 
                        Your graph will load promptly with weather information from your previously selected location.'''),
                dcc.Markdown("### Select an Attribute:"),
                dcc.Dropdown(visual_options,id='target'),
                dcc.Markdown("### Select the Number of Hours to Predict:"),
                dcc.Input(id='n_future', type="number", className="input"),
                html.Br(),
                html.Br(),
                html.Button('Predict Weather', id='predict-weather', n_clicks=0,className="button"),
                dcc.Markdown(id='pred-caption'),
                dash_table.DataTable(id='pred-table',
                    style_as_list_view=True,
                    style_cell={'padding': '2px'},
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    },
                    style_cell_conditional=[
                        {'textAlign': 'left'}
                    ],
                ),
                dcc.Graph(id='pred-graph'),
            ], className="tab",)
        ])
    ])

    # TAB 1 - DAILY WEATHER ---------------------------------------------------------------------------------
    @app.callback(
        Output('coords', 'children'), # output for coordinate str
        Output('map', 'children'), # output for markers
        Output('latitude','data'), # store latitude for other tabs/function
        Output('longitude','data'), # store longitude for other tabs/functions
        Input('map', 'clickData'), # update with new coords
        Input('map', 'n_clicks'), # update with new clicks
        prevent_initial_call=True
    )
    def map(click_data, n_clicks):
        """
        Adds markers to user-selected locations and saves location

        Args:
            click_data ():
            n_clicks (int): number of times map is clicked

        Returns:
            json.dumps(coordinates) (str): latitude, longitude displayed below map
            patched (Patch): map marker at location
            latitude (double): latitude coordinate saved for other functions/tabs
            longitude (double): longitude coordinate saved for other functions/tabs

        """
        if n_clicks > 0: # runs only if 'submit' button is pressed

            # save coordinates from map click data
            coordinates = click_data['latlng']
            latitude, longitude = coordinates.values()

            # create Patch() instance, add Marker layer at click coordinates
            patched = Patch()
            patched.append(dl.Marker(position=[latitude, longitude]))

            return json.dumps(coordinates), patched, latitude, longitude

    @app.callback(
        Output('weather','data'),             # records to display in table
        Output('weather','columns'),          # columns to display in table
        Output('date','data'),                # store date str for other tabs
        Output('table-caption','children'),   # caption with parameters
        Input('submit','n_clicks'),           # run when button is pressed
        State('latitude','data'),             # latitude from 'current-weather' tab
        State('longitude','data'),            # longitude from 'current-weather' tab
        State('date-picker','date'),          # user-selected date
        prevent_initial_call=True,
    )
    def submit(n_clicks, lat, lng, day):
        """
        Displays a table with weather information for user-selected location and date

        Args:
            n_clicks (int): number of user clicks on button 'submit'
            lat (double): latitude coordinate
            lng (double): longitude coordinate
            day (str): user-selected date

        Returns:
            weather_df (dict): records from weather dataframe to display in table
            cols (list): list of column names from weather dataframe to display in table
            chosen_date (str): date to store for use in other functions/tabs
            caption (str): caption with parameters

        """
        if n_clicks > 0: # runs only if 'submit' button is pressed

            # displays warning if location is not chosen before
            if(lat == None or lng == None or day == None):
                return None, None, None, '##### Please select a location.' # could change to have default location?

            # date to formated date for api call
            chosen_date = date.fromisoformat(day).strftime('%Y%m%d')

            # return df with current hourly weather
            loop = asyncio.new_event_loop()
            df = loop.run_until_complete(get_nasa_power_hourly_data(lat, lng, chosen_date))
            df["Time"] = pd.to_datetime(df["datetime"], format="%H:%M:%S", errors="coerce")
            df = df[["Time","Temperature (F)", "Wind Speed (mph)", "Precipitation (in)", "Cloud Cover (%)", "Relative Humidity (%)"]]

            # extract column names and records for table, round numbers to 4 significant figures
            cols = [{'name': col, 'id': col, 'type':'numeric', 'format': Format(precision=4)} for col in df.columns]
            weather_df = df.to_dict('records')

            # caption with parameters
            caption = f"for **({round(lat,2)}, {round(lng,2)})** on **{date.fromisoformat(day).strftime('%B %d, %Y')}**"

            return weather_df, cols, chosen_date, caption

    @app.callback(
        Output('map','center'), # adjust center coords to city of interest
        Output('map','zoom'), # zoom into city of interest
        Input('city','value'), # city of interest chosen by user
        prevent_initial_call=True,
    )
    def zoom_to_city(value):
        """ Updates map center and zoom to user-selected city

        Args: 
            value (str): user-selected city

        Retunrs:
            coords (list): list of latitude, longitude coordinates to update map ceneter
            zoom (int): int to update map zoom level

        """
        coords = ca_city_dict[value] # get list of coordinates from dict
        zoom = 13 
        return coords, zoom

    # TAB 2 - WEATHER GRAPHS ---------------------------------------------------------------------------------
    @app.callback(
        Output('visual','figure'),          # graph of weather attribute
        Output('caption','children'),       # caption with parameters
        Input('submit-graph','n_clicks'),   # run when button is pressed
        State('options','value'),           # weather attribute to plot
        State('graph_type','value'),        # graph type to plot
        State('latitude','data'),           # latitude from 'daily-weather' tab
        State('longitude','data'),          # longitude from 'daily-weather' tab
        State('date','data'),               # date from 'daily-weather' tab
        prevent_initial_call=True,
    )
    def submit_graph(n_clicks, attr, graph_type, lat, lng, day):
        """
        Displays a graph of the selected weather attributed for a given location and day

        Args:
            n_clicks (int): number of user clicks on button 'submit-graph'
            attr (str): weather attribute to plot
            graph_type (str): user-selected graph type (3 options)
            lat (double): latitude coordinate
            lng (double): latitude coordinate
            day (str): user-selected date

        Returns:
            graph (plotly graph): plotted weather attribute on selected graph
            caption (str): caption with parameters

        """
        if n_clicks > 0: # runs only if 'submit-graph' button is pressed

            # convert date from num string to readable date
            chosen_date = date.fromisoformat(day).strftime('%B %d, %Y')

            # caption to inform user of what parameters they selected
            caption = f" **{attr}** graph for **{chosen_date}** at **({round(lat,2)}, {round(lng,2)})**."

            # for async call of fetch_nasa_data
            loop = asyncio.new_event_loop()
            graph = loop.run_until_complete(choose_graph(attr, graph_type, lat, lng, day))

            return graph, caption


    # TAB 3 - WEATHER PREDICTIONS ---------------------------------------------------------------------------------
    @app.callback(
        Output('pred-caption','children'),      # caption with parameters
        Output('pred-graph','figure'),          # graph of real and predicted weather
        Output('pred-table','data'),            # weather_df with real and predicted values
        Output('pred-table','columns'),         # cols of weather_df for table
        Input('predict-weather','n_clicks'),    # run when button is pressed
        State('target','value'),                # weather attribute to predict and plot
        State('n_future','value'),              # number of hours into the future to predict
        State('latitude','data'),               # latitude from 'daily-weather' tab
        State('longitude','data'),              # longitude from 'daily-weather' tab
        State('date-picker','date'),            # date from 'daily-weather' tab
        prevent_initial_call=True,
    )
    def predict_weather(n_clicks, attr, n_future, lat, lng, day):
        """ 
        Displays a graph with predictions for a weather attribute for a number of hours
        using the 3 immediate days before the selected date.

        Args:
            n_clicks (int): number of user clicks on button 'predict-weather'
            attr (str): weather attribute to predict
            n_future (int): number of hours to predict
            lat (double): latitude coordinate
            lng (double): longitude coordinate
            day (str): user-selected date

        Returns:
            caption (str): caption with parameters
            graph (plotly line graph): real and predicted weather attribute values
                on a line plot over time
            weather_df (dict): records from dataframe with real and predicted data
            cols (lsit): column names from dataframe with real and predicted data
        """
        if n_clicks > 0: # runs only if 'predict-weather' button is pressed

            # calculate dates 1-2 days apart for get_nasa_power_data
            chosen_date = date.fromisoformat(day).strftime('%Y%m%d')
            date_1 = (date.fromisoformat(day) - timedelta(days=1)).strftime('%Y%m%d')
            date_2 = (date.fromisoformat(day) - timedelta(days=2)).strftime('%Y%m%d')

            # caption to inform user of what parameters they selected
            caption = f" **{attr}** predictions at **({round(lat,2)}, {round(lng,2)})**."

            # create dataframe with 3-days of weather date to fit model to
            loop = asyncio.new_event_loop()
            day1 = loop.run_until_complete(get_nasa_power_hourly_data(lat, lng, chosen_date))
            day2 = loop.run_until_complete(get_nasa_power_hourly_data(lat, lng, date_1))
            day3 = loop.run_until_complete(get_nasa_power_hourly_data(lat, lng, date_2))
            df = pd.concat([day1,day2,day3], ignore_index=True)

            # get real dataframe with known weather, get predicted data array from model
            real, predicted, = run_model(df, attr, 50, int(n_future))

            # return graph and dataframe with predicted and real data
            graph, pred_df = plot_predictions(attr, real, predicted, n_future)

            # extract column names and records to display as table
            cols = [{'name': col, 'id': col} for col in pred_df.columns]
            weather_df = pred_df.to_dict('records')

            return caption, graph, weather_df, cols

    if __name__ == '__main__':
        app.run(port=1332,debug=True)
    
    return app
    
# END OF AP --------------------------------------------------------------------------------------------------------------------------------------------------
visual_options = ["Temperature (F)", "Precipitation (in)", "Wind Speed (mph)", "Relative Humidity (%)", "Cloud Cover (%)"]
graph_options = ["Hourly Change", "Yearly Change", "Local Trends"]

# dictionary with cities and their associated (latitude,longitude) coordinates
df = pd.read_csv('california_cities_5.csv')
df = df.dropna()
ca_city_dict = dict([(city,[lat,lng]) for city,lat,lng in zip(df['City'], df['Latitude'],df['Longitude'])])

async def choose_graph(attr, graph_type, lat, lng, day):
    """
    Choose which graph to plot and send as output to dcc.Graph in dash app
    based off of the weather attribute, graph type, location, and day chosen
    by the user. Fetches weather dataframes using async functions
    
    Args:
        attr (string): the weather attribute of interest to plot and predict
        graph_type (string): the type of graph (hourly, yearly, or heatmap) to plot
        lat (double): latitude coordinate representing location
        lng (double): longitude coordinate representing location
        day (str): string date in %Y%m%d format to fetch nasa api data
        
    Returns:
        A specified graph plotted using user-defined parameters
    
    """    
    # to plot hourly change line graphs in weather attr
    if graph_type == "Hourly Change":
        hour_df = await get_nasa_power_hourly_data(lat, lng, day)
        return plot_hourly(hour_df,attr)
    
    # to plot yearly average change boxplots in weather attr
    elif graph_type == "Yearly Change":
        year_df = await get_nasa_power_hourly_data_years(lat, lng, 2014, 2024)
        return plot_yearly(year_df,attr)

    # to plot heatmaps of weather attr in nearby locations 
    else: 
        return plot_mock_heatmap(attr, lat, lng, day, num_locations=50)