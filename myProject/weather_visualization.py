#!/usr/bin/env python
# coding: utf-8


# In[1]:
import numpy as np
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly import express as px
import plotly.graph_objs as go
import datetime
from datetime import date, datetime, timedelta
import aiohttp
import asyncio
# Dictionary mapping weather attributes to their corresponding line colors for visualization
attr_colors = {"Temperature (F)": "red", 
         "Precipitation (in)":"blue", 
         "Wind Speed (mph)":"green", 
         "Relative Humidity (%)":"teal", 
         "Cloud Cover (%)":"grey"}


# Dictionary mapping weather attributes to their corresponding line colors for visualization
attr_colors = {"Temperature (F)": "red", 
         "Precipitation (in)":"blue", 
         "Wind Speed (mph)":"green", 
         "Relative Humidity (%)":"teal", 
         "Cloud Cover (%)":"grey"}

def plot_hourly(df, attr):
    """
    Generates a line plot of an hourly weather attribute over time.

    Args:
        df (pd.DataFrame): The DataFrame containing the weather data.
        attr (str): The weather attribute to plot 

    Returns:
        fig: A figure displaying the hourly trends for the selected attribute.
    """
    # create a line plot on plotly using weather dataframe
    # x is the time while y is the attribute the user chooses
    fig = px.line(df, 
                  x="datetime", y=attr, 
                  title=f"Hourly {attr} Trends",
                  labels={"datetime": "Datetime"},
                  line_shape='linear', 
                  # Set line color based on attribute type
                  color_discrete_sequence=[attr_colors[attr]])
    
    # Update the layout with x and y axis
    # x axis shows time while y axis shows weather parameter measures
    fig.update_layout(xaxis_title="Datetime", 
                      yaxis_title=attr, 
                      xaxis_tickangle=-45)
    
    # Return the figure
    return fig


# In[2]:


def plot_yearly(df, attr):
    """
    Generates a box plot to visualize yearly trends of a weather attribute.

    Args:
        df (pd.DataFrame): The DataFrame containing weather data.
        attr (str): The weather attribute to plot (must be a key in attr_colors).

    Returns:
        fig: A figure displaying the yearly trends of the selected attribute.
    """
    # Create a new 'year' column
    df["year"] = df["datetime"].dt.year
    
    # Create a box plot
    fig = px.box(df,
                 # x axis shows year while y is set using the attribute
                 x="year",
                 y=attr,
                 # Set color based on attribute type
                 color_discrete_sequence=[attr_colors[attr]],
                 # update labels
                 title=f"Yearly Average {attr} Trends",
                 labels={"year": "Year"})
    
    # return the figure
    return fig


# In[3]:


def plot_mock_heatmap(attr, lat, lon, date=None, num_locations=50, radius_miles=15):
    """
    Fetches weather data for locations in a specified radius and plots a scatter map 
    showing temperature, precipitation, or other weather attributes.

    Args:
        attr (str): The weather attribute to visualize.
        lat (float): Latitude of the central location.
        lon (float): Longitude of the central location.
        date (str, optional): Date in "YYYYMMDD" format. Defaults to today.
        num_locations (int, optional): Number of nearby locations to generate.
        radius_miles (float, optional): Radius in miles for generating nearby locations.

    Returns:
        fig: A scatter map visualization of the selected weather attribute.
    """
    from .API_calls import (fetch_nasa_data, get_nasa_power_hourly_data, 
                        get_nasa_power_hourly_data_years, 
                        generate_nearby_locations, 
                        get_nasa_power_data_nearby)
    
    # Fetch weather data for the generated locations using asyncio
    # use get_nasa_power_data_nearby() to get data for attribute for nearby locations
    weather_data = asyncio.run(get_nasa_power_data_nearby(lat, lon, date, num_locations, radius_miles))
    # print error if data is not found
    if weather_data is None or weather_data.empty:
        print("No data retrieved.")
        return
  
    # Create scatter map visualization using data taken from nearby locations
    fig = px.scatter_mapbox(
        weather_data, lat="Latitude", lon="Longitude", color=attr,
        title=f"{attr} in Nearby Locations"
    )
    
    # set style of map
    fig.update_layout(
        mapbox_style="carto-positron",
        autosize=True 
    )
    
    # return map visualization
    return fig


# In[ ]:





# In[ ]:




