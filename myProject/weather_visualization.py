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
from scipy.interpolate import griddata

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


# In[ ]:

from scipy.interpolate import griddata

def plot_heatmap(attr, lat, lon, date=None, num_locations=50, radius_miles=15):
    """
    Fetches weather data for nearby locations, applies interpolation, and overlays a 
    generated heatmap on a map.

    Args:
        attr (str): The weather attribute to visualize (e.g., temperature, precipitation).
        lat (float): Latitude of the central location.
        lon (float): Longitude of the central location.
        date (str, optional): Date in "YYYYMMDD" format. Defaults to today.
        num_locations (int, optional): Number of nearby locations to generate.
        radius_miles (float, optional): Radius in miles for generating nearby locations.

    Returns:
        fig: A Mapbox heatmap visualization.
    """
    from .API_calls import (get_nasa_power_data_nearby)

    # Fetch weather data asynchronously
    weather_data = asyncio.run(get_nasa_power_data_nearby(lat, lon, date, num_locations,
                                                          radius_miles))

    # Extract latitude, longitude, and weather attribute values
    lats = weather_data["Latitude"].values
    lons = weather_data["Longitude"].values
    values = weather_data[attr].values  

    # Define a grid for interpolation
    lat_grid = np.linspace(lats.min(), lats.max(), 150)
    lon_grid = np.linspace(lons.min(), lons.max(), 150)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Apply cubic interpolation
    interpolated_values = griddata((lats, lons), values, (lat_grid, lon_grid), method='cubic')

    # Replace NaNs in the interpolated grid with nearest neighbor values
    nearest_values = griddata((lats, lons), values, (lat_grid, lon_grid), method='nearest')
    interpolated_values = np.where(np.isnan(interpolated_values), nearest_values, interpolated_values)

    # Flatten the grids for plotting
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    values_flat = interpolated_values.flatten()

    # Define color range
    vmin, vmax = values.min(), values.max()

    # Create the heatmap layer
    fig = go.Figure(go.Densitymapbox(
        lat=lat_flat, 
        lon=lon_flat, 
        z=values_flat, 
        radius=15,  
        colorscale="viridis", 
        colorbar_title=f"{attr}",
        opacity=0.6,    
    ))

    # Overlay actual data points with color matching the heatmap
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=dict(
            size=10,
            # Use the attribute values for color
            color=values, 
            # Match heatmap colors
            colorscale="viridis", 
            # Normalize color scale
            cmin=vmin, 
            cmax=vmax,
            colorbar=dict(title=attr)
        ), 
        customdata=np.array(values),  
        hoverinfo="text",
        hovertemplate=(
            "<b>Latitude:</b> %{lat}<br>" +
            "<b>Longitude:</b> %{lon}<br>" +
            "<b>" + attr + ":</b> %{customdata:.3f} in"
        ),
        name="Data Points"
    ))

    # Configure the Mapbox layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": lat, "lon": lon},  
            zoom=10  
        ),
        title=f"{attr} Heatmap",
    )

    return fig






# In[ ]:




