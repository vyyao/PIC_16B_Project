#!/usr/bin/env python
# coding: utf-8

# In[2]:

import aiohttp
import asyncio

async def fetch_nasa_data(session, lat, lon, start_date, end_date, sem):
    """
    Fetches hourly weather data from the NASA POWER API asynchronously for a given location and time.

    Args:
        session (aiohttp.ClientSession): The active aiohttp session for making HTTP requests.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        start_date (str): The start date in YYYYMMDD format.
        end_date (str): The end date in YYYYMMDD format.
        sem (asyncio.Semaphore): A semaphore to limit the number of concurrent API requests.

    Returns:
        dict: A dictionary containing the API response data, or None if an error occurs.
    """
    # request information from nasa power
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    
    # define the parameters that we want to call
    parameters = "PRECSNO,T2MDEW,PRECTOTCORR,T2M,WS2M,RH2M,CLOUD_AMT"
    
    # Construct the API request parameters
    params = {
        "parameters": parameters,
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    # Use a semaphore to control the number of simultaneous API requests
    async with sem:  
        # Send an asynchronous request to the NASA POWER API
        async with session.get(url, params=params) as response:
            # print error if no data is found for location
            if response.status != 200:
                print(f"Error {response.status} for {lat}, {lon}, {start_date}-{end_date}")
                return None
            # Convert the API response to a dictionary and return it
            return await response.json()  


# In[3]:

import numpy as np
import requests
import pandas as pd

async def get_nasa_power_hourly_data(lat, lon, date=None):
    """
    Fetches NASA POWER hourly weather data for a single day (default behavior).

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        date (str, optional): The date in YYYYMMDD format. Defaults to the current date.

    Returns:
        pd.DataFrame: A DataFrame containing weather data, with datetime and relevant weather metrics.
    """

    # If no date is provided, use the current date in YYYYMMDD format
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    # Create a semaphore to limit the number of concurrent API requests (max 5 at a time)
    sem = asyncio.Semaphore(5)

    # Create an asynchronous HTTP session
    async with aiohttp.ClientSession() as session:
        # Fetch the weather data from NASA POWER API
        data = await fetch_nasa_data(session, lat, lon, date, date, sem)

    # If no data is returned or the expected data structure is missing, print an error and return None
    if data is None or "properties" not in data:
        print(f"No data found for {lat}, {lon} on {date}")
        return None

    # Convert the nested JSON response into a DataFrame
    df = pd.DataFrame.from_dict(data["properties"]["parameter"], orient="index").T

    # Reset the index to move datetime from the index to a column
    df.reset_index(inplace=True)

    # Rename the datetime column
    df.rename(columns={"index": "datetime"}, inplace=True)

    # Convert the datetime column to a proper timestamp format
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H", errors="coerce")

    # Drop rows where datetime conversion failed (if any)
    df.dropna(subset=["datetime"], inplace=True)

    # Rename API parameter column names to more user-friendly names
    df.rename(columns={
        "PRECSNO": "Snow_Precipitation",
        "T2MDEW": "Dew_Point_2m",
        "PRECTOTCORR": "Total_Precipitation_mm",
        "T2M": "Temperature_2m_C",
        "WS2M": "Wind_Speed_2m",
        "RH2M": "Relative Humidity (%)",
        "CLOUD_AMT" : "Cloud Cover (%)",
    }, inplace=True)

    # Add latitude and longitude columns for reference
    df["Latitude"] = lat
    df["Longitude"] = lon

    # Convert metric units to more common US units
    df["Precipitation (in)"] = df["Total_Precipitation_mm"] / 25.4  # Convert mm to inches
    df["Temperature (F)"] = (df["Temperature_2m_C"] * 9/5) + 32  # Convert Celsius to Fahrenheit
    df["Wind Speed (mph)"] = df["Wind_Speed_2m"] * 2.237  # Convert m/s to mph

    # Return the cleaned and structured DataFrame
    return df


# In[4]:


async def get_nasa_power_hourly_data_years(lat, lon, start_year, end_year):
    """
    Fetches NASA POWER API hourly data for multiple years.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        start_year (int): The starting year of data collection.
        end_year (int): The ending year of data collection.

    Returns:
        pd.DataFrame: A DataFrame containing the weather data for multiple years
    """
    # Create a list to store asynchronous API requests
    tasks = []
    
    # loop through specified range of years
    for year in range(start_year, end_year): 
        # start January 1st of the first year
        start_date = f"{year}0101"
        # end at December 31st of the last year
        end_date = f"{min(year, end_year)}1231"
        # Add an asynchronous task to fetch weather data for this year
        tasks.append(get_nasa_power_hourly_data(lat, lon, start_date))
    
    # Execute all API requests concurrently
    results = await asyncio.gather(*tasks)
    
    # Filter out None values and combine results
    all_data = [df for df in results if df is not None]

    # Return a combined DataFrame with weather data for multiple years
    # If no valid data is found, return None
    return pd.concat(all_data, ignore_index=True) if all_data else None


# In[ ]:


def generate_nearby_locations(lat, lon, radius_miles=15, num_points=10):
    """
    Generates random nearby locations within a radius using NumPy.

    Args:
    - lat (float): Latitude of the center point.
    - lon (float): Longitude of the center point.
    - radius_miles (float): Search radius in miles.
    - num_points (int): Number of locations to generate.

    Returns:
    - NumPy array of shape (num_points, 2) with latitude and longitude values.
    """
    # Generate random angles uniformly between 0 and 2π
    angles = np.random.uniform(0, 2 * np.pi, num_points)

    # Use square root scaling to distribute points uniformly across the circle
    distances = radius_miles * np.sqrt(np.random.uniform(0, 1, num_points))

    # Convert polar coordinates (distance, angle) to latitude & longitude offsets
    delta_lat = (distances / 69) * np.cos(angles)
    delta_lon = (distances / (69 * np.cos(np.radians(lat)))) * np.sin(angles)

    # Compute new latitudes and longitudes
    new_lat = lat + delta_lat
    new_lon = lon + delta_lon

    return np.column_stack((new_lat, new_lon))


# In[5]:


async def get_nasa_power_data_nearby(lat, lon, date=None, num_locations=10, radius_miles=15):
    """
    Fetches NASA POWER API hourly data for multiple nearby locations for a single day.

    Args:
        lat (float): Latitude of the central location.
        lon (float): Longitude of the central location.
        date (str, optional): Date in "YYYYMMDD" format. Defaults to today.
        num_locations (int, optional): Number of nearby locations to generate. Default is 10.
        radius_miles (float, optional): Radius in miles to generate nearby locations. Default is 15.

    Returns:
        pd.DataFrame or None: A Pandas DataFrame containing weather data for all locations
    """
    # If no date is provided, use today's date in YYYYMMDD format
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
        
    # Generate a list of random nearby locations within the specified radius
    locations = generate_nearby_locations(lat, lon, radius_miles, num_locations)
    
    # Create a list of asynchronous tasks to fetch weather data for each location using api call
    tasks = [get_nasa_power_hourly_data(location[0], location[1], date) for location in locations]
    
    # Fetch data for all locations concurrently
    results = await asyncio.gather(*tasks)
    
    # Remove any None values
    all_data = [df for df in results if df is not None]
    
    # Combine all the retrieved data into a single Pandas DataFrame
    # If no valid data was found, return None
    return pd.concat(all_data, ignore_index=True) if all_data else None











# In[ ]:




