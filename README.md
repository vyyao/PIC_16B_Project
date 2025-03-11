# About Our Project

Our project was to develop a web app that provides real-time, hyperlocal weather visualizations using NASA POWER API data for Los Angeles County. The app will allow users to explore weather conditions such as temperature, precipitation, wind speed, and humidity for specific locations. Users will be able to select a location and date to retrieve weather trends, compare past weather patterns, and analyze localized climate variations through interactive visualizations.

In addition to real-time and historical weather visualizations, we implemented a Keras-based neural network model to predict future weather conditions. The model is trained using historical weather data obtained from the NASA POWER API. It learns patterns in temperature, humidity, wind speed, and precipitation over time, allowing it to make short-term forecasts for a given location.

In order to implement this app we utilized a combination of Python tools and data visualization techniques. API requests (aiohttp, asyncio), plotly, machine learning (keras/tensorflow), and Dash. 

<img width="425" alt="Screenshot 2025-03-10 at 5 52 22 PM" src="https://github.com/user-attachments/assets/c3d72650-d0c4-42f8-b4ba-3149197d40c5" />

# Technical Components

## Using data from API calls and using aiohttp and asyncio

To get the weather data needed, we use the temporal API from NASA Power (Prediction of Worldwide Energy Sources) and implemented asynchronous programming using aiohttp and asyncio. The API provides climate and weather-related parameters such as temperature, precipitation, and wind speed, which I needed to fetch for different locations and time periods. 

How this API works is that you send a request to NASA's servers and they return the parameters you ask for in a JSON format that we can use for our data visualizations and model later on. In the request, you need to specify your location (latitude and longitude), the time period (choosing between daily, hourly, or long-term averages), the weather parameters you want, and the format (here we chose JSON). Because, fetching data one request at a time would take too long, especially when querying multiple locations or retrieving historical data spanning years, I used aiohittp and asynchio to submit asynchronous requests. By using asynchronous requests, I was able to send multiple API calls simultaneously, reducing overall wait time.

To manage this process, we implemented the get_nasa_power_hourly_data function, which handles the asynchronous fetching, processing, and structuring of weather data. It calls fetch_nasa_data, which constructs and sends the API request asynchronously. Once the API response is received, get_nasa_power_hourly_data converts the JSON data into a Pandas DataFrame, renaming columns and converting units for better usability. This structured dataset is then used for data visualizations and predictive modeling.

<img width="536" alt="Screenshot 2025-03-10 at 5 53 12 PM" src="https://github.com/user-attachments/assets/fadb9a25-28e4-47c4-88b9-ddc31f2e9063" />

```
# Apply nest_asyncio for async execution 
nest_asyncio.apply()

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

async def get_nasa_power_hourly_data(lat, lon, date=None):
    """
    Fetches NASA POWER hourly weather data for a single day (default behavior).
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    sem = asyncio.Semaphore(5)
    async with aiohttp.ClientSession() as session:
        data = await fetch_nasa_data(session, lat, lon, date, date, sem)

    if data is None or "properties" not in data:
        print(f"No data found for {lat}, {lon} on {date}")
        return None

    df = pd.DataFrame.from_dict(data["properties"]["parameter"], orient="index").T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "datetime"}, inplace=True)
    
    # Convert datetime and clean up
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H", errors="coerce")
    df.dropna(subset=["datetime"], inplace=True)
    
    # Rename columns
    df.rename(columns={
        "PRECSNO": "Snow_Precipitation",
        "T2MDEW": "Dew_Point_2m",
        "PRECTOTCORR": "Total_Precipitation_mm",
        "T2M": "Temperature_2m_C",
        "WS2M": "Wind_Speed_2m",
        "RH2M": "Relative Humidity (%)",
        "CLOUD_AMT" : "Cloud Cover (%)",
    }, inplace=True)

    # Add Lat/Lon
    df["Latitude"] = lat
    df["Longitude"] = lon

    # Convert units
    df["Precipitation (in)"] = df["Total_Precipitation_mm"] / 25.4  
    df["Temperature (F)"] = (df["Temperature_2m_C"] * 9/5) + 32  
    df["Wind Speed (mph)"] = df["Wind_Speed_2m"] * 2.237  

    return df

df = await get_nasa_power_hourly_data(34.0722, -118.4427, "20240301")
df.head()
```

<img width="1271" alt="Screenshot 2025-03-10 at 6 03 56 PM" src="https://github.com/user-attachments/assets/8519be6b-daa2-41d1-a0b9-effb455c7c91" />



# Complex data visualization using packages beyond matplotlib (plotly)

I used the Plotly library to visualize weather data retrieved from the NASA POWER API. The goal was to create interactive plots that display key weather parameters, including temperature, precipitation, and wind speed, for user-selected locations. These visualizations help users analyze both real-time hourly data and historical trends over multiple years.

To achieve this, I implemented the `get_nasa_power_hourly_data(lat, lon, specific_date)` function, which retrieves hourly weather data for a given date and location. The data is processed into a Pandas DataFrame and visualized using Plotly’s line graphs, bar charts, and choropleth maps. The line graph allows users to track temperature changes throughout the day, while the bar chart highlights yearly trends. Additionally, I developed a choropleth map to display precipitation and temperature at randomly generated locations within a 15-mile radius of the user's selection. Since generating a heatmap required excessive API calls, I optimized the process by introducing get_`nasa_power_data_for_one_day_nearby()`, which fetches weather parameters for nearby locations. This function works in conjunction with `generate_new_locations()`, which generates random coordinate sets within the set mile radius.

```
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
    # Fetch weather data for the generated locations using asyncio
    # use get_nasa_power_data_nearby() to get data for attribute for nearby locations
    weather_data = asyncio.run(get_nasa_power_data_nearby(lat, lon, date, num_locations, radius_miles))
    # print error if data is not found
    if weather_data is None or weather_data.empty:
        print("No data retrieved.")
        return
  
    # Create scatter map visualization
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

# plot precipitation for nearby locations
plot_mock_heatmap('Precipitation (in)', 34.0722, -118.4427, "20240301")
```
<img width="1246" alt="Screenshot 2025-03-10 at 6 03 28 PM" src="https://github.com/user-attachments/assets/bd8267b6-ceb2-4c28-9a76-a44f47aaa32b" />



# Machine Learning (Keras)
So how are we actually going to get the weather predictions that we will be showing in the app? Well, the next step is building the predictive model. This model is a key part of our web app, enabling users to anticipate future weather patterns based on historical data. To achieve this, we implemented a Bidirectional LSTM model on the Keras/Tensorflow platform, which learns from past weather data to predict upcoming values for a selected feature, such as temperature, wind speed, or humidity.

How the Model Works:

The run_model() function builds and trains a deep learning model using Long Short-Term Memory (LSTM) networks to make weather predictions. Below is a breakdown of its key steps:

Data Preprocessing

Before training the model, we need to prepare the input data.

- Handling Missing Data:
    - The function first removes any missing values in the selected weather feature using dropna(), ensuring that only valid data points are included in training.
- Extracting Feature Values:
    - The selected feature (e.g., temperature) is extracted from the dataset as a NumPy array. This isolates the data we want the model to learn from.
- Normalization with MinMaxScaler:
    - Since neural networks perform best when inputs are scaled, the values are normalized to the range (0,1) using MinMaxScaler. This helps stabilize training and prevents certain values from dominating the learning process.



Creating Training Sequences

To make predictions, the model needs to learn from sequences of past weather data.

- Defining input (x_train) and output (y_train) sequences:
    - The function slides over the dataset, creating input-output pairs.
    - Each input sequence consists of n_past days of data.
    - The corresponding output sequence consists of n_future days of values to predict.
- Converting to NumPy Arrays:
    - The input (x_train) and output (y_train) data are converted into NumPy arrays and reshaped to match the LSTM input format, which requires a 3D shape (samples, time steps, features).


Building the Neural Network

The model is structured using multiple LSTM layers, designed to capture long-term dependencies in time series data.

- Bidirectional LSTM Layer:
    - The first layer is a Bidirectional LSTM, meaning it processes data both forward and backward, capturing more patterns in the time series.
- Stacked LSTM Layers with Dropout:
    - Additional LSTM layers follow, each with dropout layers to reduce overfitting.
    - These layers help the model generalize better to unseen weather patterns.
- Dense Output Layer:
    - The final layer is a fully connected Dense layer with a linear activation function. This ensures the model outputs continuous numerical predictions.
    
    
Training the Model

Once the architecture is built:

- Compilation:
    - The model is compiled with:
        - adam optimizer for efficient gradient updates.
        - mean_squared_error (MSE) as the loss function, ideal for regression tasks.
        - accuracy as a metric. But, it is likely not the best metric for this task, as it focuses on a perfect match between the real and predicted features, which is not feasible and not necessary for our level of forecast. This can be seen in low accuracy levels when training the model, yet the graphs show the predicted values follow the real values well.
- Training:
    - The model is trained using 20 epochs and a batch size of 32. This determines how many times the model sees the data and how many samples it processes at once. These values were decided after testing different combinations and selecting the one that had good efficiency and performance.
    
    
Making Predictions

Once trained, the model is used to predict future weather values.

- Preparing Test Data:
    - The function extracts the most recent n_past values and scales them using the same MinMaxScaler.
- Model Prediction:
    - The trained model predicts n_future worth of values, which are then inverse-transformed to restore the original scale for easier visualizations and more interpretable results/

Visualizing Predicted Weather Trends

Once future weather attributes have been predicted by the model, it is important to present them in a clear and interpretable way. The function responsible for this visualization creates a line graph comparing real past weather data with model-generated forecasts. This allows users to observe how the predicted values align with historical trends and anticipate upcoming weather patterns.

We begin by labeling the real weather data, which comes directly from the NASA API. This distinction ensures that past observations and future predictions remain clearly identifiable in the final visualization. The next step is to generate a sequence of future timestamps corresponding to the model's predicted values. These timestamps extend just beyond the available real data, ensuring that predictions are plotted accurately in a continuous timeline.

Once the future dates have been established, they are combined with the predicted weather values in a structured dataset. This dataset consists of both real and predicted weather data, each labeled appropriately to help with differentiation in the visualization. An interactive line graph is then generated, allowing users to hover over individual data points to see precise weather values. The real and predicted values are distinguished through color coding and marker symbols, enhancing clarity.

In addition to the graphical representation, we also prepare a tabular summary of the predictions. This table presents the forecasted values alongside their respective dates in a familiar Month-Day-Year format, making it easier for users to interpret the information. We also round the predicted values so that the data in the table remains both readable and still informative enough.

# Dash

The purpose of the dash web application is to provide users a way to interact with past, present, and future weather data from the NASA POWER API at a hyperlocal scale that is relevant to their everyday lives. This is reflected in the three different tabs of the application: “Daily Weather”, “Weather Graphs”, and “Weather Predictions”. Each tab implements the aforementioned technical components with user-selected parameters. 

The first tab “Daily Weather” prompts the user for a location and date of interest. Users can zoom, pan, and select a city to explore the dashleaflet.Map object and a marker to their point of interest. To select a date, they can select a day from a dcc.DatePickerSingle object and submit these two parameters with the “Submit” button. Clicking the button will initiate the callback function ‘submit’ which will make an asynchronous call to gather NASA POWER API for the stored latitude, longitude, and date. The result of the function is a dash_table.DataTable with several weather attributes for each 24 hours of the given day displayed at the bottom of the page. Users can make several choices and submissions to update the dash_table.DataTable if they desire.

```
dcc.Tab(label='Daily Weather', children=[ 
    dcc.Markdown('''
        ## Directions: 
        1. Pan and zoom in/out on the map to find your location.
        2. Click your location on the map for current weather.
            Press the "Submit" button to confirm your selection and your weather results will load promptly.
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
], className="tab",)
```

The image below is the default layout of the first tab with directions listed at the top for users and labeled dash objects for ease of navigation. 

<img width="501" alt="Screenshot 2025-03-10 at 5 57 19 PM" src="https://github.com/user-attachments/assets/c790d3ee-1e59-402a-9ba7-c6eed48f5546" />


The image below demonstrates the zoom_to_city option and adding a marker for a point of interest to the dashleaflet.Map object. The dcc.DatePickerSingle pop-up is shown at the bottom.

<img width="507" alt="Screenshot 2025-03-10 at 5 57 29 PM" src="https://github.com/user-attachments/assets/de1e7b6c-7262-4365-ab4e-91948ebd5e44" />


The image below is the dash_table.DataTable output of the ‘submit’ callback function. Formatted NASA POWER API data is displayed for each hour of the selected day. A caption with the user-selected parameters is printed above the table to remind users what they selected for the table below.

<img width="505" alt="Screenshot 2025-03-10 at 5 57 39 PM" src="https://github.com/user-attachments/assets/20cbb34e-d2c7-4d4d-a1ca-9cd47f5f5176" />

The second tab “Weather Graphs” implements the complex plotly visualizations explained in the earlier section. Users are able to select a weather attribute they wish to visualize over a temporal or spatial scale depending on the graph type they choose. The five options for weather attributes are: "Temperature (F)", "Precipitation (in)", "Wind Speed (mph)", "Relative Humidity (%)", "Cloud Cover (%)". These weather attribute options were selected based on the most common information found on phone weather apps and were also available on the NASA POWER API. The three options for graph types are hourly changes throughout a single day, yearly average changes throughout the past ten years, and the most recent weather information for nearby locations on the specified date. The hourly changes are displayed as a line graph of the weather attribute over time (see `plot_hourly`), the yearly changes are displayed as a boxplot of the weather attribute over time (see `plot_yearly`), and the nearby location data are displayed as a map scatterplot (see `plot_mock_heatmap`). Users click the "submit graph" button to confirm their plot choices. 
```
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
], className='tab',),
```

The image below shows the default layout of the “Weather Graphs” tab. There are directions to help users understand how to display a graph and labeled dash objects for ease of navigation.

<img width="504" alt="Screenshot 2025-03-10 at 6 00 36 PM" src="https://github.com/user-attachments/assets/d9bf7574-af1b-4ea2-a18f-54496166cba1" />

The image below is an example of the hourly change in relative humidity for the location and date parameters chosen in the previous tab “Daily Weather.” A caption is provided to remind users of what parameters were used to retrieve NASA POWER API data. 

<img width="505" alt="Screenshot 2025-03-10 at 6 00 44 PM" src="https://github.com/user-attachments/assets/7b942f62-7b29-4cf6-9e76-cd3ec64479ba" />

The image below is an example of the yearly change in relative humidity graph.

<img width="500" alt="Screenshot 2025-03-10 at 6 00 52 PM" src="https://github.com/user-attachments/assets/f86a58a0-c053-41fd-a682-1d03c9eb8841" />

The image below is an example of the spatial comparison of nearby locations’ relative humidity for a given day.

<img width="504" alt="Screenshot 2025-03-10 at 6 01 04 PM" src="https://github.com/user-attachments/assets/ccf9333e-2194-4be7-8df6-4b344077f364" />

The third tab “Weather Predictions” implements the keras machine learning model described in the previous section. Users can type or select a number of hours into the future to predict a given weather attribute. Confirming their choice with the “predict weather” button will output a dash_table.DataTable with the listed hours of the next day and their predicted weather attribute. A plotly line graph is also displayed to compare the 3 days of weather data the model was fitted to and the predicted weather data highlighted in red.

```
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
```
The image below shows the default layout of the “Weather Predictions” tab. Directions and labels are written out for ease of navigation and function use. 


<img width="501" alt="Screenshot 2025-03-10 at 6 01 17 PM" src="https://github.com/user-attachments/assets/fb933d14-1260-4efe-a85e-acef3b629526" />



The image below shows the output produced after clicking the “predict weather” button. Users can also hover over the graph to inspect individual data points on the graph. 


<img width="511" alt="Screenshot 2025-03-10 at 6 01 25 PM" src="https://github.com/user-attachments/assets/e4206e01-b022-44ab-a662-724eced80db9" />



# Conclusion
Our primary objective was to develop a web application that provides Los Angeles residents with precise and reliable weather information. We aim for this project to bring users' accurate weather forecasts and awareness regarding climate changes over time. By utilizing our weather visualizations, users can examine daily trends and review historical data spanning the past decade, allowing them to observe how these trends have evolved over time. While our predictions of extreme weather events are intended to inform users, they also might evoke concern. Although we strive for accuracy in our forecasts, there is still the possibility of error. One limitation of our project is that we rely solely on data from the NASA POWER API, which may contain certain bias since we did not cross-reference with other weather stations.


# How to run our webapp
1. Download ‘final_weather_dash_app.ipynb’
2. Download all .py files in  ‘myProject’ folder from repository and keep it all in a folder called ‘myProject’
3. Download ‘assets’ folder (includes ‘header.css’, ‘typography.css’, and ‘custom-script.js’) and ‘california_cities_5.csv’
4. Download the 'Project_images' folder with all the .png files and and keep it all in a folder called ‘Project_images’
5. Put all files together in your desired directory.
6. Run all cells in ‘final_weather_dash_app.ipynb’ 
