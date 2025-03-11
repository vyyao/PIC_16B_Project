# -*- coding: utf-8 -*-

import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
#from keras import layers, losses
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
import datetime
import requests

import matplotlib.pyplot as plt
from plotly import express as px
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing, tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Function that takes a feature, and runs a model to predict the next 4 days of that feature based on 30 days of previous weather
def run_model(dataset, feature, n_past=50, n_future=7):
    """
    Trains a Bidirectional LSTM model to predict the next
    "n_future" units of time of a given weather feature based on
    "n_past" units of time of historical data.

    Args:
    dataset (DataFrame): Pandas DataFrame containing historical weather data.
    feature (str): Column name of the feature to be predicted.
    n_past (int, optional): Number of past time units(usually days)
        to use for training. Default 50.
    n_future (int, optional): Number of future time units(usually days)
        to predict. Default 7.

    Returns:
    real_temperature (DataFrame): DataFrame containing the last "n_past"
        units of time of actual feature values.
    predicted_temperature (ndarray): Array containing predicted values
        for the next "n_future" units of time.
    """
    feature = feature
    # Drop rows with missing values in the selected feature
    dataset = dataset.dropna(subset=[feature])
    dataset = dataset.reset_index(drop=True)
    # Extract the feature values as a NumPy array
    training_set = dataset[[feature]].values

    # Normalize the data to the range (0,1) for better model training stability
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)

    x_train = []
    y_train = []

    n_past = n_past # Number of days to use as training data
    n_future = n_future # Number of days into future to predict feature

    # Create sequences of "n_past" days as input and "n_future" days as output
    for i in range(0,len(training_set_scaled)-n_past-n_future+1):
        x_train.append(training_set_scaled[i : i + n_past , 0])
        y_train.append(training_set_scaled[i + n_past : i + n_past + n_future , 0 ])
    # Convert to NumPy arrays and reshape for LSTM input
    x_train , y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1) )

    # Define the LSTM model architecture
    model = Sequential([
        Bidirectional(LSTM(units=n_past, return_sequences=True, input_shape=(x_train.shape[1], 1))),
        Dropout(0.2),
        LSTM(units=n_past, return_sequences=True),
        Dropout(0.2),
        LSTM(units=n_past, return_sequences=True),
        Dropout(0.2),
        LSTM(units=n_past),
        Dropout(0.2),
        Dense(units=n_future, activation='linear')
    ])
    # Compile the model with Adam optimizer and mean squared error loss function
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
    # Note that accuracy may be misleadingly low because the prediction
    # is not 100% matching the real, even though that's not totally necessary

    # Train the model on the prepared dataset
    model.fit(x_train, y_train, epochs=20, batch_size=32)

    # Prepare test dataset using the most recent "n_past" values
    testdataset = dataset.copy()
    testdataset = testdataset[[feature]].iloc[:n_past].values
    # return n_past most recent temperatures
    real_temperature = dataset.copy().sort_values("datetime", ascending=False).iloc[-n_past:-1]
    real_temperature = real_temperature[["datetime",feature]]

    # Scale and reshape test data for prediction
    testing = sc.transform(testdataset)
    testing = np.array(testing)
    testing = np.reshape(testing,(testing.shape[1],testing.shape[0],1))

    # Make predictions using the trained model
    predicted_temperature = model.predict(testing)
    # Inverse transform the predicted values to original scale
    predicted_temperature = sc.inverse_transform(predicted_temperature)
    predicted_temperature = np.reshape(predicted_temperature,(predicted_temperature.shape[1],predicted_temperature.shape[0]))
    return real_temperature, predicted_temperature

from datetime import date, datetime, timedelta
import plotly.express as px

def plot_predictions(attr, real, predicted, n_future):
    """
    Plots a line graph of past weather attributes over the given date range selected by
    the user and the n_future days of predicted weather attributes

    Args:
        attr (str): the weather attribute of interest to plot and predict
        real (dataframe): a dataframe with the weather attribute on a specific date
        predicted (np.array): an array of predicted weather attributes of length n_future
        n_future (int): the number of days into the future with predicted weather

    Returns:
        fig (plotly line graph): real and predicted weather attributes plotted over time
        df (dataframe): a dataframe with labeled real and predicted weather points for
            specific dates
    """
    # add real labels to weather attr of dates that came directly from nasa api database
    real["Label"] = "Real"

    # create list of n_future number of new dates the model is predicting
    new_dates = [real["datetime"].iloc[0] + timedelta(hours=x) for x in range(1,n_future+1)]

    # create a temporary dataframe from a dict of dates, predicted weather, and labels
    new_rows = pd.DataFrame({"datetime": new_dates,
                             attr: predicted.flatten(),
                             "Label": "Predicted"})

    # concatenate real and predicted dataframes with all columns filled
    df = pd.concat([real, new_rows], ignore_index=True)

    # plot a line graph where real and predicted weather are differentiated by color
    fig = px.line(df,x="datetime",y=attr,color="Label",markers=True,symbol="Label",
                  hover_data = {attr:':.4f'},) # round hover data

    # create dataframe subset with only predicted dates to display in table
    df = df[df["Label"]=="Predicted"]
    df[attr] = round(df[attr],2)

    # make dates into familiar month-day-year format for display in table
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H", errors="coerce")

    # remove labels from records to display in table
    df = df[["datetime",attr]]

    return fig, df

