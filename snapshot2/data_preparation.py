# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:57:46 2024

@author: loulo
"""
import numpy as np
import os
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    
    Parameters:
    - ticker: Stock symbol
    - start_date: Start date for data retrieval (format: 'YYYY-MM-DD')
    - end_date: End date for data retrieval (format: 'YYYY-MM-DD')
    
    Returns:
    - data: DataFrame with stock data, granularity: daily
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    rounded_data = data.round(2)
    
    #print(rounded_data[['Open', 'High', 'Low', 'Close']])
    return rounded_data

def generate_candlestick_image(data):
    """
    Generate a 2D grayscale candlestick image from stock data.

    Parameters:
    - data: DataFrame containing 'Open', 'High', 'Low', 'Close' stock data

    Returns:
    - img: 2D numpy array representing the candlestick image in grayscale 84x84 + plots it 
    """
    # Ensure that the data contains exactly 28 days
    #print(len(data))
    assert len(data) == 28, "Data must contain exactly 28 days of stock prices."
    height, width = 84, 84
    img = np.ones((height, width))  # Initialize with white background

    # Normalize the data
    max_price = data[['High']].max().max()
    min_price = data[['Low']].min().min()

    # Calculate scale factors
    scale_y = height / (max_price - min_price)
    scale_x = width / len(data)

    for i, (index, row) in enumerate(data.iterrows()):
        # Calculate x positions
        x_center = int(i * scale_x) + 1  # Center of the candlestick
        x_left = max(x_center - 1, 0)  # Ensure within bounds
        x_right = min(x_center + 1, width - 1)

        # Normalize y positions
        y_open = int((row['Open'] - min_price) * scale_y)  
        y_close = int((row['Close'] - min_price) * scale_y)
        y_high = int((row['High'] - min_price) * scale_y)
        y_low = int((row['Low'] - min_price) * scale_y)

        if row['Close'] >= row['Open']:
            color = 0.5  # Light gray for upward movement
        else:
            color = 0  # Black for downward movement

        # Draw the body
        top = min(y_open, y_close)
        bottom = max(y_open, y_close)
        img[height - bottom:height - top, x_left:x_right + 1] = color  #  body

        # Draw the stick
        img[height - y_high:height - y_low, x_center] = color  #  stick
        
    #print(np.shape(img))
    #plt.imshow(img, cmap='gray', aspect='auto',interpolation='none')
    #plt.show()
    return img



def generate_image_files(ticker, start_date, end_date):
    """
    Fetches data for a given ticker between start_date and end_date, 
    then generates candlestick images for every 28-day segment within that period.
    
    Parameters:
    - ticker: The stock ticker symbol as a string.
    - start_date: The start date for data retrieval as a string (format: 'YYYY-MM-DD').
    - end_date: The end date for data retrieval as a string (format: 'YYYY-MM-DD').
    """
    # Fetch the data
    data = fetch_data(ticker, start_date, end_date)
    # Calculate the number of days to remove to make the data length a multiple of 28
    excess_days = len(data) % 28
    nb_im = len(data)//28
    print(f'fetched {len(data)}={nb_im}x28+{excess_days} days worth of data, creating {nb_im} images...')
        

    # Loop through the data 
    for i in range(0, len(data)-27):
        segment = data.iloc[i:i+28]
        # Generate the candlestick image for the segment
        img = generate_candlestick_image(segment)
        
        # Define the image file path
        directory = f'./candlestick_images/{ticker}'
        if not os.path.exists(directory):  # Check if the directory does not exist
            os.makedirs(directory)  # Create the directory
        
        image_path = f'./candlestick_images/{ticker}/{i}.png'
        
        # Save the image
        plt.imsave(image_path, img, cmap='gray', format='png')
        print(f"Saved: {image_path}")




if __name__ == "__main__":
    ticker = 'ADBE'  # Example stock ticker
    #start_date = '2013-01-01'
    #end_date = '2020-01-04'
    start_date = '2013-01-01'
    end_date = '2013-02-12'
    # Fetch data
    data = fetch_data(ticker, start_date, end_date)
    
    # Generate candlestick images
    img = generate_candlestick_image(data)

    #plt.imsave('./candlestick_images/prout.png', img, cmap='gray', format='png')
    
    
    #generate_image_files(ticker, start_date, end_date)
    
    