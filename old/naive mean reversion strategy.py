# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:53:22 2024

@author: loulo
"""
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to fetch historical data
def fetch_data(stock, start_date, end_date, inter='1d'):
    data =  yf.download(stock, start=start_date, end=end_date,interval=inter)
    num_periods = data.index.nunique()
    print(f"Imported data has {num_periods} periods.")
    return data



# Function to apply mean reversion strategy
def mean_reversion_strategy(data, n1, n2):
    # Calculate moving averages
    data['MA_n1'] = data['Close'].rolling(window=n1).mean()
    data['MA_n2'] = data['Close'].rolling(window=n2).mean()

    # Generate trading signals
    # Buy signal: when short-term MA (n1) crosses above long-term MA (n2)
    # Sell signal: when short-term MA (n1) crosses below long-term MA (n2)
    data['Signal'] = 0
    data['Signal'] = np.where(data['MA_n1'] > data['MA_n2'], 1, 0)
    data['Position'] = data['Signal'].diff()

    # Calculate returns
    data['Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Return'] * data['Position'].shift(1)

    return data['Strategy_Return'].cumsum().iloc[-1]


def plot_stock_with_moving_averages(data, n1, n2):
    # Calculate moving averages
    data['MA_n1'] = data['Close'].rolling(window=n1).mean()
    data['MA_n2'] = data['Close'].rolling(window=n2).mean()

    # Generate trading signals
    data['Signal'] = np.where(data['MA_n1'] > data['MA_n2'], 1, 0)
    data['Position'] = data['Signal'].diff()

    # Calculate returns
    data['Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Return'] * data['Position'].shift(1)

    # Plotting the stock's closing price and the moving averages
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Closing Price', color='blue')
    plt.plot(data['MA_n1'], label=f'{n1}-Period MA', color='red')
    plt.plot(data['MA_n2'], label=f'{n2}-Period MA', color='green')

    data['Cumulative_Strategy_Return'] = data['Strategy_Return'].cumsum()
    
    # Plotting the trades and the cumulative returns
    for i in range(len(data)):
        if data['Position'].iloc[i] == 1:  # Buy signal
            plt.axvline(data.index[i], color='green', linestyle='dashed', alpha=0.7)
        elif data['Position'].iloc[i] == -1:  # Sell signal
            plt.axvline(data.index[i], color='red', linestyle='dotted', alpha=0.7)
            # Display cumulative return at the sell signal
            cumulative_return = data['Cumulative_Strategy_Return'].iloc[i]
            plt.text(data.index[i], data['Close'].iloc[i], f'{cumulative_return:.4f}', 
                     horizontalalignment='left', verticalalignment='bottom', fontsize=8, color='black')

    # Adding title and labels
    plt.title('Stock Price with Moving Averages and Trades')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


##n1 and n2 are   integers representing the number of periods over which the
## moving averages are calculated, where a "period" is determined by the 
## frequency of your data.For daily data, they represent days; for hourly data, 
##they represent hours.
#n2>n1
def findn1n2(data):
    best_return = -100000
    # Range for n1 and n2 values
    for n1 in range(5, 50):  # Range from 1 to 50
        for n2 in range(n1, 50):  # Range from 1 to 50
            if n1 != n2:
                strategy_return = mean_reversion_strategy(data.copy(), n1, n2)
                if strategy_return > best_return:
                    best_return = strategy_return
                    best_n1, best_n2 = n1, n2
    return best_return,best_n1,best_n2


def find_best_n1_n2_monthly(stock_symbol,data):

    # Break the data into months
    months = data.resample('M').mean().index
    
    for start_date, end_date in zip(months[:-1], months[1:]):
        monthly_data = data[start_date:end_date]
        best_return, best_n1, best_n2 = findn1n2(monthly_data)
        # Calculate monthly volatility as the standard deviation of daily returns
        monthly_volatility = monthly_data['Close'].pct_change().std()
        print(f"Month: {start_date.strftime('%Y-%m')} - Best n1: {best_n1}, Best n2: {best_n2}, Return: {best_return}, Volatility: {monthly_volatility:.4f}")
        
def yearly_cumulative_return_with_fixed_n1_n2(stock_symbol, n1, n2):
    # Download the stock data for the year 2023
    data = yf.download(stock_symbol, start="2023-01-01", end="2024-01-01", interval='1h')

    # Break the data into months
    months = data.resample('M').mean().index
    cumulative_return = 0

    for start_date, end_date in zip(months[:-1], months[1:]):
        monthly_data = data[start_date:end_date]
        monthly_return = mean_reversion_strategy(monthly_data.copy(), n1, n2)
        cumulative_return += monthly_return
        print(f"Month: {start_date.strftime('%Y-%m')} - Return: {monthly_return}")

    print(f"Cumulative return for the year: {cumulative_return}")
    return cumulative_return

    


#######
stock_symbol = "AAPL" # Example stock symbol (apple)
start_date = "2023-01-01"
end_date = "2024-01-01"
data = fetch_data(stock_symbol, start_date, end_date,inter='1h')
#######


#best_return,best_n1,best_n2 = findn1n2(data)
#print('best n1 and n2 for given interval: ', best_n1, ',', best_n2, 'with a total return of: ', best_return*100,"%")


#print(f"Best strategy return: {best_return} with n1 = {best_n1} and n2 = {best_n2}")
#plot_stock_with_moving_averages(data,best_n1,best_n2)


#find_best_n1_n2_monthly("AAPL",data)

#yearly_cumulative_return_with_fixed_n1_n2("AAPL", 21, 19)
