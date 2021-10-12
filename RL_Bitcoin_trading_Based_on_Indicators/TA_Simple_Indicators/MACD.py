import pandas as pd
from ta.trend import macd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates


import numpy as np
import pandas as pd
import yfinance as yf

def Plot_OHCL(df, ax1_indicators=[], ax2_indicators=[]):
    df_original = df.copy()
    # necessary convert to datetime
    df["Date"] = pd.to_datetime(df.Date)
    df["Date"] = df["Date"].apply(mpl_dates.date2num)

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # We are using the style ‘ggplot’
    plt.style.use('ggplot')
    
    # figsize attribute allows us to specify the width and height of a figure in unit inches
    fig = plt.figure(figsize=(16,8)) 

    # Create top subplot for price axis
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)

    # Create bottom subplot for volume which shares its x-axis
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

    candlestick_ohlc(ax1, df.values, width=0.8/24, colorup='green', colordown='red', alpha=0.8)
    ax1.set_ylabel('Price', fontsize=12)
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    # plot all ax1 indicators
    for indicator in ax1_indicators:
        ax1.plot(df["Date"], df_original[indicator],'.',label=indicator)
    ax1.legend(loc="upper left")

    # plot all ax2 indicators
    for indicator in ax2_indicators:
        ax2.plot(df["Date"], df_original[indicator],'-',label=indicator)
    ax2.legend(loc="upper left")

    # beautify the x-labels (Our Date format)
    ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()
    
    plt.show()

def AddIndicators(df):
    # Add Moving Average Convergence Divergence (MACD) indicator
    df["MACD"] = macd(close=df["Close"], window_slow=26, window_fast=12, fillna=True)
    
    return df

if __name__ == "__main__": 
    
    #Original code  
    '''df = pd.read_csv('./pricedata.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)
    print(df)'''

    # Self Manipulated Data Without Changing Date
    '''df = pd.read_csv('./Artificial_env.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)
    print(df)'''

    # Self Manipulated Data Without Changing Date
    '''df = pd.read_csv('./Artificial_env.csv')
    df.Date = pd.date_range('2020-01-01', '2021-01-01', freq="h")[:len(df)]
    df = df.sort_values('Date')
    df.dropna(inplace=True)
    df = AddIndicators(df)
    df.to_csv('artificial_env_with_date.csv', index=False)
    print(df)'''

    # Self Manipulated Data with Date Attached
    #df = pd.read_csv('./artificial_env_with_date.csv')
    df = pd.read_csv('./Binance_BTCUSDT_1h_Modified.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)
    print(df)

    #Downloading from Yahoo Finance
    '''data = yf.download(tickers='BTC-USD',  interval = '1h',start='2020-01-01', end='2021-01-01',)
    data.pop('Adj Close')
    data = data.rename_axis('Date').reset_index()
    #data = AddIndicators(data)

    data.to_csv('2020_01_to_2021_01.csv', index=False)

    print(data)
    test_df = data[-400:]'''
    #test_df = df[-6000:-3000]
    test_df = df[-400:]

    # Add Moving Average Convergence Divergence
    Plot_OHCL(test_df, ax2_indicators=["MACD"])