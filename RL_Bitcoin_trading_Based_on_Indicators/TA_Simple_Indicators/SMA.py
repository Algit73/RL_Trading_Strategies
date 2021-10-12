import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates

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
        print(indicator)
        ax1.plot(df["Date"], df_original[indicator],'-',label=indicator)
    ax1.legend(loc="upper left")
    # plot all ax2 indicators
    for indicator in ax2_indicators:
        ax2.plot(df["Date"], df_original[indicator],'-',label=indicator)

    # beautify the x-labels (Our Date format)
    ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()
    
    plt.show()

def AddIndicators(df):
    # Add Simple Moving Average (SMA) indicators
    df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()
    
    return df

if __name__ == "__main__":   
    '''
    df = pd.read_csv('./pricedata.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)
    '''

    '''data = yf.download(tickers='BTC-USD',  interval = '1h',start='2020-01-01', end='2021-02-20',)
    #data = yf.download(tickers='BTC-USD', period = '1mo', interval = '30m')
    data.pop('Adj Close')
    data = data.rename_axis('Date').reset_index()
    data = AddIndicators(data)

    #test_df = df[-400:]
    test_df = data[-400:]'''

    #
    df = pd.read_csv('./artificial_env_with_date.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)
    #print(df)
    test_df = df[-400:]

    # Add Simple Moving Average
    Plot_OHCL(test_df, ax1_indicators=["sma7", "sma25", "sma99"])



'''
    data = yf.download(tickers='BTC-USD',  interval = '1d',start='2020-01-01', end='2021-02-20',)
    #data = yf.download(tickers='BTC-USD', period = '1mo', interval = '30m')
    data.pop('Adj Close')
    data = data.rename_axis('Date').reset_index()
    data = AddIndicators(data)

    #test_df = df[-400:]
    test_df = data[-400:]

    # Add Simple Moving Average
    Plot_OHCL(test_df, ax1_indicators=["sma7", "sma25", "sma99"])
    #plt.show()
'''