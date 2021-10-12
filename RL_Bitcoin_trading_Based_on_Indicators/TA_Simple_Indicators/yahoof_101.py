# Raw Package
import numpy as np
import pandas as pd

#Data Source
import yfinance as yf

#Data viz
import plotly.graph_objs as go

data = yf.download(tickers='BTC-USD',  interval = '1h',start='2020-01-01', end='2021-01-01',)
print(data)
data.pop('Adj Close')
#data['Date'] = data.index
data = data.rename_axis('Date').reset_index()
#data.columns = ['Date', 'High', 'Low', 'Close', 'Volume']
print(data)