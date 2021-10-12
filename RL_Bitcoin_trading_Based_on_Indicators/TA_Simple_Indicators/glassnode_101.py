import json
import requests
import pandas as pd


# insert your API key here
API_KEY = '1tbzU6aq6YGd4ZVsSLr5l1RI9Yx'

# make API request
res = requests.get('https://api.glassnode.com/v1/metrics/indicators/sopr',
    params={'a': 'BTC','i':'24h', 'api_key': API_KEY})

# convert to pandas dataframe
df = pd.read_json(res.text, convert_dates=['t'])
print(df.shape)