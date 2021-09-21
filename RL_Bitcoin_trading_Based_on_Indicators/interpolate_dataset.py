from numpy.core.records import _deprecate_shape_0_as_None
from numpy.lib.arraysetops import isin
import pandas as pd
from indicators import AddIndicators
from icecream import ic
import numpy as np
 

## 1H is the base for all other dataframes
'''data_frame_1h = pd.read_csv('./Binance_BTCUSDT_1h_Modified.csv')
data_frame_2h = pd.read_csv('./Binance_BTCUSDT_2h_Modified.csv')
data_frame_4h = pd.read_csv('./Binance_BTCUSDT_4h_Modified.csv')
data_frame_8h = pd.read_csv('./Binance_BTCUSDT_8h_Modified.csv')

df_1h_indicators_added = AddIndicators(data_frame_1h)
df_2h_indicators_added = AddIndicators(data_frame_2h)
df_4h_indicators_added = AddIndicators(data_frame_4h)
df_8h_indicators_added = AddIndicators(data_frame_8h)'''


## Generating a base DataFrame
'''dates = pd.date_range(start="2018-01-01", end="2021-8-23",freq="H")
df_general_length = dates.shape[0]

df_general = pd.DataFrame(columns = ['Date','Open','High','Low','Close','Volume','ATR'])
df_general.Date = dates
df_general.to_csv('Binance_BTCUSDT_1h_Base.csv', index=False)
'''

## Using a generated DataFrame as a base
'''df_general = pd.read_csv('./Binance_BTCUSDT_1h_Base.csv')
ic(df_general.shape)'''



## Modifing Dataframes to a standard format
### Removing miliseconds from the end of the string
'''data_frame_2h['Date'] = data_frame_2h['Date'].map(lambda x: x[:-9] + '00:00' if len(x)>19 else x )
data_frame_2h.to_csv('Binance_BTCUSDT_2h_Modified.csv', index=False)'''



## Generating a DataFrame consist of all times
'''df_general = df_general.merge(df_1h_indicators_added[['Date','ATR','Open','High','Low','Close','Volume']], on='Date', how="left")
df_general = df_general.merge(df_2h_indicators_added[['Date','ATR']], on='Date', how="left")
df_general = df_general.merge(df_4h_indicators_added[['Date','ATR']], on='Date', how="left")
df_general = df_general.merge(df_8h_indicators_added[['Date','ATR']], on='Date', how="left")
df_general.drop('Open_x')
df_general.drop('High_x')
df_general.drop('Close_x')
df_general.drop('Volume_x')
del df_general['Open_x'],df_general['Low_x'],df_general['Close_x'],df_general['Volume_x'] ,df_general['High_x']
ic(df_general)

df_general.to_csv('Binance_BTCUSDT_1h_Base_Inidc_Added.csv', index=False)'''

df_general = pd.read_csv('./Binance_BTCUSDT_1h_Base_Inidc_Added.csv')
df_general=df_general.interpolate(method ='linear', limit_direction ='forward')
del df_general['ATR']
ic(df_general)
df_general.to_csv('Binance_BTCUSDT_1h_Base_Inidc_Added.csv', index=False)

'''for index, row in df_1h_indicators_added.iterrows():

    #ic(df_general.index[df_general['2021-08-19 17:00:00']==True].tolist())
    #ic(str(df_general.index[df_general['Date']==row['Date']]))
    #list_=list_+df_general.index[df_general['Date']==row['Date']].tolist()
    #df_general.loc[df_general['Date'] == row['Date']] = row
    index_ = df_general.index[df_general['Date']==row['Date']].tolist()
    df_general['ATR'][index_] = row['ATR']
    #if(not data_frame_1h.Date.isin([str(row['Date'])]).any().any()):
     #   ic(row

'''
