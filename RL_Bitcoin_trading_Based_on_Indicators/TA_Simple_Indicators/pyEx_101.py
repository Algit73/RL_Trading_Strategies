import pyEX as p
from icecream import ic

c = p.Client(api_token='pk_3400ac35b76a4c12a4b54cba222788a2', version='stable')

sym='BTC'
timeframe='2y'
df = c.chartDF(symbol=sym, timeframe=timeframe)[['open', 'volume']]
ic(df[20:30])

crypto_df = p.cryptocurrency.cryptoBook(sym, token='pk_3400ac35b76a4c12a4b54cba222788a2', version='stable', filter='', format='json')
