import quandl
from icecream import ic


quandl.ApiConfig.api_key = r'L82LWbB4pNVznzhVy8hX'

#quandl.read_key() ## for local API key
print(quandl.ApiConfig.api_key)


data = quandl.get_table('ZACKS/FC', ticker='AAPL')
data_nse = quandl.get('NSE/OIL', start_date='2018-01-01', end_date='2020-01-01',
                  collapse='annual', transformation='rdiff',
                  rows=4)
ic(data_nse)                  
data_msf_appl = quandl.get(['WIKI/AAPL.11','WIKI/MSFT.11'])
ic(data_msf_appl)

