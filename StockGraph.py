import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from datetime import date
import datetime

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters

today = date.today()
d1 = today.strftime("%Y-%m-%d")
print(d1)

tenY = datetime.datetime.today() - datetime.timedelta(days=3653)
previousTenY = tenY.strftime("%Y-%m-%d")
print(previousTenY)

ticker = yf.Ticker("UNP")
data = ticker.history(period = "10y", interval = "1d")

data.sort_values('Date', inplace = True, ascending = True)
print(ticker.info['LongBusinessSummary'])

new = yf.download("AAPL", start='2011-02-16', end='2021-02-16', group_by='tickers')
df = new[['Close','Open','Low','High','Adj Close']].copy()
print(df.head())


plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
data['Close'].plot()
plt.tight_layout()
plt.grid()
plt.show()