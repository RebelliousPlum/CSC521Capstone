import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters


ticker = yf.Ticker("JWN")
data = ticker.history(period = "10y", interval = "1d")

data.sort_values('Date', inplace = True, ascending = True)



plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
data['Close'].plot()
plt.tight_layout()
plt.grid()
plt.show()