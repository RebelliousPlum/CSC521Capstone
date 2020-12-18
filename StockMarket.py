import yfinance as yf
import pandas as pd
import matplotlib.pyplot as pyplot



data = yf.download(tickers = "SPY AAPL MSFT", period = "10y", group_by="ticker")

tick = input("Enter stock name: ")

Open = data[tick]['Open']
Close = data[tick]['Close']
Low = data[tick]['Low'] 
High = data[tick]['High']
Volume = data[tick]['Volume']


dfOne = pd.DataFrame(Open, columns = ["Open"])
dfTwo = pd.DataFrame(Volume, columns = ["Volume"])
dfThree = pd.DataFrame(Low, columns = ["Low"])
dfFour = pd.DataFrame(High, columns = ["High"])
dfFive = pd.DataFrame(Close, columns = ["Close"])


df_all = pd.concat([dfOne, dfTwo, dfThree, dfFour,dfFive], axis = 1)

X = df_all.iloc[: , [True, True, True, True, False]]
y = df_all.iloc[:, [False,False,False,False,True]]

print(y.head())