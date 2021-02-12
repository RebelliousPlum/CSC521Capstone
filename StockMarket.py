import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import time as ti

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
gpus = tf.config.experimental.list_physical_devices('GPU')
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
tf.config.experimental.set_virtual_device_configuration(gpus[0], 
   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


ticker = yf.Ticker("UNP")
data = ticker.history(period = "10y", interval = "1d")

data.sort_values('Date', inplace = True, ascending = True)


df = data[['Close','Open','Low','High']].copy()
print(df.head())

# Seperate Testing data
train, test = df.iloc[0:-500], df.iloc[-500:len(df)]

print(len(train), len(test))


train_max = train.max()
train_min = train.min()

# Normalize data
train = (train - train_min)/(train_max - train_min)
test = (test - train_min)/(train_max - train_min)

# Format the data into data and timesteps
def dataset(X, y , time = 1):
    Xs, ys = [], []
    for i in range(len(X) - time):
        v = X.iloc[i:(i+time)].values
        Xs.append(v)
        ys.append(y.iloc[i + time])
    return np.array(Xs), np.array(ys)
    
time = 10
X_train, y_train = dataset(train, train.Close, time)
X_test, y_test = dataset(test, test.Close, time)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = 150, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units = 150, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units = 75, return_sequences = False),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer = 'adam')
model.summary()

#Begin Timer
Start_time = ti.time()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    verbose=1,
    validation_data = (X_test, y_test),
    shuffle = False
)

#End Timer
End_time = ti.time()
print("The total time taken: ", round((End_time-Start_time)/60), 'Minutes')