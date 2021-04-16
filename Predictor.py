import yfinance as yf
import math
import numpy as np
import pandas as pd
import csv
from math import sqrt

#Libraries used for plotting and graphing
import matplotlib.pyplot as plt 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters

#Library used to normalize data and obtain metrics for evaluation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Libraries used for machine learning portion
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

#Libraries used to get current date
from datetime import date
import datetime
import time as ti


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


#This is to enable GPU use for training the model
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



def StockOne():
    #Get todays date
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")

    #Get date ten years from current date
    tenY = datetime.datetime.today() - datetime.timedelta(days=3653)
    previousTenY = tenY.strftime("%Y-%m-%d")


    #Information ticker for stock
    ticker = yf.download("JWN", start=previousTenY, end=d1, group_by='tickers')

    #Tickers to determine what data is being used. The 'Open' is the opening price of stocks each day. 
    #The 'Low' and 'High' are the lowest and highest price of each stock within a day.
    #The 'Adj Close' is the closing price when factoring in corporate actions and other technical factors
    #The 'Close' is the end of the day raw cash value of the stock
    df = ticker[['Close','High','Low','Open','Adj Close']].copy()



    #Normalize the data to be within a range of (0,1) 
    scaler = MinMaxScaler(feature_range = (0,1))
    df = scaler.fit_transform(np.array(df).reshape(-1,1))


    #Split the data into an 80/20 ratio for training and testing
    training_size = int(len(df)*0.80)
    test_size = len(df) - training_size
    train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]



    #Format the data into data and timesteps for the LSTM to read
    def dataset(data, time = 1):
        Xs, ys = [], []
        for i in range(len(data) - time - 1):
            Xs.append(data[i:(i+time), :])
            ys.append(data[i,0])
        return np.array(Xs), np.array(ys)


    #Converting data into X and y with timestamps for the LSTM to read.
    time = 20
    X_train, y_train = dataset(train_data,  time)
    X_test, y_test = dataset(test_data, time)

    #Reshaping the data in order to read into the model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) 


    #RNN Model using LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units = 20, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units= 20, return_sequences = True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units = 15, return_sequences = False),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss='mean_squared_error', 
        optimizer = 'adam',
        metrics = ['mae'])
    model.summary()


    #Begin Timer to see how long it takes to train
    Start_time = ti.time()

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 3)

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=128,
        verbose=1,
        validation_data = (X_test, y_test),
        shuffle = False,
        callbacks = [es]
    )

    #End Timer 
    End_time = ti.time()
    print("The total time taken: ", round((End_time-Start_time)/60), 'Minutes')

    y_pred = model.predict(X_test)


    #Accuracy meaningless for regression problems(?)
    rmse_y_test = sqrt(mean_squared_error(y_test, y_pred))
    mse = np.mean((y_test - y_pred)**2)
    score = model.evaluate(X_test,y_test, verbose=1)
    print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
    print("Accuracy:" , {score[1]})
    print('RMSE = ', rmse_y_test)
    print('MSE = ', mse)
    print("R2 Score = {0:0.2%}".format(r2_score(y_test,y_pred)))


    #Plot Model train vs Validation loss to determine if the model is underfitting or overfitting. This is based on googling "How to determine if an LSTM architecture in underfitting or overfitting"
    plt.plot(history.history['loss'], label = 'train')
    plt.plot(history.history['val_loss'], label = 'test')
    plt.title('Model train vs Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Nordstrom loss comparison")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("Nordstrom loss")
   



    #Plot predictions
    y_pred = scaler.inverse_transform(y_pred)
    fig, ax = plt.subplots(figsize=(20, 10))

    #Reshape back to original data format
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    #plotting the predictions vs the true
    plt.plot(y_test_scaled, label="True Price")
    plt.plot(y_pred, label='Predicted Testing Price')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.title("Nordstrom Stock Prediction")
    plt.legend()
    plt.savefig("Nordstrom Stock")

def StockTwo():
    #Get todays date
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")

    #Get date ten years from current date
    tenY = datetime.datetime.today() - datetime.timedelta(days=3653)
    previousTenY = tenY.strftime("%Y-%m-%d")


    #Information ticker for stock
    ticker = yf.download("BB", start=previousTenY, end=d1, group_by='tickers')

    #Tickers to determine what data is being used. The 'Open' is the opening price of stocks each day. 
    #The 'Low' and 'High' are the lowest and highest price of each stock within a day.
    #The 'Adj Close' is the closing price when factoring in corporate actions and other technical factors
    #The 'Close' is the end of the day raw cash value of the stock
    df = ticker[['Close','High','Low','Open','Adj Close']].copy()
    df2 = ticker['Close'].copy()



    #Normalize the data to be within a range of (0,1) 
    scaler = MinMaxScaler(feature_range = (0,1))
    df = scaler.fit_transform(np.array(df).reshape(-1,1))
    df2 = scaler.fit_transform(np.array(df2).reshape(-1,1))


    #Split the data into an 80/20 ratio for training and testing
    training_size = int(len(df)*0.80)
    test_size = len(df) - training_size
    train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]



    #Format the data into data and timesteps for the LSTM to read
    def dataset(data, time = 1):
        Xs, ys = [], []
        for i in range(len(data) - time - 1):
            Xs.append(data[i:(i+time), :])
            ys.append(data[i,0])
        return np.array(Xs), np.array(ys)


    #Converting data into X and y with timestamps for the LSTM to read.
    time = 30
    X_train, y_train = dataset(train_data,  time)
    X_test, y_test = dataset(test_data, time)

    #Reshaping the data in order to read into the model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) 


    #RNN Model using LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units = 15, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units = 15, return_sequences = True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units = 10, return_sequences = True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units = 5, return_sequences = False),
    
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss='mean_squared_error', 
        optimizer = 'adam',
        metrics = ['mae'])
    model.summary()


    #Begin Timer to see how long it takes to train
    Start_time = ti.time()

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 5)

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        verbose=1,
        validation_data = (X_test, y_test),
        shuffle = False,
        callbacks = [es]
    )

    #End Timer 
    End_time = ti.time()
    print("The total time taken: ", round((End_time-Start_time)/60), 'Minutes')

    test_predict = model.predict(X_test)

    #Accuracy meaningless for regression problems(?)
    rmse_y_test = sqrt(mean_squared_error(y_test, test_predict))
    mse = np.mean((y_test - test_predict)**2)
    score = model.evaluate(X_test,test_predict, verbose=1)
    print("Mean Absolute Error: ", mean_absolute_error(y_test, test_predict))
    print('RMSE = ', rmse_y_test)
    print('MSE = ', mse)
    print("R2 Score = {0:0.2%}".format(r2_score(y_test,test_predict)))


    #Plot Model train vs Validation loss to determine if the model is underfitting or overfitting. This is based on googling "How to determine if an LSTM architecture in underfitting or overfitting"
    plt.plot(history.history['loss'], label = 'train')
    plt.plot(history.history['val_loss'], label = 'test')
    plt.title('Model train vs Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Union Pacific Loss Comparison")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("Union Pacific Loss")



 
   
def StockThree():
    #Get todays date
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")

    #Get date ten years from current date
    tenY = datetime.datetime.today() - datetime.timedelta(days=3653)
    previousTenY = tenY.strftime("%Y-%m-%d")


    #Information ticker for stock
    ticker = yf.download("DIS", start=previousTenY, end=d1, group_by='tickers')

    #Tickers to determine what data is being used. The 'Open' is the opening price of stocks each day. 
    #The 'Low' and 'High' are the lowest and highest price of each stock within a day.
    #The 'Adj Close' is the closing price when factoring in corporate actions and other technical factors
    #The 'Close' is the end of the day raw cash value of the stock
    df = ticker[['Close','High','Low','Open','Adj Close']].copy()



    #Normalize the data to be within a range of (0,1) 
    scaler = MinMaxScaler(feature_range = (0,1))
    df = scaler.fit_transform(np.array(df).reshape(-1,1))


    #Split the data into an 80/20 ratio for training and testing
    training_size = int(len(df)*0.80)
    test_size = len(df) - training_size
    train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]



    #Format the data into data and timesteps for the LSTM to read
    def dataset(data, time = 1):
        Xs, ys = [], []
        for i in range(len(data) - time - 1):
            Xs.append(data[i:(i+time), :])
            ys.append(data[i,0])
        return np.array(Xs), np.array(ys)


    #Converting data into X and y with timestamps for the LSTM to read.
    time = 30
    X_train, y_train = dataset(train_data,  time)
    X_test, y_test = dataset(test_data, time)

    #Reshaping the data in order to read into the model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) 


    #RNN Model using LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units = 10, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units = 5, return_sequences = True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units = 5, return_sequences = False),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss='mean_squared_error', 
        optimizer = 'adam',
        metrics = ['mae'])
    model.summary()


    #Begin Timer to see how long it takes to train
    Start_time = ti.time()

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 3)

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=128,
        verbose=1,
        validation_data = (X_test, y_test),
        shuffle = False,
        callbacks = [es]
    )

    #End Timer 
    End_time = ti.time()
    print("The total time taken: ", round((End_time-Start_time)/60), 'Minutes')

    y_pred = model.predict(X_test)


    #Accuracy meaningless for regression problems(?)
    rmse_y_test = sqrt(mean_squared_error(y_test, y_pred))
    mse = np.mean((y_test - y_pred)**2)
    score = model.evaluate(X_test,y_test, verbose=1)
    R2 = r2_score(y_test, y_pred)
    print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
    print('RMSE = ', rmse_y_test)
    print('MSE = ', mse)
    print("R2 Score = {0:0.2%}".format(r2_score(y_test,y_pred)))
    model.save('savedModel.h5')
    model_use = keras.models.load_model("savedModel.h5")
    modeltest = model_use.predict(X_test)
    print("TestValue = {0:0.2%}".format(r2_score(y_test,modeltest)))



    #Plot Model train vs Validation loss to determine if the model is underfitting or overfitting. This is based on googling "How to determine if an LSTM architecture in underfitting or overfitting"
    plt.plot(history.history['loss'], label = 'train')
    plt.plot(history.history['val_loss'], label = 'test')
    plt.title('Model train vs Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Disney Loss Comparison")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("Disney Loss")


    #Plot predictions
    y_pred = scaler.inverse_transform(y_pred)
    fig, ax = plt.subplots(figsize=(20, 10))

    #Reshape back to original data format
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    #plotting the predictions vs the true
    plt.plot(y_test_scaled, label="True Price")
    plt.plot(y_pred, label='Predicted Testing Price')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.title("Disney Predictions")
    plt.legend()
    plt.savefig("Disney Predictions")

StockThree()

    