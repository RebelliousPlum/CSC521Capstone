import yfinance as yf
import math
import numpy as np
import pandas as pd
import csv
from math import sqrt
import schedule
from getch import pause

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
import keras.backend as K
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


def DataDis():
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
    df2 = ticker[['Close']].copy()
    dfTemp = df.index
 



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

    def DisModel(X_train, y_train):
        #RNN Model using LSTM
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units = 15, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])),
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

        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 5)

        history = model.fit(
            X_train, y_train,
            epochs=35,
            batch_size=128,
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
        r2 = r2_score(y_test,test_predict)
        print("Mean Absolute Error: ", mean_absolute_error(y_test, test_predict))
        print('RMSE = ', rmse_y_test)
        print('MSE = ', mse)
        print("R2 Score = {0:0.2%}".format(r2_score(y_test,test_predict)))
        

        if r2 < .75:
            while r2 < .75:
                #Reset the states
                K.clear_session()

                #RNN Model using LSTM
                model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(units = 15, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])),
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

                es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 5)

                history = model.fit(
                    X_train, y_train,
                    epochs=35,
                    batch_size=128,
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
                r2 = r2_score(y_test,test_predict)
                print("Mean Absolute Error: ", mean_absolute_error(y_test, test_predict))
                print('RMSE = ', rmse_y_test)
                print('MSE = ', mse)
                print("R2 Score = {0:0.2%}".format(r2_score(y_test,test_predict)))
                model.save("DisModel.h5")
        else:
            model.save("DisModel.h5")
    
    DisModel(X_train,y_train)

    model = keras.models.load_model("DisModel.h5")


    test_predict = model.predict(X_test)
    train_predict = model.predict(X_train)

    #Transform to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    print(len(test_data))
    x_input=test_data[2415:].reshape(1,-1)
    print(x_input.shape)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=100 
    i=0
    while(i<30):
        
        if(len(temp_input)>100):
            print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    print(len(df2))
    
    listValues = scaler.inverse_transform(lst_output)
    dfTempt = dfTemp.reindex(pd.date_range("2021-04-10", "2021-05-10") )
 
    df3 = df2.tolist()
    df3.extend(lst_output)
    df3 = scaler.inverse_transform(df3).tolist()
    dfObj = pd.DataFrame(df3)

    #Last object in list 
    print(listValues[:-1])




    plt.plot(dfObj.mask(dfObj.apply(lambda x: x.index < 2517))[0], color = 'red')
    plt.plot(dfObj.mask(dfObj.apply(lambda x: x.index > 2517))[0], color = 'blue') 
    plt.ylabel('Closing Price')
    plt.xlabel('Days - 10 year data')
    plt.show()
        
    y = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    plt.scatter(listValues,y)
    plt.show()

def DataNord():
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
    df2 = ticker[['Close']].copy()
    dfTemp = df.index
 



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
    time = 20
    X_train, y_train = dataset(train_data,  time)
    X_test, y_test = dataset(test_data, time)

    #Reshaping the data in order to read into the model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) 

    def NordModel(X_train, y_train):
        #RNN Model using LSTM
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units = 20, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units = 20, return_sequences = True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units = 15, return_sequences = False),
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

        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 5)

        history = model.fit(
            X_train, y_train,
            epochs=25,
            batch_size=128,
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
        r2 = r2_score(y_test,test_predict)
        print("Mean Absolute Error: ", mean_absolute_error(y_test, test_predict))
        print('RMSE = ', rmse_y_test)
        print('MSE = ', mse)
        print("R2 Score = {0:0.2%}".format(r2_score(y_test,test_predict)))
        

        if r2 < .75:
            while r2 < .75:
                #Reset the states
                K.clear_session()

                #RNN Model using LSTM
                model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(units = 20, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(units = 20, return_sequences = True),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(units = 15, return_sequences = False),
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

                es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 5)

                history = model.fit(
                    X_train, y_train,
                    epochs=25,
                    batch_size=128,
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
                r2 = r2_score(y_test,test_predict)
                print("Mean Absolute Error: ", mean_absolute_error(y_test, test_predict))
                print('RMSE = ', rmse_y_test)
                print('MSE = ', mse)
                print("R2 Score = {0:0.2%}".format(r2_score(y_test,test_predict)))
                model.save("NordModel.h5")
        else:
            model.save("NordModel.h5")
    
    NordModel(X_train,y_train)

    model = keras.models.load_model("NordModel.h5")


    test_predict = model.predict(X_test)
    train_predict = model.predict(X_train)

    #Transform to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    print(len(test_data))
    x_input=test_data[2415:].reshape(1,-1)
    print(x_input.shape)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=100 
    i=0
    while(i<30):
        
        if(len(temp_input)>100):
            print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    print(len(df2))
    
    listValues = scaler.inverse_transform(lst_output)
    dfTempt = dfTemp.reindex(pd.date_range("2021-04-10", "2021-05-10") )
 
    df3 = df2.tolist()
    df3.extend(lst_output)
    df3 = scaler.inverse_transform(df3).tolist()
    dfObj = pd.DataFrame(df3)

    #Last object in list 
    print(listValues[:-1])




    plt.plot(dfObj.mask(dfObj.apply(lambda x: x.index < 2517))[0], color = 'red')
    plt.plot(dfObj.mask(dfObj.apply(lambda x: x.index > 2517))[0], color = 'blue') 
    plt.ylabel('Closing Price')
    plt.xlabel('Days - 10 year data')
    plt.show()
        
    y = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    plt.scatter(listValues,y)
    plt.show()

def DataBB():
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
    df2 = ticker[['Close']].copy()
    dfTemp = df.index
 



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

    def BBModel(X_train, y_train):
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

        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 3)

        history = model.fit(
            X_train, y_train,
            epochs=30,
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
        r2 = r2_score(y_test,test_predict)
        print("Mean Absolute Error: ", mean_absolute_error(y_test, test_predict))
        print('RMSE = ', rmse_y_test)
        print('MSE = ', mse)
        print("R2 Score = {0:0.2%}".format(r2_score(y_test,test_predict)))
        

        if r2 < .75:
            while r2 < .75:
                #Reset the states
                K.clear_session()

                #RNN Model using LSTM
                model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(units = 15, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(units = 15, return_sequences = True),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(units = 10, return_sequences = True),
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
                r2 = r2_score(y_test,test_predict)
                print("Mean Absolute Error: ", mean_absolute_error(y_test, test_predict))
                print('RMSE = ', rmse_y_test)
                print('MSE = ', mse)
                print("R2 Score = {0:0.2%}".format(r2_score(y_test,test_predict)))
                model.save("BBModel.h5")
        else:
            model.save("BBModel.h5")
    
    BBModel(X_train, y_train)

    model = keras.models.load_model("BBModel.h5")


    test_predict = model.predict(X_test)
    train_predict = model.predict(X_train)

    #Transform to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    print(len(test_data))
    x_input=test_data[2415:].reshape(1,-1)
    print(x_input.shape)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=100 
    i=0
    while(i<30):
        
        if(len(temp_input)>100):
            print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    print(len(df2))
    
    listValues = scaler.inverse_transform(lst_output)
 
    df3 = df2.tolist()
    df3.extend(lst_output)
    df3 = scaler.inverse_transform(df3).tolist()
    dfObj = pd.DataFrame(df3)

    #Last object in list 
    print(listValues[:-1])




    plt.plot(dfObj.mask(dfObj.apply(lambda x: x.index < 2517))[0], color = 'red')
    plt.plot(dfObj.mask(dfObj.apply(lambda x: x.index > 2517))[0], color = 'blue') 
    plt.ylabel('Closing Price')
    plt.xlabel('Days - 10 year data')
    plt.show()
        
    y = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    plt.yticks(y)
    plt.scatter(listValues,y)
    plt.show()


DataDis()
DataNord()
DataBB()
""" schedule.every().day.at("16:17").do(DataBB)

while True:
    schedule.run_pending()
    ti.sleep(1)
    out = input('Press f key to exit\n')
    if out:
        exit(0)
 """


