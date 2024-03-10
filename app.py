from datetime import date, timedelta
from nsepy import get_history
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yahooFinance

import streamlit as st

#TITLE 
st.title('STOCK MARKET DATA ANALSYIS USING LSTM')

#EXTRACTING DATA
end = date.today() - timedelta(days=1)
start = end - timedelta(days=365*10)  # 10 years before the end date
symbol = st.text_input('Enter stock ticker', 'MRF.NS')
GetStockInformation = yahooFinance.Ticker(symbol)
df = GetStockInformation.history(start=start, end=end)

#DISPLAYING THE DESCRIPTIVE STATISTICS OF DATA
st.subheader('Data from 10 years ago to yesterday')
st.write(df.describe())

#VISUALIZATION OF CLOSE PRICE
st.subheader('CLOSING PRICE')
fig = plt.figure(figsize=(12,8))
plt.plot(df.Close)
st.pyplot(fig)

#SCALING THE DATA
df1 = df.reset_index()['Close']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

#SPLITTING THE DATA
training_size = int(len(df1) * 0.70)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size,:], df1[training_size:len(df1),:1]

#LOADING THE MODEL
import tensorflow
model = tensorflow.keras.models.load_model('model_h5.h5')

#FUNCTION TO CREATE DATASET
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 150
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

#TESTING THE DATA
Y = ytest.reshape(-1, 1)
y_test = scaler.inverse_transform(Y)

#VISUALIZING THE ACTUAL VALUES VS PREDICTED VALUES
st.subheader('ACTUAL VALUES vs PREDICTED VALUES')
fig = plt.figure(figsize=(12,8))
plt.plot(y_test)
plt.plot(test_predict,color="red")
st.pyplot(fig)

#PREDICTING THE NEXT 30 DAYS CLOSING PRICE
x_input = test_data[-150:].reshape(1,-1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps = 150
i = 0
while(i < 30):
    
    if(len(temp_input) > 150):
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i += 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i += 1

forecasted_vals = scaler.inverse_transform(lst_output)

#VISUALIZING THE FORECASTED VALUES
st.subheader('FORECASTING CLOSE PRICE FOR NEXT 30 DAYS')
fig = plt.figure(figsize=(12,8))
day_new = np.arange(1,151)
day_pred = np.arange(151,181)
plt.plot(day_new, scaler.inverse_transform(df1[-150:]))
plt.plot(day_pred, forecasted_vals, color="red")
st.pyplot(fig)

#DISPLAYING THE FORECASTED VALUES
st.subheader('FORECASTED VALUES')
date_range = pd.date_range(end + timedelta(days=1), end + timedelta(days=30))
forecast_df = pd.DataFrame({'Date': date_range, 'Close Price': forecasted_vals.flatten()})
st.write(forecast_df)
