#install following libraries and import them as shown
import investpy
import math
import pandas as pd
import pandas_datareader as pdd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

#fetching dataset using investpy
gethistory = investpy.get_stock_historical_data(stock='SAIL',
                                        country='India',
                                        from_date='01/01/2010',
                                        to_date='31/12/2020')
#saving dataset in .csv format
gethistory.to_csv('SAIL.csv')

#getting .csv file
df = pd.read_csv('SAIL.csv')

#plotting dataset
plt.figure(figsize=(16,8))
plt.title('Close Price Movement')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price in Rs',fontsize=18)
plt.show

#filtering data set
data = df.filter(['Close'])
dataset = data.values
len(dataset)
#90% data to train as asked for 1 year data prediction out of 10 years data
training_data_size = math.ceil(len(dataset)*.9)
training_data_size

#Min-Max Scaler 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

#train data
train_data = scaled_data[0:training_data_size, :]
x_train = []
y_train = []
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])
  if i<=61:
    print(x_train)
    print(y_train)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))
x_train.shape

#LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=1)

test_data = scaled_data[training_data_size - 60: ,:]
x_test = []
y_test = dataset[training_data_size:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#root mean squared value for prediction
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse
train = data[:training_data_size]
valid = data[training_data_size:]
valid['predictions'] = predictions

#plot final Chart to compare Prediction and actual data
plt.figure(figsize=(16,8))
plt.title('Model LM')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','predictions']])
plt.legend(['Train','val','predictions'],loc='upper right')
plt.show