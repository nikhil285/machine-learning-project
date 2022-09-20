import pandas
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(2)

print("Creating model...")
batch_size = 1
look_back = 5
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print("Model creation complete!")

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# load the dataset
print("Loading data...")
dataframe = read_csv('data.txt', usecols=[2], engine='python', delimiter=',')
print("Data loading complete!")

number_of_predictions = str(int(input("How many predictions would you like?: "))+look_back+1)
if(number_of_predictions == 7):
    print("Graph of singular data point prediction not available, but training/test graph will be displayed.")
# normalize the dataset
print("Normalizing data...")
dataset = dataframe.values
dataset = dataframe.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset)*0.80)
train = dataset[:train_size]
test = dataset[train_size:len(dataset)]
# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

dataset = dataframe.values
dataset = dataframe.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
prediction_data = dataset[:(int(number_of_predictions))]
prediction_setX, prediction_setY = create_dataset(prediction_data, look_back)
# reshape input to be [samples, time steps, features]
prediction_setX = numpy.reshape(prediction_setX, (prediction_setX.shape[0], prediction_setX.shape[1], 1))
print("Data normalization complete!")

print("Beginning training...")
# create and fit the LSTM network. Uses training set
model.fit(trainX, trainY, epochs=1000, batch_size=batch_size, verbose=2)
print("Training completed. Moving on to testing...")
model.reset_states()
train_predict = model.predict(trainX, batch_size=batch_size)
train_predict = scaler.inverse_transform(train_predict)
trainY = scaler.inverse_transform([trainY])
train_score = math.sqrt(mean_squared_error(trainY[0], train_predict[:,0]))

model.reset_states()
#Test LSTM network using test set
test_predict = model.predict(testX, batch_size=batch_size)
# invert test predictions
test_predict = scaler.inverse_transform(test_predict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
test_score = math.sqrt(mean_squared_error(testY[0], test_predict[:,0]))

print("Testing completed. Moving on to predictions and validation...")
model.reset_states()
# make predictions, number of predictions determined by user input earlier. This is the validation set
prediction = model.predict(prediction_setX, batch_size=batch_size)
prediction = scaler.inverse_transform(prediction)
print("Prediction(s): " + str(prediction))
# calculate root mean squared error
prediction_setY = scaler.inverse_transform([prediction_setY])
prediction_score = math.sqrt(mean_squared_error(prediction_setY[0], prediction[:,0]))
print('Train Score: %.2f RMSE' % (train_score))
print('Test Score: %.2f RMSE' % (test_score))
print('Prediction(s) Score: %.2f RMSE' % (prediction_score))

def percent_error(prediction, prediction_setY):
    values = []
    experimental_list = []
    accepted_list = []
    avg = 0
    for r in prediction:
        for c in r:
            experimental_list.append(c)
    for r in prediction_setY:
        for c in r:
            accepted_list.append(c)
    for x in range(len(prediction)):
        accepted = accepted_list[x]
        experimental = experimental_list[x]
        values.append((abs(accepted-experimental)/(accepted))*100)
    for i in range(len(values)):
        avg += values[i]
    avg = avg/len(values)
    accuracy = 100-avg
    print("The average percent error of the predictions made is: " + str(avg))
    print("The accuracy of the model is: " + str(accuracy))

percent_error(prediction, prediction_setY)

print('Predictions complete!')
print("Beginning plotting step...")
predict_plot = numpy.empty_like(dataset)
predict_plot[:, :] = numpy.nan
predict_plot[look_back:len(prediction)+look_back, :] = prediction
# shift test predictions for plotting
# plot baseline and predictions
prediction_setX = scaler.inverse_transform(prediction_setX[:,0])
n = numpy.arange(0, len(prediction))
plt.plot(n, prediction_setX, '-r')
plt.plot(n, prediction)
plt.show()

train_predict_plot = numpy.empty_like(dataset)
train_predict_plot[:, :] = numpy.nan
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
test_predict_plot = numpy.empty_like(dataset)
test_predict_plot[:, :] = numpy.nan
test_predict_plot[len(train_predict)+(look_back*2)+1:len(dataset)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()

print("Process Complete with " + str(int(number_of_predictions)-(look_back+1)) + " predictions made.")