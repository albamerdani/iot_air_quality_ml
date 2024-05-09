import datetime

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import os
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


path_pm2 = os.path.join("../../dataset", "matje_pm2.csv")
path_pm10 = os.path.join("../../dataset", "matje_pm10.csv")
path_temp = os.path.join("../../dataset", "matje_temp.csv")
path_hum = os.path.join("../../dataset", "matje_hum.csv")
path_pressure = os.path.join("../../dataset", "matje_pressure.csv")


# transform a time series dataset into a supervised learning dataset
def series_to_supervised2(data, n_in=2, n_out=1, dropnan=True):
    n_vars = 2 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = random_forest_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions


def predict_pm25(num_prediction: int):
    print("Dust with pm2 Prediction Time - " + str(datetime.datetime.now()))
    time1 = datetime.datetime.now()
    #load the dataset
    series_pm2 = read_csv(path_pm2, header=1, index_col=0)
    values_pm2 = series_pm2.values
    # transform the time series data into supervised learning
    data_pm2 = series_to_supervised2(values_pm2, 1, 2)
    # evaluate
    mae, y, yhat = walk_forward_validation(data_pm2, num_prediction)
    print('MAE: %.3f' % mae)
    time2 = datetime.datetime.now()
    diff = time2 - time1
    print(diff)
    print("Dust with pm2 Prediction Time - " + str(datetime.datetime.now()))
    # plot expected vs predicted
    pyplot.plot(y, label='Expected')
    pyplot.plot(yhat, label='Predicted')
    pyplot.legend()
    pyplot.show()


def predict_pm10(num_prediction: int):
    print("Dust with pm10 Prediction Time - " + str(datetime.datetime.now()))
    time1 = datetime.datetime.now()
    series_pm10 = read_csv(path_pm10, header=1, index_col=0)
    values_pm10 = series_pm10.values
    data_pm10 = series_to_supervised2(values_pm10, 1, 2)
    mae_pm10, y_pm10, yhat_pm10 = walk_forward_validation(data_pm10, num_prediction)
    print('MAE: %.3f' % mae_pm10)
    time2 = datetime.datetime.now()
    diff = time2 - time1
    print(diff)
    print("Dust with pm10 Prediction Time - " + str(datetime.datetime.now()))
    pyplot.plot(y_pm10, label='Expected')
    pyplot.plot(yhat_pm10, label='Predicted')
    pyplot.legend()
    pyplot.show()


def predict_temp(num_prediction: int):
    print("Temperature Prediction Time - " + str(datetime.datetime.now()))
    time1 = datetime.datetime.now()
    series_temp = read_csv(path_temp, header=1, index_col=0)
    values_temp = series_temp.values
    data_temp = series_to_supervised2(values_temp, 1, 2)
    mae_temp, y_temp, yhat_temp = walk_forward_validation(data_temp, num_prediction)
    print('MAE: %.3f' % mae_temp)
    #accuracy = accuracy_score(y_temp, yhat_temp)
    #print('Accuracy Score: %.3f' % accuracy)
    time2 = datetime.datetime.now()
    diff = time2 - time1
    print(diff)
    print("Temperature Prediction Time - " + str(datetime.datetime.now()))
    pyplot.plot(y_temp, label='Expected')
    pyplot.plot(yhat_temp, label='Predicted')
    pyplot.legend()
    pyplot.show()


def predict_humidity(num_prediction: int):
    print("Humidity Prediction Time - " + str(datetime.datetime.now()))
    time1 = datetime.datetime.now()
    series_hum = read_csv(path_hum, header=1, index_col=0)
    values_hum = series_hum.values
    data_hum = series_to_supervised2(values_hum, 1, 2)
    mae_hum, y_hum, yhat_hum = walk_forward_validation(data_hum, num_prediction)
    print('MAE: %.3f' % mae_hum)
    time2 = datetime.datetime.now()
    diff = time2 - time1
    print(diff)
    print("Humidity Prediction Time - " + str(datetime.datetime.now()))
    pyplot.plot(y_hum, label='Expected')
    pyplot.plot(yhat_hum, label='Predicted')
    pyplot.legend()
    pyplot.show()


def predict_pressure(num_prediction: int):
    print("Pressure Prediction. Time - " + str(datetime.datetime.now()))
    time1 = datetime.datetime.now()
    series_pressure = read_csv(path_pressure, header=1, index_col=0)
    values_pressure = series_pressure.values
    data_pressure = series_to_supervised2(values_pressure, 1, 2)
    mae_pressure, y_pressure, yhat_pressure = walk_forward_validation(data_pressure, num_prediction)
    print('MAE: %.3f' % mae_pressure)
    time2 = datetime.datetime.now()
    diff = time2 - time1
    print(diff)
    print("Pressure Prediction. Time - " + str(datetime.datetime.now()))
    pyplot.plot(y_pressure, label='Expected')
    pyplot.plot(yhat_pressure, label='Predicted')
    pyplot.legend()
    pyplot.show()


predict_temp(10)
predict_pm25(30)
predict_pm10(20)

