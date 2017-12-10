# define imports and configurations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas
import numpy as np
np.set_printoptions(threshold=np.nan)
from pomegranate import *
pandas.options.display.max_rows = 500
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
# pomegranate.utils.enable_gpu()


def load_data(year="2017", airport=None):
    data = pandas.read_csv("./data/processed/" + year + ".csv")
    # define delay as DepDelay > 30 or ArrDelay>30 -- which implies traveler anxious at airport or miss connection flight
    data["Delay"] = ((data["DepDelay"] > 30) | (
        data["ArrDelay"] > 30) | data['Cancelled']) * 1

    # discretize ArrTime, DepTime, AirTime,Distance
    # ArrTime, DepTime are divided by 100, because 22:30 is sored as 2230.
    # AirTime is divided into hours
    # Distance is divided by 250
    data = data.sort_values(['Year', 'Month', 'DayofMonth', 'CRSDepTime'], ascending=[
                            True, True, True, True])
    data["ArrTime"] = data["ArrTime"] // 100
    data["DepTime"] = data["DepTime"] // 100
    data["AirTime"] = data["AirTime"] // 60
    data["Distance"] = data["Distance"] // 250
    label_set = pandas.factorize(data["UniqueCarrier"])
    data["UniqueCarrier"] = label_set[0]
    # for keep data from certain airport

    if airport:
        data = data.query("Origin==\'" + airport + "\'")
        # data = data[data["Origin"]=="IAH"]
    return data

# utility for model scoring
# the model has two states, 0 or 1. The state 0 and 1 may mean different things for each run, so it may need to be flipped.


def score_model(model, data, truth, flip=False):
    y_pred = model.predict(data)[:, 3]
    print(y_pred.shape)
    print(truth.shape)
    print("F1 score : %f" % f1_score(y_pred, truth),
          "Accuracy : %f" % accuracy_score(y_pred, truth),
          "Recall : %f" % recall_score(y_pred, truth),
          "Precision : %f" % precision_score(y_pred, truth))
# score the model given validation serires


def score_model_given_series(model, validate_x, validate_y):
    # is 0 delay or is 1 delay
    zero_score = score_model(model, validate_x[0], validate_y[0])
    one_score = score_model(model, validate_x[0], validate_y[0], True)
    flip = None
#     if zero_score > one_score:
#         flip = False
#     else:
#         flip = True
    sum_score = 0
    for i in range(len(validate_x)):
        score = score_model(model, validate_x[i], validate_y[i], flip)
        sum_score += score
    print(sum_score / len(validate_x))
# splice_series to day


def splice_series(input_data, split=0.8):
    # split is the split ratio for train vs validate
    x = []
    y = []
    for year in input_data.Year.unique():
        for month in input_data.Month.unique():
            for day in input_data.DayofMonth.unique():
                hold = input_data[(input_data.Month == month) & (
                    input_data.DayofMonth == day) & (input_data.Year == year)]
                hold = hold.sort_values(axis=0, by="CRSDepTime")[features]
                if len(hold > 0):
                    x.append(hold.loc[:, hold.columns !=
                                      'Delay'].as_matrix().astype(int).tolist())
                    y.append(hold.loc[:, hold.columns ==
                                      'Delay'].as_matrix().astype(int).flatten())
    x = np.array(x)
    y = np.array(y)
    indexer = np.random.permutation(x.shape[0])
    x = x[indexer]
    y = y[indexer]
    train_x = x[:int(x.shape[0] * split)]
    train_y = y[:int(x.shape[0] * split)]
    validate_x = x[int(x.shape[0] * split):]
    validate_y = y[int(x.shape[0] * split):]
    return train_x, train_y, validate_x, validate_y
# Train BayesianNetwork and Treat Each Flight as Independent Flight


def BN_preparation(input_data):
    #     input_data = input_data[['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier',
    #        'FlightNum', 'OriginAirportID', 'DestAirportID',
    #        'AirTime', 'Distance',
    #        'CRSDepTime', 'Delay']]
    #     input_data["ArrTime"] = input_data["ArrTime"]//100
    #     input_data["DepTime"].apply(lambda x: x//100)
    input_data["AirTime"].apply(lambda x: x // 60)
    return input_data


data = load_data("2017", "ATL")


BN_data = data[["OriginAirportID", "CRSDepTime",
                "UniqueCarrier", "Delay", "FlightNum"]]
train = BN_data.as_matrix()
test_x = BN_data[int(BN_data.shape[0] * 0.8):]
test_y = BN_data[int(BN_data.shape[0] * 0.8):]
test_x.Delay = None
test_x = test_x.as_matrix()

model = BayesianNetwork.from_samples(BN_data, algorithm='chow-liu', n_jobs=-1)
model.fit(BN_data.as_matrix())
model.bake()
print("Training Done")
y_pred = model.predict(test_x)[:, 3]
# print("Y_pred",y_pred)
truth = test_y.Delay*1
truth = truth.as_matrix().tolist()
# print("Truth",truth)


print("F1 score : %f" % f1_score(y_pred, truth),
      "Accuracy : %f" % accuracy_score(y_pred, truth),
      "Recall : %f" % recall_score(y_pred, truth),
      "Precision : %f" % precision_score(y_pred, truth))
# score_model(model, test_x[:100], test_y[:100].Delay)
#score_model(model.predict(test_x), test_y)
