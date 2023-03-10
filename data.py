import os
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def get_data(data_path):
    train_data = pd.read_csv(os.path.join(data_path, "Train_Data_Final.csv"))
    train_data["DateTime"] = train_data["Date"] + ' ' + train_data["Time"]
    train_data['date&time'] = train_data['DateTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    train_data["timestamp"] = pd.to_datetime(train_data['date&time']).astype('int64')/10**18
    le = preprocessing.LabelEncoder()
    train_data["RH_type"] = le.fit_transform(train_data["RH_type"])
    train_data = train_data.drop(["Date", "Time", "date&time", "DateTime"], axis=1)
    return train_data, le


def get_test_data(data_path):
    test_data = pd.read_csv(os.path.join(data_path, "Test_Data.csv"))
    test_data["DateTime"] = test_data["Date"] + ' ' + test_data["Time"]
    test_data['date&time'] = test_data['DateTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    test_data["timestamp"] = pd.to_datetime(test_data['date&time']).astype('int64')/10**18
    # le = preprocessing.LabelEncoder()
    # train_data["RH_type"] = le.fit_transform(train_data["RH_type"])
    test_data = test_data.drop(["Date", "Time", "date&time", "DateTime"], axis=1)
    test_set_ids = test_data.pop("id")
    test_set = test_data.values
    return test_set, test_set_ids


def get_whole_data(train_data):
    # Analysis for train set
    train_set_ids = train_data.pop("id")

    train_set_labels = train_data.pop("RH_type")
    return train_data, train_set_labels


def get_partitioned_data(train_data):
    train_set, test_set = train_test_split(train_data, test_size=0.15, random_state=8420)

    # Analysis for train set
    train_set_ids = train_set.pop("id")
    test_set_ids = test_set.pop("id")

    train_set_labels = train_set.pop("RH_type")
    test_set_labels = test_set.pop("RH_type")
    return (train_set, train_set_labels), (test_set, test_set_labels)


def prepare_data(train_data):
    train_set, test_set = train_test_split(train_data, test_size=0.15, random_state=8420)

    # Analysis for train set
    train_set_ids = train_set.pop("id")
    test_set_ids = test_set.pop("id")

    train_set_labels = train_set.pop("RH_type")
    test_set_labels = test_set.pop("RH_type")

    X_train = train_set.values
    Y_train = train_set_labels.values
    X_test = test_set.values
    Y_test = test_set_labels.values
    return (X_train, Y_train), (X_test, Y_test)


def get_normalized_data(train_data):
    train_set, test_set = train_test_split(train_data, test_size=0.15, random_state=8420)

    # Analysis for train set
    train_set_ids = train_set.pop("id")
    test_set_ids = test_set.pop("id")

    train_set_labels = train_set.pop("RH_type")
    test_set_labels = test_set.pop("RH_type")

    train_set = (train_set - train_set.mean()) / train_set.std()
    test_set = (test_set - test_set.mean()) / test_set.std()
    X_train = train_set.values
    Y_train = train_set_labels.values
    X_test = test_set.values
    Y_test = test_set_labels.values
    return (X_train, Y_train), (X_test, Y_test)


if __name__ == "__main__":
    path = "data"
    data, le = get_data(path)
    train, test = prepare_data(data)
