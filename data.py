import os

import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from preprocessing import engineered_data, outlier_bound, remove_outlier
from utils import plot_outliers
from models import lightgbm_model_final


def get_data(data_path):
    train_data = pd.read_csv(os.path.join(data_path, "Train_Data_Final.csv"))
    train_data["DateTime"] = train_data["Date"] + ' ' + train_data["Time"]
    train_data['date&time'] = train_data['DateTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    train_data["timestamp"] = pd.to_datetime(train_data['date&time']).astype('int64') / 10 ** 18
    le = preprocessing.LabelEncoder()
    train_data["RH_type"] = le.fit_transform(train_data["RH_type"])
    train_data = train_data.drop(["Date", "Time", "date&time", "DateTime"], axis=1)
    return train_data, le


def get_test_data(data_path):
    test_data = pd.read_csv(os.path.join(data_path, "Test_Data.csv"))
    test_data["DateTime"] = test_data["Date"] + ' ' + test_data["Time"]
    test_data['date&time'] = test_data['DateTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    test_data["timestamp"] = pd.to_datetime(test_data['date&time']).astype('int64') / 10 ** 18
    # le = preprocessing.LabelEncoder()
    # train_data["RH_type"] = le.fit_transform(train_data["RH_type"])
    test_data = test_data.drop(["Date", "Time", "date&time", "DateTime", "index"], axis=1)
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
    # print(train_set.head())

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


def get_final_data(data_path, do_remove_outlier=False, do_feature_engineering=False):
    data_, le_ = get_data(data_path)
    cols = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
            'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
            'T', 'AH', 'timestamp']
    if do_remove_outlier:
        lb, ub = outlier_bound(data_, col='T')
        data_ = remove_outlier(data_, ub, lb)

    (X_train, Y_train), (X_test, Y_test) = get_partitioned_data(data_)
    if do_feature_engineering:
        X_train = engineered_data(X_train)
        X_test = engineered_data(X_test)
    if do_remove_outlier:
        plot_outliers(frame=X_train, cols=cols, com_col=Y_train)
    return (X_train, Y_train), (X_test, Y_test)


if __name__ == "__main__":
    path = "data"
    data, le = get_data(path)
    # train, test = prepare_data(data)
    train, test = get_partitioned_data(data)
    train_new = engineered_data(train[0])
