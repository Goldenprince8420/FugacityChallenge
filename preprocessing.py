import numpy as np
from sklearn.model_selection import train_test_split


def remove_outlier(frame, upper_bound, lower_bound):
    frame = frame.drop(upper_bound[0])
    frame = frame.drop(lower_bound[0])
    return frame


def outlier_bound(frame, col):
    # finding the 1st quartile
    q1 = np.quantile(frame[col], 0.25)

    # finding the 3rd quartile
    q3 = np.quantile(frame[col], 0.75)

    med = np.median(frame[col])

    # finding the iqr region
    iqr = q3 - q1

    # finding upper and lower whiskers
    upper_bound = np.where(frame[col] >= q3 + (1.5 * iqr))
    lower_bound = np.where(frame[col] <= q1 - (1.5 * iqr))
    return upper_bound, lower_bound


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


def engineered_data(frame):
    # T engineering
    T_coeff = 100 * np.exp((7.5 * frame['T'])) / (237.7 + frame['T'])
    frame["T_coeff"] = T_coeff
    # AH Engineering
    AH_coeff = frame['AH'] * 461.5 * frame['T'] / 1
    frame["AH_coeff"] = AH_coeff
    # print(frame)
    custom_coeff = np.exp(-1 * frame["CO(GT)"]) + 0.8 ** (frame["PT08.S4(NO2)"] + frame["PT08.S1(CO)"] +
                                                          frame["PT08.S2(NMHC)"] + frame["PT08.S3(NOx)"] +
                                                          frame["PT08.S5(O3)"]) + 1.3 * (frame["NOx(GT)"] +
                                                                                         frame["NO2(GT)"])
    frame["Custom_coeff"] = custom_coeff
    # frame_cols = frame.columns
    # scaler = StandardScaler()
    # frame[frame_cols] = scaler.fit_transform(frame)
    return frame
