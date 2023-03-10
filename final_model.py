from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf

from utils import *
from data import *
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

path = "data"
data = get_data(path)
(X_train, Y_train), (X_test, Y_test) = get_partitioned_data(data)
# X_train = X_train.astype(np.float32)
# Y_train = Y_train.astype(np.float32)


clf = TabNetClassifier()  #TabNetRegressor()
clf.fit(
  X_train, Y_train,
  eval_set=[(X_test, Y_test)]
)
preds = clf.predict(X_test)

