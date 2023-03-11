from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from data import *
from preprocessing import remove_outlier, outlier_bound, engineered_data
from bayes_opt import BayesianOptimization
import time
import lightgbm as lgb


def optimize(acc_score, X, Y):
    # Run Bayesian Optimization
    start = time.time()

    def gbm_cl_bo(max_depth, learning_rate, n_estimators, subsample, num_leaves, max_bin):
        params_gbm = {'max_depth': round(max_depth), 'learning_rate': learning_rate,
                      'n_estimators': round(n_estimators), 'subsample': subsample, 'num_leaves': round(num_leaves),
                      'max_bin': round(max_bin)}
        # params_gbm['bagging_freq'] = round(bagging_freq)
        scores = cross_val_score(lgb.LGBMClassifier(random_state=123, **params_gbm),
                                 X, Y, scoring=acc_score, cv=5).mean()
        score = scores.mean()
        return score

    params_gbm = {
        'max_depth': (6, 10),
        # 'max_features':(0.8, 1),
        'learning_rate': (0.09, 0.25),
        'n_estimators': (100, 150),
        'subsample': (0.93, 1),
        'num_leaves': (36, 54),
        'max_bin': (300, 400),
        # 'bagging_freq': (35, 45)
    }
    gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm, random_state=111)
    gbm_bo.maximize(init_points=40, n_iter=10)
    print('It takes %s minutes' % ((time.time() - start) / 60))
    params_gbm = gbm_bo.max['params']
    params_gbm['max_depth'] = round(params_gbm['max_depth'])
    params_gbm['n_estimators'] = round(params_gbm['n_estimators'])
    params_gbm['max_bin'] = round(params_gbm['max_bin'])
    params_gbm['num_leaves'] = round(params_gbm['num_leaves'])
    return params_gbm


def optimization_pipeline(data_path):
    # idx = 15
    data_, le_ = get_data(data_path)
    # (X_train, Y_train), (X_test, Y_test) = get_partitioned_data(data_)
    # cols = X_train.columns
    lb, ub = outlier_bound(data_, col='T')
    data_ = remove_outlier(data_, ub, lb)
    (X_train, Y_train), (X_test, Y_test) = get_partitioned_data(data_)
    X_train = engineered_data(X_train)
    X_test = engineered_data(X_test)
    acc_score = make_scorer(accuracy_score)
    params = optimize(acc_score, X_train, Y_train)
    return params


if __name__ == "__main__":
    path = "data"
    parameters = optimization_pipeline(path)
