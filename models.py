from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from supervised.automl import AutoML
from supervised.preprocessing.eda import EDA

from data import *
from sklearn.model_selection import GridSearchCV
from utils import *


def logistic_regression(X_train, Y_train, X_test, Y_test):
    clf_logr = LogisticRegression()
    # create a dictionary of all values we want to test for parameters
    params_logr = {}
    # use gridsearch to test all values for parameters
    clf_logr = GridSearchCV(clf_logr, params_logr, cv=5)
    clf_logr.fit(X_train, Y_train)

    Y_pred = clf_logr.predict(X_train)
    model_evaluate(Y_train, Y_pred, set_name="Train")
    Y_pred = clf_logr.predict(X_test)
    model_evaluate(Y_test, Y_pred, set_name="Test")
    return clf_logr


def decision_tree(X_train, Y_train, X_test, Y_test):
    clf_dt = DecisionTreeClassifier()
    # create a dictionary of all values we want to test for parameters
    params_dt = {}
    # use gridsearch to test all values for parameters
    clf_dt = GridSearchCV(clf_dt, params_dt, cv=5)
    clf_dt.fit(X_train, Y_train)

    Y_pred = clf_dt.predict(X_train)
    model_evaluate(Y_train, Y_pred, set_name="Train")
    Y_pred = clf_dt.predict(X_test)
    model_evaluate(Y_test, Y_pred, set_name="Test")
    return clf_dt


def support_vector_machine(X_train, Y_train, X_test, Y_test):
    clf_svc = SVC()
    # create a dictionary of all values we want to test for parameters
    params_svc = {}
    # use gridsearch to test all values for parameters
    clf_svc = GridSearchCV(clf_svc, params_svc, cv=5)
    clf_svc.fit(X_train, Y_train)

    Y_pred = clf_svc.predict(X_train)
    model_evaluate(Y_train, Y_pred, set_name="Train")
    Y_pred = clf_svc.predict(X_test)
    model_evaluate(Y_test, Y_pred, set_name="Test")
    return clf_svc


def random_forest(X_train, Y_train, X_test, Y_test):
    clf_rf = RandomForestClassifier()
    # create a dictionary of all values we want to test for parameters
    params_rf = {}
    # use gridsearch to test all values for parameters
    clf_rf = GridSearchCV(clf_rf, params_rf, cv=5)
    clf_rf.fit(X_train, Y_train)

    Y_pred = clf_rf.predict(X_train)
    model_evaluate(Y_train, Y_pred, set_name="Train")
    Y_pred = clf_rf.predict(X_test)
    model_evaluate(Y_test, Y_pred, set_name="Test")
    return clf_rf


def xgboost(X_train, Y_train, X_test, Y_test):
    clf_xgb = XGBClassifier(
        booster="dart",
        eta=0.1,
        max_depth=8,
        sampling_method="gradient_based",
        max_bin=300,
        normalize_type="forest",
        sample_type="weighted",
        objective="multi:softmax",

    )
    # create a dictionary of all values we want to test for parameters
    params_xgb = {}
    # use gridsearch to test all values for parameters
    clf_xgb = GridSearchCV(clf_xgb, params_xgb, cv=5)
    clf_xgb.fit(X_train, Y_train)

    Y_pred = clf_xgb.predict(X_train)
    model_evaluate(Y_train, Y_pred, set_name="Train")
    Y_pred = clf_xgb.predict(X_test)
    model_evaluate(Y_test, Y_pred, set_name="Test")
    return clf_xgb


def catboost(X_train, Y_train, X_test, Y_test):
    clf_cat = CatBoostClassifier(iterations=100)
    # create a dictionary of all values we want to test for parameters
    params_cat = {}
    # use gridsearch to test all values for parameters
    clf_cat = GridSearchCV(clf_cat, params_cat, cv=5)
    clf_cat.fit(X_train, Y_train)

    Y_pred = clf_cat.predict(X_train)
    model_evaluate(Y_train, Y_pred, set_name="Train")
    Y_pred = clf_cat.predict(X_test)
    model_evaluate(Y_test, Y_pred, set_name="Test")
    return clf_cat


def lightgbm(X_train, Y_train, X_test, Y_test):
    clf_lgbm = lgb.LGBMClassifier(
        boosting_type='dart',
        objective="multiclass",
        num_iterations=200,
        # learning_rate=0.07,
        num_leaves=48,
        max_bin=300,
        bagging_freq=40,
        tree_learner="voting",
    )
    # create a dictionary of all values we want to test for parameters
    params_lgbm = {}
    # use gridsearch to test all values for parameters
    clf_lgbm = GridSearchCV(clf_lgbm, params_lgbm, cv=5)
    clf_lgbm.fit(X_train, Y_train)

    Y_pred = clf_lgbm.predict(X_train)
    model_evaluate(Y_train, Y_pred, set_name="Train")
    Y_pred = clf_lgbm.predict(X_test)
    model_evaluate(Y_test, Y_pred, set_name="Test")
    return clf_lgbm


def ensemble_model(estimators, X_train, Y_train, X_test, Y_test):
    # create our voting classifier, inputting our models
    clf_ensemble = VotingClassifier(estimators, voting="hard")
    clf_ensemble.fit(X_train, Y_train)
    Y_pred = clf_ensemble.predict(X_train)
    model_evaluate(Y_train, Y_pred, set_name="Train")
    Y_pred = clf_ensemble.predict(X_test)
    model_evaluate(Y_test, Y_pred, set_name="Test")
    return clf_ensemble


def automl_model(X_train, Y_train, X_test, Y_test):
    clf_automl = AutoML(total_time_limit=10)
    clf_automl.fit(X_train, Y_train)

    Y_pred = clf_automl.predict(X_train)
    model_evaluate(Y_train, Y_pred, set_name="Train")
    Y_pred = clf_automl.predict(X_test)
    model_evaluate(Y_test, Y_pred, set_name="Test")
    return clf_automl


def automl_exploration(data):
    EDA.extensive_eda(data, data["RH_type"], save_path="./data/results")


if __name__ == "__main__":
    path = "data"
    data = get_data(path)
    train, test = prepare_data(data)
    X_train, Y_train = train
    X_test, Y_test = test

    # logr_classifier = logistic_regression(X_train, Y_train, X_test, Y_test)
    # dt_classifier = decision_tree(X_train, Y_train, X_test, Y_test)
    # svc_classifier = support_vector_machine(X_train, Y_train, X_test, Y_test)
    # rf_classifier = random_forest(X_train, Y_train, X_test, Y_test)
    xgb_classifier = xgboost(X_train, Y_train, X_test, Y_test)
    # cat_classifier = catboost(X_train, Y_train, X_test, Y_test)
    lgbm_classifier = lightgbm(X_train, Y_train, X_test, Y_test)

    xgb = xgb_classifier.best_estimator_
    # cat = cat_classifier.best_estimator_
    lgbm = lgbm_classifier.best_estimator_
    #
    # # create a dictionary of our models
    estimators = [("xgb", xgb), ("lgbm", lgbm)]

    ensemble_classifier = ensemble_model(estimators, X_train, Y_train, X_test, Y_test)

    # automl_exploration(data)
    # automl_classifier = automl_model(X_train, Y_train, X_test, Y_test)
