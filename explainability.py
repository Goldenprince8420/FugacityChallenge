from lime import lime_tabular
from data import *
import shap
import lightgbm as lgb
from utils import model_evaluate


def lightgbm_model_final(X_train, Y_train, X_test, Y_test):
    clf_lgbm = lgb.LGBMClassifier(
        boosting_type='dart',
        objective="multiclass",
        num_iterations=200,
        # learning_rate=0.12,
        num_leaves=48,
        max_bin=300,
        bagging_freq=40,
        tree_learner="voting",
    )
    # create a dictionary of all values we want to test for parameters
    # params_lgbm = {}
    # # use gridsearch to test all values for parameters
    # clf_lgbm = GridSearchCV(clf_lgbm, params_lgbm, cv=5)
    clf_lgbm.fit(X_train, Y_train)

    Y_pred = clf_lgbm.predict(X_train)
    model_evaluate(Y_train, Y_pred, set_name="Train")
    Y_pred = clf_lgbm.predict(X_test)
    model_evaluate(Y_test, Y_pred, set_name="Test")
    return clf_lgbm

if __name__ == "__main__":
    idx = 15
    path = "data"
    data = get_data(path)
    # train, test = prepare_data(data)
    # X_train, Y_train = train
    # X_test, Y_test = test
    (X_train, Y_train), (X_test, Y_test) = get_partitioned_data(data)
    explainer = lime_tabular.LimeTabularExplainer(X_train.values,
                                                feature_names=X_train.columns,
                                                class_names=[0, 1, 2, 3, 4], kernel_width=3)

    clf = lightgbm_model_final(X_train, Y_train, X_test, Y_test)
    predict_fn = lambda x: clf.predict_proba(x).astype(float)
    # Choose a local instance
    chosen = X_test.iloc[idx]
    # print(chosen)
    exp = explainer.explain_instance(chosen, predict_fn, num_features=5)
    exp.save_to_file(file_path="./data/results/explanation.html")


    shap_values = shap.TreeExplainer(clf).shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    shap.dependence_plot("T", shap_values[0], X_test)
