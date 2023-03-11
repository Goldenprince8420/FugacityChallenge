from lime import lime_tabular
from data import *
from models import lightgbm_model_final
import shap

from preprocessing import engineered_data


def explain_lime(frame, fn, index):
    chosen = frame.iloc[index]
    explainer = lime_tabular.LimeTabularExplainer(frame.values,
                                                  feature_names=frame.columns,
                                                  class_names=[0, 1, 2, 3, 4], kernel_width=3)
    exp = explainer.explain_instance(chosen, fn, num_features=5)
    exp.save_to_file(file_path="./data/results/explanation.html")


def explain_shap(classifier, test_data):
    shap_values = shap.TreeExplainer(classifier).shap_values(test_data)
    shap.summary_plot(shap_values, test_data)
    shap.dependence_plot("T", shap_values[0], test_data)


if __name__ == "__main__":
    idx = 15
    path = "data"
    data, le = get_data(path)

    # train, test = prepare_data(data)
    # X_train, Y_train = train
    # X_test, Y_test = test
    (X_train, Y_train), (X_test, Y_test) = get_partitioned_data(data)
    cols = X_train.columns
    # plot_outliers(frame=X_train, cols=cols, com_col=Y_train)
    # X_train = engineered_data(X_train)
    # X_test = engineered_data(X_test)
    clf = lightgbm_model_final(X_train, Y_train, X_test, Y_test)
    predict_fn = lambda x: clf.predict_proba(x).astype(float)
    idx = 15
    explain_lime(X_train, predict_fn, idx)
    explain_shap(clf, X_test)
