from data import *
from models import lightgbm_model_final
from explainability import explain_lime, explain_shap
from optimization import optimization_pipeline


def main_pipeline(data_path_, do_optimize=False):
    (X_train, Y_train), (X_test, Y_test) = get_final_data(data_path_)
    parameters = {}
    if do_optimize:
        parameters = optimization_pipeline(path)
    clf = lightgbm_model_final(X_train, Y_train, X_test, Y_test, parameters)
    predict_fn = lambda x: clf.predict_proba(x).astype(float)
    idx = 15
    explain_lime(X_train, predict_fn, idx)
    explain_shap(clf, X_test)


if __name__ == "__main__":
    data_path = "data"
    main_pipeline(data_path)
