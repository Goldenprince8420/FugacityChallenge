from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import pandas as pd
import pprint
from time import time


def model_evaluate(Y_train, Y_pred, set_name="Train"):
    print(set_name + ' Accuracy:\t{:0.1f}%'.format(accuracy_score(Y_train, Y_pred) * 100))

    # classification report
    print('\n')
    print(classification_report(Y_train, Y_pred))

    # confusion matrix
    confmat = confusion_matrix(Y_train, Y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()


# Reporting util for different optimizers
def report_perf(optimizer, X, y, title="model", callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers

    optimizer = a sklearn or a skopt optimizer
    X = the training set
    y = our target
    title = a string label for the experiment
    """
    start = time()

    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)

    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           + u"\u00B1" + " %.3f") % (time() - start,
                                     len(optimizer.cv_results_['params']),
                                     best_score,
                                     best_score_std))
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params
