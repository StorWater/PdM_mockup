import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def metrics_threshold(y_score, y, thres_list=np.linspace(0, 1, 101)):
    """[summary]

    TODO: documentation

    Parameters
    ----------
    y_score : [type]
        [description]
    y : [type]
        [description]
    thres_list : [type], optional
        [description], by default np.linspace(0, 1, 101)

    Returns
    -------
    [type]
        [description]
    """
    tp, tn, fp, fn = [], [], [], []

    for thres in thres_list:

        # Predictions depending on threshold
        y_pred = np.where(y_score > thres, 1, 0)

        # Confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)

        # Save: Cij: known i, predicted j class
        tn.append(conf_matrix[0, 0])
        tp.append(conf_matrix[1, 1])
        fp.append(conf_matrix[0, 1])
        fn.append(conf_matrix[1, 0])

    # Save metrics
    tp = np.array(tp)
    tn = np.array(tn)
    fp = np.array(fp)
    fn = np.array(fn)

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    fpr = fp / (fp + tn)

    return dict(
        thres=thres_list, prec=prec, rec=rec, fpr=fpr, tp=tp, tn=tn, fp=fp, fn=fn
    )