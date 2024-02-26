import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error, matthews_corrcoef, roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from scipy.stats import pearsonr


def score_regression(pred, target):
    filter_nan = ~np.isnan(target)
    target = target[filter_nan]
    pred = pred[filter_nan]

    R_test = pearsonr(target, pred)[0]
    r2_test = r2_score(target, pred)
    mse_test = mean_squared_error(target, pred)  #in version 23.1, rmse_testset = mean_squared_error(Y_test, Y_pred_test, squared=False)
    rmse_test = np.sqrt(mse_test)
    
    return pd.DataFrame([{'rmse_test': rmse_test, 'R_test': R_test, 'r2_test': r2_test}])

def score_classification(preds, target):
    pred, pred_prob = preds

    filter_nan = ~np.isnan(target)
    target = target[filter_nan]
    pred_prob = pred_prob[filter_nan]
    pred = pred[filter_nan]

    # pred = (pred_prob>=.5).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(target, pred_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
#     acc = metrics.accuracy_score(y_test, y_pred)
#     precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(target, pred)
    acc = (1-np.abs(target-pred)).mean()
    mcc = metrics.matthews_corrcoef(target, pred)
    cf_mat = metrics.confusion_matrix(target, pred)
    TPR = cf_mat[1,1] / cf_mat[1].sum()
    TNR = cf_mat[0,0] / cf_mat[0].sum()
#     sensitivity = metrics.recall_score(y_test, y_pred)
    
    return pd.DataFrame([{'AUC': auc, 'F1-score': f1, 'Accuracy': acc, 'MCC': mcc,
                           'Sensitivity': TPR, 'Specificity': TNR}])

def sensitivity_specificity(y_true, y_pred, average=None, classes=None):
    if classes is None:
        np.unique(classes)
    print('unique classes: ', classes)
    print(np.eye(np.max(classes) + 1))
    y_true_onehot = np.eye(np.max(classes) + 1)[y_true]
    y_pred_onehot = np.eye(np.max(classes) + 1)[y_pred]
    
    if average=='micro':
        y_true_onehot=y_true_onehot.ravel()
        y_pred_onehot=y_pred_onehot.ravel()

    P = y_true_onehot.sum(0)
    N = (1-y_true_onehot).sum(0)

    #sensitivity == True Positive Rate (TPR)
    TP = (y_pred_onehot*y_true_onehot).sum(0)
    TPR = TP / P
#         print('Sensitivity (TPR): ', TPR)

    #sensitivity == True Positive Rate (TPR)
    TN = ((1-y_true_onehot)*(1-y_pred_onehot)).sum(0)
    TNR = TN / N
#         print('Specificity (TNR): ', TNR)
    if average=='macro':
        return TPR.mean(), TNR.mean()
    else:
        return TPR, TNR

    
# 1. ovr sets given class to positive (1) and all other classes to negative (0) --> binary ROC_AUC - iterate over all classes
# uses ther probability of the given class as the binary probability of positive (1)
# the negative probability is then just the sum of the probabilities for the other classes
# 2. macro averaging takes the mean ROC_AUC across all classes
# 3. micro averaging takes all classes and probabilities, computing the TPR & FPR and computing the ROC_AUC 
#the averaging is indirect (micro)
# micro averaging is better for imbalanced datasets
# 4. accuracy
# 5. balanced accuracy
# 6. sensitivity, specificity
def score_multiclass(pred_prob, target, classes=None):
    #flatten if given Nx1 column vector
    if target.ndim>1:
        target = target.flatten()

    filter_nan = ~np.isnan(target)
    filter_nan = filter_nan.flatten()
    target = target[filter_nan]
    target = target.astype(np.int32)
    pred_prob = pred_prob[filter_nan]
    print(target.shape)
    print(pred_prob.shape)

    print('unique target classes: ', np.unique(target))
    pred = pred_prob.argmax(1).astype(np.int32)
    print('unique pred classes: ', np.unique(pred))
    if classes is None:
        classes = np.unique(target)
    print(classes)
    #handle exception when not all classes are represented in the true target set
    try:
        roc_auc_macro = roc_auc_score(target, pred_prob, multi_class='ovr', average='macro', labels=classes)
        roc_auc_micro = roc_auc_score(target, pred_prob, multi_class='ovr', average='micro', labels=classes)
    except ValueError:
        roc_auc_macro = np.nan
        roc_auc_micro = np.nan
        pass
    acc = accuracy_score(target, pred)
    acc_balanced = balanced_accuracy_score(target, pred)
    sensitivity, specificity = sensitivity_specificity(target, pred, average='macro', classes=classes)
    
    data = {'ROC_AUC_macro': roc_auc_macro, 'ROC_AUC_micro': roc_auc_micro, 'Accuracy': acc, 
            'Balanced Accuracy': acc_balanced, 'Sensitivity': sensitivity, 'Specificity': specificity}
                
    return pd.DataFrame(data, index=[0])
