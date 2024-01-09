from logging import Logger
import os
import pickle
import joblib
from typing import Dict, List, Union
from copy import deepcopy
import numpy as np
from data import Dataset
from utils import save_model, get_savepath, output 
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
import xgboost as xgb
from xgboost.sklearn import XGBRegressor, XGBClassifier
from sklearn.svm import SVR, SVC
import skopt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error, make_scorer, matthews_corrcoef, roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler


ModelType = Union[RandomForestRegressor, RandomForestClassifier, XGBRegressor, XGBClassifier, LGBMRegressor, LGBMClassifier, SVR, SVC, KNeighborsRegressor, KNeighborsClassifier]
model_classes = {
    'regression': 
    {'RF': RandomForestRegressor,
     'XGB': XGBRegressor,
     'LGB': LGBMRegressor,
     'SVM': SVR,
     'KNN': KNeighborsRegressor,
     'wKNN': KNeighborsRegressor},
    'classification': 
    {'RF': RandomForestClassifier,
     'XGB': XGBClassifier,
     'LGB': LGBMClassifier,
     'SVM': SVC, 
     'KNN': KNeighborsClassifier,
     'wKNN': KNeighborsClassifier
                  },
    'multiclass': 
    {'RF': RandomForestClassifier,
     'XGB': XGBClassifier,
     'LGB': LGBMClassifier,
     'SVM': SVC, 
     'KNN': KNeighborsClassifier,
     'wKNN': KNeighborsClassifier
                  }
}
default_hyparam_space = {'RF': {'n_estimators': Integer(25,750),
                      'max_depth': Integer(3,10), 
                      'min_samples_leaf': Integer(2,7),
                      'min_samples_split': Integer(2,10)},
               'XGB': {'n_estimators': Integer(25,500), 
                       'max_depth': Integer(3,12),
                    #    'colsample_bytree': Real(.6, .9, 'log-uniform'),  
                       'learning_rate': Real(1e-3, 3e-1, 'log-uniform'), 
                       'subsample': Real(.5, .9, 'log-uniform'),
                       'min_child_weight': (1e-5,5)
                       },
               'LGB': {'n_estimators': Integer(25,500), 
                       'subsample': Real(.5, .9, 'log-uniform'),
                       'learning_rate': Real(1e-3, 3e-1, 'log-uniform'),
                       'max_depth': Integer(3,12),
                    #    'colsample_bytree': Real(.6, .9, 'log-uniform'), 
                    #    'subsample_freq': [3], 
                       'min_child_weight': (1e-5,5)},
               'SVM': {'C': Real(1e-2, 5e+2, 'log-uniform'),
                       'gamma': Real(1e-3, 1e+1, 'log-uniform'),
                       #'degree': Integer(1, 8),  # integer valued parameter
                       'kernel': ['rbf'],
                       'probability': [True]},  # categorical parameter},
               'KNN': {'n_neighbors': Integer(3,12), 'weights': ['uniform']},
               'wKNN': {'n_neighbors': Integer(3,12), 'weights': ['distance']} 
}


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
    if (len(target.shape)>1) & (target.shape[1]==1):
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




class Learner:
    def __init__(self, model_type, task_type, model_params):
        """
        Creates an instance of an sklearn or sklearn-wrapped model (in ['RF', 'XGB', 'LGB', 'SVM', 'KNN', 'wKNN'])

        :param task_type: The type of task (regression, classification (binary), or multiclass)
        :param model_type: The type of model.
        :param model_params: A dictionary of parameters for the given model. 
        :return: A model class instance. 
        """
        try:
            model_type in ['RF', 'XGB', 'LGB', 'SVM', 'KNN', 'wKNN']
        except ValueError:
            raise ValueError(f'Model type "{model_type}" not supported')
        
        try:
            task_type in ['regression', 'classification', 'multiclass']
        except ValueError:
            raise ValueError(f'Task type "{task_type}" not supported')
        
        self.model_type = model_type
        self.task_type = task_type
        self.model = model_classes[task_type][model_type](**model_params)
        self.classes = None
        print(self.model)
    
    def predict(self, 
                features: np.ndarray, 
                logger: Logger = None):

        """
        Predicts using a scikit-learn model.

        :param model: The trained scikit-learn model to make predictions with.
        :param task_type: The type of task (regression or classification).
        :param features: The data features used as input for the model.
        :return: A list of lists of floats containing the predicted values.
        """    
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print
        
        debug('Predicting from built model')

        if self.task_type == 'regression':
            preds = self.model.predict(features)
            return [preds]
        elif self.task_type == 'classification':
            preds = self.model.predict(features)
            probs = self.model.predict_proba(features)[:,1]
            return [preds, probs]
        elif self.task_type == 'multiclass':
            preds = self.model.predict(features)
            probs = self.model.predict_proba(features)
            return [preds, probs]

    def score(self, preds, target):
        if isinstance(target, pd.DataFrame):
            target=target.values
        if self.task_type =='regression':
            preds =  [p.reshape((-1, target.shape[-1])) for p in preds]
            return score_regression(preds[-1], target)
        elif self.task_type == 'classification':
            preds =  [p.reshape((-1, target.shape[-1])) for p in preds]
            return score_classification(preds, target)
        elif self.task_type == 'multiclass':
            return score_multiclass(preds[-1], target, classes=self.classes)
    
    def validate(self, 
                 features: np.ndarray, 
                 target: np.ndarray, 
                 logger: Logger = None):

        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print
        
        debug('Running model validation')

        preds = []
        scores= self.score(preds, target)
        return scores
    
    def build(self, features: np.ndarray,
            targets: np.ndarray,
            checkpoint: str = None,
            logger: Logger = None) -> ModelType:
        """
        Builds/fits an sklearn or sklearn-wrapped model (in ['RF', 'XGB', 'LGB', 'SVM', 'KNN', 'wKNN'])

        :param features: The training data features.
        :param targets: The training data targets.
        :param logger: A logger to record output.
        :return: A fitted model class instance. 
        """
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print
        
        filter_nan = ~np.isnan(targets)
        filter_nan = filter_nan.flatten()
        debug('Building model')
        y = targets[filter_nan]
        # print(np.unique(y))
        if self.task_type in ['classification', 'multiclass']:
            y = y.astype(np.int32)
            self.classes = np.unique(y)
        y = y.ravel()
        self.model.fit(features[filter_nan], y)

        if checkpoint is not None:
            print('saving model to ', checkpoint)
            save_model(self.model, save_path = checkpoint)


    def run_hyperopt(self, features, targets, tune_args, checkpoint=None, return_val_score=False):
        if isinstance(targets, pd.DataFrame):
            targets=targets.values
        targets = targets.ravel()
        param_space = tune_args['param_space']
        if len(param_space)==0:
            param_space = default_hyparam_space[self.model_type]
        if tune_args['metric'] =='mcc':
            metric = make_scorer(matthews_corrcoef)
        else:
            metric = 'roc_auc'
        self.opt = BayesSearchCV(
            self.model,
            param_space,
            n_iter=tune_args['n_iter'],
            n_points=1,
            n_jobs=-1,
            cv=tune_args['folds'],
            # scoring='roc_auc',
            scoring=metric,
            random_state=0,
            verbose=5
        )

        self.opt.fit(features, targets)

        self.model = self.opt.best_estimator_
        best_val_score = self.opt.best_score_
        if checkpoint is not None:
            save_model(self.model , save_path = checkpoint)
        print('Best val score: ', best_val_score)
        if return_val_score:
            return best_val_score
        else:
            pass
        


    def pre_process(self,X):
        if self.model_type in ['SVM', 'KNN', 'wKNN']:
            if not hasattr(self, 'scaler'):
                self.scaler = RobustScaler().fit(X)
            return self.scaler.transform(X) 
        else:
            return X  
    # workflows: 
    #           build 
    #           validate    (must provide model)
    #           predict     (must provide model)
    #           build_validate
    #           build_predict   
    #           validate_predict
    #           build_validate_predict
    #           hyperopt -- hyper parameter tuning
    def run_workflow(self, workflow, dataset: Dataset, wfargs):
        saved_model, checkpoint,  results_path = wfargs['saved_model'], wfargs['checkpoint'], wfargs['results_path']
        wf_list = workflow.split('_')
        wf_start = wf_list[0]
        self.score_dict = {}
        self.n_targets = dataset.n_targets
        self.target_name = dataset.activity_tag_ls[0]
        if workflow=='prospective':
            #first build on train set (don't save model -- checkpoint==None)
            _, X, Y = dataset.get_data(return_separate=True, split='train', shuffle_data = False)
            self.build(features=X.values, targets=Y.values, checkpoint=None)
            #second predict/score on test set
            id, X, Y = dataset.get_data(return_separate=True, split='test', shuffle_data = False)
            preds = self.predict(features=X)
            # print('target cols', Y.columns.tolist())
            # print(np.unique(Y.values))
            scores = self.score(preds, Y.values)
            save_path = get_savepath(results_path, '%s_test_scores.csv'%(self.target_name))
            output(scores, self.task_type, preds=preds[-1], id=id, col_tag=dataset.activity_tag_ls[0], save_path=save_path)
            #third build on full dataset and save model to checkpoint
            _, X, Y = dataset.get_data(return_separate=True, shuffle_data = False)
            # print('saving model to ', checkpoint)
            self.build(features=X.values, targets=Y.values, checkpoint=checkpoint)

        elif wf_start == 'build':
            _, X, Y = dataset.get_data(return_separate=True, split='train', shuffle_data = False)
            self.build(features=X.values, targets=Y.values, checkpoint=checkpoint)
            if 'validate' in workflow:
                _, X, Y = dataset.get_data(return_separate=True, split='val')
                scores = self.validate(features=X, targets=Y, folds=5)
                save_path = get_savepath(results_path, '%s_val_scores.csv'%(self.target_name))
                output(scores, self.task_type, save_path=save_path)
            if 'predict' in workflow:
                id, X, Y = dataset.get_data(return_separate=True, split='test')
                preds = self.predict(features=X)
                scores = self.score(preds, Y.values)
                save_path = get_savepath(results_path, '%s_test_scores.csv'%(self.target_name))
                output(scores, self.task_type, preds=preds[-1], id=id, col_tag=dataset.activity_tag_ls[0], save_path=save_path)
        elif wf_start =='hyptune':
            self.tune_args = wfargs['tune_args'] #{'folds': wfargs['folds'], 'split_type': wfargs['split_type'], 'n_iter': wfargs['n_iter'], 'param_space': wfargs['param_space']}
            id, X, Y = dataset.get_data(return_separate=True)
            X = self.pre_process(X)
            best_val_score = self.run_hyperopt(X, Y, self.tune_args, checkpoint=checkpoint, return_val_score=True)
            preds = self.predict(features=X)
            scores = self.score(preds, Y.values)
            scores['best_val_%s'%(self.tune_args['metric'])] = best_val_score
            save_path = get_savepath(results_path, '%s_hyptune_scores.csv'%(self.target_name))
            output(scores, self.task_type, save_path=save_path)
            if 'predict' in workflow:
                id, X, Y = dataset.get_data(return_separate=True, split='test')
                X = self.pre_process(X)
                preds = self.predict(features=X)
                scores = self.score(preds, Y.values)
                save_path = get_savepath(results_path, '%s_test_scores.csv'%(self.model_type))
                output(scores, self.task_type, preds=preds[-1], id=id, col_tag=dataset.activity_tag_ls[0], save_path=save_path)
        else:
            try:
                self.model = joblib.load(saved_model)
            except:
                # print('No pre-built model provided, cannot continue work flow.')
                raise TypeError("No pre-built model provided, cannot continue work flow.") from None
            if wf_start =='validate':
                _, X, Y = dataset.get_data(return_separate=True)
                scores = self.validate(features=X, targets=Y, folds=5)
                save_path = get_savepath(results_path, '%s_val_scores.csv'%(self.target_name))
                output(scores, self.task_type, save_path=save_path)
                if 'predict' in workflow:
                    id, X, Y = dataset.get_data(return_separate=True, split='test')
                    preds = self.predict(features=X)
                    scores = self.score(preds, Y.values)
                    save_path = get_savepath(results_path, '%s_test_scores.csv'%(self.target_name))
                    output(scores, self.task_type, preds=preds[-1],  id=id, col_tag=dataset.activity_tag_ls[0], save_path=save_path)
            elif workflow =='predict':
                id, X, Y = dataset.get_data(return_separate=True)
                preds = self.predict(features=X)
                scores = self.score(preds, Y.values)
                save_path = get_savepath(results_path, '%s_test_scores.csv'%(self.target_name))
                output(scores, self.task_type, preds=preds[-1], id=id, col_tag=dataset.activity_tag_ls[0], save_path=save_path)
            else:
                print('%s is not a defined workflow, please choose from workflows listed in main script.'%workflow)

            