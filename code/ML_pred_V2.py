from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools
import datetime
import math
import pickle
import os, string,shutil
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs,Torsions
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import Fragments
from rdkit.Chem import EState
#from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.preprocessing import RobustScaler
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd 
import numpy as np
from scipy.stats import pearsonr
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import joblib
from sklearn.utils import shuffle
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score,RepeatedKFold,train_test_split
from rdkit import RDLogger 
from utils import save_model, to_str, str_to_dict, to_ls
from data import Dataset
from learner import Learner

import argparse
from argparse import *
import sys
import json 

RDLogger.DisableLog('rdApp.*') 

def main():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        epilog="--------------Example on HowTo Run ADME_pred.py codes ---------------------"+
                                        "python ML_pred.py input_file -a pIC50 -fp FCFP4_MACCS -d -m RF -w build ")
        parser.add_argument("data_path",type=str, help="path to input molecule smiles/sdf data")
        parser.add_argument("--val_data_path",type=str, help="path to validation molecule smiles/sdf data")
        parser.add_argument("--test_data_path",type=str, help="path to test molecule smiles/sdf data")
        parser.add_argument("--feature_path",type=str, help="path to input molecule features")
        parser.add_argument("--val_feature_path",type=str, help="path to validation molecule features")
        parser.add_argument("--test_feature_path",type=str, help="path to test molecule features")
        parser.add_argument("--structure_col", type=str, default=None, help="column name for the strucutre string / smiles. Defualts to first column if none provided.")
        parser.add_argument("-a","--activity", nargs="*", type=str, default=None, help="define the experimental activity tag ")
        parser.add_argument("-fp", "--fingerprints",type=str, default=None, help="define the fingerprint sets to use with _ as the delimiter "+
                        "The available fingerprints are MACCS, ECFP4, ECFP6, FCFP4, FCFP6, APFP, Torstion ") #default FP: FCFP4_MACCS
        parser.add_argument("-d", "--descriptors",type=str, default=None, help="define the descriptor sets to use with _ as the delimiter "+
                        "The available descriptors are rdkit, fragment, autocorr2d ")
        parser.add_argument("--model_type",type=str, help="specify the ML algorithms to use "+
                        "The available ML algorithms are 'RF','XGB','LGB','SVM','KNN','wKNN' ")
        parser.add_argument("--task_type",type=str, help="specify the task type (regression or classification)", default='classification')
        parser.add_argument("-w", "--workflow",type=str, help="specify the task to implement "+
                        "The available modes are build, validation, prediction ")
        parser.add_argument("-cp", "--checkpoint_path",type=str, help="Specify path to save checkpoints / built models")
        parser.add_argument("-rp", "--results_path",type=str, help="Specify path to save results / output")
        parser.add_argument("-m", "--model",type=str, help="specify the saved ML models ")
        parser.add_argument("--hyp_params", type=str, default = '{"random_state": 30, "class_weight": "balanced"}', help='Set the hyper parameters for the chosen model, e.g. ''{"p0": 10, "p1": 2}''')
        parser.add_argument("--tune_paramspace", type=str, default = None, help='Set the hyper parameter space for tuning, e.g. ''{"p0": 10, "p1": 2}''')
        parser.add_argument('--tune_args', type=str, help="Set arguments for hyper parameter tuning", 
                        default=None)
        parser.add_argument("--filter_protac", action='store_true', help="set flag to filter out protac/large molecules (weight > 700)")

        args=parser.parse_args()
        
        # Print the full command
        # print(" ".join(sys.argv))
        # Print parsed arguments
        print("Parsed arguments:", args)

        '''
                (build)  python ML_pred.py input.sdf -a 'XXX' -algo LGB -w build
        (validation)  python ML_pred.py input.sdf -a 'XXX' -w validation
        (prediction) python ML_pred.py input.sdf -a 'XXX' -w prediction -m XXX.rds -algo SVM

        '''
        ##########################
        # command line arguments #
        ##########################
        input_file = args.data_path
        input_file_name = os.path.splitext(os.path.basename(input_file))[0]
        # data_path = os.path.dirname(input_file)
        model_type = args.model_type
        task_type = args.task_type
        # checkpoint_path = args.checkpoint_path
        # results_path = args.results_path
        workflow = args.workflow
        # if args.model is not None:
        #     saved_model = args.model
        #     saved_model_name = os.path.splitext(os.path.basename(saved_model))[0]

        ##########################
        # model setting
        ##########################
        model_params = str_to_dict(args.hyp_params) #args.params given as json (dict) with named params. Default params if none provided.
        # model_params = args.hyp_params
        print(args.tune_args) 
        tune_args = str_to_dict(args.tune_args)
        tune_args['param_space'] =  str_to_dict(args.tune_paramspace)
        print(tune_args) 
        datargs = {'descriptors': args.descriptors,
                'fingerprints': args.fingerprints,
                'structure_col': args.structure_col,
                'activity':  to_ls(args.activity),
                'feature_path': args.feature_path,
                'val_feature_path': args.val_feature_path,
                'test_feature_path': args.test_feature_path,
                'workflow': workflow,
                'filter_protac': args.filter_protac
                }

        wfargs = {'saved_model': args.model,
                'checkpoint': args.checkpoint_path,
                'results_path': args.results_path,
                'tune_args': tune_args
                }
        #####################################################        
        # main part #
        #####################################################


        #####################################################        
        # create data class isntance and get molecule features (e.g. descriptors & fingerprints) #
        #####################################################
        dataset = Dataset(args.data_path, args.val_data_path, args.test_data_path, **datargs)
        #load data/features
        dataset.load_data()


        ############################################################
        # Instantiate or load learner class and run workflow
        ############################################################
        learner_model = Learner(model_type, task_type, model_params)
        # if saved_model is not None:
        #     learner_model.model = saved_model
        learner_model.run_workflow(workflow, dataset, wfargs)

if __name__ == '__main__':
    main()