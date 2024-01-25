import joblib
import sys, os
import pandas as pd
import numpy as np
import json

from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import AllChem

def to_str(s):
    return(s or "")

def get_savepath(p, id_ext):
    if p is not None:
        return os.path.join(p, id_ext)
    else:
        return None

def output(scores, task_type, preds=None, id=None, col_tag=None, save_path=None):
    print(scores)
    if save_path is not None:
        if preds is not None:
            if id is None:
                pred_df = pd.DataFrame(index=np.arange(len(preds)))
            else:
                pred_df = id
            if len(col_tag) == 1:
                col_tag = col_tag[0]
                if task_type == 'multiclass':
                    for j in range(preds.shape[1]):
                        pred_df[col_tag+'_pred_%d'%j] = preds[:,j]
                else:
                    pred_df[col_tag+'_pred'] = preds
            else:
                scores.insert(0, 'Target', col_tag)
                if task_type == 'multiclass':
                    for c, p in zip(col_tag, preds):
                        pred_df[c] = list(p)
            scores.to_csv(save_path,  index=False, sep=',')
            pred_df.to_csv(save_path.replace('scores', 'preds'), index=False, sep=',')

            

    

def save_model(model, save_path, xp_id=None, algo=None):
    #to_str handles case where no path is passed (None type)
    # save_path = os.path.join(to_str(save_path), "%s_%s_model.rds"%(xp_id,algo))
    joblib.dump(model, save_path)

# def csv2sdf(csv_file, smilesCol = 'SMILES', molCol = 'Mol'):
#     data = pd.read_csv(csv_file) 

#     PandasTools.AddMoleculeColumnToFrame(data,smilesCol=smilesCol, molCol=molCol) 
#     data = data.dropna(subset=molCol)  # remove bad structures

#     PandasTools.WriteSDF(data, '%s.sdf' %csv_file[:-4],  molColName=molCol, idName='RowID',properties=list(data.columns))
    
#     sdFile=Chem.SDMolSupplier('%s.sdf' %csv_file[:-4])
#     return sdFile


def to_dict(d):
    return(d or {})

def str_to_dict(s):
    if s is None:
        return {}
    else:
        return json.loads(to_dict(s))


def to_ls(obj):
    if obj is None:
        return []
    elif isinstance(obj, str):
        return [obj]
    elif isinstance(obj, list):
        return obj
  