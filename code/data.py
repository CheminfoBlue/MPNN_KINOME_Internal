import os
from typing import Any
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from utils import to_str
from transforms import compute_features
from sklearn.utils import shuffle

def get_smiles_col(df, smilesCol):
    if smilesCol is None:
            smilesCol = df.columns[0]
    return smilesCol

def get_mol_data(p, smilesCol='SMILES', molCol='Mol'):
    p = to_str(p)
    extension = os.path.splitext(p)[1]
    if extension == '.csv':
        df = pd.read_csv(p)
        smilesCol = get_smiles_col(df, smilesCol)
        df[smilesCol] = df[smilesCol].astype('str')
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol=smilesCol, molCol=molCol)
        df = df.dropna(subset=molCol) # remove bad structures
        molCol_data = df.pop(molCol)
        df.insert(0, molCol, molCol_data) 
        smilesCol_ix = np.where(df.columns==smilesCol)[0][0]
        MolInfoCols = df.columns.tolist()[:(smilesCol_ix+1)]
        return MolInfoCols, df
    elif extension == '.sdf':
        if smilesCol is None:
            smilesCol = 'SMILES'
        df = PandasTools.LoadSDF(p, smilesName=smilesCol,molColName=molCol, includeFingerprints=False)
        df = df.dropna(subset=molCol) # remove bad structures
        df.drop(['ID'], axis=1, inplace=True)
        molCol_data = df.pop(molCol)
        df.insert(0, molCol, molCol_data)
        smilesCol_ix = np.where(df.columns==smilesCol)[0][0]
        MolInfoCols = df.columns.tolist()[:(smilesCol_ix+1)]
        return MolInfoCols, df
    else:
        print('data type (extension) not supported')
        
wf_split_map = {'build': 'train', 'validate': 'val', 'test': 'test', 'hyptune': 'train', 'prospective': 'data'}

class Dataset:
    def __init__(self, data_path, val_data_path=None, test_data_path=None, **kwargs):
        """
        :param data_path: path to input data 
        :param val_data_path: path to separate validation data
        :param test_data_path: path to separate test data
        :return: A Dataset class instance. 
        """
        #get descriptor and fingerprints types, as well as activity tag (label type name)
        self.descType, self.fpType, self.smilesCol, self.activity_tag_ls = kwargs['descriptors'], kwargs['fingerprints'], kwargs['structure_col'], kwargs['activity']
        self.feature_path, self.val_feature_path, self.test_feature_path, self.workflow =  kwargs['feature_path'], kwargs['val_feature_path'], kwargs['test_feature_path'], kwargs['workflow']
        self.filter_protac = kwargs['filter_protac']
        # self.n_targets = len(self.activity_tag_ls)
#         self.activity_tag = self.activity_tag_ls[0]
        self.molCol = 'Mol'
    
        print('Activities: ', self.activity_tag_ls)

        if (val_data_path is not None) | (test_data_path is not None):
            base_set = wf_split_map[self.workflow.split('_')[0]]
            self.feature_paths = {base_set: self.feature_path}
            self.data_paths = {base_set: data_path}
            self.MolInfoCols, mol_data = get_mol_data(data_path, smilesCol=self.smilesCol)
            self.mol_data = {base_set: mol_data}
            if val_data_path is not None:
                self.data_paths['val'] = val_data_path
                _, self.mol_data['val'] = get_mol_data(val_data_path, smilesCol=self.smilesCol)
                self.feature_paths['val'] = self.val_feature_path
            if test_data_path is not None:
                self.data_paths['test'] = test_data_path
                _, self.mol_data['test'] = get_mol_data(test_data_path, smilesCol=self.smilesCol)
                self.feature_paths['test'] = self.test_feature_path
        else:
            base_set = 'data'#wf_split_map[self.workflow.split('_')[0]]
            self.feature_paths = {base_set: self.feature_path}
            self.data_paths = {base_set: data_path}
            self.MolInfoCols, mol_data = get_mol_data(data_path, smilesCol=self.smilesCol)
            self.mol_data = {base_set: mol_data}
        
        self.smilesCol = get_smiles_col(self.mol_data[base_set], self.smilesCol)

        # self.MolInfoCols = [self.smilesCol, self.molCol]
#         if self.workflow == 'prospective':
#             self.MolInfoCols += ['Split']
        
        #if no activities provided, we assume all target cols (any beyond the structure/smiles col) are select targets
        self.target_cols = [c for c in self.mol_data[base_set].columns if c not in self.MolInfoCols]
        if self.activity_tag_ls:
            self.target_cols = [c for c in self.activity_tag_ls if c in self.target_cols]
        # print('activities ls: ', self.activity_tag_ls)
        self.n_targets = len(self.target_cols)
        self.MolInfoCols.remove(self.molCol)

    def load_data(self):
        self.data = {}
        for split, p in self.feature_paths.items():
            mol_df = self.mol_data[split]
            print(mol_df.loc[:,self.MolInfoCols].columns.tolist())
            if p is not None:
                print('Loading %s features from %s'%(split, p))
                data_df = pd.read_csv(p)
                print(self.MolInfoCols+self.target_cols)
                data_df = pd.concat([mol_df[self.MolInfoCols+self.target_cols], data_df], axis=1)
                # data_df.dropna(axis=0, how='any', subset=[self.smilesCol]+self.target_cols, inplace=True)
#                 self.data[split] = data_df
                self.data[split] = data_df
            elif self.mol_data[split] is not None:
                print('Processing %s smiles and computing features'%split)
                input_data_path = self.data_paths[split]
#                 mol_df = self.mol_data[split]
                
                
                input_file_name = os.path.splitext(os.path.basename(input_data_path))[0]
                data_path = os.path.dirname(input_data_path)

                data_df = compute_features(mol_df[self.molCol].tolist(), self.descType, self.fpType)
                featureData_file = os.path.join(data_path,'%s_featureData.csv'%input_file_name)
                data_df.reset_index(drop=True, inplace=True)
                data_df.to_csv(featureData_file,header=True, index=False,sep=',')
                data_df.drop(['Smiles'], axis=1, inplace=True)
#                 data_df.dropna(axis=0, how='any', inplace=True)
                data_df = pd.concat([mol_df[self.MolInfoCols+self.target_cols], data_df], axis=1)
                # data_df.dropna(axis=0, how='any', subset=[self.smilesCol]+self.target_cols, inplace=True)
                self.data[split] = data_df
#                 self.data[split+'_features'] = data_df
#                 self.data[split+'_input'] = mol_df[[self.smilesCol]+self.target_cols]
            else:
                print('No features provided, and no input data provided to generate features.')
            

    def get_data(self, split=None, return_separate=True, shuffle_data=False):
        if self.workflow == 'prospective':
            data_df =  self.data['data'].copy()
            if self.filter_protac:
                print('removing %d potential protacs - mols with weight > 700'%((data_df['MolWeight']>700).sum()))
                data_df = data_df[data_df['MolWeight']<=700]
            print('data cols: ', data_df.columns)
            if split is not None:
                data_df = data_df[data_df.Split==split]
            print('The size of %s set is '%to_str(split), len(data_df))
        else:
            try:
                data_df =  self.data[split].copy()
                print('The size of %s set is '%split, len(data_df))
            except:
                data_df =  next(iter(self.data.values())).copy()
                print('The size of the data set is ', len(data_df))
        
        if (self.workflow == 'predict') & ('Split' in data_df.columns.tolist()):
            data_df = data_df[data_df.Split=='test']
            print('The size of %s set is '%to_str(split), len(data_df))

        if shuffle_data:
                # shuffle the data
                data_df = shuffle(data_df, random_state=42) 
        if return_separate:  
            try:
                Y = data_df[self.target_cols]
            except:
                print('Cannot return features and targets separately - no targets provided. /n returning None type')
                Y = None
            # X = data_df[ data_df.columns[~data_df.columns.isin([self.smilesCol]+self.target_cols)]]
            X = data_df[ data_df.columns[~data_df.columns.isin(self.MolInfoCols+self.target_cols)]]
            id = data_df[self.MolInfoCols]
            return id, X, Y 
        else:
            return data_df 
        

    # def add_datattr(self, split=None, )
        


            

        




