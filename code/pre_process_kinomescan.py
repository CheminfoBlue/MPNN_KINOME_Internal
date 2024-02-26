import pandas as pd
import numpy as np
import re
import argparse
import os
from argparse import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_path",type=str, help="path to multiclass Kinome data")
parser.add_argument("--kinase_info", default = 'kinases-on-panel-468.csv', type=str, help="path to kinase-eurofin group mappings")
parser.add_argument("--adhoc_run", action='store_true', help="set flag to save all class predicted probabilities")

args=parser.parse_args()
adhoc_run = args.adhoc_run

data_dir = os.path.split(args.data_path)[0]
args.kinase_info = os.path.join(data_dir, args.kinase_info)
df = pd.read_csv(args.data_path)

# fp_file_path = os.path.join(data_dir, 'KinomeData_ECFP6_FPs_all.csv')
# df_fps_current = pd.read_csv(fp_file_path)
kinase_select = pd.read_csv(args.kinase_info)
select_kinase = kinase_select.Full.tolist()
select_targets = select_kinase+['S(10)', 'S(35)']
#ensure current target cols are in order of select_targets
if not adhoc_run:
    current_file_path = os.path.join(data_dir, 'kinome_data_multiclass_current.csv')
    df_current = pd.read_csv(current_file_path)
    if (df_current.columns[3:]!=select_targets).any():
        df_current = df_current.loc[:, ['Compound Name', 'Split', 'Structure']+select_targets]


def poc_filter(df, select_poc=None):
    df_subset = df.copy(deep=True)
    col_split_filter = df_subset.columns.str.contains('kinomescan', case=False)
    id_cols = df_subset.columns[~col_split_filter]
    target_cols = df_subset.columns[col_split_filter]
    
    poc_cols = target_cols[~target_cols.str.contains('KinomeScan Kd', case=False)]
    if select_poc is not None:
        #escape non-alpha characters to match string literals
        select_poc_esc = select_poc.copy()
        select_poc_esc = [re.escape(poc) for poc in select_poc]
        #filter on select poc list
        poc_cols = poc_cols[poc_cols.str.contains('|'.join(select_poc_esc), case=False)]
        df_subset = df_subset[id_cols.tolist()+poc_cols.tolist()]
    #rename target cols
    df_subset.columns = df_subset.columns.str.split('KinomeScan Gene Symbol: ').str[-1]
    df_subset.columns = df_subset.columns.str.split('KinomeScan Selectivity;Mean;').str[-1]
    
    df_subset = df_subset.loc[:,id_cols.tolist()+select_poc]
    return df_subset


def binning(x, partitions = [10, 35], choicelist=[2, 1, 0]):
    condlist=[]
    L = len(choicelist)
    for i in range(L):
        c = choicelist[i]
        if i == 0:
            print('x <= %d --> %d'%(partitions[0],c))
            condlist+=[x<=partitions[0]]
        elif i+1 == L:
            print('x > %d --> %d'%(partitions[-1], c))
            condlist+=[x>partitions[-1]]
        else:
            p0 = partitions[i-1]
            p1 = partitions[i]
            print('x > %d AND x <= %d --> %d'%(p0,p1, c))
            condlist+=[(x>p0)&(x<=p1)]
            
    x_bin=np.select(condlist, choicelist, np.nan)
    
    return x_bin




#0.a. remove empty columns/targets, and empty structures
df = df[~df['Structure'].isna()]
df = df.loc[:,~df.isna().all(0)]

#0.b. get & add BLU Id (temporal stamp)
#remove overlapping between current and new data - by Compound Name / BLU ID
if not adhoc_run:
    filter_overlap = df['Compound Name'].isin(df_current['Compound Name'])
    print('Removing %d overlapping records between current and new'%(filter_overlap.sum())) if filter_overlap.sum()>0 else print('No overlapping records between current and new')
    df = df.loc[~filter_overlap,:]
#add ascending temporal order
df.sort_values('Compound Name', key=lambda col: col.str.split('BLU').str[-1], ascending=True, inplace=True)

#1. filter on PoC targets <--> filter out Kd target columns
#allows for subset of PoC targets -- select_poc
df_subset = poc_filter(df, select_poc=select_targets)

#2. filter out molecules without any hits across all targets
df_subset = df_subset.dropna(axis=0, how='all', subset = select_targets, inplace=False).reset_index(drop=True)


#3. apply binning - keep float format to allow for nan/blanks
#PoC>35 --> 0
#10<PoC<=35 --> 1
#PoC<=10 --> 2
df_subset_binned = df_subset.copy()
df_subset_binned[select_kinase] = binning(df_subset_binned[select_kinase].values, partitions = [10, 35], choicelist=[2, 1, 0])


#4. define target id-name-shortname df map
# target_name_id_map = pd.DataFrame(data=np.array([['target'+str(i+1) for i in range(len(poc_cols))], poc_cols]).T,
#             columns=['Target Id', 'Target Name'])
# target_name_id_map['Target Name (short)'] = target_name_id_map['Target Name'].str.split('Gene Symbol: ').str[-1].str.split(';Location').str[0]

#get S(10)  and S(35) scores 
# S10 = (df_subset[poc_cols]<=10).sum(1) / df_subset[poc_cols].count(axis=1, numeric_only=True)
# S35 = ((df_subset[poc_cols]>10.0) & (df_subset[poc_cols]<=35.0)).sum(1) / df_subset[poc_cols].count(axis=1, numeric_only=True)

#5. compute fingerprints for new data and append to current fingerprints 
# mols = [Chem.MolFromSmiles(smi) for smi in df_subset_binned['Structure'].tolist()]
# ecfp6_fps = compute_features(mols, descType=None, fpType='ECFP6')
# ecfp6_fps.reset_index(drop=True, inplace=True)
# ecfp6_fps.drop(['Smiles'], axis=1, inplace=True)
# df_fps_current = pd.concat([df_fps_current, ecfp6_fps], axis=0, ignore_index=True)

#6. update splits (old df_current --> train & new df_subset_binned --> test) and append -- save updated 'current' files
if adhoc_run:
    df_subset_binned.insert(1, 'Split', 'train')
    df_subset_binned.to_csv(os.path.splitext(args.data_path)[0]+'_preprocessed.csv', index=False)
else:
    df_current['Split'] = 'train'
    df_subset_binned.insert(1, 'Split', 'test')
    if (df_current.columns==df_subset_binned.columns).all():
        df_current = pd.concat([df_current, df_subset_binned], axis=0, ignore_index=True)
        df_current.to_csv(current_file_path, index=False)
        # df_fps_current.to_csv(fp_file_path, index=False)
        print('Updated current kinome profile with %d new records'%len(df_subset_binned))
    else:
        print('mismatching columns, DO NOT APPEND! \nFiles have not been updated.')

