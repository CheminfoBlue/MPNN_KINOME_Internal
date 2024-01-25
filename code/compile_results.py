import os
from datetime import date, timedelta
import re
import pandas as pd
import numpy as np
import argparse
import joblib
from argparse import *
from learner import MultiOutputClassifier


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("results_path",type=str, help="path to multiclass Kinome data")
parser.add_argument("--sscore_target_map_path", default = None, type=str, help="path to kinase-eurofin group mappings")
parser.add_argument("--save_all_preds", action='store_true', help="set flag to save all class predicted probabilities")
parser.add_argument("--append_results", action='store_true', help="set flag to append results to respective current files")

args=parser.parse_args()
# args=parser.parse_args(["./results/LGB"])

'''
        (without sscore_target_map)  python compile_results.py ./results/LGB/
        (with sscore_target_map)  python compile_results.py ./results/LGB/ --sscore_target_map_path ./data/kinases-on-panel-468_eurofin_map.csv
'''
current_date = date.today().isoformat()  #(date.today()+timedelta(days=30)).isoformat() 
default_sscore_target_group_dict = {'AAK1': 'AAK1',
 'ABL2': 'ABL2',
 'ACVR1': 'ACVR1',
 'ACVR1B': 'ACVR1B',
 'ACVR2A': 'ACVR2A',
 'ACVR2B': 'ACVR2B',
 'ACVRL1': 'ACVRL1',
 'ADCK3': 'ADCK3',
 'ADCK4': 'ADCK4',
 'AKT1': 'AKT1',
 'AKT2': 'AKT2',
 'AKT3': 'AKT3',
 'ALK': 'ALK',
 'AMPK-alpha1': 'AMPK-alpha1',
 'AMPK-alpha2': 'AMPK-alpha2',
 'ANKK1': 'ANKK1',
 'ARK5': 'ARK5',
 'ASK1': 'ASK1',
 'ASK2': 'ASK2',
 'AURKA': 'AURKA',
 'AURKB': 'AURKB',
 'AURKC': 'AURKC',
 'AXL': 'AXL',
 'BIKE': 'BIKE',
 'BLK': 'BLK',
 'BMPR1A': 'BMPR1A',
 'BMPR1B': 'BMPR1B',
 'BMPR2': 'BMPR2',
 'BMX': 'BMX',
 'BRAF': 'BRAF',
 'BRK': 'BRK',
 'BRSK1': 'BRSK1',
 'BRSK2': 'BRSK2',
 'BTK': 'BTK',
 'BUB1': 'BUB1',
 'CAMK1': 'CAMK1',
 'CAMK1B': 'CAMK1B',
 'CAMK1D': 'CAMK1D',
 'CAMK1G': 'CAMK1G',
 'CAMK2A': 'CAMK2A',
 'CAMK2B': 'CAMK2B',
 'CAMK2D': 'CAMK2D',
 'CAMK2G': 'CAMK2G',
 'CAMK4': 'CAMK4',
 'CAMKK1': 'CAMKK1',
 'CAMKK2': 'CAMKK2',
 'CASK': 'CASK',
 'CDC2L1': 'CDC2L1',
 'CDC2L2': 'CDC2L2',
 'CDC2L5': 'CDC2L5',
 'CDK11': 'CDK11',
 'CDK2': 'CDK2',
 'CDK3': 'CDK3',
 'CDK5': 'CDK5',
 'CDK7': 'CDK7',
 'CDK8': 'CDK8',
 'CDK9': 'CDK9',
 'CDKL1': 'CDKL1',
 'CDKL2': 'CDKL2',
 'CDKL3': 'CDKL3',
 'CDKL5': 'CDKL5',
 'CHEK1': 'CHEK1',
 'CHEK2': 'CHEK2',
 'CIT': 'CIT',
 'CLK1': 'CLK1',
 'CLK2': 'CLK2',
 'CLK3': 'CLK3',
 'CLK4': 'CLK4',
 'CSK': 'CSK',
 'CSNK1A1': 'CSNK1A1',
 'CSNK1A1L': 'CSNK1A1L',
 'CSNK1D': 'CSNK1D',
 'CSNK1E': 'CSNK1E',
 'CSNK1G1': 'CSNK1G1',
 'CSNK1G2': 'CSNK1G2',
 'CSNK1G3': 'CSNK1G3',
 'CSNK2A1': 'CSNK2A1',
 'CSNK2A2': 'CSNK2A2',
 'CTK': 'CTK',
 'DAPK1': 'DAPK1',
 'DAPK2': 'DAPK2',
 'DAPK3': 'DAPK3',
 'DCAMKL1': 'DCAMKL1',
 'DCAMKL2': 'DCAMKL2',
 'DCAMKL3': 'DCAMKL3',
 'DDR1': 'DDR1',
 'DDR2': 'DDR2',
 'DLK': 'DLK',
 'DMPK': 'DMPK',
 'DMPK2': 'DMPK2',
 'DRAK1': 'DRAK1',
 'DRAK2': 'DRAK2',
 'DYRK1A': 'DYRK1A',
 'DYRK1B': 'DYRK1B',
 'DYRK2': 'DYRK2',
 'EGFR': 'EGFR',
 'EIF2AK1': 'EIF2AK1',
 'EPHA1': 'EPHA1',
 'EPHA2': 'EPHA2',
 'EPHA3': 'EPHA3',
 'EPHA4': 'EPHA4',
 'EPHA5': 'EPHA5',
 'EPHA6': 'EPHA6',
 'EPHA7': 'EPHA7',
 'EPHA8': 'EPHA8',
 'EPHB1': 'EPHB1',
 'EPHB2': 'EPHB2',
 'EPHB3': 'EPHB3',
 'EPHB4': 'EPHB4',
 'EPHB6': 'EPHB6',
 'ERBB2': 'ERBB2',
 'ERBB3': 'ERBB3',
 'ERBB4': 'ERBB4',
 'ERK1': 'ERK1',
 'ERK2': 'ERK2',
 'ERK3': 'ERK3',
 'ERK4': 'ERK4',
 'ERK5': 'ERK5',
 'ERK8': 'ERK8',
 'ERN1': 'ERN1',
 'FAK': 'FAK',
 'FER': 'FER',
 'FES': 'FES',
 'FGFR1': 'FGFR1',
 'FGFR2': 'FGFR2',
 'FGFR3': 'FGFR3',
 'FGFR4': 'FGFR4',
 'FGR': 'FGR',
 'FLT1': 'FLT1',
 'FLT4': 'FLT4',
 'FRK': 'FRK',
 'FYN': 'FYN',
 'GAK': 'GAK',
 'GCN2(Kin.Dom.2,S808G)': 'GCN2(Kin.Dom.2,S808G)',
 'GRK1': 'GRK1',
 'GRK2': 'GRK2',
 'GRK3': 'GRK3',
 'GRK4': 'GRK4',
 'GRK7': 'GRK7',
 'GSK3A': 'GSK3A',
 'GSK3B': 'GSK3B',
 'HASPIN': 'HASPIN',
 'HCK': 'HCK',
 'HIPK1': 'HIPK1',
 'HIPK2': 'HIPK2',
 'HIPK3': 'HIPK3',
 'HIPK4': 'HIPK4',
 'HPK1': 'HPK1',
 'HUNK': 'HUNK',
 'ICK': 'ICK',
 'IGF1R': 'IGF1R',
 'IKK-alpha': 'IKK-alpha',
 'IKK-beta': 'IKK-beta',
 'IKK-epsilon': 'IKK-epsilon',
 'INSR': 'INSR',
 'INSRR': 'INSRR',
 'IRAK1': 'IRAK1',
 'IRAK3': 'IRAK3',
 'IRAK4': 'IRAK4',
 'ITK': 'ITK',
 'JAK1(JH1domain-catalytic)': 'JAK1(JH1domain-catalytic)',
 'JAK1(JH2domain-pseudokinase)': 'JAK1(JH2domain-pseudokinase)',
 'JAK2(JH1domain-catalytic)': 'JAK2(JH1domain-catalytic)',
 'JAK3(JH1domain-catalytic)': 'JAK3(JH1domain-catalytic)',
 'JNK1': 'JNK1',
 'JNK2': 'JNK2',
 'JNK3': 'JNK3',
 'LATS1': 'LATS1',
 'LATS2': 'LATS2',
 'LCK': 'LCK',
 'LIMK1': 'LIMK1',
 'LIMK2': 'LIMK2',
 'LKB1': 'LKB1',
 'LOK': 'LOK',
 'LRRK2': 'LRRK2',
 'LTK': 'LTK',
 'LYN': 'LYN',
 'LZK': 'LZK',
 'MAK': 'MAK',
 'MAP3K1': 'MAP3K1',
 'MAP3K15': 'MAP3K15',
 'MAP3K2': 'MAP3K2',
 'MAP3K3': 'MAP3K3',
 'MAP3K4': 'MAP3K4',
 'MAP4K2': 'MAP4K2',
 'MAP4K3': 'MAP4K3',
 'MAP4K4': 'MAP4K4',
 'MAP4K5': 'MAP4K5',
 'MAPKAPK2': 'MAPKAPK2',
 'MAPKAPK5': 'MAPKAPK5',
 'MARK1': 'MARK1',
 'MARK2': 'MARK2',
 'MARK3': 'MARK3',
 'MARK4': 'MARK4',
 'MAST1': 'MAST1',
 'MEK1': 'MEK1',
 'MEK2': 'MEK2',
 'MEK3': 'MEK3',
 'MEK4': 'MEK4',
 'MEK5': 'MEK5',
 'MEK6': 'MEK6',
 'MELK': 'MELK',
 'MERTK': 'MERTK',
 'MET': 'MET',
 'MINK': 'MINK',
 'MKK7': 'MKK7',
 'MKNK1': 'MKNK1',
 'MKNK2': 'MKNK2',
 'MLCK': 'MLCK',
 'MLK1': 'MLK1',
 'MLK2': 'MLK2',
 'MLK3': 'MLK3',
 'MRCKA': 'MRCKA',
 'MRCKB': 'MRCKB',
 'MST1': 'MST1',
 'MST1R': 'MST1R',
 'MST2': 'MST2',
 'MST3': 'MST3',
 'MST4': 'MST4',
 'MTOR': 'MTOR',
 'MUSK': 'MUSK',
 'MYLK': 'MYLK',
 'MYLK2': 'MYLK2',
 'MYLK4': 'MYLK4',
 'MYO3A': 'MYO3A',
 'MYO3B': 'MYO3B',
 'NDR1': 'NDR1',
 'NDR2': 'NDR2',
 'NEK1': 'NEK1',
 'NEK10': 'NEK10',
 'NEK11': 'NEK11',
 'NEK2': 'NEK2',
 'NEK3': 'NEK3',
 'NEK4': 'NEK4',
 'NEK5': 'NEK5',
 'NEK6': 'NEK6',
 'NEK7': 'NEK7',
 'NEK9': 'NEK9',
 'NIK': 'NIK',
 'NIM1': 'NIM1',
 'NLK': 'NLK',
 'OSR1': 'OSR1',
 'p38-alpha': 'p38-alpha',
 'p38-beta': 'p38-beta',
 'p38-delta': 'p38-delta',
 'p38-gamma': 'p38-gamma',
 'PAK1': 'PAK1',
 'PAK2': 'PAK2',
 'PAK3': 'PAK3',
 'PAK4': 'PAK4',
 'PAK6': 'PAK6',
 'PAK7': 'PAK7',
 'PCTK1': 'PCTK1',
 'PCTK2': 'PCTK2',
 'PCTK3': 'PCTK3',
 'PDGFRA': 'PDGFRA',
 'PDGFRB': 'PDGFRB',
 'PDPK1': 'PDPK1',
 'PFCDPK1(P.falciparum)': 'PFCDPK1(P.falciparum)',
 'PFPK5(P.falciparum)': 'PFPK5(P.falciparum)',
 'PFTAIRE2': 'PFTAIRE2',
 'PFTK1': 'PFTK1',
 'PHKG1': 'PHKG1',
 'PHKG2': 'PHKG2',
 'PIK3C2B': 'PIK3C2B',
 'PIK3C2G': 'PIK3C2G',
 'PIK3CA': 'PIK3CA',
 'PIK3CB': 'PIK3CB',
 'PIK3CD': 'PIK3CD',
 'PIK3CG': 'PIK3CG',
 'PIK4CB': 'PIK4CB',
 'PIKFYVE': 'PIKFYVE',
 'PIM1': 'PIM1',
 'PIM2': 'PIM2',
 'PIM3': 'PIM3',
 'PIP5K1A': 'PIP5K1A',
 'PIP5K1C': 'PIP5K1C',
 'PIP5K2B': 'PIP5K2B',
 'PIP5K2C': 'PIP5K2C',
 'PKAC-alpha': 'PKAC-alpha',
 'PKAC-beta': 'PKAC-beta',
 'PKMYT1': 'PKMYT1',
 'PKN1': 'PKN1',
 'PKN2': 'PKN2',
 'PKNB(M.tuberculosis)': 'PKNB(M.tuberculosis)',
 'PLK1': 'PLK1',
 'PLK2': 'PLK2',
 'PLK3': 'PLK3',
 'PLK4': 'PLK4',
 'PRKCD': 'PRKCD',
 'PRKCE': 'PRKCE',
 'PRKCH': 'PRKCH',
 'PRKCI': 'PRKCI',
 'PRKCQ': 'PRKCQ',
 'PRKD1': 'PRKD1',
 'PRKD2': 'PRKD2',
 'PRKD3': 'PRKD3',
 'PRKG1': 'PRKG1',
 'PRKG2': 'PRKG2',
 'PRKR': 'PRKR',
 'PRKX': 'PRKX',
 'PRP4': 'PRP4',
 'PYK2': 'PYK2',
 'QSK': 'QSK',
 'RAF1': 'RAF1',
 'RET': 'RET',
 'RIOK1': 'RIOK1',
 'RIOK2': 'RIOK2',
 'RIOK3': 'RIOK3',
 'RIPK1': 'RIPK1',
 'RIPK2': 'RIPK2',
 'RIPK4': 'RIPK4',
 'RIPK5': 'RIPK5',
 'ROCK1': 'ROCK1',
 'ROCK2': 'ROCK2',
 'ROS1': 'ROS1',
 'RPS6KA4(Kin.Dom.1-N-terminal)': 'RPS6KA4(Kin.Dom.1-N-terminal)',
 'RPS6KA4(Kin.Dom.2-C-terminal)': 'RPS6KA4(Kin.Dom.2-C-terminal)',
 'RPS6KA5(Kin.Dom.1-N-terminal)': 'RPS6KA5(Kin.Dom.1-N-terminal)',
 'RPS6KA5(Kin.Dom.2-C-terminal)': 'RPS6KA5(Kin.Dom.2-C-terminal)',
 'RSK1(Kin.Dom.1-N-terminal)': 'RSK1(Kin.Dom.1-N-terminal)',
 'RSK1(Kin.Dom.2-C-terminal)': 'RSK1(Kin.Dom.2-C-terminal)',
 'RSK2(Kin.Dom.1-N-terminal)': 'RSK2(Kin.Dom.1-N-terminal)',
 'RSK2(Kin.Dom.2-C-terminal)': 'RSK2(Kin.Dom.2-C-terminal)',
 'RSK3(Kin.Dom.1-N-terminal)': 'RSK3(Kin.Dom.1-N-terminal)',
 'RSK3(Kin.Dom.2-C-terminal)': 'RSK3(Kin.Dom.2-C-terminal)',
 'RSK4(Kin.Dom.1-N-terminal)': 'RSK4(Kin.Dom.1-N-terminal)',
 'RSK4(Kin.Dom.2-C-terminal)': 'RSK4(Kin.Dom.2-C-terminal)',
 'S6K1': 'S6K1',
 'SBK1': 'SBK1',
 'SGK': 'SGK',
 'SgK110': 'SgK110',
 'SGK2': 'SGK2',
 'SGK3': 'SGK3',
 'SIK': 'SIK',
 'SIK2': 'SIK2',
 'SLK': 'SLK',
 'SNARK': 'SNARK',
 'SNRK': 'SNRK',
 'SRC': 'SRC',
 'SRMS': 'SRMS',
 'SRPK1': 'SRPK1',
 'SRPK2': 'SRPK2',
 'SRPK3': 'SRPK3',
 'STK16': 'STK16',
 'STK33': 'STK33',
 'STK35': 'STK35',
 'STK36': 'STK36',
 'STK39': 'STK39',
 'SYK': 'SYK',
 'TAK1': 'TAK1',
 'TAOK1': 'TAOK1',
 'TAOK2': 'TAOK2',
 'TAOK3': 'TAOK3',
 'TBK1': 'TBK1',
 'TEC': 'TEC',
 'TESK1': 'TESK1',
 'TGFBR1': 'TGFBR1',
 'TGFBR2': 'TGFBR2',
 'TIE1': 'TIE1',
 'TIE2': 'TIE2',
 'TLK1': 'TLK1',
 'TLK2': 'TLK2',
 'TNIK': 'TNIK',
 'TNK1': 'TNK1',
 'TNK2': 'TNK2',
 'TNNI3K': 'TNNI3K',
 'TRKA': 'TRKA',
 'TRKB': 'TRKB',
 'TRKC': 'TRKC',
 'TRPM6': 'TRPM6',
 'TSSK1B': 'TSSK1B',
 'TSSK3': 'TSSK3',
 'TTK': 'TTK',
 'TXK': 'TXK',
 'TYK2(JH1domain-catalytic)': 'TYK2(JH1domain-catalytic)',
 'TYK2(JH2domain-pseudokinase)': 'TYK2(JH2domain-pseudokinase)',
 'TYRO3': 'TYRO3',
 'ULK1': 'ULK1',
 'ULK2': 'ULK2',
 'ULK3': 'ULK3',
 'VEGFR2': 'VEGFR2',
 'VPS34': 'VPS34',
 'VRK2': 'VRK2',
 'WEE1': 'WEE1',
 'WEE2': 'WEE2',
 'WNK1': 'WNK1',
 'WNK2': 'WNK2',
 'WNK3': 'WNK3',
 'WNK4': 'WNK4',
 'YANK1': 'YANK1',
 'YANK2': 'YANK2',
 'YANK3': 'YANK3',
 'YES': 'YES',
 'YSK1': 'YSK1',
 'YSK4': 'YSK4',
 'ZAK': 'ZAK',
 'ZAP70': 'ZAP70',
 'ABL1-nonphosphorylated': 'ABL1-nonphosphorylated/ABL1-phosphorylated',
 'ABL1-phosphorylated': 'ABL1-nonphosphorylated/ABL1-phosphorylated',
 'CDK4': 'CDK4/CDK4-cyclinD1/CDK4-cyclinD3',
 'CDK4-cyclinD1': 'CDK4/CDK4-cyclinD1/CDK4-cyclinD3',
 'CDK4-cyclinD3': 'CDK4/CDK4-cyclinD1/CDK4-cyclinD3',
 'FLT3': 'FLT3/FLT3-autoinhibited',
 'FLT3-autoinhibited': 'FLT3/FLT3-autoinhibited',
 'KIT': 'KIT/KIT-autoinhibited',
 'KIT-autoinhibited': 'KIT/KIT-autoinhibited',
 'CSF1R': 'CSF1R/CSF1R-autoinhibited',
 'CSF1R-autoinhibited': 'CSF1R/CSF1R-autoinhibited'}

def sscore(df_binned, k=10):
    #map k (PoC cutoff) to the right bin boundary
    poc_bin_dict = {10: 2,
                    35: 1}
    c = poc_bin_dict[k]
    
    sscore = df_binned.copy()
    #set (binned) hits >= c to 1, and hits < c to 0
    #0*x ensures that nan values are maintained and not set to 0
    sscore = sscore.apply(lambda x: np.where(x>=c, 1, 0*x))
    sscore = sscore.mean(1).values
    return sscore

if args.sscore_target_map_path is not None:
    #get kinase-eurofin mapping (eurofin groups some kinase for sscore calculations)
    sscore_target_map = pd.read_csv(args.sscore_target_map_path)
    #create dictionary mapping 
    sscore_target_group_dict = dict(zip(sscore_target_map['Full'], sscore_target_map['Eurofins_SScore']))
else:
    sscore_target_group_dict = default_sscore_target_group_dict

def is_sscore_target(x):
    return ('S(10)' in x)|('S(35)' in x)

def is_res_path(p):
    return ('_scores.csv' in p)|('_preds.csv' in p)|('_model.rds' in p)

def save_results(res, res_pth, update_append=False):
    current_res = None 
    if update_append:
        try:
            current_res = pd.read_csv(res_pth)
        except:
            print('No current/existing results file to append to \n %s DNE!'%res_pth)
    current_res = pd.concat([current_res, res], axis=0)
    current_res.to_csv(res_pth, index=False)
        
        
kinase_info = pd.read_csv('./data/kinases-on-panel-468.csv')
kinase_ls_master = kinase_info.Full.tolist()
target_ls_master = kinase_ls_master+['S(10)', 'S(35)']
# results_path = './results/LGB'
results_path = args.results_path


# res_path_ls = [os.path.join(results_path, p) for p in os.listdir(results_path) if ('model.rds' not in p)&('_all.csv' not in p)]
res_path_ls = [os.path.join(results_path, p) for p in os.listdir(results_path) if is_res_path(p)]
# preds_path_ls = [p for p in res_path_ls if ('_preds.csv' in p)]
preds_path_ls = [os.path.join(results_path, t+'_test_preds.csv') for t in target_ls_master]
# scores_path_ls = [p for p in res_path_ls if ('_scores.csv' in p)]
scores_path_ls = [os.path.join(results_path, t+'_test_scores.csv') for t in target_ls_master]
# target_ls = [os.path.split(p)[-1].split('_test_preds.csv')[0] for p in preds_path_ls]
models_path_ls = [os.path.join(results_path, t+'_model.rds') for t in target_ls_master]
target_ls = target_ls_master
sscore_ls = [t for t in target_ls if is_sscore_target(t)]
kinome_ls = [t for t in target_ls if (t not in sscore_ls)]

preds_ls  = [pd.read_csv(p) for p in preds_path_ls if not is_sscore_target(p)]
scores_ls  = [pd.read_csv(p) for p in scores_path_ls if not is_sscore_target(p)]
kinase_pred_models = [joblib.load(p) for p in models_path_ls if not is_sscore_target(p)]

sscore_preds_ls  = [pd.read_csv(p) for p in preds_path_ls if is_sscore_target(p)]
sscore_scores_ls  = [pd.read_csv(p) for p in scores_path_ls if is_sscore_target(p)]

id_cols = preds_ls[0].columns[~preds_ls[0].columns.str.contains('pred')].tolist()
id_cols_select = id_cols.copy()
try:
    id_cols_select.remove('Split')
except:
    print('No split column present')

try:
    id_cols_select.remove('MolWeight')
except:
    print('No MolWeight column present')

pred_label_ls = [pd.concat([p.loc[:, p.columns.isin(id_cols)], 
                            pd.DataFrame({t: p.loc[:, ~p.columns.isin(id_cols)].idxmax(axis=1, numeric_only=True).str.split('pred_').str[-1].astype(np.int32)})], axis=1) 
                 for t, p in zip(kinome_ls, preds_ls)]
scores_df = pd.concat(scores_ls, axis=0).reset_index(drop=True)
scores_df.insert(0, 'Target', kinome_ls)
save_results(scores_df, res_pth=os.path.join(results_path, 'test_scores_all.csv'), update_append=False) 
# scores_df.to_csv(os.path.join(results_path, 'test_scores_all.csv'), index=False)
# print('KinomePred model scores - median over targets: ', scores_df.median(axis=0, numeric_only=True))

sscores_scores_df = pd.concat(sscore_scores_ls, axis=0).reset_index(drop=True)
sscores_scores_df.insert(0, 'Target', sscore_ls) 
# sscores_scores_df.to_csv(os.path.join(results_path, 'S-score_scores_all.csv'), index=False)
save_results(sscores_scores_df, res_pth=os.path.join(results_path, 'S-score_scores_all.csv'), update_append=False) 

if args.save_all_preds:
    print('saving all pred probs')
    test_preds_all = preds_ls.copy()
    test_preds_all = [pd.concat([p.loc[:,id_cols_select], pd.DataFrame({t: p.loc[:,~p.columns.isin(id_cols)].values.tolist()})], axis=1) for t, p in zip(target_ls, preds_ls)]
    test_preds_all = pd.concat(test_preds_all, axis=1)
    test_preds_all = test_preds_all.loc[:,~test_preds_all.columns.duplicated()]
    test_preds_all.insert(0, 'run_date', current_date)
    # test_preds_all.to_csv(os.path.join(results_path, 'test_pred_probs_all.csv'), index=False)
    save_results(test_preds_all, res_pth=os.path.join(results_path, 'test_pred_probs_all.csv'), update_append=args.append_results)

#get structure and predicted probs for class 2 (PoC<=10) only
preds_ls = [preds.loc[:,id_cols_select+[preds.columns[-1]]] for preds in preds_ls]
#concatenate all the kinase class 2 probabilities
preds_df = pd.concat(preds_ls, axis=1)
preds_df = preds_df.loc[:,~preds_df.columns.duplicated()]
preds_df.insert(0, 'run_date', current_date)
# preds_df.to_csv(os.path.join(results_path, 'test_pred_probs_class2_all.csv'), index=False)
save_results(preds_df, res_pth=os.path.join(results_path, 'test_pred_probs_class2_all.csv'), update_append=args.append_results)

#compile predicted kinome profile
sscore_pred_df = pd.concat(sscore_preds_ls, axis=1)
sscore_pred_df = sscore_pred_df.loc[:,~sscore_pred_df.columns.duplicated()]

kinome_pred_df = pd.concat(pred_label_ls, axis=1)
kinome_pred_df = kinome_pred_df.loc[:,~kinome_pred_df.columns.duplicated()]
#get selected target cols for sscore calculation
select_targets = kinome_ls #list(sscore_target_group_dict.keys())
sscore_df = kinome_pred_df.copy()[select_targets]
# kinome_df = kinome_df.loc[:, ~kinome_df.columns.isin(list(set(target_ls)-set(select_targets)))]
#group according to the kinase-eurofin map data specifications (most kinase make up their own group)
#get within-group maximums -- applied to yield a hit in sscore calculation if there's a hit for at least one group member
sscore_df = sscore_df.groupby(sscore_target_group_dict, axis=1).max()
#get SScore S(k), where k is your chosen PoC cutoff (right endpoint - inclusive)
kinome_pred_df['S(10)'] = sscore(sscore_df, k=10)
kinome_pred_df['S(35)'] = sscore(sscore_df, k=35)
kinome_pred_df['S(10)_direct'] = sscore_pred_df['S(10)_pred'].values
kinome_pred_df['S(35)_direct'] = sscore_pred_df['S(35)_pred'].values
try:
    kinome_pred_df.drop(columns=['Split'], inplace=True)
except:
    print('No split column present')
try:
    kinome_pred_df.drop(columns=['MolWeight'], inplace=True)
except:
    print('No split column present')
kinome_pred_df.insert(0, 'run_date', current_date)
# kinome_pred_df.to_csv(os.path.join(results_path, 'kinome_pred_all.csv'), index=False)
save_results(kinome_pred_df, res_pth=os.path.join(results_path, 'kinome_pred_all.csv'), update_append=args.append_results)


#save combined model -- all 468 kinase pred models as a single multi-output model
multiout_model = MultiOutputClassifier(n_outputs=len(kinase_pred_models))
multiout_model.classifiers = kinase_pred_models
joblib.dump(multiout_model, os.path.join(results_path, 'model_combined.rds'))

for p in models_path_ls:
    if not is_sscore_target(p):
        os.remove(p)


for p in preds_path_ls:
    if not is_sscore_target(p):
        os.remove(p)


for p in scores_path_ls:
    if not is_sscore_target(p):
        os.remove(p)