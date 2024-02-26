import os
from datetime import date, timedelta
import re
import pandas as pd
import numpy as np
import argparse
from argparse import *
from learner import score_regression, score_multiclass


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("results_path",type=str, help="path to multiclass Kinome data")
parser.add_argument("--sscore_target_map_path", default = None, type=str, help="path to kinase-eurofin group mappings")
parser.add_argument("--save_all_preds", action='store_true', help="set flag to save all class predicted probabilities")
parser.add_argument("--append_results", action='store_true', help="set flag to append results to respective current files")

args=parser.parse_args()
print(args)
# args=parser.parse_args(["../results/MPNN"])

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

def save_results(res, res_pth, update_append=False):
    current_res = None 
    if update_append:
        try:
            current_res = pd.read_csv(res_pth)
        except:
            print('No current/existing results file to append to \n %s DNE!'%p)
    current_res = pd.concat([current_res, res], axis=0)
    current_res.to_csv(res_pth, index=False)
        
def clean_mpnn_structures(smiles_ls):
    smiles_ls_clean = [smiles[2:-2] for smiles in smiles_ls]
    smiles_ls_clean = [smiles.replace('\\\\', '\\') for smiles in smiles_ls_clean] 
    return smiles_ls_clean


def str_ls_to_float_ls(x):
    if ', ' in x:
      delim = ', ' 
    else:
        delim = ' '
        x = '['+x.strip('[]').strip()+']'
        x = ' '.join(x.split())
    # print('delim: ', delim)
    return [float(val) for val in x.strip('[]').split(delim)]

# results_path = './results/LGB'
results_path = args.results_path


kinome_exp = pd.read_csv('./data/kinome_data_multiclass_current.csv')
kinome_exp_test = kinome_exp[kinome_exp.Split=='test'].copy().reset_index(drop=True)
del kinome_exp
kinase_cols = kinome_exp_test.columns[~kinome_exp_test.columns.isin(['Compound Name', 'Split', 'Structure', 'S(10)', 'S(35)'])].tolist()

kinome_test_preds = pd.read_csv(os.path.join(results_path, 'kinome_test_preds.csv'))
sscore_test_preds = pd.read_csv(os.path.join(results_path, 'SScore_test_preds.csv'))

kinome_test_preds.rename({'smiles': 'Structure'}, axis=1, inplace=True)
sscore_test_preds.rename({'smiles': 'Structure'}, axis=1, inplace=True)

kinome_test_preds.Structure = clean_mpnn_structures(kinome_test_preds.Structure.tolist())
sscore_test_preds.Structure = clean_mpnn_structures(sscore_test_preds.Structure.tolist())

kinome_test_preds[kinase_cols]=kinome_test_preds[kinase_cols].applymap(lambda x: str_ls_to_float_ls(x))

#Add compound name and run date
if all(kinome_test_preds.Structure == kinome_exp_test.Structure)==all(sscore_test_preds.Structure == kinome_exp_test.Structure):
    print('Smiles match original test set.')
    kinome_test_preds = pd.concat([kinome_exp_test[['Compound Name']], kinome_test_preds], axis=1)
    kinome_test_preds.insert(0, 'run_date', current_date)
    sscore_test_preds = pd.concat([kinome_exp_test[['Compound Name']], sscore_test_preds], axis=1)   
    sscore_test_preds.insert(0, 'run_date', current_date)                          
else:
    print('Smiles do not match original test set, cannot proceed.')



if args.save_all_preds:
    test_preds_all = pd.concat([kinome_test_preds, sscore_test_preds], axis=1)
    test_preds_all = test_preds_all.loc[:,~test_preds_all.columns.duplicated()]
    save_results(test_preds_all, res_pth=os.path.join(results_path, 'test_pred_probs_all.csv'), update_append=args.append_results)


test_preds_class2 = kinome_test_preds.copy()
test_preds_class2[kinase_cols] = test_preds_class2[kinase_cols].applymap(lambda x: x[2])
save_results(test_preds_class2, res_pth=os.path.join(results_path, 'test_pred_probs_class2_all.csv'), update_append=args.append_results)




kinome_pred_scores_ls =  []
for kinase in kinase_cols:
    score = score_multiclass(np.vstack(kinome_test_preds[kinase].values), kinome_exp_test[kinase].values, classes=np.array([0,1,2]))
    kinome_pred_scores_ls += [score]
kinome_pred_scores = pd.concat(kinome_pred_scores_ls, axis=0).reset_index(drop=True)
kinome_pred_scores.insert(0, 'Target', kinase_cols)
save_results(kinome_pred_scores, res_pth=os.path.join(results_path, 'test_scores_all.csv'), update_append=False) 


sscore_pred_scores_ls =  []
sscore_pred_scores_ls += [score_regression(sscore_test_preds['S(10)'].values, kinome_exp_test['S(10)'].values)]
sscore_pred_scores_ls += [score_regression(sscore_test_preds['S(35)'].values, kinome_exp_test['S(35)'].values)]
sscore_pred_scores = pd.concat(sscore_pred_scores_ls, axis=0).reset_index(drop=True)
sscore_pred_scores.insert(0, 'Target', ['S(10)', 'S(35)'])
save_results(sscore_pred_scores, res_pth=os.path.join(results_path, 'S-score_scores_all.csv'), update_append=False) 


kinome_sscore_test_preds = kinome_test_preds.copy()
kinome_sscore_test_preds[kinase_cols] = kinome_sscore_test_preds[kinase_cols].applymap(lambda x: np.array(x).argmax())

sscore_df = kinome_sscore_test_preds.copy()[kinase_cols]
# kinome_df = kinome_df.loc[:, ~kinome_df.columns.isin(list(set(target_ls)-set(select_targets)))]
#group according to the kinase-eurofin map data specifications (most kinase make up their own group)
#get within-group maximums -- applied to yield a hit in sscore calculation if there's a hit for at least one group member
sscore_df = sscore_df.groupby(sscore_target_group_dict, axis=1).max()
#get SScore S(k), where k is your chosen PoC cutoff (right endpoint - inclusive)
kinome_sscore_test_preds['S(10)'] = sscore(sscore_df, k=10)
kinome_sscore_test_preds['S(35)'] = sscore(sscore_df, k=35)
kinome_sscore_test_preds['S(10)_direct'] = sscore_test_preds['S(10)'].values
kinome_sscore_test_preds['S(35)_direct'] = sscore_test_preds['S(35)'].values
save_results(kinome_sscore_test_preds, res_pth=os.path.join(results_path, 'kinome_pred_all.csv'), update_append=args.append_results)