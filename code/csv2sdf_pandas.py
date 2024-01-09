import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import AllChem 
import sys,os

csv_file = sys.argv[1]
data = pd.read_csv(csv_file) 

PandasTools.AddMoleculeColumnToFrame(data,smilesCol='SMILES', molCol='Mol') 
data = data.dropna()  # remove bad structures
#data = data.drop(['level_0'], axis=1) # remove index


PandasTools.WriteSDF(data, '%s.sdf' %csv_file[:-4],  molColName='Mol', idName='RowID',properties=list(data.columns))