import joblib
import os
import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from Mold2_pywrapper import Mold2
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.AtomPairs import Pairs,Torsions
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import Fragments
from rdkit.Chem import EState
from utils import to_str

'''
smiles_list ex: smiles_list = [
        # erlotinib
        "n1cnc(c2cc(c(cc12)OCCOC)OCCOC)Nc1cc(ccc1)C#C",
        # midecamycin
        "CCC(=O)O[C@@H]1CC(=O)O[C@@H](C/C=C/C=C/[C@@H]([C@@H](C[C@@H]([C@@H]([C@H]1OC)O[C@H]2[C@@H]([C@H]([C@@H]([C@H](O2)C)O[C@H]3C[C@@]([C@H]([C@@H](O3)C)OC(=O)CC)(C)O)N(C)C)O)CC=O)C)O)C",
        # selenofolate
        "C1=CC(=CC=C1C(=O)NC(CCC(=O)OCC[Se]C#N)C(=O)O)NCC2=CN=C3C(=N2)C(=O)NC(=N3)N",
        # cisplatin
        "N.N.Cl[Pt]Cl"
    ]
'''

# def standardize(mol, activity_tag, smiles_tag='SMILES'):
def standardize(mol, smiles_tag='SMILES'):
    try:
#         activity_label = mol.GetProp(activity_tag)
        name_label = mol.GetProp('_Name')
        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol) 

        # if many fragments, get the "parent" (the actual mol we are interested in) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # try to Canonicalize tautomers
        te = rdMolStandardize.TautomerEnumerator() 
        mol_final = te.Canonicalize(uncharged_parent_clean_mol)
#         mol_final = uncharged_parent_clean_mol

        #Cleanup and possibly other standardizations may remove original properties. 
        #we have to re-add smiles and the activity label
        if len(mol_final.GetPropsAsDict())==0:
            smiles_final = Chem.MolToSmiles(mol_final)
            mol_final.SetProp(smiles_tag, smiles_final)
#             mol_final.SetProp(activity_tag, str(activity_label))
            mol_final.SetProp('_Name', name_label)
    except:
        try:
            mol_final = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        except:
            mol_final = mol

    return mol_final


def descriptor_calc(mol, descriptors):
    # rdkit, fragment, autocorr2d
    # rdkit---> 155;
    # fragment ---> 85
    # autocorr2d ---> 192
    # mold2 --> 777
    # mol = standardize(mol)
    MDlist = []
    if 'rdkit' in descriptors: ## 31 (phschem) + 19 (topo) + 38 (MolSurf) + 25 (EState) + 42 (MQNs) = 155
        ##########################################################
        ##### physiochemical desc ---> 31 descriptors 
        ##########################################################
        # Lipinski descriptors (19)
        try:
            MDlist.append(rdMolDescriptors.CalcFractionCSP3(mol))
            MDlist.append(Lipinski.NHOHCount(mol))                # new!!!
            MDlist.append(Lipinski.NOCount(mol))                  # new!!!
            MDlist.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
            MDlist.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAromaticRings(mol))
            MDlist.append(rdMolDescriptors.CalcNumHBA(mol))
            MDlist.append(rdMolDescriptors.CalcNumHBD(mol))
            #MDlist.append(rdMolDescriptors.CalcNumLipinskiHBA(mol)) # Lipinski.NumHAcceptors(x)
            #MDlist.append(rdMolDescriptors.CalcNumLipinskiHBD(mol)) # Lipinski.NumHDonors(x)          
            MDlist.append(rdMolDescriptors.CalcNumHeteroatoms(mol)) # Lipinski.NumHeteroatoms(x)
            MDlist.append(rdMolDescriptors.CalcNumRotatableBonds(mol)) # Lipinski.NumRotatableBonds(x)
            MDlist.append(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
            MDlist.append(rdMolDescriptors.CalcNumRings(mol)) # Lipinski.RingCount(x)
            MDlist.append(rdMolDescriptors.CalcNumHeavyAtoms(mol)) # new!!!
            MDlist.append(Descriptors.HeavyAtomMolWt(mol))  # new!! redundant to CalcNumHeavyAtoms(mol)

            # Crippen descriptors (2)
            for d in rdMolDescriptors.CalcCrippenDescriptors(mol): #2 descr
                MDlist.append(d)

            # Other physiochemical descriptors (10)
            MDlist.append(rdMolDescriptors.CalcExactMolWt(mol)) # Descriptors.ExactMolWt
            #MDlist.append(Descriptors.MolWt(mol))  # redundant to Descriptors.ExactMolWt
            MDlist.append(Descriptors.FpDensityMorgan1(mol))   # new !!
            MDlist.append(Descriptors.FpDensityMorgan2(mol))   # new !!
            MDlist.append(Descriptors.FpDensityMorgan3(mol))   # new !!
            MDlist.append(Descriptors.MinAbsPartialCharge(mol)) # new !!
            MDlist.append(Descriptors.MaxAbsPartialCharge(mol)) # new !!
            MDlist.append(Descriptors.MinPartialCharge(mol)) # new !!
            MDlist.append(Descriptors.MaxPartialCharge(mol)) # new !!
            MDlist.append(Descriptors.NumValenceElectrons(mol)) # new !!
            MDlist.append(Descriptors.NumRadicalElectrons(mol)) # new !!
            
            ##########################################################
            ##### topological (shape) desc  ----> 19 descriptors
            ##########################################################
            MDlist.append(rdMolDescriptors.CalcHallKierAlpha(mol))  # GraphDescriptors.HallKierAlpha(x)
            MDlist.append(rdMolDescriptors.CalcKappa1(mol))         # GraphDescriptors.Kappa1(x)
            MDlist.append(rdMolDescriptors.CalcKappa2(mol))
            MDlist.append(rdMolDescriptors.CalcKappa3(mol))
            MDlist.append(GraphDescriptors.Chi0(mol))           # new!!!!
            MDlist.append(rdMolDescriptors.CalcChi0n(mol))     # GraphDescriptors.Chi0n(x)
            MDlist.append(rdMolDescriptors.CalcChi0v(mol))     #
            MDlist.append(GraphDescriptors.Chi1(mol))           # new!!!!
            MDlist.append(rdMolDescriptors.CalcChi1n(mol))     #
            MDlist.append(rdMolDescriptors.CalcChi1v(mol))     #
            MDlist.append(rdMolDescriptors.CalcChi2n(mol))     #
            MDlist.append(rdMolDescriptors.CalcChi2v(mol))     #
            MDlist.append(rdMolDescriptors.CalcChi3n(mol))     #
            MDlist.append(rdMolDescriptors.CalcChi3v(mol))     #
            MDlist.append(rdMolDescriptors.CalcChi4n(mol))     #
            MDlist.append(rdMolDescriptors.CalcChi4v(mol))     #
            MDlist.append(GraphDescriptors.BalabanJ(mol))      # new!!!!
            MDlist.append(GraphDescriptors.BertzCT(mol))       # new!!!!
            #MDlist.append(GraphDescriptors.Ipc(mol))           # new!!!!
              
            ##########################################################
            # MolSurf descriptors (38) (MOE-like approximate molecular surface area descriptors)
            ##########################################################
            MDlist.append(rdMolDescriptors.CalcTPSA(mol))       # MolSurf.TPSA
            MDlist.append(rdMolDescriptors.CalcLabuteASA(mol))
            for d in rdMolDescriptors.PEOE_VSA_(mol): #14 descr
                MDlist.append(d)
            for d in rdMolDescriptors.SMR_VSA_(mol): #10 descr
                MDlist.append(d)
            for d in rdMolDescriptors.SlogP_VSA_(mol): #12 descr
                MDlist.append(d)

            ##########################################################
            # EState.Estate (# Hybrid EState-VSA descriptors-- like MOE VSA descriptors) ----> 25 descriptors
            ##########################################################
            MDlist.append(EState.EState_VSA.EState_VSA1(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.EState_VSA2(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.EState_VSA3(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.EState_VSA4(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.EState_VSA5(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.EState_VSA6(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.EState_VSA7(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.EState_VSA8(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.EState_VSA9(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.EState_VSA10(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.EState_VSA11(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.VSA_EState1(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.VSA_EState2(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.VSA_EState3(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.VSA_EState4(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.VSA_EState5(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.VSA_EState6(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.VSA_EState7(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.VSA_EState8(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.VSA_EState9(mol))   # new!!!!
            MDlist.append(EState.EState_VSA.VSA_EState10(mol))   # new!!!!
            MDlist.append(EState.EState.MaxAbsEStateIndex(mol))   # new!!!!
            MDlist.append(EState.EState.MaxEStateIndex(mol))   # new!!!!
            MDlist.append(EState.EState.MinAbsEStateIndex(mol))   # new!!!!
            MDlist.append(EState.EState.MinEStateIndex(mol))   # new!!!!

            # MQNs descriptors (42) ---> molecular quantum numbers
            for d in rdMolDescriptors.MQNs_(mol): #42 descr
                MDlist.append(d)

        except:
            print('The rdkit descriptor calculation failed!')

        ##########################################################
        # Fragment descriptors  ----> 85 descriptors
        ##########################################################
    if 'fragment' in descriptors:
        try:
            MDlist.append(Fragments.fr_Al_COO(mol)) 
            MDlist.append(Fragments.fr_Al_OH(mol)) 
            MDlist.append(Fragments.fr_Al_OH_noTert(mol)) 
            MDlist.append(Fragments.fr_ArN(mol)) 
            MDlist.append(Fragments.fr_Ar_COO(mol)) 
            MDlist.append(Fragments.fr_Ar_N(mol)) 
            MDlist.append(Fragments.fr_Ar_NH(mol)) 
            MDlist.append(Fragments.fr_Ar_OH(mol)) 
            MDlist.append(Fragments.fr_COO(mol)) 
            MDlist.append(Fragments.fr_COO2(mol)) 
            MDlist.append(Fragments.fr_C_O(mol)) 
            MDlist.append(Fragments.fr_C_O_noCOO(mol)) 
            MDlist.append(Fragments.fr_C_S(mol)) 
            MDlist.append(Fragments.fr_HOCCN(mol)) 
            MDlist.append(Fragments.fr_Imine(mol)) 
            MDlist.append(Fragments.fr_NH0(mol)) 
            MDlist.append(Fragments.fr_NH1(mol)) 
            MDlist.append(Fragments.fr_NH2(mol)) 
            MDlist.append(Fragments.fr_N_O(mol)) 
            MDlist.append(Fragments.fr_Ndealkylation1(mol)) 
            MDlist.append(Fragments.fr_Ndealkylation2(mol)) 
            MDlist.append(Fragments.fr_Nhpyrrole(mol)) 
            MDlist.append(Fragments.fr_SH(mol)) 
            MDlist.append(Fragments.fr_aldehyde(mol)) 
            MDlist.append(Fragments.fr_alkyl_carbamate(mol)) 
            MDlist.append(Fragments.fr_alkyl_halide(mol)) 
            MDlist.append(Fragments.fr_allylic_oxid(mol)) 
            MDlist.append(Fragments.fr_amide(mol)) 
            MDlist.append(Fragments.fr_amidine(mol)) 
            MDlist.append(Fragments.fr_aniline(mol)) 
            MDlist.append(Fragments.fr_aryl_methyl(mol)) 
            MDlist.append(Fragments.fr_azide(mol)) 
            MDlist.append(Fragments.fr_azo(mol)) 
            MDlist.append(Fragments.fr_barbitur(mol)) 
            MDlist.append(Fragments.fr_benzene(mol)) 
            MDlist.append(Fragments.fr_benzodiazepine(mol)) 
            MDlist.append(Fragments.fr_bicyclic(mol)) 
            MDlist.append(Fragments.fr_diazo(mol)) 
            MDlist.append(Fragments.fr_dihydropyridine(mol)) 
            MDlist.append(Fragments.fr_epoxide(mol)) 
            MDlist.append(Fragments.fr_ester(mol)) 
            MDlist.append(Fragments.fr_ether(mol)) 
            MDlist.append(Fragments.fr_furan(mol)) 
            MDlist.append(Fragments.fr_guanido(mol)) 
            MDlist.append(Fragments.fr_halogen(mol)) 
            MDlist.append(Fragments.fr_hdrzine(mol)) 
            MDlist.append(Fragments.fr_hdrzone(mol)) 
            MDlist.append(Fragments.fr_imidazole(mol)) 
            MDlist.append(Fragments.fr_imide(mol)) 
            MDlist.append(Fragments.fr_isocyan(mol)) 
            MDlist.append(Fragments.fr_isothiocyan(mol)) 
            MDlist.append(Fragments.fr_ketone(mol)) 
            MDlist.append(Fragments.fr_ketone_Topliss(mol)) 
            MDlist.append(Fragments.fr_lactam(mol)) 
            MDlist.append(Fragments.fr_lactone(mol)) 
            MDlist.append(Fragments.fr_methoxy(mol)) 
            MDlist.append(Fragments.fr_morpholine(mol)) 
            MDlist.append(Fragments.fr_nitrile(mol)) 
            MDlist.append(Fragments.fr_nitro(mol)) 
            MDlist.append(Fragments.fr_nitro_arom(mol)) 
            MDlist.append(Fragments.fr_nitro_arom_nonortho(mol)) 
            MDlist.append(Fragments.fr_nitroso(mol)) 
            MDlist.append(Fragments.fr_oxazole(mol)) 
            MDlist.append(Fragments.fr_oxime(mol)) 
            MDlist.append(Fragments.fr_para_hydroxylation(mol)) 
            MDlist.append(Fragments.fr_phenol(mol)) 
            MDlist.append(Fragments.fr_phenol_noOrthoHbond(mol)) 
            MDlist.append(Fragments.fr_phos_acid(mol)) 
            MDlist.append(Fragments.fr_phos_ester(mol)) 
            MDlist.append(Fragments.fr_piperdine(mol)) 
            MDlist.append(Fragments.fr_piperzine(mol)) 
            MDlist.append(Fragments.fr_priamide(mol)) 
            MDlist.append(Fragments.fr_prisulfonamd(mol)) 
            MDlist.append(Fragments.fr_pyridine(mol)) 
            MDlist.append(Fragments.fr_quatN(mol)) 
            MDlist.append(Fragments.fr_sulfide(mol)) 
            MDlist.append(Fragments.fr_sulfonamd(mol)) 
            MDlist.append(Fragments.fr_sulfone(mol)) 
            MDlist.append(Fragments.fr_term_acetylene(mol)) 
            MDlist.append(Fragments.fr_tetrazole(mol)) 
            MDlist.append(Fragments.fr_thiazole(mol)) 
            MDlist.append(Fragments.fr_thiocyan(mol)) 
            MDlist.append(Fragments.fr_thiophene(mol)) 
            MDlist.append(Fragments.fr_unbrch_alkane(mol)) 
            MDlist.append(Fragments.fr_urea(mol)) 
        except:
            print("The fragment descriptor calculation failed!")
            

    if 'autocorr2d' in descriptors:
        # AutoCorr2D
        try:
            for d in rdMolDescriptors.CalcAUTOCORR2D(mol):  #192 descr
                MDlist.append(d)
        except:
            print('The AutoCorr2D descriptor calculation failed!')
    
    if 'mold2' in descriptors:
        mold2 = Mold2()
        if type(mol) is list:
            # print('is list')
            MDlist += mold2.calculate(mol, show_banner=False).values.tolist()[0]
        else:
            # print('not list')
            MDlist += mold2.calculate([mol], show_banner=False).values.tolist()[0]
    return MDlist

def compute_descriptors(smiles_list, desc = 'mold2'):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    mold2 = Mold2()
    mold2_desc = mold2.calculate(mols)
    print(mold2_desc)
    return(mold2_desc)

#####################
# calculate fingerprints #
#####################

def fingerprint_calc(mol,fingerprints):
    # MACCS, ECFP4, ECFP6, FCFP4, FCFP6, APFP, Torsion
    FPlist = []
    if 'MACCS' in fingerprints: ## rdkit MACCS has 167 bit, but the bit 0 is always 0 --> ignored
        try:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            maccs_fplist = maccs_fp.ToBitString()[1:]
        except:
            maccs_fplist = ""  
        FPlist.append(maccs_fplist)
    
    if 'ECFP4' in fingerprints:
        try:
            ecfp4_fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,useFeatures=False,nBits=1024)
            ecfp4_fplist = ecfp4_fp.ToBitString()
        except:
            ecfp4_fplist = ""
        FPlist.append(ecfp4_fplist)
        
    if 'ECFP6' in fingerprints:
        try:
            ecfp6_fp = AllChem.GetMorganFingerprintAsBitVect(mol,3,useFeatures=False,nBits=2048)
            ecfp6_fplist = ecfp6_fp.ToBitString()
        except:
            ecfp6_fplist = ""
        FPlist.append(ecfp6_fplist)
    
    if 'FCFP4' in fingerprints:
        try:
            fcfp4_fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,useFeatures=True,nBits=1024)
            fcfp4_fplist = fcfp4_fp.ToBitString()
        except:
            fcfp4_fplist = ""
        FPlist.append(fcfp4_fplist)
        
    if 'FCFP6' in fingerprints:
        try:
            fcfp6_fp = AllChem.GetMorganFingerprintAsBitVect(mol,3,useFeatures=True,nBits=2048)
            fcfp6_fplist = fcfp6_fp.ToBitString()
        except:
            fcfp6_fplist = ""
        FPlist.append(fcfp6_fplist)
    
    if 'APFP' in fingerprints:
        try:
            apfp_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol,nBits=2048)
            apfp_fplist = apfp_fp.ToBitString()
        except:
            apfp_fplist =""
        FPlist.append(apfp_fplist)
        
    if 'Torsion' in fingerprints:
        try:
            torstion_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol,nBits=2048)
            torstion_fplist = torstion_fp.ToBitString()
        except:
            torstion_fplist = ""
        FPlist.append(torstion_fplist)
    
    FP_bit= []
    for i in FPlist:
        for bit in i:
            FP_bit.append(int(bit))
    return FPlist,FP_bit


#####################################
# Descriptor + FPs names
#####################################
descriptor_names = {'rdkit':['FractionCSP3','NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings',
                            'NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors','NumHDonors',
                            'NumHeteroatoms','NumRotatableBonds','NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings',
                            'RingCount','HeavyAtomCount','HeavyAtomMolWt','MolLogP','MolMR','ExactMolWt','FpDensityMorgan1',
                            'FpDensityMorgan2','FpDensityMorgan3','MinAbsPartialCharge','MaxAbsPartialCharge','MinPartialCharge','MaxPartialCharge',
                            'NumValenceElectrons','NumRadicalElectrons','HallKierAlpha','Kappa1','Kappa2','Kappa3','Chi0','Chi0n','Chi0v','Chi1',
                            'Chi1n','Chi1v','Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v','BalabanJ','BertzCT','TPSA','LabuteASA',
                            'PEOE_VSA1','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5','PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','PEOE_VSA10',
                            'PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','SMR_VSA1','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5','SMR_VSA6',
                            'SMR_VSA7','SMR_VSA8','SMR_VSA9','SMR_VSA10','SlogP_VSA1','SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5',
                            'SlogP_VSA6','SlogP_VSA7','SlogP_VSA8','SlogP_VSA9','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','EState_VSA1',
                            'EState_VSA2','EState_VSA3','EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8','EState_VSA9',
                            'EState_VSA10','EState_VSA11','VSA_EState1','VSA_EState2','VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6',
                            'VSA_EState7','VSA_EState8','VSA_EState9','VSA_EState10','MaxAbsEStateIndex','MaxEStateIndex','MinAbsEStateIndex',
                            'MinEStateIndex','MQNs1','MQNs2','MQNs3','MQNs4','MQNs5','MQNs6','MQNs7','MQNs8','MQNs9','MQNs10','MQNs11','MQNs12',
                            'MQNs13','MQNs14','MQNs15','MQNs16','MQNs17','MQNs18','MQNs19','MQNs20','MQNs21','MQNs22','MQNs23','MQNs24','MQNs25',
                            'MQNs26','MQNs27','MQNs28','MQNs29','MQNs30','MQNs31','MQNs32','MQNs33','MQNs34','MQNs35','MQNs36','MQNs37','MQNs38',
                            'MQNs39','MQNs40','MQNs41','MQNs42'],
                   'fragment':['fr_Al_COO','fr_Al_OH','fr_Al_OH_noTert','fr_ArN','fr_Ar_COO','fr_Ar_N','fr_Ar_NH','fr_Ar_OH','fr_COO',
                               'fr_COO2','fr_C_O','fr_C_O_noCOO','fr_C_S','fr_HOCCN','fr_Imine','fr_NH0','fr_NH1','fr_NH2','fr_N_O',
                               'fr_Ndealkylation1','fr_Ndealkylation2','fr_Nhpyrrole','fr_SH','fr_aldehyde','fr_alkyl_carbamate',
                               'fr_alkyl_halide','fr_allylic_oxid','fr_amide','fr_amidine','fr_aniline','fr_aryl_methyl','fr_azide',
                               'fr_azo','fr_barbitur','fr_benzene','fr_benzodiazepine','fr_bicyclic','fr_diazo','fr_dihydropyridine',
                               'fr_epoxide','fr_ester','fr_ether','fr_furan','fr_guanido','fr_halogen','fr_hdrzine','fr_hdrzone',
                               'fr_imidazole','fr_imide','fr_isocyan','fr_isothiocyan','fr_ketone','fr_ketone_Topliss','fr_lactam',
                               'fr_lactone','fr_methoxy','fr_morpholine','fr_nitrile','fr_nitro','fr_nitro_arom','fr_nitro_arom_nonortho',
                               'fr_nitroso','fr_oxazole','fr_oxime','fr_para_hydroxylation','fr_phenol','fr_phenol_noOrthoHbond','fr_phos_acid',
                               'fr_phos_ester','fr_piperdine','fr_piperzine','fr_priamide','fr_prisulfonamd','fr_pyridine','fr_quatN','fr_sulfide',
                               'fr_sulfonamd','fr_sulfone','fr_term_acetylene','fr_tetrazole','fr_thiazole','fr_thiocyan','fr_thiophene','fr_unbrch_alkane','fr_urea'],
                   'autocorr2d':[f'autocorr2d_{i+1}' for i in range(192)],
                   'mold2':[f'mold2_{i+1}' for i in range(777)]}


def descriptor_fp_names(descriptors,fps):
    # rdkit, fragment, autocorr2d
    # MACCS, ECFP4, ECFP6, FCFP4, FCFP6, APFP, Torsion
    desc_fp_names = []
    if 'rdkit' in descriptors:
        desc_fp_names.extend(descriptor_names['rdkit'])
    if 'fragment' in descriptors:
        desc_fp_names.extend(descriptor_names['fragment'])
    if 'autocorr2d' in descriptors:
        desc_fp_names.extend(descriptor_names['autocorr2d'])
    if 'mold2' in descriptors:
        # print('adding mo=ld2 descriptor names')
        # print(descriptor_names['Mold2'])
        desc_fp_names.extend(descriptor_names['mold2'])
    if '3Dbasic' in descriptors:
        desc_fp_names.extend(descriptor_names['3Dbasic'])
        
    if 'MACCS' in fps:
        desc_fp_names.extend([f'MACCS_{i+1}' for i in range(166)])
    if 'ECFP4' in fps:
        desc_fp_names.extend([f'ECFP4_{i+1}' for i in range(1024)])
    if 'ECFP6' in fps:
        desc_fp_names.extend([f'ECFP6_{i+1}' for i in range(2048)])
    if 'FCFP4' in fps:
        desc_fp_names.extend([f'FCFP4_{i+1}' for i in range(1024)])
    if 'FCFP6' in fps:
        desc_fp_names.extend([f'FCFP6_{i+1}' for i in range(2048)])
    if 'APFP' in fps:
        desc_fp_names.extend([f'APFP_{i+1}' for i in range(2048)])
    if 'Torsion' in fps:
        desc_fp_names.extend([f'Torsion_{i+1}' for i in range(2048)])
    return desc_fp_names



# def compute_features(sdFile, descType, fpType, activity_tag):
def compute_features(mols, descType, fpType):
    # descType = descType.lower()
    try:
        (descType in list(descriptor_names.keys())) | (descType is None)
        print("Descriptors: ", descType)
    except ValueError:
        print('Descriptor type %s is either not supported'%descType)
        raise
    # try:
    #     (fpType in list(descriptor_names.keys())) | (fpType is None)
    # except ValueError:
    #     print('Descriptor type %s is either not supported'%descType)
        raise
    #
    i=1
    outdict = {}
    # print("Descriptors: ", descType)
    print("Fingerprints: ", fpType)
    for Rmol in mols:
        outlist = []
        if Rmol is not None:
            mol = standardize(Rmol)
            smi = Chem.MolToSmiles(mol)
            try:
                molName = mol.GetProp('Name')
            except:
                try:
                    molName = mol.GetProp('_Name')        
                except:
                    molName = "mol_%i"%i

#             try:
#                 activity = mol.GetProp('%s' % activity_tag)
#             except KeyError:
#                 activity = '0.00000'

            i=i+1
            
            outlist.append(smi)
#             outlist.append(activity)
            if descType is not None:
                MDlist = descriptor_calc(mol,descType)
                outlist.extend(MDlist)
            if fpType is not None:
                FPlist,FP_bit = fingerprint_calc(mol,fpType)
                outlist.extend(FP_bit)           

            outdict[molName] = outlist
    
    #############################################
    # save calculated descriptors + fps + other properites        
    #############################################
    desc_fp_names = descriptor_fp_names(to_str(descType),to_str(fpType))
    desc_fp_names = ['Smiles']+desc_fp_names
    #for key, value in outdict.items() :
        #print (key, value)
    df_out = pd.DataFrame.from_dict(outdict,orient='index',columns=desc_fp_names)
    #convert activity back to int
#     df_out[activity_tag] = df_out[activity_tag].astype(np.int64)
    return df_out