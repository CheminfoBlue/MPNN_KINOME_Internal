#!/bin/bash


#SBATCH --job-name=MPNN_WeightedCE_Kinase_PoC_MultiTask_Predict

#SBATCH --partition=gpu-t4-4x

#SBATCH --output=MPNN_WeightedCE_Kinase_PoC_MultiTask_Predict_out.log

#SBATCH --error=MPNN_WeightedCE_Kinase_PoC_MultiTask_Predict_error.log

#SBATCH -N 1

source activate chemprop-BP
python ./code/get_test_data.py | chemprop_predict --smiles_columns Structure --checkpoint_path ./results/MPNN/KinomePred/fold_0/model_0/model.pt --preds_path ./results/MPNN/KinomePred/test_preds.csv --drop_extra_columns
