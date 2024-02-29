#!/bin/bash


#SBATCH --job-name=MPNN_PoC_SScore_MultiTask_Predict

#SBATCH --partition=gpu-t4-4x

#SBATCH --output=MPNN_PoC_SScore_MultiTask_Predict_out.log

#SBATCH --error=MPNN_PoC_SScore_MultiTask_Predict_error.log

#SBATCH -N 1

source activate chemprop-BP
python ./code/get_test_data.py | chemprop_predict --smiles_columns Structure --checkpoint_path ./results/MPNN/SScorePred/fold_0/model_0/model.pt --preds_path ./results/MPNN/SScorePred/test_preds.csv --drop_extra_columns
