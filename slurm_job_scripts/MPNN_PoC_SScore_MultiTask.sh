#!/bin/bash


#SBATCH --job-name=MPNN_PoC_SScore_MultiTask

#SBATCH --partition=gpu-t4-4x

#SBATCH --output=MPNN_PoC_SScore_MultiTask_out.log

#SBATCH --error=MPNN_PoC_SScore_MultiTask_error.log

#SBATCH -N 1

source activate chemprop-BP
chemprop_train --data_path ./data/kinome_data_multiclass_current.csv --split_sizes 0.9 0.1 0.0 --split_type predetermined --split_column Split --test_split_key test --smiles_columns Structure --target_columns 'S(10)' 'S(35)' --dataset_type regression --save_dir ./results/MPNN/SScorePred --save_smiles_splits --show_individual_scores --save_preds