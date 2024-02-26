#!/bin/bash


#SBATCH --job-name=MPNN_weightedCE_Kinase_PoC_MultiTask

#SBATCH --partition=gpu-t4-4x

#SBATCH --output=MPNN_weightedCE_Kinase_PoC_MultiTask_out.log

#SBATCH --error=MPNN_weightedCE_Kinase_PoC_MultiTask_error.log

#SBATCH -N 1

source activate chemprop-BP
chemprop_train --data_path ./data/kinome_data_multiclass_current.csv --split_sizes 0.9 0.1 0.0 --split_type predetermined --split_column Split --test_split_key test --smiles_columns Structure --ignore_columns 'Compound Name' 'Split' 'S(10)' 'S(35)' --dataset_type multiclass --multiclass_num_classes 3 --class_weighted_loss --save_dir ./results/MPNN/KinomePred --save_smiles_splits --show_individual_scores --save_preds