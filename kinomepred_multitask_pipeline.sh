#!/bin/bash

source activate MPNNKinomePred
python ./code/pre_process_kinomescan.py ./data/KinomeData_new.csv
sbatch ./slurm_job_scripts/MPNN_weightedCE_Kinase_PoC_MultiTask
sbatch ./slurm_job_scripts/MPNN_PoC_SScore_MultiTask.sh

# Wait for the Slurm jobs to finish
wait 

python ./code/compile_results.py ./results/MPNN/ --save_all_preds
