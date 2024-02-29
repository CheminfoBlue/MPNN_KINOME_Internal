#!/bin/bash

#SBATCH --job-name=compile_prediction_results
#SBATCH --partition=cpu1
#SBATCH --output=compile_prediction_results_out.log
#SBATCH --error=compile_prediction_results_error.log
#SBATCH -N 1

source activate MPNNKinomePred
python ./code/compile_results.py ./results/MPNN/ --save_all_preds