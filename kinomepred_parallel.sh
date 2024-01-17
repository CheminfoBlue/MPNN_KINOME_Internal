#!/bin/bash

source activate KinomePred
python ./code/pre_process_kinomescan.py ./data/KinomeData_new.csv
python ./code/KinomePred_MultiProcess_batched.py > KinomePred_out.log
python ./code/compile_results.py ./results/LGB/ --save_all_preds
