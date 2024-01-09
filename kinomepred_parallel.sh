#!/bin/bash

source activate dilipred
python ./code/pre_process_kinomescan.py ./data/kinome_scan-2023_12_15.csv
python ./code/KinomePred_MultiProcess_batched.py > KinomePred_out.log
python ./code/compile_results.py ./results/LGB/ --save_all_preds
