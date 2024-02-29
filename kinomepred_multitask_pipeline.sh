#!/bin/bash

# Run pre-processing and wait to continue pipeline
source activate MPNNKinomePred && python ./code/pre_process_kinomescan.py ./data/KinomeData_new.csv
wait

#Submit the prediction jobs (both kinome and sscore) and get job IDs
KINOME_PREDICT_JOB_ID=$(sbatch --parsable ./slurm_job_scripts/MPNN_weightedCE_Kinase_PoC_MultiTask_Predict.sh)
SSCORE_PREDICT_JOB_ID=$(sbatch --parsable ./slurm_job_scripts/MPNN_PoC_SScore_MultiTask_Predict.sh)
echo "KINOME_PREDICT_JOB_ID: $KINOME_PREDICT_JOB_ID"
echo "SSCORE_PREDICT_JOB_ID: $SSCORE_PREDICT_JOB_ID"

#Submit the prediction results compilation job and get job ID, with prediction jobs as run-dependencies
COMPILE_RES_JOB_ID=$(sbatch --parsable --dependency=afterok:"${KINOME_PREDICT_JOB_ID}:${SSCORE_PREDICT_JOB_ID}" ./slurm_job_scripts/compile_prediction_results.sh)
echo "COMPILE_RES_JOB_ID: $COMPILE_RES_JOB_ID"

#Submit training jobs (both kinome and sscore), with results compilation job as run-dependency
KINOME_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:${COMPILE_RES_JOB_ID} ./slurm_job_scripts/MPNN_weightedCE_Kinase_PoC_MultiTask_Train_FULL.sh)
SSCORE_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:${COMPILE_RES_JOB_ID} ./slurm_job_scripts/MPNN_PoC_SScore_MultiTask_Train_FULL.sh)
echo "KINOME_TRAIN_JOB_ID: $KINOME_TRAIN_JOB_ID"
echo "SSCORE_TRAIN_JOB_ID: $SSCORE_TRAIN_JOB_ID" 
