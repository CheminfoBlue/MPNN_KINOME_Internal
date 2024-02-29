# MPNN_KINOME_Internal

[Installation & Getting Started](#installation) <br>
  Create KinomePred conda environment
  ```
  conda env create -f environment.yml
  ```

[Kinome Prediction Pipeline](#run_pipeline) <br>
Activate MPNNKinomePred env
```
conda activate MPNNKinomePred
```
Run the bash script kinomepred_multitask_pipeline.sh to run the full Kinome prediction pipeline
```
bash kinomepred_multitask_pipeline.sh
```
[Prediction](#run_pipeline) <br>
Run the bash script kinome_prediction.sh to run prediction using the multi-task model.
```
sbatch ./slurm_job_scripts/MPNN_weightedCE_Kinase_PoC_MultiTask_Predict.sh
```
Or you may run the multi-output prediction on a different set of prospective data <prospective_data_path>
```
source activate chemprop-BP
chemprop_predict <prospective_data_path> --smiles_columns Structure --checkpoint_path ./results/MPNN/KinomePred/fold_0/model_0/model.pt --preds_path <preds_path>
```
