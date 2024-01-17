# KINOME_Pred_Internal

[Installation & Getting Started](#installation) <br>
  Create KinomePred conda environment
  ```
  conda env create -f environment.yml
  ```
  Go to root folder and un-pack the data zip file
  ```
  cd ./data/
  unzip init_kinomepred_data.zip && rm init_kinomepred_data.zip
  ```

[Kinome Prediction Pipeline](#run_pipeline) <br>
Activate KinomePred env
```
conda activate KinomePred
```
Run the bash script kinomepred_parallel.sh to run the full Kinome prediction pipeline
```
nohup bash kinomepred_parallel.sh & 
```
