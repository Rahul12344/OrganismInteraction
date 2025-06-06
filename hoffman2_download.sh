#!/bin/bash
#$ -cwd
#$ -V
#$ -pe shared 2
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
#$ -l h_rt=8:00:00,h_data=4G
# Email address to notify
#$ -M $USER_EMAIL
# Notify when
#$ -m bea

# load the job environment:
. /u/local/Modules/default/init/modules.sh
module use /u/project/CCN/apps/modulefiles

# Load the FSL module
# Your script content goes here...

module load anaconda3
conda activate organismtraining
python datadownloads/download_client.py --dataset=$DATASET --data_path=$DATA_PATH --temp_download_path=$TEMP_DOWNLOAD_PATH --num_threads=10 --api_key=$API_KEY --subsample=$SUBSAMPLE
python datadownloads/data_sampler.py --input_path=$DATA_PATH --output_path=$OUTPUT_PATH --train_ratio=$TRAIN_RATIO --dev_ratio=$DEV_RATIO --test_ratio=$TEST_RATIO --pos_neg_ratio=$POS_NEG_RATIO --seed=$SEED