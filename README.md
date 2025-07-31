## Automated virus-interacting-protein detection from PubMed abstracts

We introduce a novel way of identifying virus-interacting proteins. We use a fine-tuned version of BlueBERT (citation) to predict if an abstract from PubMed contains a mention of a virus-interacting protein.

This package provides an implementation of running this method.

## Setup
To set up local environment, run
```
./setup_env.sh
```
which will create a virtual environment called `organismtraining`.

In the root directory of the project, download the BlueBERT model from [HuggingFace](https://huggingface.co/bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12/tree/main) and set this path to `$MODEL` e.g. `export MODEL=PATH_TO_BLUEBERT`.

## Downloading Datasets
We provide a script for downloading each of the datasets discussed.

### Train, validation, test download
To get the labeled dataset for the model,
```
python datadownloads/download_client.py
\ --dataset=standard # one of "standard - for train/test/dev dataset, recapture, or negative"
\ --data_path=$OUTPUT # output path of dataset download
\ --temp_download_path=$TEMP # temporary path for dataset download for txt files
\ --num_threads=10 # number of threads with which to download
\ --api_key=$PUBMED_API_KEY
```

### "Recapture" evaluation set download
To get the unlabeled dataset for the model,
```
python datadownloads/download_client.py
\ --dataset=recapture # one of "standard - for train/test/dev dataset, recapture, or negative"
\ --data_path=$OUTPUT # output path of dataset download
\ --temp_download_path=$TEMP # temporary path for dataset download for txt files
\ --num_threads=10 # number of threads with which to download
\ --api_key=$PUBMED_API_KEY
```

### "Negative" evaluation set download
To get the labeled dataset for the model,
```
python datadownloads/download_client.py
\ --dataset=negative # one of "standard - for train/test/dev dataset, recapture, or negative"
\ --data_path=$OUTPUT # output path of dataset download
\ --temp_download_path=$TEMP # temporary path for dataset download for txt files
\ --num_threads=10 # number of threads with which to download
\ --api_key=$PUBMED_API_KEY
```

### Creating data split

To create a split for training,
```
python datadownloads/dataset_splitter.py
\ --input_path=$INPUT # path to downloaded dataset
\ --output_path=$OUTPUT
\ --train_ratio=$TRAIN_RATIO
\ --dev_ratio=$DEV_RATIO
\ --test_ratio=$TEST_RATIO # must sum to 1.0
\ --pos_neg_ratio=$RATIO # ratio of pos to neg
\ --seed=$SEED
```

If none of the ratios are specified, a 60:20:20 split will be created.

## Training
To train the model,
```
python organismtraining/interaction_detection_evaluator.py
\ --dataset_path=$DATASET
\ --model_path=$MODEL
\ --retrain
```

To save the predictions from the model,
```
python organismtraining/interaction_detection_evaluator.py --dataset_path=$DATASET --model_path=$MODEL --retrain --save_output_path=$SAVE_OUTPUT_PATH
```

## Evaluation
To evaluate the model on some dataset,
```
python organismtraining/interaction_detection_evaluator.py
\ --dataset_path=$EVALUATION_DATASET
\ --model_path=$MODEL
\ --predict_set_path=$PREDICTION_DATASET_FILE
```

To save the predictions from the model,
```
python organismtraining/interaction_detection_evaluator.py --dataset_path=$EVALUATION_DATASET --model_path=$MODEL --save_output_path=$SAVE_OUTPUT_PATH --predict_set_path=$PREDICTION_DATASET_FILE
```

## Metrics on frozen CSV
To chart/evaluate metrics on frozen predictions,
```
python organismtraining/interaction_detection_evaluator.py --dataset_path=$DATASET --model_path=$MODEL
--from_prediction_csv=$PREDICTION_CSV
```