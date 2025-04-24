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
\ --output_path=$OUTPUT
\ --dataset_name="labeled_dataset"
```

To create a 60:20:20 split for training,
```
python datadownloads/dataset_splitter.py
\ --output_path=$OUTPUT
```

### "Recapture" evaluation set download
To get the labeled dataset for the model,
```
python datadownloads/download_client.py
\ --output_path=$OUTPUT
\ --dataset_name="recapture_dataset"
```

### "Standard" evaluation set download
To get the labeled dataset for the model,
```
python datadownloads/download_client.py
\ --output_path=$OUTPUT
\ --dataset_name="negative_dataset"
```

## Training
To train the model,
```
python organismtraining/interaction_detection_evaluator.py
\ --dataset_path=$DATASET
\ --model_path=$MODEL
\ --retrain=True
\ --output_path=$OUTPUT
```

## Evaluation
To evaluate the model on some dataset,
```
python organismtraining/interaction_detection_evaluator.py
\ --dataset_path=$EVALUATION_DATASET
\ --model_path=$MODEL
\ --retrain=False
```