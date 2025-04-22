## Automated virus-interacting-protein detection from PubMed abstracts

We introduce a novel way of identifying virus-interacting proteins. We use a fine-tuned version of BlueBERT (citation) to predict if an abstract from PubMed contains a mention of a virus-interacting protein.

This package provides an implementation of running this method.

## Setup
To set up local environment, run
```
./setup_env.sh
```
which will create a virtual environment called `organismtraining`.

In the root directory of the project, download the BlueBERT model from [HuggingFace](https://huggingface.co/bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12/tree/main) and set this path to `$MODEL`.

## Downloading Datasets

### Train, validation, test download

### "Recapture" evaluation set download

### "Standard" evaluation set download

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