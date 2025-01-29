## Automated virus-interacting-protein detection from PubMed abstracts

We introduce a novel way of identifying virus-interacting proteins. We use a fine-tuned version of BlueBERT (citation) to predict if an abstract from PubMed contains a mention of a virus-interacting protein.

## Setup Guide
1. Run `pip install -r requirements.txt`
2. Run `python download_client.py`
3. Upload the generated `train.csv`, `test.csv` and `dev.tsv` to a cloud compute service e.g. Google Cloud
4. Upload `bluebert.ipynb` to the same cloud compute service
5. Update the corresponding file paths for `train.csv`, `test.csv` and `dev.tsv` in `bluebert.ipynb`
6. Run each cell of `bluebert.ipynb`