import pandas as pd
from datetime import datetime

_ENSEMBL_ID_COLUMN = "Ensembl Gene ID"
_HGNC_SYMBOL_COLUMN = "HGNC symbol"

_INVALID_PUBMED_IDS = {"interactions information", "retracted"}

def get_true_positive_ids(dataset_path: str) -> set:
    """
    Get the true positive IDs from the dataset.
    """
    with open(dataset_path, "r") as f:
        lines = f.readlines()[1:]

    for line in lines:
        ensid_pubmedids_viruses = line.split("\t")
        ens_id = ensid_pubmedids_viruses[0]
        pubmedids_viruses = ensid_pubmedids_viruses[1].rstrip(",\n").split(",")
        for pubmed_id_virus in pubmedids_viruses:
            pubmed_id = pubmed_id_virus.split("-")[0]
            if pubmed_id in _INVALID_PUBMED_IDS:
                continue
            pubmed_ids.append(int(pubmed_id))

    return set(pubmed_ids)

def get_enriched_ensembl_ids(dataset_path: str) -> set:
    """
    Get the enriched Ensembl IDs from the dataset.
    """
    with open(dataset_path, "r") as f:
        lines = f.readlines()
    ensembl_ids = [line.rstrip('\n') for line in lines]
    return set(ensembl_ids)


def get_hgnc_symbol_from_ensembl_id_or_none(ensembl_id: str, hgnc_symbol_map_df: pd.DataFrame) -> str | None:
    """
    Get the HGNC symbol from an Ensembl ID.
    """
    candidate_hgnc_symbols = hgnc_symbol_map_df[hgnc_symbol_map_df[_ENSEMBL_ID_COLUMN] == ensembl_id.strip()][_HGNC_SYMBOL_COLUMN].tolist()
    if not candidate_hgnc_symbols:
        return None
    return candidate_hgnc_symbols[0]

def convert_date(date, date_format="%Y-%m-%d") -> int:
    date_time_str = datetime.strptime(date, date_format)
    return 10000*date_time_str.year + 100*date_time_str.month + date_time_str.day

