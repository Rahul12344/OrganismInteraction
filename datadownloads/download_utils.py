

def get_dataset_pubmed_and_ensembl_ids(
    dataset_name: str,
    neg: bool=True
) -> tuple[list[str], list[str]]:
    """Gets a list of Pubmed and Ensembl ideas for a given dataset.

    Args:
        dataset_name (str): Name of the dataset
        neg (bool, optional): Defaults to True.
    """
    pubmed_ids = []
    ens_ids = []
    if dataset_name == "virus":
        pubmed_ids, ens_ids = _get_virus_ids()

    if dataset_name == "recapture_virus":
        pubmed_ids, ens_ids = _get_virus_recapture_ids()

    return list(set(ens_ids)), list(set(pubmed_ids))


def get_hgncs(ens_ids: list[str]) -> list[str]:
    """Gets the list of corresponding HGNC gene symbols from ENS ID."""
    mart_export_df = pd.read_csv("c["filenames"]["mart-export"]", sep="\t")

    hgncs = []
    for ens_id in ens_ids:
        hgnc_symbols = mart_export_df[
            mart_export_df["Ensembl Gene ID"] == ens_id.strip()
        ]["HGNC symbol"].tolist()
        if len(hgnc_symbols) > 0:
            hgncs.append(hgnc_symbols[0])
    return hgncs


def convert_date(date, date_format="%Y-%m-%d") -> int:
    date_time_str = datetime.strptime(date, date_format)
    return 10000 * date_time_str.year + 100 * date_time_str.month + date_time_str.day