import os, sys
import logging
import pandas as pd
import requests
from xml.etree import ElementTree as ET
from tqdm import tqdm
from time import sleep
import random
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import queue
except ImportError:
    import Queue as queue

from datadownloads.consts import MONTHS
from pubmed_query import QueryPubmed
from datadownloads.download_client import DownloadWorker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    dataset: str
    data_directory: str
    filenames: Dict[str, str]
    neg: bool = False

class PubMedAPI:
    """Handles interactions with the PubMed API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.max_retries = 3
        self.retry_delay = 5

    def fetch_abstract_dates(self, abstract_ids: List[str]) -> Dict[str, int]:
        """Fetch publication dates for a list of abstract IDs."""
        abstract_dates = {}

        for abstract_id in tqdm(abstract_ids, desc="Fetching abstract dates"):
            for attempt in range(self.max_retries):
                try:
                    response = self._make_request(abstract_id)
                    date = self._parse_pub_date(response)
                    if date:
                        abstract_dates[abstract_id] = date
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to fetch date for abstract {abstract_id}: {e}")
                    sleep(self.retry_delay)

        return abstract_dates

    def _make_request(self, abstract_id: str) -> str:
        """Make a request to the PubMed API."""
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "api_key": self.api_key,
            "id": abstract_id,
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.text

    def _parse_pub_date(self, xml_text: str) -> Optional[int]:
        """Parse publication date from XML response."""
        try:
            xml_tree = ET.fromstring(xml_text)
            for article in xml_tree.findall(".//PubmedArticle"):
                pmid = article.find(".//PMID").text
                pub_date_node = article.find(".//PubDate")

                year = pub_date_node.find(".//Year")
                month = pub_date_node.find(".//Month")
                day = pub_date_node.find(".//Day")

                if year is not None:
                    month = MONTHS.get(month.text, "01") if month is not None else "01"
                    day = day.text if day is not None else "01"
                    date_string = f"{year.text}-{month.zfill(2)}-{day.zfill(2)}"
                    return convert_date(date_string)
        except Exception as e:
            logger.error(f"Error parsing publication date: {e}")
        return None

class DatasetLoader:
    """Handles loading and processing of dataset files."""

    @staticmethod
    def get_abstract_ids(config: DatasetConfig) -> Tuple[List[str], List[str]]:
        """Get abstract IDs and ENS IDs based on dataset configuration."""
        if config.dataset == "virus":
            return DatasetLoader._load_virus_dataset(config)
        elif config.dataset == "malaria":
            return DatasetLoader._load_malaria_dataset(config)
        elif config.dataset == "recapture_virus":
            return DatasetLoader._load_recapture_virus_dataset(config)
        else:
            raise ValueError(f"Unsupported dataset type: {config.dataset}")

    @staticmethod
    def _load_virus_dataset(config: DatasetConfig) -> Tuple[List[str], List[str]]:
        """Load virus dataset."""
        pubmed_ids = []
        ens_ids = []

        with open(os.path.join(config.data_directory, config.filenames['virus-ids']), "r") as f:
            for line in f.readlines()[1:]:
                ensid_pubmedids_viruses = line.split("\t")
                ens_id = ensid_pubmedids_viruses[0]
                pubmedids_viruses = ensid_pubmedids_viruses[1].rstrip(",\n").split(",")

                for pubmed_id_virus in pubmedids_viruses:
                    pubmed_id = pubmed_id_virus.split("-")[0]
                    if pubmed_id not in ["interactions information", "retracted"]:
                        pubmed_ids.append(int(pubmed_id))
                ens_ids.append(ens_id)

        return list(set(ens_ids)), list(set(pubmed_ids))

    @staticmethod
    def _load_malaria_dataset(config: DatasetConfig) -> Tuple[List[str], List[str]]:
        """Load malaria dataset."""
        malaria_df = pd.read_csv(config.filenames["malaria-ids"], sep=",")
        ens_ids = malaria_df["ENSID"].tolist()
        pubmed_ids = []

        for pubmed_ids_str in malaria_df["Pubmed_IDs"].tolist():
            pubmed_ids.extend(pubmed_ids_str.split("*"))

        return ens_ids, pubmed_ids

    @staticmethod
    def _load_recapture_virus_dataset(config: DatasetConfig) -> Tuple[List[str], List[str]]:
        """Load recapture virus dataset."""
        if config.neg:
            with open(config.filenames["enriched-ids"], "r") as f:
                neg_ens_ids = [line.rstrip("\n") for line in f.readlines()]

            ens_ids, pubmed_ids = DatasetLoader._load_virus_dataset(config)
            neg_ens_ids.extend(ens_ids)
            neg_ens_ids = set(neg_ens_ids)

            mart_export_df = pd.read_csv(config.filenames["mart-export"], sep="\t")
            all_ens_ids = set(mart_export_df["Ensembl Gene ID"].values.tolist())
            ens_ids = random.sample(list(all_ens_ids - neg_ens_ids), 5000)
        else:
            with open(config.filenames["enriched-ids"], "r") as f:
                ens_ids = [line.rstrip("\n") for line in f.readlines()]
            _, pubmed_ids = DatasetLoader._load_virus_dataset(config)

        return ens_ids, pubmed_ids

    @staticmethod
    def get_hgncs(config: DatasetConfig, ens_ids: List[str]) -> List[str]:
        """Get HGNC symbols for ENS IDs."""
        mart_export_df = pd.read_csv(config.filenames["mart-export"], sep="\t")
        hgncs = []

        for ens_id in ens_ids:
            hgnc_symbols = mart_export_df[
                mart_export_df["Ensembl Gene ID"] == ens_id.strip()
            ]["HGNC symbol"].tolist()
            if hgnc_symbols:
                hgncs.append(hgnc_symbols[0])

        return hgncs

class DownloadClient:
    """Client for downloading and processing PubMed data."""

    def __init__(self, config: DatasetConfig, download_path: str, num_threads: int):
        self.config = config
        self.download_path = download_path
        self.num_threads = num_threads
        self.api = PubMedAPI(api_key="49c77251ac91cbaa16ec5ae4269ab17d9d09")
        self.dataset_loader = DatasetLoader()

    def download(self):
        """Main download process."""
        # Get dataset IDs
        ens_ids, pubmed_ids = self.dataset_loader.get_abstract_ids(self.config)
        hgncs = self.dataset_loader.get_hgncs(self.config, ens_ids[:])

        # Query PubMed
        query_client = QueryPubmed(hgncs=hgncs)
        abstract_XML_ids = query_client.query(
            dataset=self.config.dataset,
            XML_ids_range=pubmed_ids,
            by_hgnc=True,
            by_date=False,
            return_limit=100,
        )

        # Process results
        query_client.add_entries_to_df(self.config, pubmed_ids, abstract_XML_ids, self.config.dataset)

        # Get final XML IDs
        df = pd.read_csv(os.path.join(self.config.data_directory, "dataset_df.csv"))
        if self.config.dataset == "recapture_virus" and self.config.neg:
            abstract_XML_ids = [
                ids[:-4] for ids in df[df["dataset"] == "negative_recapture"]["file_name"].values.tolist()
            ]
        else:
            abstract_XML_ids = [
                ids[:-4] for ids in df[df["dataset"] == self.config.dataset]["file_name"].values.tolist()
            ]

        # Download abstracts
        self._download_abstracts(abstract_XML_ids, pubmed_ids)

    def _download_abstracts(self, abstract_XML_ids: List[str], pubmed_ids: List[str]):
        """Download abstracts using multiple workers."""
        queuer = queue.Queue(maxsize=0)

        # Start workers
        workers = []
        for _ in range(self.num_threads):
            worker = DownloadWorker(
                queuer=queuer,
                XML_ids=pubmed_ids,
                dataset=self.config.dataset,
                download_path=self.download_path,
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)

        # Queue work
        for abstract_id in abstract_XML_ids:
            logger.info(f"Queueing {self.config.dataset} abstract {abstract_id}")
            queuer.put(abstract_id)

        # Wait for completion
        queuer.join()

def convert_date(date: str, date_format: str = "%Y-%m-%d") -> int:
    """Convert date string to integer format."""
    date_time_str = datetime.strptime(date, date_format)
    return 10000 * date_time_str.year + 100 * date_time_str.month + date_time_str.day
