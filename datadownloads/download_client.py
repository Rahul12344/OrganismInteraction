import os, sys
from typing import Any
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
import shutil
import argparse
import random
from threading import Thread
from os import path
from collections import defaultdict


try:
    import queue
except ImportError:
    import Queue as queue

from datadownloads.pubmed_query import PubMedAbstractIdFetcher, PubMedDownloader
from utils.data_utils import get_true_positive_ids, get_enriched_ensembl_ids, get_hgnc_symbol_from_ensembl_id_or_none


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

_ABSTRACT_ID_RANGE_PARAM = "abstract_id_range"
_HGNC_SYMBOL_PARAM = "hgnc_symbols"
_ABSTRACT_IDS_PARAM = "abstract_ids"

class DownloadWorker(Thread):
    def __init__(
        self,
        thread_queue: queue.Queue,
        api_key: str,
        temp_download_path: str
    ):
        '''
        Class for downloading data from a source.

        Temporarily stores the downloaded abstracts in memory for further parsing into a dataframe.
        '''
        Thread.__init__(self)
        self._thread_queue = thread_queue
        self._download_path = temp_download_path
        self._pubmed_downloader = PubMedDownloader(api_key)

    def run(self):
        '''
        Runs the thread.
        '''
        while True:
            try:
                item_to_download = self._thread_queue.get(timeout=1)  # Add timeout to prevent hanging
                logger.info(f'Downloading abstracts related to {item_to_download}')
                try:
                    if path.isfile(f"{self._download_path}/{item_to_download}.txt"):
                        logger.info(f'Already downloaded {item_to_download}')
                    else:
                        text = self._pubmed_downloader(item_to_download)
                        with open(f"{self._download_path}/{item_to_download}.txt", "w") as f:
                            f.write(text)
                        logger.info('Downloaded abstracts related to {0}'.format(item_to_download))
                    sleep(1)
                except Exception as e:
                    logger.error("Failed to download:{0}".format(str(e)))
                    self._thread_queue.put(item_to_download)
                    logger.info(f'Re-queueing {item_to_download}')
                    sleep(1)
                finally:
                    self._thread_queue.task_done()
            except queue.Empty:
                # No more items in queue, exit the thread
                break

class DownloadClient:
    """Client for downloading and processing PubMed data."""

    def __init__(
        self,
        dataset: str,
        temp_download_path: str,
        data_path: str,
        num_threads: int,
        api_key: str,
    ):
        self._dataset = dataset
        self._temp_download_path = temp_download_path
        # make directories for both paths if they don't exist
        Path(temp_download_path).mkdir(parents=True, exist_ok=True)
        Path(data_path).mkdir(parents=True, exist_ok=True)
        self._data_path = data_path
        self._num_threads = num_threads
        self._api_key = api_key
        self._id_fetcher = PubMedAbstractIdFetcher(api_key)

    def __call__(self):
        """Main download process."""
        # Query PubMed
        custom_params = self._get_custom_params(self._dataset)
        abstract_XML_ids = self._id_fetcher(
            dataset=self._dataset,
            **custom_params
        )

        # Download abstracts
        self._download_abstracts(abstract_XML_ids, custom_params)

    def _download_abstracts(self, abstract_XML_ids: list[str], custom_params: dict[str, Any]):
        """Download abstracts using multiple workers."""
        if not abstract_XML_ids:
            raise ValueError("No abstract IDs to download")

        thread_queue = queue.Queue(maxsize=0)

        # Start workers
        workers = []
        for _ in range(self._num_threads):
            worker = DownloadWorker(
                thread_queue=thread_queue,
                api_key=self._api_key,
                temp_download_path=self._temp_download_path,
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)

        # Queue work
        for abstract_id in abstract_XML_ids:
            logger.info(f"Queueing {self._dataset} abstract {abstract_id}")
            thread_queue.put(abstract_id)

        # Wait for completion
        thread_queue.join()

        # Process results
        self._process_results(abstract_XML_ids, custom_params)

    def _get_custom_params(self, dataset: str) ->dict[str, Any]:
        """Get custom parameters for the dataset."""
        if dataset == "standard":
            true_positive_ids = get_true_positive_ids("dataset/VIP_ids.txt")
            return {
                _ABSTRACT_IDS_PARAM: true_positive_ids,
                _ABSTRACT_ID_RANGE_PARAM: [min(true_positive_ids), max(true_positive_ids)]
            }
        elif dataset == "recapture":
            mart_export_df = pd.read_csv("dataset/mart_export.tsv", sep="\t")
            hgnc_symbols = [get_hgnc_symbol_from_ensembl_id_or_none(ensembl_id, mart_export_df) for ensembl_id in get_enriched_ensembl_ids("dataset/mass-spec_VIPs.txt")]
            hgnc_symbols = [str(hgnc_symbol) for hgnc_symbol in hgnc_symbols if hgnc_symbol is not None]
            return {
                _HGNC_SYMBOL_PARAM: hgnc_symbols,
            }
        elif dataset == "negative":
            return {
                _ABSTRACT_IDS_PARAM: true_positive_ids,
                _ABSTRACT_ID_RANGE_PARAM: [min(true_positive_ids), max(true_positive_ids)]
            }
        else:
            raise ValueError(f"Unsupported dataset type: {dataset}")

    def _process_results(self, abstract_XML_ids: list[str], custom_params: dict[str, Any]):
        """Process results."""
        file_text, file_ids = [], []
        for file in os.listdir(self._temp_download_path):
            with open(os.path.join(self._temp_download_path, file), "r") as f:
                file_text.append(f.read())
                file_ids.append(file.split(".")[0])

        if os.path.exists(self._temp_download_path):
            shutil.rmtree(self._temp_download_path)

        if self._dataset == "standard":
            self._process_standard_dataset(file_text, file_ids, custom_params.get(_ABSTRACT_IDS_PARAM, []))
        elif self._dataset == "recapture":
            self._process_recapture_virus_dataset(file_text, file_ids)
        elif self._dataset == "negative":
            self._process_negative_virus_dataset(file_text, file_ids)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset}")


    def _process_standard_dataset(self, file_text: list[str], file_ids: list[str], true_positive_ids: set[str]):
        """Process standard dataset."""
        # Collect all data first
        data = []
        for file_id, text in zip(file_ids, file_text):
            label = 1 if file_id in true_positive_ids else 0
            data.append({
                "abstract_id": file_id,
                "abstract_text": text,
                "label": label
            })

        # Create DataFrame in one go
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self._data_path, "standard_dataset.tsv"), index=False, sep="\t")

    def _process_recapture_virus_dataset(self, file_text: list[str], file_ids: list[str]):
        """Process recapture virus dataset."""
        # Collect all data first
        data = []
        for file_id, text in zip(file_ids, file_text):
            data.append({
                "abstract_id": file_id,
                "abstract_text": text
            })

        # Create DataFrame in one go
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self._data_path, "recapture_virus_dataset.tsv"), index=False, sep="\t")

    def _process_negative_virus_dataset(self, file_text: list[str], file_ids: list[str]):
        """Process negative virus dataset."""
        # Collect all data first
        data = []
        for file_id, text in zip(file_ids, file_text):
            data.append({
                "abstract_id": file_id,
                "abstract_text": text
            })

        # Create DataFrame in one go
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self._data_path, "negative_virus_dataset.tsv"), index=False, sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to download")
    parser.add_argument("--data_path", type=str, required=True, help="Path to save the downloaded data")
    parser.add_argument("--temp_download_path", type=str, required=True, help="Path to save the temporary downloaded data")
    parser.add_argument("--num_threads", type=int, required=True, help="Number of threads to use for downloading")
    parser.add_argument("--api_key", type=str, required=True, help="API key for PubMed")
    args = parser.parse_args()

    download_client = DownloadClient(
        dataset=args.dataset,
        data_path=args.data_path,
        temp_download_path=args.temp_download_path,
        num_threads=args.num_threads,
        api_key=args.api_key
    )
    download_client()