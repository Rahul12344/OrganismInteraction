import os, sys
import logging
import pandas as pd
import requests
from xml.etree import ElementTree as ET
from tqdm import tqdm
from time import sleep
import random
from datetime import datetime

try:
    import queue
except ImportError:
    import Queue as queue

from pubmed_query import QueryPubmed
from download_worker import DownloadWorker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# three letter month abbreviation to number
months = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


def get_abstract_ids(dataset, neg=True):
    pubmed_ids = []
    ens_ids = []
    if dataset == "virus":
        with open(c["filenames"]["virus-ids"], "r") as f:
            lines = f.readlines()[1:]

        for line in lines:
            ensid_pubmedids_viruses = line.split("\t")
            ens_id = ensid_pubmedids_viruses[0]
            pubmedids_viruses = ensid_pubmedids_viruses[1].rstrip(",\n").split(",")
            for pubmed_id_virus in pubmedids_viruses:
                pubmed_id = pubmed_id_virus.split("-")[0]
                if pubmed_id != "interactions information" and pubmed_id != "retracted":
                    pubmed_ids.append(int(pubmed_id))
            ens_ids.append(ens_id)

    if dataset == "malaria":
        malaria_df = pd.read_csv(c["filenames"]["malaria-ids"], sep=",")
        ens_ids = malaria_df["ENSID"].tolist()
        all_pubmed_ids = malaria_df["Pubmed_IDs"].tolist()

        pubmed_ids = []
        for all_pubmed_id in all_pubmed_ids:
            sub_ids = all_pubmed_id.split("*")
            for sub_id in sub_ids:
                pubmed_ids.append(sub_id)

    if dataset == "recapture_virus":
        if neg:
            with open(c["filenames"]["enriched-ids"], "r") as f:
                lines = f.readlines()
            neg_ens_ids = [line.rstrip("\n") for line in lines]

            with open(c["filenames"]["virus-ids"], "r") as f:
                lines = f.readlines()[1:]

            for line in lines:
                ensid_pubmedids_viruses = line.split("\t")
                ens_id = ensid_pubmedids_viruses[0]
                pubmedids_viruses = ensid_pubmedids_viruses[1].rstrip(",\n").split(",")
                for pubmed_id_virus in pubmedids_viruses:
                    pubmed_id = pubmed_id_virus.split("-")[0]
                    if (
                        pubmed_id != "interactions information"
                        and pubmed_id != "retracted"
                    ):
                        pubmed_ids.append(int(pubmed_id))
                neg_ens_ids.append(ens_id)

            neg_ens_ids = set(neg_ens_ids)

            mart_export_df = pd.read_csv(c["filenames"]["mart-export"], sep="\t")
            ens_ids = set(mart_export_df["Ensembl Gene ID"].values.tolist())
            ens_ids = set(ens_ids - neg_ens_ids)
            ens_ids = random.sample(ens_ids, 5000)

        else:
            with open(c["filenames"]["enriched-ids"], "r") as f:
                lines = f.readlines()
            ens_ids = [line.rstrip("\n") for line in lines]

    return list(set(ens_ids)), list(set(pubmed_ids))


def get_hgncs(c, ens_ids):
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


class DownloadClient:
    """
    Class for downloading data from a source.
    """

    def __init__(self, c, num_threads):
        self.c = c
        self.num_threads = num_threads

        self.download_path = c["download_dir"]

    def fetch_pubmed_abstract_dates(self):
        ens_ids, abstract_ids = get_abstract_ids(self.c)
        abstract_ids = [str(abstract_id) for abstract_id in abstract_ids]
        abstract_dates = {}

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        for abstract_id in tqdm(abstract_ids):
            params = {
                "db": "pubmed",
                "retmode": "xml",
                "api_key": "49c77251ac91cbaa16ec5ae4269ab17d9d09",
                "id": abstract_id,
            }

            response = requests.get(base_url, params=params)

            while response.status_code != 200:
                sleep(5)
                params = {
                    "db": "pubmed",
                    "retmode": "xml",
                    "api_key": "49c77251ac91cbaa16ec5ae4269ab17d9d09",
                    "id": abstract_id,
                }

                response = requests.get(base_url, params=params)

                # raise Exception(f"Failed to retrieve data from PubMed API. Status code: {response.status_code}")

            xml_tree = ET.fromstring(response.text)

            for article in xml_tree.findall(".//PubmedArticle"):
                pmid = article.find(".//PMID").text
                pub_date_node = article.find(".//PubDate")
                year = pub_date_node.find(".//Year")
                month = pub_date_node.find(".//Month")
                if month is None:
                    month = "1"
                else:
                    month = months[month.text]
                day = pub_date_node.find(".//Day")
                if day is None:
                    day = "1"
                else:
                    day = day.text
                if year is not None:
                    date_string = f"{year.text}-{month.zfill(2)}-{day.zfill(2)}"
                    abstract_dates[pmid] = convert_date(date_string)

        return abstract_dates

    def download(self):
        dataset = self.c["dataset"]

        # Query client for pubmed abstracts, populate HGNCs using mart export dataframe
        ens_ids, pubmed_ids = get_abstract_ids(self.c)
        hgncs = get_hgncs(self.c, ens_ids[:])
        print(hgncs)
        queryClient = QueryPubmed(hgncs=hgncs)

        # XML_id_range/valid_XML_ids from
        abstract_XML_ids = queryClient.query(
            dataset=dataset,
            XML_ids_range=pubmed_ids,
            by_hgnc=True,
            by_date=False,
            return_limit=100,
        )
        print(abstract_XML_ids)
        queryClient.add_entries_to_df(self.c, pubmed_ids, abstract_XML_ids, dataset)

        df = pd.read_csv(os.path.join(self.c["data-directory"], "dataset_df.csv"))
        # print(df)
        if dataset == "recapture_virus" and self.c["neg"]:
            abstract_XML_ids = [
                ids[:-4]
                for ids in df[df["dataset"] == "negative_recapture"][
                    "file_name"
                ].values.tolist()
            ]
        else:
            abstract_XML_ids = [
                ids[:-4]
                for ids in df[df["dataset"] == dataset]["file_name"].values.tolist()
            ]
        print(abstract_XML_ids)

        queuer = queue.Queue(maxsize=0)
        for _ in range(self.num_threads):
            worker = DownloadWorker(
                c=self.c,
                queuer=queuer,
                XML_ids=pubmed_ids,
                dataset=dataset,
                download_path=self.download_path,
            )
            worker.daemon = True
            worker.start()

        for _, abstract_id in enumerate(abstract_XML_ids):
            logger.info(f"Queueing {dataset} abstract {abstract_id}")
            queuer.put(abstract_id)

        queuer.join()
