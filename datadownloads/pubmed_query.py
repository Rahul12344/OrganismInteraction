import requests
import xml.etree.ElementTree as ET
import logging
import random
from tqdm import tqdm
import pandas as pd
import os

from utils.data_utils import convert_date
from utils.consts import MONTH_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

_TERM_NAMES = 'virus'
_IDLIST_PREFIX = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_DOWNLOAD_PREFIX = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
_LARGEST_DATE = "2019/10/31"
_MAX_RETRIES = 5
_RETRY_DELAY = 5

_DATABASE_NAME = 'pubmed'
_RETRIEVE_MODE = 'xml'
_ABSTRACT_RETRIEVE_MODE = 'text'
_ABSTRACT_RETRIEVE_TYPE = 'abstract'

_AND_CONNECTOR = '+'

class PubMedAbstractDateFetcher:
    """Handles interactions with the PubMed API."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    def __call__(self, abstract_ids: list[str]) -> dict[str, int]:
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
            "db": _DATABASE_NAME,
            "retmode": _RETRIEVE_MODE,
            "api_key": self._api_key,
            "id": abstract_id,
        }
        response = requests.get(_DOWNLOAD_PREFIX, params=params)
        response.raise_for_status()
        return response.text

    def _parse_pub_date(self, xml_text: str) -> int | None:
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
                    month = MONTH_MAP.get(month.text, "01") if month is not None else "01"
                    day = day.text if day is not None else "01"
                    date_string = f"{year.text}-{month.zfill(2)}-{day.zfill(2)}"
                    return convert_date(date_string)
        except Exception as e:
            logger.error(f"Error parsing publication date: {e}")
        return None

class PubMedAbstractIdFetcher:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def __call__(
        self,
        dataset: str,
        ands:list[str]=[],
        nots:list[str]=[],
        start:int=0,
        return_limit:int=10000,
        id_return_limit:int=100000,
        date:str=None,
        **kwargs
    ):
        if dataset not in ['standard', 'recapture', 'negative']:
            raise ValueError("dataset must be one of 'standard', 'recapture', or 'negative'")
        if dataset == 'standard':
            return self._get_abstract_ids_for_standard_set(
                dataset=dataset,
                ands=ands,
                nots=nots,
                start=start,
                return_limit=return_limit,
                id_return_limit=id_return_limit,
                date=date,
                **kwargs
            )
        elif dataset == 'recapture':
            return self._get_abstract_ids_for_recapture_set(
                dataset=dataset,
                ands=ands,
                nots=nots,
                start=start,
                return_limit=return_limit,
                id_return_limit=id_return_limit,
                date=date,
                **kwargs
            )


    def _get_abstract_ids_for_standard_set(
        self,
        dataset: str,
        ands:list[str]=[],
        nots:list[str]=[],
        start:int=0,
        return_limit:int=10000,
        id_return_limit:int=100000,
        date:str=None,
        **kwargs
    ) -> list[str]:
        abstract_id_range = kwargs.get('abstract_id_range', [])
        if not isinstance(abstract_id_range, list) or not all(isinstance(abstract_id, int) for abstract_id in abstract_id_range):
            raise ValueError("abstract_id_range must be a list of ints")
        if not abstract_id_range or len(abstract_id_range) != 2 or abstract_id_range[0] >= abstract_id_range[1]:
            raise ValueError("abstract_id_range is required, please provide a lower and upper bound of the range")

        positive_ids = list(kwargs.get('abstract_ids', []))
        if not isinstance(positive_ids, list) or not all(isinstance(abstract_id, int) for abstract_id in positive_ids):
            raise ValueError("abstract_ids must be a list of str")
        if not positive_ids:
            raise ValueError("abstract_ids is required, please provide a list of positive IDs")

        abstract_ids = []
        smallest_id = abstract_id_range[0]
        largest_id = abstract_id_range[1]
        print(smallest_id, largest_id)
        for start_index in tqdm(range(smallest_id, largest_id+1, return_limit), desc="Fetching abstract IDs for standard set"):
            params = self._create_params(
                ands=f'{start_index}:{start_index+return_limit}[uid]',
                return_limit=return_limit,
            )
            r = requests.get(_IDLIST_PREFIX, params=params)
            print(r.url)
            root = ET.fromstring(r.content)
            for ids in root.findall('IdList'):
                for unformatted_id in ids.findall('Id'):
                    formatted_id = unformatted_id.text
                    try:
                        if int(formatted_id) >= smallest_id and int(formatted_id) <= largest_id:
                            #print(formatted_id)
                            abstract_ids.append(formatted_id)
                    except _:
                        logger.error(f"Invalid ID: {formatted_id}")
                        continue

        abstract_ids.extend([str(abstract_id) for abstract_id in positive_ids])
        abstract_ids = sorted(list(set(abstract_ids)))
        return abstract_ids

    def _get_abstract_ids_for_recapture_set(
        self,
        dataset: str,
        ands:list[str]=[],
        nots:list[str]=[],
        start:int=0,
        return_limit:int=10000,
        id_return_limit:int=100000,
        date:str=None,
        **kwargs
    ) -> list[str]:
        hgnc_symbols = kwargs.get('hgnc_symbols', [])
        if not isinstance(hgnc_symbols, list) or not all(isinstance(symbol, str) for symbol in hgnc_symbols):
            raise ValueError("hgnc_symbols must be a list of strings")
        if not hgnc_symbols:
            raise ValueError("hgnc_symbols is required, please provide a list of HGNC symbols")

        abstract_ids = []
        for hgnc in tqdm(hgnc_symbols[:], desc="Fetching abstract IDs for recapture set"):
            updated_ands = f'{_AND_CONNECTOR.join(ands)}+{hgnc}' if ands else hgnc
            params = self._create_params(
                ands=updated_ands,
            )
            r = requests.get(_IDLIST_PREFIX, params=params)
            root = ET.fromstring(r.content)
            for ids in root.findall('IdList'):
                for unformatted_id in ids.findall('Id'):
                    formatted_id = unformatted_id.text
                    abstract_ids.append(formatted_id)

        return abstract_ids

    def _get_abstract_ids_for_negative_set(
        self,
        dataset: str,
        ands:list[str]=[],
        nots:list[str]=[],
        start:int=0,
        return_limit:int=10000,
        id_return_limit:int=100000,
        date:str=None,
        **kwargs
    ) -> list[str]:
        abstract_id_range = kwargs.get('abstract_id_range', [])
        if not isinstance(hgnc_symbols, list) or not all(isinstance(symbol, int) for symbol in hgnc_symbols):
            raise ValueError("abstract_id_range must be a list of strings")
        if not abstract_id_range or len(abstract_id_range) != 2 or abstract_id_range[0] >= abstract_id_range[1]:
            raise ValueError("abstract_id_range is required, please provide a lower and upper bound of the range")

        abstract_ids, id_count = [], 0
        smallest_id = abstract_id_range[0]
        largest_id = abstract_id_range[1]

        params = self._create_params(
            start=start_index,
        )
        r = requests.get(_IDLIST_PREFIX, params=params)
        root = ET.fromstring(r.content)
        while root.find('RetMax') and root.find('RetMax').text != "0":
            for ids in root.findall('IdList'):
                for unformatted_id in ids.findall('Id'):
                    formatted_id = unformatted_id.text
                    if int(formatted_id) > largest_id:
                        abstract_ids.append(formatted_id)
                        id_count += 1
                        if id_count >= id_return_limit:
                            return abstract_ids

            start_index += id_return_limit
            params = self._create_params(
                start=start_index,
            )
            r = requests.get(_IDLIST_PREFIX, params=params)
            root = ET.fromstring(r.content)


        return abstract_ids

    def _create_params(
        self,
        ands:str='',
        nots:str='',
        start:int=0,
        return_limit:int=50000,
        date:str=None
    ) -> str:
        if date is not None:
            params = {
                'db': _DATABASE_NAME,
                'term': f'({_TERM_NAMES}+human{ands}){nots} AND ({date}[PDAT]:3000[PDAT])',
                'retstart': start,
                "api_key": self._api_key,
                'retmax': return_limit,
            }
        else:
            params = {
                'db': _DATABASE_NAME,
                'term': f'({_TERM_NAMES}+human+{ands}){nots}',
                'retstart': start,
                "api_key": self._api_key,
                'retmax': return_limit,
            }


        return "&".join("%s=%s" % (k,v) for k,v in params.items())

class PubMedDownloader:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def __call__(self, uid: str) -> str:
        '''
        Downloads abstracts from Pubmed.

        Arguments:
            - uids: The uids to download.
            - XML_valid_ids: The valid IDs from the XML.
            - path: The path to download to.
            - database: The database to download from.
        '''

        params = {
            "db": _DATABASE_NAME,
            "retmode": _ABSTRACT_RETRIEVE_MODE,
            'rettype': _ABSTRACT_RETRIEVE_TYPE,
            "api_key": self._api_key,
            "id": uid
        }


        try:
            r = requests.get(_DOWNLOAD_PREFIX, params=params)
            logger.info('Retrieved abstract {0}'.format(uid))
            return r.text
        except requests.HTTPError as e:
            logger.error('Error Code: {0}'.format(e.code))
            logger.error('Error: {0}'.format(e.read()))
            raise requests.HTTPError