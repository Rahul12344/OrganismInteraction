import requests
import xml.etree.ElementTree as ET
import logging
import random
from tqdm import tqdm
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
_DATASET = 'virus'
_IDLIST_PREFIX = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_DOWNLOAD_PREFIX = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
_LARGEST_DATE = "2019/10/31"

class QueryPubmed:
    '''
    Class for querying Pubmed.

    Arguments:
        - hgncs: The HGNC symbols to query, if provided.
    '''

    def __init__(self, hgncs):
        self.hgncs = hgncs

    def query(self, dataset, XML_ids_range=[], ands='', nots='', db='pubmed', start=0, return_limit=50000, by_hgnc=False, outside_range=False, by_date=True, id_return_limit=100000):
        '''
        Queries Pubmed for valid IDs.

        Arguments:
            - XML_ids_range: The XML IDs to query.
            - dataset: The dataset to query.
            - ands: The ANDs to use in the query.
            - nots: The NOTs to use in the query.
            - db: The database to query.
            - start: The start index of the query.
            - return_limit: The number of results to return.
            - by_hgnc: Whether to query by HGNC symbol.
            - outside_range: Whether to query outside the range of XML IDs.
            - id_return_limit: The number of IDs to return.
        '''

        XML_ids = []


        if by_hgnc:
            for hgnc in tqdm(self.hgncs[:]):
                ANDs = f'{ands}+AND+{hgnc}'
                base_url, params = self.create_query(ands=ANDs, nots=nots, start=start, return_limit=return_limit)
                payload_str = "&".join("%s=%s" % (k,v) for k,v in params.items())
                r = requests.get(base_url, params=payload_str)
                try:
                    root = ET.fromstring(r.content)
                    for IdList in root.findall('IdList'):
                        for ID in IdList.findall('Id'):
                            XML_ids.append(ID.text)
                except:
                    pass
        else:
            if by_date:
                total_num = 0
                while total_num < 50000:
                    base_url, params = self.create_query(ands=ands, nots=nots, start=start, return_limit=return_limit, date=self.LARGEST_DATE)
                    r = requests.get(base_url, params=params)
                    root = ET.fromstring(r.content)
                    for IdList in root.findall('IdList'):
                        for ID in IdList.findall('Id'):
                            XML_ids.append(ID.text)
                            total_num += 1
            else:
                XML_ids_range.sort()
                smallest_id = XML_ids_range[0]
                largest_id = XML_ids_range[len(XML_ids_range)-1]


                if not outside_range:
                    for start in range(0, largest_id, return_limit):
                        base_url, params = self.create_query(ands=ands, nots=nots, start=start, return_limit=return_limit)
                        r = requests.get(base_url, params=params)
                        root = ET.fromstring(r.content)
                        for IdList in root.findall('IdList'):
                            for ID in IdList.findall('Id'):
                                if int(ID.text) >= smallest_id and int(ID.text) <= largest_id:
                                    XML_ids.append(ID.text)
                    else:
                        num = 0
                        start = 0
                        r = requests.get(self.create_query("hgnc", start))
                        try:
                            root = ET.fromstring(r.content)
                            for IdList in root.findall('IdList'):
                                for ID in IdList.findall('Id'):
                                    if int(ID.text) > largest_id:
                                        XML_ids.append(ID.text)
                                        num += 1
                                    if num >= id_return_limit:
                                        return XML_ids
                        except Exception as e:
                            print("err")
                        empty_response = False
                        while root.find('RetMax') != None and root.find('RetMax').text != "0":
                            start += 100000
                            r = requests.get(self.create_query("hgnc", start))
                            try:
                                root = ET.fromstring(r.content)
                                for IdList in root.findall('IdList'):
                                    for ID in IdList.findall('Id'):
                                        if int(ID.text) > largest_id:
                                            XML_ids.append(ID.text)
                                            num += 1
                            except Exception as e:
                                print("err")
                            if num >= id_return_limit:
                                return XML_ids


        return XML_ids

    def create_query(self, hgnc='', ands='', nots='', db='pubmed', start=0, return_limit=100000, date=None):
        '''
        Creates a query for Pubmed.

        Arguments:
            - dataset: The dataset to query.
            - hgnc: The HGNC symbol to query, if provided.
            - ands: The ANDs to use in the query.
            - nots: The NOTs to use in the query.
            - db: The database to query.
            - start: The start index of the query.
            - return_limit: The number of results to return.
        '''

        if date is not None:
            params = {
                'db': db,
                'term': f'({_DATASET} AND human{ands}){nots} AND ({date}[PDAT]:3000[PDAT])',
                'retstart': start,
                "api_key": '49c77251ac91cbaa16ec5ae4269ab17d9d09',
                'retmax': return_limit,
            }
        else:
            params = {
                'db': db,
                'term': f'({dataset}+AND+human{ands}){NOTS}',
                'retstart': start,
                "api_key": '49c77251ac91cbaa16ec5ae4269ab17d9d09',
                'retmax': return_limit,
            }


        return _IDLIST_PREFIX, params

    def add_entries_to_df(self, c, XML_valid_ids, uids, dataset):
        df = pd.read_csv(os.path.join(c['data-directory'], 'dataset_df.csv'))

        rows = []
        for uid in tqdm(uids):
            is_pos = 1 if int(uid) in XML_valid_ids else 0
            if c['neg'] and dataset == 'recapture_virus':
                rows.append((f'{uid}.txt', is_pos, 'negative_recapture'))
            else:
                rows.append((f'{uid}.txt', is_pos, dataset))
        df2 = pd.DataFrame(rows, columns=['file_name', 'label', 'dataset'])

        df = df.append(df2, ignore_index = True)
        df.to_csv(os.path.join(c['data-directory'], 'dataset_df.csv'), index=False)

    def download(self, dataset, uid, XML_valid_ids, path, database='pubmed'):
        '''
        Downloads abstracts from Pubmed.

        Arguments:
            - uids: The uids to download.
            - XML_valid_ids: The valid IDs from the XML.
            - path: The path to download to.
            - database: The database to download from.
        '''

        params = {
            "db": "pubmed",
            "retmode": "text",
            'rettype': 'abstract',
            "api_key": '49c77251ac91cbaa16ec5ae4269ab17d9d09',
            "id": uid
        }


        try:
            r = requests.get(f"{self.DOWNLOAD_PREFIX}", params=params)
            with open(f"{path}/text/{uid}.txt", "w") as f:
                f.write(r.text)
            #self.add_entry_to_df(uid, is_pos, dataset)
            logger.info('Added abstracts related to {0}'.format(uid))
        except requests.HTTPError as e:
            logger.error('Error Code: {0}'.format(e.code))
            logger.error('Error: {0}'.format(e.read()))
            raise requests.HTTPError