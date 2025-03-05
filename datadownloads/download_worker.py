from threading import Thread
import logging
import os
from os import path
import csv
from collections import defaultdict
import argparse
import random

try:
   import queue
except ImportError:
   import Queue as queue

import time

from queries.queries import PubmedQuerier
from tokenizer import labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class DownloadWorker(Thread):
    def __init__(self, thread_queue: queue.Queue, xml_ids: list[str], download_path, dataset):
        '''
        Class for downloading data from a source.

        Arguments:
            - queuer: The thread queue to use.
            - XML_ids: The positive XML ids to download.
            - download_path: The path to download the data to.
            - dataset: The dataset to download.
        '''
        Thread.__init__(self)
        self._thread_queue = thread_queue
        self._xml_ids = xml_ids
        self.download_path = download_path
        self.dataset = dataset
        self.querier = PubmedQuerier(xml_ids)

    def _download(self, uid):
        '''
        Downloads the abstracts related to the given uid.

        Arguments:
            - uid: The uid to download the abstracts for.
        '''
        self.querier.download(self.dataset, uid, self.XML_ids, self.download_path)

    def run(self):
        '''
        Runs the thread.
        '''
        while True:
            item_to_download = self._thread_queue.get()
            logger.info(f'Downloading abstracts related to {item_to_download}')
            try:
                if path.isfile(f"{self.download_path}/text/{item_to_download}.txt"):
                    logger.info(f'Already downloaded {item_to_download}')
                else:
                    self._download(str(item_to_download))
                    logger.info('Downloaded abstracts related to {0}'.format(item_to_download))
                    self.queuer.task_done()
                    time.sleep(1)
            except Exception as e:
                    logger.error("Failed to download:{0}".format(str(e)))
                    self.queuer.put(item_to_download)
                    logger.info(f'Re-queueing {item_to_download}')
                    time.sleep(1)
                    self.queuer.task_done()


