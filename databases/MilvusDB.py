import logging
import os
import time

from dotenv import load_dotenv, find_dotenv

from databases.database import Database
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

class MilvusDB(Database):
    def __init__(self,
                 index_name: str = "pdf-flood",
                 api_key: str = None,
                 url: str = None,
                 port: int = None,
                 ensure_exists: bool = True,
                 ) -> None:
        super().__init__()
        self.index_name = index_name

        if api_key is None:
            _ = load_dotenv(find_dotenv())
            api_key = os.environ["Milvus_API_KEY"]
        if url is None:
            _ = load_dotenv(find_dotenv())
            url = os.environ["Milvus_URL"]
        if port is None:
            _ = load_dotenv(find_dotenv())
            port = os.environ["Milvus_PORT"]

        connections.connect(
            #user="DEFAULT",
            api_key=api_key,
            url=url,
            port=port
        )

    def create(self) -> None:
        """
        Create new Milvus index if one does not exist
        """
        logging.info("Initializing new Milvus Collection...")
        try:
            logging.info("Created new Milvus Collection")
        except Exception as e:
            logging.info(f"Error, there is likely already an existing Milvus database with that name. See error:{e}")

    def clear(self) -> None:
        """Delete all vectors in a collection. The fastest way to do this is by deleting and recreating the collection

        :return:
        """
        self.create()
        logging.info(f"All vectors have been deleted")

    async def upload(self, batch: list) -> None:
        """Method to upload vectors to Milvus

        :param batch: Batch to upload
        :return:
        """
        logging.info(f"Uploading {len(batch)} docs to Milvus")

        batched = self.preprocess(batch)

        start_time = time.time()

        logging.info(f"Uploaded to Milvus in {time.time() - start_time}")

    def query(self, text: str, top_k: int = 5, include_metadata: bool = True,
              include_vectors: bool = False):
        """Method to query Milvus database

        :param text: Text to query with
        :param top_k: How many results to return
        :param query_filter: Filter associated with the search, ignore this part until properly implemented
        :param include_metadata: Whether to include payload within the response to the search
        :param include_vectors: Whether to include vectors within the response to the search
        :return:
        """
        result = []
        return result

    def preprocess(self, batch):
        """Method to be used within the MilvusDB class in order to process data via the other format

        :param batch:
        :return:
        """
        logging.info("Milvus preprocessing complete")
        return []