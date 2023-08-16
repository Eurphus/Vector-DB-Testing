import logging
import os
import time

from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance

from databases.database import Database


class QDrantDB(Database):
    """ QDrant Database Class for interacting with a defined QDrant collection
    Args:
        index_name (str): Name of QDrant collection
        api_key (str): API key for QDrant server
        url (str): URL of QDrant server
        ensure_exists (bool): Whether to create collection if it does not exist
    """
    def __init__(self,
                 index_name: str = "pdf-flood",
                 api_key: str = None,
                 url: str = None,
                 ensure_exists: bool = True,
                 ) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.index_name = index_name

        if api_key is None:
            _ = load_dotenv(find_dotenv())
            api_key = os.environ["QDRANT_API_KEY"]
        if url is None:
            _ = load_dotenv(find_dotenv())
            url = os.environ["QDRANT_URL"]

        self.client = QdrantClient(
            api_key=api_key,
            url=url
        )
        if ensure_exists:
            self.create()

    def create(self, on_disk_payload: bool = False) -> None:
        """
        Create new QDrant index if one does not exist

        Args:
            on_disk_payload (bool): Whether to store payload data on disk or in memory
        """
        logging.info("Initializing new QDrant Collection...")
        try:
            self.client.create_collection(
                collection_name=self.index_name,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=Distance.COSINE
                ),
                on_disk_payload=on_disk_payload,
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        always_ram=True,
                    ),
                )
            )
            logging.info("Created new QDrant Collection")
        except Exception as e:
            logging.info(f"Error, there is likely already an existing QDrant database with that name. See error:{e}")

    def clear(self) -> None:
        """Delete all vectors in a collection. The fastest way to do this is by deleting and recreating the collection

        :return:
        """
        self.client.delete_collection(
            collection_name=self.index_name
        )
        time.sleep(1)

        self.create()
        logging.info(f"All vectors have been deleted")

    async def upload(self, batch: list) -> None:
        """Method to upload vectors to QDrant

        :param batch: Batch to upload
        :return:
        """
        logging.info(f"Uploading {len(batch)} docs to QDrant")

        start_time = time.time()
        batch = self.preprocess(batch)
        self.client.upsert(
            collection_name=self.index_name,
            points=batch
        )

        logging.info(f"Uploaded to QDrant in {time.time() - start_time}")

    def query(self, text: str, top_k: int = 5, query_filter: any = None, include_metadata: bool = True,
              include_vectors: bool = False):
        """Method to query QDrant database

        :param text: Text to query with
        :param top_k: How many results to return
        :param query_filter: Filter associated with the search, ignore this part until properly implemented
        :param include_metadata: Whether to include payload within the response to the search
        :param include_vectors: Whether to include vectors within the response to the search
        :return:
        """
        result = self.client.search(
            collection_name=self.index_name,
            query_vector=self.encode(text),
            limit=top_k,
            with_payload=include_metadata,
            with_vectors=include_vectors
        )
        return result

    def preprocess(self, batch):
        """Method to be used within the QDrantDB class in order to process data via the other format

        :param batch:
        :return:
        """
        ids = []
        vectors = []
        payloads = []
        for doc in batch:
            ids.append(int(doc['id']))
            vectors.append(doc['values'])
            payloads.append({
                "metadata": doc['metadata'],
                "chunk_part": doc['metadata']['chunk'],
                "document_id": doc['metadata']['filename']
            })
        logging.info("QDrant preprocessing complete")
        return models.Batch.construct(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )

    def indexing(self, enable: bool) -> None:
        if enable:
            logging.info("Enabling indexing...")
            self.client.update_collection(
                collection_name=self.index_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000,
                    memmap_threshold=20000
                )
            )
        else:
            logging.info("Disabling indexing...")
            self.client.update_collection(
                collection_name=self.index_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                    memmap_threshold=20000
                )
            )
