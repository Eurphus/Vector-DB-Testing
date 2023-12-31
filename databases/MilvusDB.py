import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from databases.database import Database


class MilvusDB(Database):
    """ Milvus Database Class for interacting with a defined Milvus index
    Args:
        index_name (str): Name of Milvus index
        url (str): URL of Milvus server
        port (int): Port of Milvus server
        user (str): Username of Milvus server
        password (str): Password of Milvus server
        ensure_exists (bool): Whether to create index if it does not exist
    """

    def __init__(self,
                 index_name: str = "pdf_flood",
                 url: str = None,
                 port: int = None,
                 user: str = None,
                 password: str = None,
                 ensure_exists: bool = True,
                 ) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.index_name = index_name

        if url is None:
            _ = load_dotenv(find_dotenv())
            url = os.environ["MILVUS_URL"]
        if port is None:
            _ = load_dotenv(find_dotenv())
            port = os.environ["MILVUS_PORT"]
        if user is None:
            _ = load_dotenv(find_dotenv())
            user = os.environ["MILVUS_USER"]
        if password is None:
            _ = load_dotenv(find_dotenv())
            password = os.environ["MILVUS_PASSWORD"]

        connections.connect(
            uri=url + ":" + port,
            token=user + ":" + password
        )
        if ensure_exists:
            self.create()
            time.sleep(1)


        self.collection = Collection(name=self.index_name, using="default")

    def create(self) -> None:
        """
        Create new Milvus index if one does not exist
        """
        self.logger.info("Initializing new Milvus Collection...")
        try:
            if utility.has_collection(self.index_name):
                self.logger.info("Collection already exists, skipping creation.")
                return
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    description="Unique document ID, right now it has a crap system."
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    description="Vectors of the document",
                    dim=768,
                ),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.JSON,
                    description="Important metadata associated with the document"
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=5000,
                    description="Text of the document"
                )
            ]

            schema = CollectionSchema(
                fields=fields,
                description="PDF Flood Index",
                enable_dynamic_field=True
            )

            collection = Collection(
                name=self.index_name,
                schema=schema,
                using="default",
                #  num_shards=2
            )

            collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "IP",
                    "index_type": "AUTOINDEX"
                },
            )
            collection.load()
            self.logger.info("Created new Milvus Collection")
        except Exception as e:
            self.logger.critical(f"Something went wrong. See error: {e}")

    def clear(self) -> None:
        """Delete all vectors in a collection. The fastest way to do this is by deleting and recreating the collection

        :return:
        """
        self.collection.drop()
        self.create()
        self.logger.info(f"All vectors have been deleted")

    async def upload(self, batch: list) -> None:
        """Method to upload vectors to Milvus

        :param batch: Batch to upload
        :return:
        """
        self.logger.info(f"Uploading {len(batch)} docs to Milvus")

        start_time = time.time()
        batch = self.preprocess(batch)

        self.collection.insert(batch)

        self.logger.info(f"Uploaded to Milvus in {time.time() - start_time}")

    def query(self, text: Optional[str], top_k: int = 5, include_metadata: bool = True,
              include_vectors: bool = False, postprocess: bool = False, pre_vectorized: bool = False):
        """Method to query Milvus database

        :param postprocess: Whether to process the results or not
        :param text: Text to query with
        :param top_k: How many results to return
        :param include_metadata: Whether to include payload within the response to the search
        :param include_vectors: Whether to include vectors within the response to the search
        :return:
        """
        output_fields = ["id"]
        if include_metadata:
            output_fields.append("metadata")
            output_fields.append("text")
        if include_vectors:
            output_fields.append("embedding")

        results = self.collection.search(
            data=[text if pre_vectorized else self.encode(text)],
            limit=top_k,
            output_fields=output_fields,
            anns_field="embedding",
            param={},
            expr=None
        )
        return self.postprocess(results[0]) if postprocess else results[0]

    def preprocess(self, batch):
        """Method to be used within the Milvus class in order to change the format of the batch to be uploaded

        :param batch: Batch to be uploaded
        :return:
        """
        ids = []
        vectors = []
        metadata = []
        text = []
        for doc in batch:
            ids.append(int(doc['id']))
            vectors.append(doc['values'])
            text.append(doc['metadata']['text'])
            del doc['metadata']['text']
            del doc['metadata']['created_at']
            metadata.append(doc['metadata'])
        self.logger.info("Milvus preprocessing complete")

        return [ids, vectors, metadata, text]

    def postprocess(self, query, include_metadata: bool = True, include_text: bool = True):
        """Method to change the format of the query results

        :param include_text: Whether to include the text of the document in the results
        :param include_metadata: Whether to include the metadata of the document in the results
        :param query: Query results
        :return:
        """
        results = []
        for doc in query:
            results.append({
                'id': doc.id,
                'score': doc.distance,
                'metadata': doc.entity.metadata if include_metadata else None,
                'text': doc.entity.text if include_text else None,
            })
        self.logger.info("Milvus postprocessing complete")
        return results