import logging
import os
import time

import pinecone
from dotenv import load_dotenv, find_dotenv

from databases.database import Database


class PineconeDB(Database):
    """Pinecone Database Class for interacting with a defined Pinecone index
    Args:
        index_name (str): Name of Pinecone index
        api_key (str): API key for Pinecone server
        environment (str): Environment of Pinecone server
        pool_threads (int): Number of threads to use for Pinecone server
        default_namespace (str): Default namespace for Pinecone server
        ensure_exists (bool): Whether to create index if it does not exist
        GRPC (bool): Whether to use GRPC for Pinecone server
    """

    def __init__(self,
                 index_name: str = "pdf-flood",
                 api_key: str = None,
                 environment: str = 'us-west4-gcp-free',
                 pool_threads: int = 30,
                 default_namespace: str = "development",
                 ensure_exists: bool = True,
                 GRPC: bool = False
                 ) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.index_name = index_name
        self.default_namespace = default_namespace

        _ = load_dotenv(find_dotenv())
        if api_key is None:
            api_key = os.environ["PINECONE_API_KEY"]
        pinecone.init(
            api_key=api_key,
            environment=environment
        )
        if ensure_exists:
            self.create()

        if GRPC:
            self.index = pinecone.GRPCIndex(
                index_name=index_name
            )
        else:
            self.index = pinecone.Index(
                index_name=index_name,
                pool_threads=pool_threads
            )

    def create(self, metric: str = "cosine", replicas: int = 2):
        """
        Create new pinecone index if one does not exist

        Args:
            metric (str): Mathematical Algorithm for querying
            replicas (int): Numbers of replicas of each added vector
        """
        if self.index_name not in pinecone.list_indexes():
            self.logger.info("Initializing new pinecone index...")
            pinecone.create_index(
                name=self.index_name,
                dimension=self.model_dimensions,
                metric="cosine",
                replicas=2
            )

            self.logger.info("Created new pinecone index")

    def clear(self, namespace: str = None) -> None:
        """Delete all vectors in a namespace

        :param namespace: Namespace to clear vectors from
        :return:
        """
        if namespace is None:
            namespace = self.default_namespace
        self.index.delete(
            namespace=namespace,
            delete_all=True
        )
        self.logger.info(f"All vectors in namespace {namespace} have been deleted")

    async def upload(self, batch: list, namespace: str = None) -> None:
        """Method to upload vectors to pinecone

        :param batch: Batch to upload
        :param namespace: Namespace to upload vectors under
        :return:
        """
        self.logger.info(f"Uploading {len(batch)} docs to pinecone")
        start_time = time.time()
        if namespace is None:
            namespace = self.default_namespace
        self.index.upsert(
            vectors=batch,
            namespace=namespace,
            async_req=False,
            show_progress=False,
            batch_size=None  # What is the upper limit? Find the absolute max before the API rejects due to 2MB+ uploads
        )

        self.logger.info(f"Uploaded to pinecone in {time.time() - start_time}")

    def query(self, text: str, top_k: int = 5, include_values: bool = False, include_metadata: bool = True):
        result = self.index.query(
            vector=self.encode(text),
            top_k=top_k,
            namespace=self.default_namespace,
            include_metadata=include_metadata,
            include_values=include_values
        )
        return result
