import logging
import time
import pinecone
from databases.database import Database
import os
from dotenv import load_dotenv, find_dotenv


class PineconeDB(Database):
    def __init__(self,
                 index_name: str = "pdf-flood",
                 api_key: str = None,
                 environment: str = 'us-west4-gcp-free',
                 pool_threads: int = 30,
                 default_namespace: str = "development",
                 ensure_exists: bool = False,
                 GRPC: bool = False
                 ) -> None:
        super().__init__()
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
            print("Initializing new pinecone index...")
            pinecone.create_index(
                name=self.index_name,
                dimension=self.model_dimensions,
                metric="cosine",
                replicas=2
            )

            print("Created new pinecone index")

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
        logging.info(f"All vectors in namespace {namespace} have been deleted")

    async def upload(self, batch: list, namespace: str = None) -> None:
        """Method to upload vectors to pinecone

        :param batch: Batch to upload
        :param namespace: Namespace to upload vectors under
        :return:
        """
        logging.info(f"Uploading {len(batch)} docs to pinecone")
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

        logging.info(f"Uploaded to pinecone in {time.time() - start_time}")

    def query(self, text: str, top_k: int = 5):
        result = self.index.query(
            vector=self.encode(text),
            top_k=top_k
        )
        return result
