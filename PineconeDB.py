import asyncio
import logging
import os
import time

import pinecone
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment='us-west4-gcp-free'
)
index_name = "pdf-flood"
index = pinecone.Index(index_name, pool_threads=30)


def create_pineconedb():
    if index_name not in pinecone.list_indexes():
        print("Initalizing new pinecone index...")
        pinecone.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            replicas=2
        )

        print("Created new pinecone index")
        asyncio.sleep(2)

def clear_vectors():
    index.delete(
        namespace="PDF_Testing",
        delete_all=True
    )
    logging.info(f"All vectors in namespace {index_name} have been deleted")

def pinecone_upload(docs):
    logging.info(f"Uploading {len(docs)} docs to pinecone")
    start_time = time.time()
    create_pineconedb()
    with index as upload:
        upload.upsert(
            vectors=docs,
            namespace="PDF_Testing",
            async_req=False,
            show_progress=False,
            batch_size=None
        )
    logging.info(f"Uploaded to pinecone in {time.time()-start_time}")
