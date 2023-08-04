import os
import time

import openai
from dotenv import find_dotenv, load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain.vectorstores import Qdrant


_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)
index_name = "pdf-flood"

def create_QdrantDB():
    try:
        print("Initalizing new qdrant collection...")
        client.create_collection(
            collection_name=index_name,
            vectors_config=models.VectorParams(
                size=768,
                distance=Distance.COSINE
            )
        )
        print("Created new qdrant collection")
    except:
        print("Existing DB Found...")

    starting_time = time.time()
    # qdrant = Qdrant.from_documents(
    #     load_pdfs(),
    #     embeddings,
    #     url=os.environ["QDRANT_URL"],
    #     api_key=os.environ["QDRANT_API_KEY"],
    #     prefer_grpc=True,
    #     collection_name=index_name
    # )
    print(f"Embedding PDFS took {time.time() - starting_time}s total")

def search_qdrant(query, k=5):
    results = client.search(
        collection_name=index_name,
        #query_vector=embeddings.embed_query(query),
        limit=k,
    )
    return results

def qdrant_upload(docs):
    print(docs[0])
