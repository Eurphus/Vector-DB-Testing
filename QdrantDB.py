import os
import time

import openai
from dotenv import find_dotenv, load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

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

def search_qdrant(query, k=5):
    results = client.search(
        collection_name=index_name,
        #query_vector=embeddings.embed_query(query),
        limit=k,
    )
    return results

def qdrant_upload(docs):
    vectors = []
    payload = []
    for doc in docs:
        vectors.append(doc['values'])
        payload.append(doc['metadata'])
    print(vectors[1])
    print(payload[1])
    client.upload_collection(

    )
