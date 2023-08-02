import os
import time

import openai
from dotenv import find_dotenv, load_dotenv
from functions import load_pdfs, embeddings
from langchain.vectorstores import Milvus

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]
index_name = "pdf-flood"

def create_MilvusDB():
    starting_time = time.time()

    vector_db = Milvus.from_documents(
        load_pdfs(),
        embeddings,
        connection_args={
            "host": os.environ["MILVUS_URL"],
            "port": os.environ["MILVUS_URL_PORT"],
            "api_key": os.environ["MILVUS_API_KEY"]
        }
    )
    print(f"Embedding PDFS took {time.time() - starting_time}s total")

def search_milvus(query, k=5):
    results = {}
    return results


create_WeaviateDB()

while True:
    print(search_milvus(input("Prompt: "), 3))