import os
import time

import pinecone
from dotenv import load_dotenv, find_dotenv
from functions import load_pdfs, embeddings
from langchain.vectorstores import Pinecone

_ = load_dotenv(find_dotenv())

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment='us-west4-gcp-free'
)
index_name = "pdf-flood"


def create_pineconedb():
    if index_name not in pinecone.list_indexes():
        print("Initalizing new pinecone index...")
        pinecone.create_index(
            name=index_name,
            dimension=768,
            metric="cosine"
        )

        print("Created new pinecone index")
    starting_time = time.time()

    pineconedb = Pinecone.from_documents(
        load_pdfs(),
        embeddings,
        index_name=index_name
    )
    print(f"Embedding PDFS took {time.time() - starting_time}s total")


def search_pinecone(query, k=5):
    index = pinecone.Index(index_name=index_name)
    return index.query(embeddings.embed_query(query), top_k=k, include_metadata=True)


create_pineconedb()
while True:
    print(search_pinecone(input("Prompt: "), 3))
