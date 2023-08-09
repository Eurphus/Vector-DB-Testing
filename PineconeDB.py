import logging
import os
import time

import pinecone
import torch
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings = SentenceTransformer(model_name_or_path="sentence-transformers/all-mpnet-base-v2", device=device)

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
            metric="cosine",
            replicas=2
        )

        print("Created new pinecone index")
    starting_time = time.time()
    time.sleep(5)

    # pineconedb = Pinecone.from_documents(
    #     load_pdfs(),
    #     embeddings,
    #     index_name=index_name
    # )
    #print(f"Embedding PDFS took {time.time() - starting_time}s total")


def search_pinecone(query, k=5):
    index = pinecone.Index(index_name=index_name)
    return index.query(embeddings.embed_query(query), top_k=k, include_metadata=True)

def pinecone_upload(docs):
    create_pineconedb()
    with pinecone.Index(index_name, pool_threads=30) as index:
        index.upsert(
            vectors=docs,
            namespace="PDF_Testing",
            async_req=False,
            show_progress=False,
            batch_size=10
        )
