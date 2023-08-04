import os
import time

import pinecone
import torch
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "all-mpnet-base-v2"
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
            metric="cosine"
        )

        print("Created new pinecone index")
    starting_time = time.time()

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
    for doc in docs:
        doc['values'] = embeddings.encode(doc['text'])
        doc['id'] = doc['metadata']['file_name']
        doc['metadata']['text'] = doc['text']
        del doc['text']

    with pinecone.Index(index_name, pool_threads=30) as index:
        index.upsert(
            vectors=docs,
            async_req=True
        )

    return
