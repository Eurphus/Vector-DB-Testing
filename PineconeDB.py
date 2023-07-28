import os

import pinecone
from dotenv import load_dotenv, find_dotenv
from llama_index.vector_stores import PineconeVectorStore
from llama_index import StorageContext, ServiceContext, VectorStoreIndex
from functions import load_pdfs, embeddings

_ = load_dotenv(find_dotenv())

def create_pineconedb():
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment='us-west4-gcp-free'
    )
    index_name = "pdf-flood"

    if index_name not in pinecone.list_indexes():
        print("Initalizing new pinecone index...")
        pinecone.create_index(
            name=index_name,
            dimension=embeddings.get_sentence_embedding_dimension(),
            metric="cosine"
        )
        print("Created new pinecone index")

    pinecone_index = pinecone.Index(index_name)
    pinecone_store = PineconeVectorStore(pinecone_index=pinecone_index, index_name=index_name)

    service_context = ServiceContext.from_defaults(
        embed_model="local:sentence-transformers/all-mpnet-base-v2"
    )

    print("Starting embedding and uploading process...")
    index = VectorStoreIndex.from_documents(
        documents=load_pdfs(),
        service_context=service_context,
        vector_store=pinecone_store,
        storage_context=StorageContext.from_defaults(vector_store=pinecone_store),
        show_progress=True
    )

    while True:
        print(index.as_retriever().retrieve(input("Prompt: ")))

create_pineconedb()
