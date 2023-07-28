import os

import openai
from dotenv import find_dotenv, load_dotenv
from llama_index.node_parser import SimpleNodeParser
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from llama_index.vector_stores import QdrantVectorStore
from llama_index import StorageContext, VectorStoreIndex
from llama_index.indices.service_context import ServiceContext, set_global_service_context
from functions import load_pdfs, embeddings


_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]


# service_context = ServiceContext.from_defaults(
#     embed_model="local:sentence-transformers/all-mpnet-base-v2",
#     llm_predictor=None,
#     llm=None
# )
# set_global_service_context(service_context)

def create_QdrantDB():
    client = QdrantClient(
        # location=":memory:",
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    index_name = "pdf-flood"
    # print(client.get_collections().collections[0])
    try:
        print("Initalizing new qdrant collection...")
        client.create_collection(
            collection_name=index_name,
            vectors_config=models.VectorParams(
                size=embeddings.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )
        print("Created new qdrant collection")
    except:
        print("Existing DB Found...")



    Qdrant_store = QdrantVectorStore(client=client, collection_name=index_name)
    #
    # print("Starting embedding and uploading process...")
    # index = VectorStoreIndex(
    #     nodes=load_pdfs(2),
    #     service_context=ServiceContext.from_defaults(
    #         embed_model="local:sentence-transformers/all-mpnet-base-v2"
    #     ),
    #     storage_context=StorageContext.from_defaults(vector_store=Qdrant_store),
    #     vector_store=Qdrant_store,
    #     show_progress=True
    # )

    # Create a StorageContext with the vector store
    storage_context = StorageContext.from_defaults(vector_store=Qdrant_store)

    # Create a VectorStoreIndex from the documents and the storage context
    index = VectorStoreIndex.from_documents(
        load_pdfs(1), storage_context=storage_context,
                                            service_context=ServiceContext.from_defaults(
             embed_model="local:sentence-transformers/all-mpnet-base-v2"),
                                            )

    while True:
        print(index.as_retriever().retrieve(input("Prompt: ")))


create_QdrantDB()
