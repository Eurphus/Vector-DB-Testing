import gc
import os
import re
import time
from typing import Optional, Iterable

import pandas as pd
import torch
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, PyPDFLoader, UnstructuredPDFLoader, \
    TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfFileReader
from sentence_transformers import SentenceTransformer

from PineconeDB import pinecone_upload
from QdrantDB import qdrant_upload


def get_file_metadata(filename: str) -> Optional[dict]:
    df = pd.read_csv("metadata.csv")
    filename = filename.replace(".pdf", "").replace("test", "").replace('\\', "")
    located = df.loc[df['digest'] == filename]
    if located.empty:
        return None
    metadata = {
        'file_name': filename,
        'size': str(located['file_size'].iloc[0]),  # Has to be str or int for some reason, was not working
        'created_at': located['date'].iloc[0][:10]
    }
    return metadata


# embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": device})


# Recursive method to reduce junk \n and whitespace characters. Reduces all spaces to just one.
def clean_text(text):
    fixed = text.replace("  ", " ")
    # Recursive method to reduce junk whitespace
    if "  " in fixed:
        return clean_text(fixed)
    else:
        # Remove line skipping associated with PDF loading
        fixed = fixed.replace("\n", "").replace(".,", ".")

        # Remove unknown chars related to PDF loading
        fixed = re.sub(r'\b(\w+)-(\w+)\b', r'\1\2', fixed)
        fixed = re.sub(r'\(cid:\d+\)', ' ', fixed)
        if "  " in fixed:
            return clean_text(fixed)

        # Remove whitespace if text starts with it
        if fixed.startswith(" "):
            fixed = fixed[1:]
        return fixed


# Loads pdfs from directory and splits them into nodes
def load_pdfs():
    starting_time = time.time()
    print("Loading PDFS...")
    loader = DirectoryLoader(
        path="./test/",
        loader_cls=UnstructuredPDFLoader,
        show_progress=True,
        use_multithreading=True
    ).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )

    docs = text_splitter.split_documents(loader)
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata = get_file_metadata(doc.metadata['source'])
    # print(docs)

    print(f"Parsing PDF's took {time.time() - starting_time}s total")
    return docs


def encode_docs(vectordb='Pinecone', batch_size=50, max_num=0):
    directory_path = os.getcwd() + "/data/"
    directory = os.listdir(directory_path)

    dir_length = len(directory)
    if max_num != 0:
        dir_length = max_num
    loops = 0
    while dir_length > loops * batch_size:
        loops += 1

        limit = batch_size * loops
        lang_documents: List[Document] = []
        if limit > dir_length:
            limit = dir_length

        for i in range((loops - 1) * batch_size, limit):
            filename = directory[i]
            if filename.endswith(".pdf") is not True:
                continue

            print(filename)
            try:
                loaded_pdf = UnstructuredPDFLoader(file_path=f"./data/{filename}").load()
            except:
                continue
            # print(loaded_pdf)
            loaded_pdf[0].metadata = get_file_metadata(filename)
            lang_documents.extend(loaded_pdf)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20
        )
        docs = text_splitter.split_documents(lang_documents)

        # Simplify process before querying
        final_docs = []
        for data in docs:
            doc = {
                'text': clean_text(data.page_content),
                'metadata': data.metadata,
            }
            final_docs.append(doc)

        if vectordb == 'Pinecone':
            pinecone_upload(final_docs)
        elif vectordb == 'Qdrant':
            qdrant_upload(final_docs)

    # Garbage Collection
    gc.collect()

encode_docs()
