import re
import time
from typing import Optional

import pandas as pd
import torch
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, PyPDFLoader, UnstructuredPDFLoader, \
    TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "all-mpnet-base-v2"
#embeddings = SentenceTransformer(model_name_or_path="sentence-transformers/all-mpnet-base-v2", device=device)
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": device})


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
    #print(docs)

    print(f"Parsing PDF's took {time.time() - starting_time}s total")
    return docs

