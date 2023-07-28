import re
import time
from typing import Optional

import pandas as pd
import torch
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index import Document
from sentence_transformers import SentenceTransformer


def get_file_metadata(filename: str) -> Optional[dict]:
    df = pd.read_csv("metadata.csv")
    filename = filename.replace(".pdf", "").replace("data", "").replace('\\', "")
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
model_name="all-MiniLM-L6-v2"
embeddings = SentenceTransformer(model_name_or_path="sentence-transformers/all-mpnet-base-v2", device=device)


# Recursive method to reduce junk \n and whitespace characters. Reduces all spaces to just one.
def clean_text(text):
    fixed = text.replace("  ", " ")
    # Recursive method to reduce junk whitespace
    if "  " in fixed:
        return clean_text(fixed)
    else:
        # Remove line skipping associated with PDF loading
        fixed = fixed.replace("\n", "").replace(".,", ".")

        # Remove unknown dashes related to PDF loading
        fixed = re.sub(r'\b(\w+)-(\w+)\b', r'\1\2', fixed)

        # Remove whitespace if text starts with it
        if fixed.startswith(" "):
            fixed = fixed[1:]
        return fixed


# Loads pdfs from directory and splits them into nodes
def load_pdfs(numPDFS=-1):
    starting_time = time.time()
    print("Loading PDFS...")
    if numPDFS < 0:
        loader = SimpleDirectoryReader(
            input_dir="./data/",
            #file_metadata=get_file_metadata,
        )
    else:
        loader = SimpleDirectoryReader(
            input_dir="./data/",
            #file_metadata=get_file_metadata,
            num_files_limit=numPDFS,
        )
    docs = loader.load_data()

    #return docs

    extracted_docs = []
    for doc in docs:
        extracted_docs.append(
            Document(
                text=doc.text,
                excluded_embed_metadata_keys=['start_char_idx', 'metadata_template', "document_id", 'filename'],
                excluded_llm_metadata_keys=['start_char_idx', 'metadata_template', "document_id", 'filename'],
                metadata={
                    'filename': doc.metadata['file_name'],
                },
            )
        )
        return extracted_docs
    print(extracted_docs)
    print("Done Loading PDFS")

    for i in range(len(docs)):
        docs[i].text = clean_text(docs[i].text)

    # These steps are required if we want the uploaded data to not have a bunch of junk
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=512,
        include_prev_next_rel=False,
        include_metadata=False,
    )

    nodes = node_parser.get_nodes_from_documents(docs)
    print(nodes)
    print(f"Parsing PDF's took {time.time() - starting_time}s total")
    return docs

print(load_pdfs(1))