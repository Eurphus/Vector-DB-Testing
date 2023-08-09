import asyncio
import concurrent.futures
import datetime
import gc
import json
import logging
import math
import os
import re
import time
from typing import Optional

import PyPDF2 as PyPDF2
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from PineconeDB import pinecone_upload
# Current Run config
current_source = "JSON"
batch_size = 20
max_num = 100
device_override = 'cpu'

# List of configurable options
model_name = "all-mpnet-base-v2"
data_directory = "data"
metadata_file = "metadata.csv"
JSON_file = "bigdata.json"
max_workers_num = min(32, (os.cpu_count() or 1))
max_chunk_size = 800  # Maximum chunk size
overlap = 20  # Allowed overlap between chunks
logging_folder = os.getcwd() + "/logs/"
date_F = datetime.datetime.now().strftime("%d.%b_%Y_%H.%M.%S")

logging.basicConfig(
    level=logging.INFO,
    filename=logging_folder + f"{date_F}.txt",
    # datefmt='%Y-%m-%d_%H:%M:%S'
)

if device_override is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = device_override
logging.info(f"Using device {device} for embedding "
             f"Using {max_workers_num} workers"
             f"Using batches of {batch_size} with a max of {max_num} files"
             f"Model Name={model_name}, data directory={data_directory}, max chunk size={max_chunk_size}, overlap={overlap}")
# Define a embedding model for each worker to prevent re-defining them constantly
# Can't use a single one with using CUDA or else issues arise
# embeddings = SentenceTransformer(model_name_or_path=model_name, device=device)#[]
# encoder = embeddings
embeddings = []
for i in range(max_workers_num):
    embeddings.append(SentenceTransformer(model_name_or_path=model_name, device=device))

def get_file_metadata(filename: str) -> Optional[dict]:
    # Scan metadata.csv for associated data
    df = pd.read_csv(metadata_file)
    filename = filename.replace(".pdf", "").replace(data_directory, "").replace('\\', "")
    located = df.loc[df['digest'] == filename]
    if located.empty:
        logging.error(f"Could not find {filename} in metadata.csv")
        return {}
    return {
        'file_name': filename,
        'size': str(located['file_size'].iloc[0]),
        # Has to be str or int for some reason, was not working before. # Size is in bytes
        'created_at': located['date'].iloc[0][:10]
    }


# Recursive method to reduce junk \n and whitespace characters. Reduces all spaces to just one.
def clean_text(text):
    fixed = text.replace("  ", " ")
    # Recursive method to reduce junk whitespace
    if "  " in fixed:
        return clean_text(fixed)
    else:
        # Remove line skipping associated with PDF loading
        fixed = fixed.replace("\n", "").replace(".,", ".").replace("\t", "")

        # Remove hypens between two words where it should not exist due to PDF loading
        fixed = re.sub(r'\b(\w+)-(\w+)\b', r'\1\2', fixed)

        # Forgot what this one removes
        fixed = re.sub(r'\(cid:\d+\)', ' ', fixed)
        if "  " in fixed:
            return clean_text(fixed)

        # Remove starting whitespace
        if fixed.startswith(" "):
            fixed = fixed[1:]
        return fixed


def process_pdf(filename, encoder):
    encoder = encoder%max_workers_num
    logging.info(f"Using encoder #{encoder}")
    encoder = embeddings[encoder]
    text = ""
    directory_path = os.getcwd() + f"/{data_directory}/"
    file_path = directory_path + filename

    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Extract metadata
            metadata = get_file_metadata(filename)

            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

            if len(text) == 0:
                logging.warning(f"PDF with filename {filename} did not detect any text. Consider deletion")
                return {}
            text = clean_text(text)

            # Cut down text from PDF into reasonable chunks
            chunks = []
            start_idx = 0

            while start_idx < len(text):
                end_idx = start_idx + max_chunk_size
                if end_idx >= len(text):
                    tmp = text[start_idx:]
                else:
                    tmp = text[start_idx:end_idx]
                try:
                    vectors = encoder.encode(tmp, show_progress_bar=False).tolist()
                except Exception as e:
                    logging.warning(f"An error occured while embedding. File {filename} and end {end_idx}: {e}")
                    continue

                chunks.append({
                    'id': filename,
                    'metadata': {
                        'file_name': filename,
                        'size': str(metadata['size']),  # Has to be str or int for some reason, was not working
                        'created_at': metadata['created_at'],
                        'text': tmp
                    },
                    'values': vectors
                })
                start_idx = end_idx - overlap
            return chunks
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return []


def encode_docs(source=None, batch_size=50, max_num=0):
    starting_time = time.time()

    # Get current path
    directory_path = os.getcwd() + f"/{data_directory}/"
    directory = os.listdir(directory_path)

    dir_length = len(directory)

    if max_num != 0:
        dir_length=max_num

    loops = 0
    pbar = tqdm(total=math.ceil(dir_length / batch_size), position=1, desc=f"Loading chunks of {batch_size}")
    logging.info(f"Encoding documents...")
    all_docs = []

    # While loop which will loop through the directory in batches of batch_size
    while dir_length > loops * batch_size:
        loops += 1

        limit = batch_size * loops
        batched = []
        if limit > dir_length:
            limit = dir_length

        # Returns the files only in the current batch
        batched_files = directory[(loops - 1) * batch_size:limit]


        #Multi-threading processing of PDFs
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_num) as executor:
            futures = [executor.submit(process_pdf, filename=filename, encoder=i) for i, filename in enumerate(batched_files)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                batched.extend(result)
        # for doc in batched_files:
        #     batched.extend(process_pdf(doc))

        # Upload to VectorDB based off of preference
        source_time = time.time()
        logging.info(f"Uploading batch {loops} to {source}...")
        try:
            if source == 'Pinecone':
                asyncio.run(pinecone_upload(batched))
            elif source == 'JSON':
                convertJSON(batch=batched)
            elif source == None:
                all_docs.append(batched)
            logging.info(f"Uploaded batch {loops} to {source} in {time.time() - source_time} seconds")
        except Exception as e:
            logging.warning(f"Batch failed to upload batch {loops}. \nError:{e}")

        # Clear memory
        batched = None
        gc.collect()
        pbar.update(1)

    logging.info(f"Encoded {dir_length} documents in {time.time() - starting_time} seconds")
    pbar.close()
    if source is None:
        return all_docs


def convertJSON(filename="\\bigdata.json", batch=None):
    if batch is None:
        batch = []
    directory_path = os.getcwd() + filename

    with open(directory_path, "w") as file:
        json.dump(batch, file)


def fromJSON(filename="\\bigdata.json", limit=None):
    directory_path = os.getcwd() + filename
    with open(directory_path, "w") as file:
        data = json.load(file)

    # Only grab a certain # of lines
    if limit is not None:
        data = data[:limit]
    return data
