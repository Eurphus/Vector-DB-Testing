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

from PineconeDB import pinecone_upload, clear_vectors

# Current Run config
device_override = 'cuda'

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
    filename=logging_folder + f"{date_F}.txt"
)

if device_override is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = device_override

# Define a embedding model for each worker to prevent re-defining them constantly
# Can't use a single one with using CUDA or else issues arise
# CPU can be defined once and not worried about, as it is thread-safe. The GPU counterpart is not.
if device == 'cuda':
    embeddings = []
    for i in range(max_workers_num):
        embeddings.append(SentenceTransformer(model_name_or_path=model_name, device=device))
else:
    embeddings = SentenceTransformer(model_name_or_path=model_name, device=device)

clear_vectors()

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
        'size': str(located['file_size'].iloc[0]), # Has to be str or int for some reason, was not working before. # Size is in bytes
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
                logging.warning(f"PDF with filename {filename} did not detect any text. Deleting")
                os.remove(file_path)
                return {}
            text = clean_text(text)

            # Cut down text from PDF into reasonable chunks
            chunks = []
            start_idx = 0
            count = 0
            while start_idx < len(text):
                count+=1
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
                    'id': f"{filename}-{count}",
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
        return {}


def encode_docs(source, batch_size, max_num=0):
    logging.info(f"""Using device {device} for embedding 
                 Using {max_workers_num} workers
                 Using batches of {batch_size} with a max of {max_num} files
                 Model Name={model_name}, data directory={data_directory}, max chunk size={max_chunk_size}, overlap={overlap}""")
    starting_time = time.time()

    # Get current path
    directory_path = os.getcwd() + f"/{data_directory}/"
    directory = os.listdir(directory_path)

    dir_length = len(directory)

    if max_num != 0 and dir_length > max_num:
        dir_length = max_num

    pbar = tqdm(total=dir_length, desc=f"Encoding files")

    logging.info(f"Encoding documents...")
    batched_files = directory[:dir_length]
    batched = []

    # Allow the use of a single chunk of code for encoding for both types of encoding
    encoder_func = lambda i : embeddings if device == 'cpu' else embeddings[i%max_workers_num]

    # Multi-threading processing of PDFs
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_num) as executor:
        futures = [executor.submit(process_pdf, filename=filename, encoder=encoder_func(i)) for i, filename in
                   enumerate(batched_files)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None or result is []:
                continue
            batched.extend(result)
            while len(batched) >= batch_size:
                try:
                    if source == 'Pinecone':
                        pinecone_upload(batched[:batch_size])
                    elif source == 'JSON':
                        convertJSON(batch=batched[:batch_size])
                    batched = batched[batch_size:]
                    gc.collect()
                except Exception as e:
                    logging.warning(f"Failed to upload. \nError:{e}")
            pbar.update(1)
    try:
        if source == 'Pinecone':
            pinecone_upload(batched)
        elif source == 'JSON':
            convertJSON(batch=batched)
        batched = []
        gc.collect()
    except Exception as e:
        logging.warning(f"Failed to upload. \nError:{e}")

    logging.info(f"Encoded {dir_length} documents in {time.time() - starting_time} seconds")
    pbar.close()


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
